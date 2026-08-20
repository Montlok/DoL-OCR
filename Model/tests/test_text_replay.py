# -*- coding: utf-8 -*-

"""Focused safety and supervision tests for traditional-Mongolian CE replay."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from Model.ocr.position_contract import BOUNDARY_V1
from Model.posttrain.text_replay import (
    TEXT_REPLAY_PUBLIC_SPLITS,
    TextReplayCollator,
    TextReplayDataset,
    TextReplayPartition,
    canonical_text_replay_contract_sha256,
    text_replay_ngram_sha256,
    validate_text_replay_partition_contract,
)


@dataclass
class _Encoded:
    input_ids: list[int]


class _NativeEncoder:
    mode = "native"

    def __init__(
        self,
        *,
        fallback_text: str | None = None,
        canonicalized_text: str | None = None,
        fail_text: str | None = None,
    ) -> None:
        self.stats = {"native": 0, "byte_fallback": 0, "canonicalized": 0}
        self.fallback_text = fallback_text
        self.canonicalized_text = canonicalized_text
        self.fail_text = fail_text

    def encode_with_features(self, text: str) -> _Encoded:
        if text == self.fail_text:
            raise ValueError("round-trip mismatch")
        if text == self.fallback_text:
            self.stats["byte_fallback"] += 1
        if text == self.canonicalized_text:
            self.stats["canonicalized"] += 1
        self.stats["native"] += 1
        return _Encoded([1000 + (ord(character) % 500) for character in text])


def _row(
    sample_id: str,
    document_id: str,
    text: str,
    *,
    split: str = "train",
) -> dict[str, object]:
    try:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        digest = "0" * 64
    return {
        "schema_version": 1,
        "id": sample_id,
        "document_id": document_id,
        "source": "reviewed-corpus",
        "text": text,
        "utf8_sha256": digest,
        "split": split,
    }


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _dataset(path: Path, **kwargs) -> TextReplayDataset:
    return TextReplayDataset(
        path,
        split="train",
        native_encoder=kwargs.pop("native_encoder", _NativeEncoder()),
        bos_id=1,
        eos_id=2,
        max_seq_len=kwargs.pop("max_seq_len", 64),
        exclusion_document_ids=kwargs.pop("exclusion_document_ids", set()),
        exclusion_ngram_sha256=kwargs.pop("exclusion_ngram_sha256", set()),
        **kwargs,
    )


class TextReplayDatasetTest(unittest.TestCase):
    def test_full_supervision_contract_hashes_stats_and_boundary_collation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replay.jsonl"
            rows = [
                _row("train-1", "doc-a", "ᠮᠣᠩ"),
                _row("train-2", "doc-a", "ᠪᠢᠴᠢᠭ"),
                _row("val-1", "doc-b", "ᠨᠣᠮ", split="sft_validation"),
            ]
            _write(path, rows)
            dataset = _dataset(path)

            self.assertEqual(len(dataset), 2)
            first = dataset[0]
            self.assertEqual(first["input_ids"][0], 1)
            self.assertEqual(first["input_ids"][-1], 2)
            self.assertEqual(first["labels"], first["input_ids"])
            self.assertEqual(first["attention_mask"], [1] * len(first["input_ids"]))
            self.assertEqual(
                first["metadata"]["text_utf8_sha256"],
                rows[0]["utf8_sha256"],
            )
            self.assertEqual(
                first["metadata"]["document_sha256"],
                dataset.document_sha256["doc-a"],
            )
            self.assertEqual(dataset.token_stats["samples"], 2)
            self.assertEqual(dataset.token_stats["documents"], 1)
            self.assertEqual(dataset.token_stats["total_content_tokens"], 8)

            contract = dataset.dataset_contract
            self.assertEqual(contract["contract_sha256"], dataset.contract_sha256)
            self.assertEqual(
                canonical_text_replay_contract_sha256(contract),
                dataset.contract_sha256,
            )
            self.assertEqual(len(contract["all_rows_canonical_sha256"]), 64)
            self.assertEqual(len(contract["selected_rows_canonical_sha256"]), 64)
            self.assertEqual(set(dataset.text_sha256), {"train-1", "train-2"})

            batch = TextReplayCollator(pad_id=0)([dataset[0], dataset[1]])
            self.assertEqual(batch["position_contract"], BOUNDARY_V1)
            self.assertEqual(
                batch["dataset_contract_sha256"],
                dataset.contract_sha256,
            )
            self.assertNotIn("word_pos", batch)
            self.assertNotIn("morph_depth", batch)
            self.assertEqual(tuple(batch["input_ids"].shape), (2, 7))
            self.assertEqual(batch["attention_mask"][0, -2:].tolist(), [0, 0])
            self.assertEqual(batch["labels"][0, -2:].tolist(), [-100, -100])

    def test_exact_schema_unicode_hash_duplicates_and_split_leakage_are_rejected(self):
        cases: list[tuple[list[dict[str, object]], str]] = []
        extra = _row("a", "doc-a", "ᠠ")
        extra["extra"] = 1
        cases.append(([extra], "row fields must be exact"))
        boolean_schema = _row("a", "doc-a", "ᠠ")
        boolean_schema["schema_version"] = True
        cases.append(([boolean_schema], "schema_version must be 1"))
        wrong_hash = _row("a", "doc-a", "ᠠ")
        wrong_hash["utf8_sha256"] = "0" * 64
        cases.append(([wrong_hash], "does not match"))
        cases.extend(
            [
                ([_row("a", "doc-a", "x\x00y")], "NUL"),
                ([_row("a", "doc-a", "x\ufffdy")], r"U\+FFFD"),
                ([_row("a", "doc-a", "x\ud800y")], "surrogate"),
                (
                    [_row("a", "doc-a", "ᠠ", split="validation")],
                    "split must be one of",
                ),
                (
                    [_row("a", "doc-a", "ᠠ"), _row("a", "doc-b", "ᠡ")],
                    "duplicate text replay id",
                ),
                (
                    [
                        _row("a", "doc-a", "ᠠ", split="train"),
                        _row("b", "doc-a", "ᠡ", split="sft_validation"),
                    ],
                    "crosses splits",
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            for index, (rows, error) in enumerate(cases):
                with self.subTest(error=error):
                    path = Path(tmp) / f"bad-{index}.jsonl"
                    _write(path, rows)
                    with self.assertRaisesRegex(ValueError, error):
                        _dataset(path)

    def test_reviewed_exclusions_native_only_and_overlength_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replay.jsonl"
            text = "ᠮᠣᠩᠭᠣᠯ ᠪᠢᠴᠢᠭ"
            _write(path, [_row("a", "doc-a", text)])

            with self.assertRaisesRegex(ValueError, "reviewed exclusion"):
                _dataset(path, exclusion_document_ids={"doc-a"})
            exclusion_hash = next(iter(text_replay_ngram_sha256(text)))
            with self.assertRaisesRegex(ValueError, "exclusion n-gram"):
                _dataset(path, exclusion_ngram_sha256={exclusion_hash})

            non_native = _NativeEncoder()
            non_native.mode = "native_fallback"
            with self.assertRaisesRegex(ValueError, "mode must be native"):
                _dataset(path, native_encoder=non_native)
            with self.assertRaisesRegex(ValueError, "used fallback"):
                _dataset(path, native_encoder=_NativeEncoder(fallback_text=text))
            with self.assertRaisesRegex(ValueError, "changed text"):
                _dataset(
                    path,
                    native_encoder=_NativeEncoder(canonicalized_text=text),
                )
            with self.assertRaisesRegex(ValueError, "round-trip failed"):
                _dataset(path, native_encoder=_NativeEncoder(fail_text=text))
            with self.assertRaisesRegex(ValueError, "segment the document upstream"):
                _dataset(path, max_seq_len=5)

    def test_four_way_partition_contract_and_global_leakage_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "partition.jsonl"
            rows = [
                _row("train-1", "doc-train", "ᠠᠪ", split="train"),
                _row(
                    "sft-1",
                    "doc-sft",
                    "ᠡᠢ",
                    split="sft_validation",
                ),
                _row(
                    "kl-1",
                    "doc-kl",
                    "ᠣᠤ",
                    split="kl_selection",
                ),
                _row(
                    "formal-1",
                    "doc-formal",
                    "ᠪᠦ",
                    split="formal_monitor",
                ),
            ]
            _write(path, rows)
            partition = TextReplayPartition(
                path,
                native_encoder=_NativeEncoder(),
                bos_id=1,
                eos_id=2,
                max_seq_len=64,
                exclusion_document_ids=set(),
                exclusion_ngram_sha256=set(),
            )

            self.assertEqual(set(partition.datasets), set(TEXT_REPLAY_PUBLIC_SPLITS))
            contracts = {
                split: partition.dataset(split).contract_sha256
                for split in TEXT_REPLAY_PUBLIC_SPLITS
            }
            self.assertEqual(len(set(contracts.values())), 4)
            self.assertEqual(
                partition.partition_contract["split_contract_sha256"],
                contracts,
            )
            self.assertTrue(
                partition.partition_contract["leakage_contract"][
                    "global_cross_split_leakage_validated"
                ]
            )
            self.assertEqual(
                validate_text_replay_partition_contract(
                    partition.partition_contract
                ),
                partition.partition_contract,
            )
            tampered = partition.partition_contract
            tampered["split_contract_sha256"]["formal_monitor"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "contract_sha256 differs"):
                validate_text_replay_partition_contract(tampered)

            _write(path, rows[:-1])
            with self.assertRaisesRegex(ValueError, "empty required splits"):
                TextReplayPartition(
                    path,
                    native_encoder=_NativeEncoder(),
                    bos_id=1,
                    eos_id=2,
                    max_seq_len=64,
                    exclusion_document_ids=set(),
                )

    def test_partition_rejects_exact_and_ngram_cross_split_leakage(self):
        base_rows = [
            _row("train-1", "doc-train", "ᠠᠪ", split="train"),
            _row("sft-1", "doc-sft", "ᠡᠢ", split="sft_validation"),
            _row("kl-1", "doc-kl", "ᠣᠤ", split="kl_selection"),
            _row("formal-1", "doc-formal", "ᠪᠦ", split="formal_monitor"),
        ]

        def admit(path: Path) -> TextReplayPartition:
            return TextReplayPartition(
                path,
                native_encoder=_NativeEncoder(),
                bos_id=1,
                eos_id=2,
                max_seq_len=128,
                exclusion_document_ids=set(),
            )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "partition.jsonl"
            exact_rows = list(base_rows)
            exact_rows[2] = _row(
                "kl-1",
                "doc-kl",
                "ᠡᠢ",
                split="kl_selection",
            )
            _write(path, exact_rows)
            with self.assertRaisesRegex(ValueError, "exact content crosses splits"):
                admit(path)

            ngram_rows = list(base_rows)
            ngram_rows[0] = _row(
                "train-1",
                "doc-train",
                "abcdefghijklmnX",
                split="train",
            )
            ngram_rows[1] = _row(
                "sft-1",
                "doc-sft",
                "Zabcdefghijklmn",
                split="sft_validation",
            )
            _write(path, ngram_rows)
            with self.assertRaisesRegex(ValueError, "content window crosses splits"):
                admit(path)


if __name__ == "__main__":
    unittest.main()
