# -*- coding: utf-8 -*-

"""Fail-closed tests for immutable OCR alignment shard receipts."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from Model.ocr.alignment_contract import (
    build_ocr_alignment_data_contract,
    default_ocr_alignment_contract_path,
    load_and_validate_ocr_alignment_data_contract,
    validate_ocr_alignment_data_contract_payload,
    write_ocr_alignment_data_contract,
)


class OCRAlignmentContractTest(unittest.TestCase):
    @staticmethod
    def _row(image: Path, input_ids: list[int]) -> dict:
        image_bytes = image.read_bytes()
        return {
            "input_ids": input_ids,
            "images": [str(image.resolve())],
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "image_size_bytes": len(image_bytes),
        }

    def _fixture(self, root: Path) -> tuple[Path, Path, dict]:
        shards = root / "align"
        shards.mkdir()
        image_a = root / "a.png"
        image_b = root / "b.png"
        image_a.write_bytes(b"image-a")
        image_b.write_bytes(b"image-bb")
        (shards / "shard-00000.jsonl").write_text(
            json.dumps(self._row(image_a, [2, 3])) + "\n",
            encoding="utf-8",
        )
        (shards / "shard-00001.jsonl").write_text(
            json.dumps(self._row(image_b, [2, 4, 3])) + "\n",
            encoding="utf-8",
        )
        token_contract = {
            "target_encoding": "native",
            "tokenization_contract_version": 3,
            "tokenizer_manifest_canonical_sha256": "a" * 64,
        }
        receipt = shards / "ocr_data_contract.json"
        write_ocr_alignment_data_contract(
            receipt,
            build_ocr_alignment_data_contract(shards, token_contract),
        )
        return shards, receipt, token_contract

    def test_sharded_receipt_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shards, receipt, token_contract = self._fixture(Path(tmp))
            validated = load_and_validate_ocr_alignment_data_contract(
                receipt,
                shards,
                token_contract,
            )
            self.assertEqual(validated["data_layout"], "sharded_jsonl")
            self.assertEqual(validated["data_file_count"], 2)
            self.assertEqual(validated["image_reference_count"], 2)
            self.assertEqual(validated["image_total_size_bytes"], 15)
            self.assertEqual(
                default_ocr_alignment_contract_path(shards),
                receipt,
            )

    def test_mutated_added_or_removed_shard_is_rejected(self) -> None:
        for name in ("mutated", "added", "removed"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                shards, receipt, token_contract = self._fixture(Path(tmp))
                if name == "mutated":
                    row = json.loads(
                        (shards / "shard-00000.jsonl").read_text(
                            encoding="utf-8"
                        )
                    )
                    row["image_sha256"] = "f" * 64
                    (shards / "shard-00000.jsonl").write_text(
                        json.dumps(row) + "\n",
                        encoding="utf-8",
                    )
                elif name == "added":
                    image = Path(tmp) / "c.png"
                    image.write_bytes(b"image-c")
                    (shards / "shard-00002.jsonl").write_text(
                        json.dumps(self._row(image, [2, 3])) + "\n",
                        encoding="utf-8",
                    )
                else:
                    (shards / "shard-00001.jsonl").unlink()
                with self.assertRaisesRegex(
                    ValueError,
                    "resolved data shards differ",
                ):
                    load_and_validate_ocr_alignment_data_contract(
                        receipt,
                        shards,
                        token_contract,
                    )

    def test_summary_only_legacy_metadata_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "schema_version differs",
        ):
            validate_ocr_alignment_data_contract_payload(
                {
                    "streaming": {
                        "corpus_manifest_sha256": "f" * 64,
                    }
                },
                {
                    "target_encoding": "native",
                    "tokenization_contract_version": 3,
                },
            )

    def test_legacy_jsonl_without_image_binding_cannot_be_certified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "legacy.jsonl"
            data.write_text('{"input_ids":[2,3]}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "image_sha256|images"):
                build_ocr_alignment_data_contract(
                    data,
                    {
                        "target_encoding": "native",
                        "tokenization_contract_version": 3,
                    },
                )

    def test_glob_requires_explicit_receipt_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit"):
            default_ocr_alignment_contract_path("align/shard-*.jsonl")


if __name__ == "__main__":
    unittest.main()
