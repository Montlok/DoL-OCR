# -*- coding: utf-8 -*-

"""Strict traditional-Mongolian text CE replay data contract."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from Model.config import IGNORE_INDEX
from Model.ocr.position_contract import BOUNDARY_V1
from Model.posttrain.ocr_manifest_builder import file_sha256


TEXT_REPLAY_SCHEMA_VERSION = 1
TEXT_REPLAY_CONTRACT_VERSION = 1
TEXT_REPLAY_PARTITION_CONTRACT_VERSION = 1
TEXT_REPLAY_EXCLUSION_NGRAM_CODEPOINTS = 13
TEXT_REPLAY_PUBLIC_SPLITS = (
    "train",
    "sft_validation",
    "kl_selection",
    "formal_monitor",
)
TEXT_REPLAY_VALIDATION_SPLITS = TEXT_REPLAY_PUBLIC_SPLITS[1:]

_ROW_FIELDS = frozenset(
    {
        "schema_version",
        "id",
        "document_id",
        "source",
        "text",
        "utf8_sha256",
        "split",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PARTITION_CONTRACT_FIELDS = frozenset(
    {
        "contract_version",
        "kind",
        "public_splits",
        "manifest_file_sha256",
        "all_rows_canonical_sha256",
        "split_contract_sha256",
        "split_selected_rows_canonical_sha256",
        "split_document_sha256_canonical_sha256",
        "split_text_sha256_canonical_sha256",
        "split_token_stats",
        "shared_contract",
        "leakage_contract",
        "contract_sha256",
    }
)


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def text_replay_ngram_sha256(text: str) -> frozenset[str]:
    """Hash the full text and every fixed 13-code-point contamination window."""

    _validate_text(text, "text")
    digests = {_utf8_sha256(text)}
    width = TEXT_REPLAY_EXCLUSION_NGRAM_CODEPOINTS
    if len(text) >= width:
        digests.update(
            _utf8_sha256(text[start:start + width])
            for start in range(len(text) - width + 1)
        )
    return frozenset(digests)


def canonical_text_replay_contract_sha256(contract: Mapping[str, object]) -> str:
    """Recompute a dataset contract hash without trusting its embedded digest."""

    if not isinstance(contract, Mapping):
        raise TypeError("text replay contract must be a mapping")
    payload = dict(contract)
    embedded = payload.pop("contract_sha256", None)
    digest = canonical_json_sha256(payload)
    if embedded is not None and embedded != digest:
        raise ValueError("text replay contract_sha256 is invalid")
    return digest


def validate_text_replay_partition_contract(
    contract: Mapping[str, object],
) -> dict[str, object]:
    """Validate a persisted four-way partition contract and its canonical hash."""

    if not isinstance(contract, Mapping) or set(contract) != set(
        _PARTITION_CONTRACT_FIELDS
    ):
        raise ValueError("text replay partition contract fields differ")
    normalized = copy.deepcopy(dict(contract))
    if normalized["contract_version"] != TEXT_REPLAY_PARTITION_CONTRACT_VERSION:
        raise ValueError("unsupported text replay partition contract version")
    if normalized["kind"] != "dol_text_ce_replay_partition":
        raise ValueError("unsupported text replay partition contract kind")
    if normalized["public_splits"] != list(TEXT_REPLAY_PUBLIC_SPLITS):
        raise ValueError("text replay partition public splits differ")
    for field in ("manifest_file_sha256", "all_rows_canonical_sha256"):
        if (
            not isinstance(normalized[field], str)
            or _SHA256_RE.fullmatch(normalized[field]) is None
        ):
            raise ValueError(f"text replay partition {field} is invalid")
    for field in (
        "split_contract_sha256",
        "split_selected_rows_canonical_sha256",
        "split_document_sha256_canonical_sha256",
        "split_text_sha256_canonical_sha256",
        "split_token_stats",
    ):
        value = normalized[field]
        if not isinstance(value, Mapping) or set(value) != set(
            TEXT_REPLAY_PUBLIC_SPLITS
        ):
            raise ValueError(f"text replay partition {field} splits differ")
    for field in (
        "split_contract_sha256",
        "split_selected_rows_canonical_sha256",
        "split_document_sha256_canonical_sha256",
        "split_text_sha256_canonical_sha256",
    ):
        if any(
            not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None
            for value in normalized[field].values()
        ):
            raise ValueError(f"text replay partition {field} contains invalid SHA")
    if len(set(normalized["split_contract_sha256"].values())) != len(
        TEXT_REPLAY_PUBLIC_SPLITS
    ):
        raise ValueError("text replay split contract SHA values must be distinct")
    if not isinstance(normalized["shared_contract"], Mapping):
        raise ValueError("text replay partition shared contract is invalid")
    leakage = normalized["leakage_contract"]
    if not isinstance(leakage, Mapping) or leakage != {
        "document_id_cross_split": "forbidden",
        "exact_text_sha256_cross_split": "forbidden",
        "ngram_sha256_cross_split": "forbidden",
        "ngram_codepoints": TEXT_REPLAY_EXCLUSION_NGRAM_CODEPOINTS,
        "global_cross_split_leakage_validated": True,
    }:
        raise ValueError("text replay partition leakage contract differs")
    embedded = normalized.pop("contract_sha256")
    if not isinstance(embedded, str) or _SHA256_RE.fullmatch(embedded) is None:
        raise ValueError("text replay partition contract_sha256 is invalid")
    if canonical_json_sha256(normalized) != embedded:
        raise ValueError("text replay partition contract_sha256 differs")
    return copy.deepcopy(dict(contract))


class TextReplayDataset(Dataset):
    """Load one exact-schema JSONL corpus through the strict native encoder."""

    def __init__(
        self,
        path: str | Path,
        *,
        split: str,
        native_encoder,
        bos_id: int,
        eos_id: int,
        max_seq_len: int,
        exclusion_document_ids: Collection[str],
        exclusion_ngram_sha256: Collection[str] | None = None,
    ) -> None:
        source_path = Path(path)
        if source_path.is_symlink():
            raise ValueError("text replay JSONL must not be a symlink")
        split = _validate_identifier(split, "split")
        if split not in TEXT_REPLAY_PUBLIC_SPLITS:
            raise ValueError(
                "text replay split must be one of "
                + ", ".join(TEXT_REPLAY_PUBLIC_SPLITS)
            )
        _validate_token_id("bos_id", bos_id)
        _validate_token_id("eos_id", eos_id)
        if bos_id == eos_id:
            raise ValueError("bos_id and eos_id must differ")
        if type(max_seq_len) is not int or max_seq_len < 3:
            raise ValueError("max_seq_len must be an integer of at least 3")
        _validate_native_encoder(native_encoder)

        excluded_documents = _validated_identifier_set(
            exclusion_document_ids,
            "exclusion_document_ids",
        )
        excluded_ngrams = _validated_sha256_set(
            exclusion_ngram_sha256 or (),
            "exclusion_ngram_sha256",
        )

        all_rows = _load_rows(source_path)
        seen_ids: set[str] = set()
        document_splits: dict[str, str] = {}
        for row_index, row in enumerate(all_rows):
            where = f"rows[{row_index}]"
            sample_id = str(row["id"])
            document_id = str(row["document_id"])
            row_split = str(row["split"])
            if sample_id in seen_ids:
                raise ValueError(f"duplicate text replay id {sample_id!r}")
            seen_ids.add(sample_id)
            previous_split = document_splits.get(document_id)
            if previous_split is not None and previous_split != row_split:
                raise ValueError(
                    f"document {document_id!r} crosses splits: "
                    f"{previous_split!r} vs {row_split!r}"
                )
            document_splits[document_id] = row_split
            if document_id in excluded_documents:
                raise ValueError(
                    f"{where} document_id {document_id!r} hits reviewed exclusion"
                )
            if excluded_ngrams:
                overlap = text_replay_ngram_sha256(str(row["text"])) & excluded_ngrams
                if overlap:
                    raise ValueError(
                        f"{where} text hits reviewed exclusion n-gram SHA-256 "
                        f"{min(overlap)}"
                    )

        selected = [row for row in all_rows if row["split"] == split]
        if not selected:
            raise ValueError(f"text replay split {split!r} is empty")

        examples: list[dict[str, Any]] = []
        encoded_rows: list[tuple[dict[str, Any], list[int]]] = []
        for row_index, row in enumerate(selected):
            text = str(row["text"])
            token_ids = _strict_native_encode(
                native_encoder,
                text,
                where=f"selected[{row_index}]",
            )
            if bos_id in token_ids or eos_id in token_ids:
                raise ValueError(
                    f"selected[{row_index}] native content contains structural "
                    "BOS/EOS token IDs"
                )
            input_ids = [bos_id, *token_ids, eos_id]
            if len(input_ids) > max_seq_len:
                raise ValueError(
                    f"selected[{row_index}] sequence length {len(input_ids)} "
                    f"exceeds max_seq_len={max_seq_len}; segment the document "
                    "upstream instead of truncating"
                )
            encoded_rows.append((row, token_ids))
            examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": list(input_ids),
                    "metadata": {
                        "id": row["id"],
                        "document_id": row["document_id"],
                        "source": row["source"],
                        "split": row["split"],
                        "text_utf8_sha256": row["utf8_sha256"],
                        "content_tokens": len(token_ids),
                        "sequence_tokens": len(input_ids),
                    },
                }
            )

        document_hashes = _document_hashes(selected)
        for example in examples:
            document_id = str(example["metadata"]["document_id"])
            example["metadata"]["document_sha256"] = document_hashes[document_id]

        content_lengths = [len(token_ids) for _, token_ids in encoded_rows]
        sequence_lengths = [length + 2 for length in content_lengths]
        token_stats = {
            "samples": len(examples),
            "documents": len(document_hashes),
            "total_content_tokens": sum(content_lengths),
            "total_sequence_tokens": sum(sequence_lengths),
            "min_content_tokens": min(content_lengths),
            "max_content_tokens": max(content_lengths),
            "mean_content_tokens": {
                "numerator": sum(content_lengths),
                "denominator": len(content_lengths),
            },
        }
        canonical_all = [_canonical_row(row) for row in all_rows]
        canonical_selected = [_canonical_row(row) for row in selected]
        contract_base: dict[str, object] = {
            "contract_version": TEXT_REPLAY_CONTRACT_VERSION,
            "kind": "dol_text_ce_replay_dataset",
            "row_schema_version": TEXT_REPLAY_SCHEMA_VERSION,
            "split": split,
            "all_rows_canonical_sha256": canonical_json_sha256(canonical_all),
            "selected_rows_canonical_sha256": canonical_json_sha256(
                canonical_selected
            ),
            "selected_row_ids": [str(row["id"]) for row in selected],
            "document_sha256": [
                {
                    "document_id": document_id,
                    "sha256": document_hashes[document_id],
                }
                for document_id in sorted(document_hashes)
            ],
            "text_sha256": [
                {"id": str(row["id"]), "sha256": str(row["utf8_sha256"])}
                for row in selected
            ],
            "token_stats": token_stats,
            "sequence_contract": {
                "bos_id": bos_id,
                "eos_id": eos_id,
                "max_seq_len": max_seq_len,
                "labels": "all_sequence_tokens",
                "overlength": "fail_upstream_segmentation_required",
                "position_contract": BOUNDARY_V1,
            },
            "native_encoder_contract": {
                "mode": "native",
                "fallback": "forbidden",
                "canonicalization": "forbidden",
                "roundtrip": "encoder_verified_exact",
            },
            "exclusion_contract": {
                "document_ids_canonical_sha256": canonical_json_sha256(
                    sorted(excluded_documents)
                ),
                "ngram_sha256_canonical_sha256": canonical_json_sha256(
                    sorted(excluded_ngrams)
                ),
                "ngram_codepoints": TEXT_REPLAY_EXCLUSION_NGRAM_CODEPOINTS,
                "source": "externally_reviewed_sets_only",
            },
        }
        contract_sha256 = canonical_json_sha256(contract_base)

        self.path = source_path
        self.split = split
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.max_seq_len = max_seq_len
        self._examples = examples
        self._document_sha256 = document_hashes
        self._text_sha256 = {
            str(row["id"]): str(row["utf8_sha256"]) for row in selected
        }
        self._token_stats = token_stats
        self._contract = {**contract_base, "contract_sha256": contract_sha256}
        for example in self._examples:
            example["metadata"]["dataset_contract_sha256"] = contract_sha256

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return copy.deepcopy(self._examples[index])

    @property
    def dataset_contract(self) -> dict[str, object]:
        return copy.deepcopy(self._contract)

    @property
    def contract_sha256(self) -> str:
        return str(self._contract["contract_sha256"])

    @property
    def document_sha256(self) -> dict[str, str]:
        return dict(self._document_sha256)

    @property
    def text_sha256(self) -> dict[str, str]:
        return dict(self._text_sha256)

    @property
    def token_stats(self) -> dict[str, object]:
        return copy.deepcopy(self._token_stats)


class TextReplayPartition:
    """Admit one immutable four-way text replay manifest fail-closed."""

    def __init__(
        self,
        path: str | Path,
        *,
        native_encoder,
        bos_id: int,
        eos_id: int,
        max_seq_len: int,
        exclusion_document_ids: Collection[str],
        exclusion_ngram_sha256: Collection[str] | None = None,
    ) -> None:
        source_path = Path(path)
        if source_path.is_symlink():
            raise ValueError("text replay JSONL must not be a symlink")
        manifest_file_sha256 = file_sha256(source_path)
        rows = _load_rows(source_path)
        rows_by_split = {
            split: [row for row in rows if row["split"] == split]
            for split in TEXT_REPLAY_PUBLIC_SPLITS
        }
        missing = [
            split for split, split_rows in rows_by_split.items() if not split_rows
        ]
        if missing:
            raise ValueError(
                "text replay partition has empty required splits: "
                + ", ".join(missing)
            )

        exact_text_owner: dict[str, tuple[str, str]] = {}
        ngram_owner: dict[str, tuple[str, str]] = {}
        for row in rows:
            split = str(row["split"])
            sample_id = str(row["id"])
            text_sha256 = str(row["utf8_sha256"])
            previous = exact_text_owner.get(text_sha256)
            if previous is not None and previous[0] != split:
                raise ValueError(
                    "text replay exact content crosses splits: "
                    f"{previous[0]}:{previous[1]} vs {split}:{sample_id}"
                )
            exact_text_owner[text_sha256] = (split, sample_id)
            for digest in text_replay_ngram_sha256(str(row["text"])):
                previous = ngram_owner.get(digest)
                if previous is not None and previous[0] != split:
                    raise ValueError(
                        "text replay 13-code-point content window crosses splits: "
                        f"{previous[0]}:{previous[1]} vs {split}:{sample_id}"
                    )
                ngram_owner[digest] = (split, sample_id)

        common = {
            "native_encoder": native_encoder,
            "bos_id": bos_id,
            "eos_id": eos_id,
            "max_seq_len": max_seq_len,
            "exclusion_document_ids": exclusion_document_ids,
            "exclusion_ngram_sha256": exclusion_ngram_sha256,
        }
        datasets = {
            split: TextReplayDataset(source_path, split=split, **common)
            for split in TEXT_REPLAY_PUBLIC_SPLITS
        }
        split_contract_sha256 = {
            split: datasets[split].contract_sha256
            for split in TEXT_REPLAY_PUBLIC_SPLITS
        }
        if len(set(split_contract_sha256.values())) != len(
            TEXT_REPLAY_PUBLIC_SPLITS
        ):
            raise ValueError(
                "all four text replay splits must have distinct dataset contracts"
            )

        shared_fields = (
            "all_rows_canonical_sha256",
            "sequence_contract",
            "native_encoder_contract",
            "exclusion_contract",
        )
        first_contract = datasets[TEXT_REPLAY_PUBLIC_SPLITS[0]].dataset_contract
        for split in TEXT_REPLAY_PUBLIC_SPLITS[1:]:
            contract = datasets[split].dataset_contract
            for field in shared_fields:
                if contract[field] != first_contract[field]:
                    raise ValueError(
                        f"text replay {field} differs across partition splits"
                    )

        canonical_rows = [_canonical_row(row) for row in rows]
        all_rows_canonical_sha256 = canonical_json_sha256(canonical_rows)
        if first_contract["all_rows_canonical_sha256"] != (
            all_rows_canonical_sha256
        ):
            raise ValueError("text replay manifest changed during partition admission")
        if file_sha256(source_path) != manifest_file_sha256:
            raise ValueError("text replay manifest bytes changed during admission")
        partition_base: dict[str, object] = {
            "contract_version": TEXT_REPLAY_PARTITION_CONTRACT_VERSION,
            "kind": "dol_text_ce_replay_partition",
            "public_splits": list(TEXT_REPLAY_PUBLIC_SPLITS),
            "manifest_file_sha256": manifest_file_sha256,
            "all_rows_canonical_sha256": all_rows_canonical_sha256,
            "split_contract_sha256": split_contract_sha256,
            "split_selected_rows_canonical_sha256": {
                split: datasets[split].dataset_contract[
                    "selected_rows_canonical_sha256"
                ]
                for split in TEXT_REPLAY_PUBLIC_SPLITS
            },
            "split_document_sha256_canonical_sha256": {
                split: canonical_json_sha256(datasets[split].document_sha256)
                for split in TEXT_REPLAY_PUBLIC_SPLITS
            },
            "split_text_sha256_canonical_sha256": {
                split: canonical_json_sha256(datasets[split].text_sha256)
                for split in TEXT_REPLAY_PUBLIC_SPLITS
            },
            "split_token_stats": {
                split: datasets[split].token_stats
                for split in TEXT_REPLAY_PUBLIC_SPLITS
            },
            "shared_contract": {
                field: copy.deepcopy(first_contract[field])
                for field in shared_fields
            },
            "leakage_contract": {
                "document_id_cross_split": "forbidden",
                "exact_text_sha256_cross_split": "forbidden",
                "ngram_sha256_cross_split": "forbidden",
                "ngram_codepoints": TEXT_REPLAY_EXCLUSION_NGRAM_CODEPOINTS,
                "global_cross_split_leakage_validated": True,
            },
        }
        partition_sha256 = canonical_json_sha256(partition_base)
        self.path = source_path
        self._datasets = datasets
        self._contract = validate_text_replay_partition_contract(
            {
                **partition_base,
                "contract_sha256": partition_sha256,
            }
        )

    def dataset(self, split: str) -> TextReplayDataset:
        if split not in TEXT_REPLAY_PUBLIC_SPLITS:
            raise ValueError(f"unknown text replay split {split!r}")
        return self._datasets[split]

    @property
    def datasets(self) -> dict[str, TextReplayDataset]:
        return dict(self._datasets)

    @property
    def partition_contract(self) -> dict[str, object]:
        return copy.deepcopy(self._contract)

    @property
    def contract_sha256(self) -> str:
        return str(self._contract["contract_sha256"])


class TextReplayCollator:
    """Pad replay rows while leaving boundary_v1 positions model-derived."""

    def __init__(
        self,
        *,
        pad_id: int,
        ignore_index: int = IGNORE_INDEX,
        pad_to_multiple_of: int | None = None,
    ) -> None:
        _validate_token_id("pad_id", pad_id)
        if type(ignore_index) is not int:
            raise ValueError("ignore_index must be an integer")
        if pad_to_multiple_of is not None and (
            type(pad_to_multiple_of) is not int or pad_to_multiple_of <= 0
        ):
            raise ValueError("pad_to_multiple_of must be a positive integer")
        self.pad_id = pad_id
        self.ignore_index = ignore_index
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
        if not rows:
            raise ValueError("text replay collator rows cannot be empty")
        normalized: list[tuple[list[int], dict[str, object]]] = []
        dataset_contracts: list[str] = []
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise TypeError(f"rows[{index}] must be a mapping")
            if any(key in row for key in ("word_pos", "morph_depth", "token_offsets")):
                raise ValueError("boundary_v1 replay rows must not materialize positions")
            try:
                input_ids = list(row["input_ids"])
                attention_mask = list(row["attention_mask"])
                labels = list(row["labels"])
            except (KeyError, TypeError) as exc:
                raise ValueError(f"rows[{index}] has invalid replay fields") from exc
            if not input_ids or not all(type(value) is int for value in input_ids):
                raise ValueError(f"rows[{index}].input_ids must be non-empty integers")
            if attention_mask != [1] * len(input_ids):
                raise ValueError(f"rows[{index}] attention_mask must be all ones")
            if labels != input_ids:
                raise ValueError(f"rows[{index}] labels must supervise every token")
            metadata = row.get("metadata", {})
            if not isinstance(metadata, Mapping):
                raise ValueError(f"rows[{index}].metadata must be a mapping")
            dataset_contract = metadata.get("dataset_contract_sha256")
            if (
                not isinstance(dataset_contract, str)
                or _SHA256_RE.fullmatch(dataset_contract) is None
            ):
                raise ValueError(
                    f"rows[{index}] has no valid text dataset contract"
                )
            dataset_contracts.append(dataset_contract)
            normalized.append((input_ids, dict(metadata)))

        if len(set(dataset_contracts)) != 1:
            raise ValueError("one text replay batch cannot mix dataset contracts")

        width = max(len(input_ids) for input_ids, _ in normalized)
        if self.pad_to_multiple_of is not None:
            remainder = width % self.pad_to_multiple_of
            if remainder:
                width += self.pad_to_multiple_of - remainder

        input_batch: list[list[int]] = []
        attention_batch: list[list[int]] = []
        label_batch: list[list[int]] = []
        for input_ids, _ in normalized:
            padding = width - len(input_ids)
            input_batch.append(input_ids + [self.pad_id] * padding)
            attention_batch.append([1] * len(input_ids) + [0] * padding)
            label_batch.append(input_ids + [self.ignore_index] * padding)
        return {
            "input_ids": torch.tensor(input_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_batch, dtype=torch.long),
            "labels": torch.tensor(label_batch, dtype=torch.long),
            "metadata": [metadata for _, metadata in normalized],
            "position_contract": BOUNDARY_V1,
            "dataset_contract_sha256": dataset_contracts[0],
        }


def _load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        with path.open("r", encoding="utf-8", errors="strict") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw, object_pairs_hook=strict_object)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(
                        f"{path}:{line_number}: invalid strict JSON: {exc}"
                    ) from exc
                rows.append(_validate_row(row, f"{path}:{line_number}"))
    except FileNotFoundError as exc:
        raise ValueError(f"text replay JSONL does not exist: {path}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"text replay JSONL is not valid UTF-8: {path}") from exc
    if not rows:
        raise ValueError("text replay JSONL is empty")
    return rows


def _validate_row(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{where}: row must be a JSON object")
    if set(value) != _ROW_FIELDS:
        missing = sorted(_ROW_FIELDS - set(value))
        extra = sorted(set(value) - _ROW_FIELDS)
        raise ValueError(
            f"{where}: row fields must be exact; missing={missing}, extra={extra}"
        )
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != TEXT_REPLAY_SCHEMA_VERSION
    ):
        raise ValueError(f"{where}: schema_version must be 1")
    sample_id = _validate_identifier(value["id"], f"{where}.id")
    document_id = _validate_identifier(
        value["document_id"],
        f"{where}.document_id",
    )
    source = _validate_identifier(value["source"], f"{where}.source")
    split = _validate_identifier(value["split"], f"{where}.split")
    if split not in TEXT_REPLAY_PUBLIC_SPLITS:
        raise ValueError(
            f"{where}.split must be one of "
            + ", ".join(TEXT_REPLAY_PUBLIC_SPLITS)
        )
    text = _validate_text(value["text"], f"{where}.text")
    digest = value["utf8_sha256"]
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"{where}.utf8_sha256 must be lowercase SHA-256")
    actual = _utf8_sha256(text)
    if digest != actual:
        raise ValueError(f"{where}.utf8_sha256 does not match text bytes")
    return {
        "schema_version": TEXT_REPLAY_SCHEMA_VERSION,
        "id": sample_id,
        "document_id": document_id,
        "source": source,
        "text": text,
        "utf8_sha256": digest,
        "split": split,
    }


def _validate_native_encoder(encoder) -> None:
    if getattr(encoder, "mode", None) != "native":
        raise ValueError("text replay encoder mode must be native")
    if not callable(getattr(encoder, "encode_with_features", None)):
        raise ValueError("native encoder must expose encode_with_features")
    stats = getattr(encoder, "stats", None)
    if not isinstance(stats, Mapping):
        raise ValueError("native encoder must expose auditable stats")
    for key in ("native", "byte_fallback", "canonicalized"):
        if type(stats.get(key)) is not int or stats[key] < 0:
            raise ValueError(f"native encoder stats[{key!r}] must be non-negative")
    if stats["byte_fallback"] != 0:
        raise ValueError("native encoder has already used byte fallback")


def _strict_native_encode(encoder, text: str, *, where: str) -> list[int]:
    stats = encoder.stats
    before_native = int(stats["native"])
    before_fallback = int(stats["byte_fallback"])
    before_canonicalized = int(stats["canonicalized"])
    try:
        encoded = encoder.encode_with_features(text)
    except Exception as exc:
        raise ValueError(f"{where} native encoder round-trip failed: {exc}") from exc
    if int(stats["byte_fallback"]) != before_fallback or before_fallback != 0:
        raise ValueError(f"{where} native encoder used fallback")
    if int(stats["canonicalized"]) != before_canonicalized:
        raise ValueError(f"{where} native encoder changed text during round-trip")
    if int(stats["native"]) != before_native + 1:
        raise ValueError(f"{where} native encoder did not record one native route")
    input_ids = getattr(encoded, "input_ids", None)
    if not isinstance(input_ids, list) or not input_ids:
        raise ValueError(f"{where} native encoder returned no token IDs")
    if any(type(token_id) is not int or token_id < 0 for token_id in input_ids):
        raise ValueError(f"{where} native encoder returned invalid token IDs")
    return list(input_ids)


def _document_hashes(rows: Sequence[Mapping[str, object]]) -> dict[str, str]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        document_id = str(row["document_id"])
        grouped.setdefault(document_id, []).append(
            {
                "id": str(row["id"]),
                "source": str(row["source"]),
                "text_utf8_sha256": str(row["utf8_sha256"]),
                "split": str(row["split"]),
            }
        )
    return {
        document_id: canonical_json_sha256(
            {"document_id": document_id, "rows": grouped[document_id]}
        )
        for document_id in sorted(grouped)
    }


def _canonical_row(row: Mapping[str, object]) -> dict[str, object]:
    return {field: row[field] for field in sorted(_ROW_FIELDS)}


def _validated_identifier_set(values: Collection[str], name: str) -> set[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Collection):
        raise TypeError(f"{name} must be a collection of strings")
    return {_validate_identifier(value, name) for value in values}


def _validated_sha256_set(values: Collection[str], name: str) -> set[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Collection):
        raise TypeError(f"{name} must be a collection of SHA-256 strings")
    result: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            raise ValueError(f"{name} values must be lowercase SHA-256")
        result.add(value)
    return result


def _validate_identifier(value: object, where: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{where} must be a non-empty stripped string")
    _reject_invalid_unicode(value, where)
    return value


def _validate_text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value or not any(
        not character.isspace() for character in value
    ):
        raise ValueError(f"{where} must contain non-whitespace text")
    _reject_invalid_unicode(value, where)
    return value


def _reject_invalid_unicode(value: str, where: str) -> None:
    for character in value:
        codepoint = ord(character)
        if character == "\x00":
            raise ValueError(f"{where} contains NUL")
        if character == "\ufffd":
            raise ValueError(f"{where} contains U+FFFD")
        if 0xD800 <= codepoint <= 0xDFFF:
            raise ValueError(f"{where} contains a surrogate code point")


def _validate_token_id(name: str, value: object) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _utf8_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="strict")).hexdigest()


__all__ = [
    "TEXT_REPLAY_CONTRACT_VERSION",
    "TEXT_REPLAY_EXCLUSION_NGRAM_CODEPOINTS",
    "TEXT_REPLAY_PARTITION_CONTRACT_VERSION",
    "TEXT_REPLAY_PUBLIC_SPLITS",
    "TEXT_REPLAY_SCHEMA_VERSION",
    "TEXT_REPLAY_VALIDATION_SPLITS",
    "TextReplayCollator",
    "TextReplayDataset",
    "TextReplayPartition",
    "canonical_text_replay_contract_sha256",
    "text_replay_ngram_sha256",
    "validate_text_replay_partition_contract",
]
