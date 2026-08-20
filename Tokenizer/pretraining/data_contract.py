# -*- coding: utf-8 -*-

"""Immutable lineage receipts for pre-tokenized RDT JSONL shards.

The JSONL rows consumed by :mod:`scripts.train_rdt` already contain token ids.
Consequently, validating only the tokenizer used at *training* time is not
enough: the rows may have been produced by a different tokenizer
implementation or bundle.  This module binds the producer's tokenizer
identity to every emitted shard byte and lets the trainer verify that receipt
before allocating a model.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any


PRETRAINING_DATA_CONTRACT_VERSION = 2
PRETRAINING_DATA_CONTRACT_KIND = "pretokenized_rdt_jsonl"
PRETRAINING_PRODUCER_GENERIC_BUILDER = "generic_pretraining_builder"
PRETRAINING_PRODUCER_OCR_ALIGN_TEXT = "ocr_alignment_text_converter"
PRETRAINING_PRODUCER_ALGORITHM_VERSION = 1
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PRODUCER_SOURCE_FILES = {
    PRETRAINING_PRODUCER_GENERIC_BUILDER: (
        "Tokenizer/tools/build_pretraining_data.py"
    ),
    PRETRAINING_PRODUCER_OCR_ALIGN_TEXT: (
        "scripts/build_text_rows_from_align.py"
    ),
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "producer_kind",
        "producer_algorithm",
        "data_file_count",
        "data_files",
        "data_total_size_bytes",
        "data_sha256",
        "tokenizer_bundle",
        "tokenizer_algorithm",
    }
)
_DERIVED_KEYS = frozenset({"contract_canonical_sha256"})


def canonical_json_sha256(value: Any) -> str:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pretraining_producer_algorithm_contract(
    producer_kind: str,
) -> dict[str, Any]:
    """Fingerprint the concrete row producer named by a receipt.

    The shared tokenizer algorithm contract intentionally excludes CLI row
    producers: otherwise a change to either producer would invalidate rows
    emitted by both.  A receipt instead names exactly one known producer and
    binds that producer's current source bytes independently.
    """

    source_file = _PRODUCER_SOURCE_FILES.get(producer_kind)
    if source_file is None:
        raise ValueError(f"unknown pretraining producer_kind: {producer_kind!r}")
    source_path = _REPO_ROOT / source_file
    if not source_path.is_file():
        raise FileNotFoundError(
            f"pretraining producer source file is missing: {source_path}"
        )
    return {
        "contract_version": PRETRAINING_PRODUCER_ALGORITHM_VERSION,
        "source_file": source_file,
        "source_sha256": file_sha256(source_path),
    }


def resolve_pretraining_data_files(
    data_spec: str | Path | Sequence[str | Path],
) -> list[Path]:
    """Resolve shards with the same path semantics as the RDT dataloader."""

    if isinstance(data_spec, (str, Path)):
        raw = str(data_spec)
        path = Path(raw)
        if path.is_dir():
            candidates = sorted(path.glob("*.jsonl"))
        elif any(character in raw for character in "*?["):
            candidates = sorted(
                Path(candidate)
                for candidate in glob.glob(raw, recursive=True)
            )
        elif path.is_file():
            candidates = [path]
        else:
            candidates = []
    else:
        candidates = [Path(candidate) for candidate in data_spec]

    if not candidates:
        raise FileNotFoundError(
            f"no pretraining JSONL files resolved from: {data_spec!r}"
        )
    resolved: list[Path] = []
    for candidate in candidates:
        path = candidate.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"pretraining shard is missing: {candidate}")
        resolved.append(path)
    if len(resolved) != len(set(resolved)):
        raise ValueError("pretraining data spec resolves duplicate files")
    names = [path.name for path in resolved]
    if len(names) != len(set(names)):
        raise ValueError(
            "pretraining shard basenames must be unique for a relocatable receipt"
        )
    return resolved


def _data_entries(
    data_spec: str | Path | Sequence[str | Path],
) -> list[dict[str, Any]]:
    return [
        {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in resolve_pretraining_data_files(data_spec)
    ]


def build_pretraining_data_contract(
    data_spec: str | Path | Sequence[str | Path],
    *,
    producer_kind: str,
    tokenizer_bundle: dict[str, Any],
    tokenizer_algorithm: dict[str, Any],
) -> dict[str, Any]:
    """Build a receipt over the exact ordered shard set."""

    entries = _data_entries(data_spec)
    core = {
        "schema_version": PRETRAINING_DATA_CONTRACT_VERSION,
        "kind": PRETRAINING_DATA_CONTRACT_KIND,
        "producer_kind": producer_kind,
        "producer_algorithm": pretraining_producer_algorithm_contract(
            producer_kind
        ),
        "data_file_count": len(entries),
        "data_files": entries,
        "data_total_size_bytes": sum(
            int(entry["size_bytes"]) for entry in entries
        ),
        "data_sha256": canonical_json_sha256(entries),
        "tokenizer_bundle": tokenizer_bundle,
        "tokenizer_algorithm": tokenizer_algorithm,
    }
    return {
        **core,
        "contract_canonical_sha256": canonical_json_sha256(core),
    }


def validate_pretraining_data_contract_payload(
    payload: dict[str, Any],
    *,
    tokenizer_bundle: dict[str, Any],
    tokenizer_algorithm: dict[str, Any],
) -> dict[str, Any]:
    """Validate receipt structure and its tokenizer producer identity."""

    if not isinstance(payload, dict):
        raise ValueError("pretraining data receipt must be an object")
    unknown = set(payload) - _RECEIPT_KEYS - _DERIVED_KEYS
    missing = _RECEIPT_KEYS - set(payload)
    errors: list[str] = []
    if unknown:
        errors.append("unknown receipt fields: " + ", ".join(sorted(unknown)))
    if missing:
        errors.append("missing receipt fields: " + ", ".join(sorted(missing)))

    core = {key: payload.get(key) for key in _RECEIPT_KEYS}
    if (
        type(core["schema_version"]) is not int
        or core["schema_version"] != PRETRAINING_DATA_CONTRACT_VERSION
    ):
        errors.append("schema_version differs from runtime")
    if core["kind"] != PRETRAINING_DATA_CONTRACT_KIND:
        errors.append(f"kind must be {PRETRAINING_DATA_CONTRACT_KIND!r}")
    producer_kind = core["producer_kind"]
    if not isinstance(producer_kind, str) or not producer_kind:
        errors.append("producer_kind must be a non-empty string")
        expected_producer_algorithm: dict[str, Any] = {}
    else:
        try:
            expected_producer_algorithm = (
                pretraining_producer_algorithm_contract(producer_kind)
            )
        except (FileNotFoundError, ValueError) as exc:
            errors.append(str(exc))
            expected_producer_algorithm = {}
    if core["producer_algorithm"] != expected_producer_algorithm:
        errors.append(
            "producer_algorithm differs from the named current producer"
        )

    entries = core["data_files"]
    if not isinstance(entries, list) or not entries:
        errors.append("data_files must be a non-empty list")
        entries = []
    normalized_entries: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != {
            "name",
            "size_bytes",
            "sha256",
        }:
            errors.append(f"data_files[{index}] has an invalid schema")
            continue
        name = entry.get("name")
        size = entry.get("size_bytes")
        digest = entry.get("sha256")
        if not isinstance(name, str) or not name or Path(name).name != name:
            errors.append(f"data_files[{index}].name must be a basename")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            errors.append(f"data_files[{index}].size_bytes is invalid")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            errors.append(f"data_files[{index}].sha256 is invalid")
        normalized_entries.append(entry)
    names = [
        entry.get("name")
        for entry in normalized_entries
        if isinstance(entry.get("name"), str)
    ]
    if len(names) != len(set(names)):
        errors.append("data_files basenames must be unique")
    if (
        type(core["data_file_count"]) is not int
        or core["data_file_count"] != len(normalized_entries)
    ):
        errors.append("data_file_count differs from data_files")
    expected_size = sum(
        int(entry["size_bytes"])
        for entry in normalized_entries
        if isinstance(entry.get("size_bytes"), int)
        and not isinstance(entry.get("size_bytes"), bool)
    )
    if (
        type(core["data_total_size_bytes"]) is not int
        or core["data_total_size_bytes"] != expected_size
    ):
        errors.append("data_total_size_bytes differs from data_files")
    expected_data_sha = canonical_json_sha256(normalized_entries)
    if core["data_sha256"] != expected_data_sha:
        errors.append("data_sha256 differs from the shard manifest")
    if core["tokenizer_bundle"] != tokenizer_bundle:
        errors.append("tokenizer_bundle differs from the current runtime")
    if core["tokenizer_algorithm"] != tokenizer_algorithm:
        errors.append("tokenizer_algorithm differs from the current runtime")

    expected_contract_sha = canonical_json_sha256(core)
    if payload.get("contract_canonical_sha256") != expected_contract_sha:
        errors.append("contract_canonical_sha256 differs from the receipt")
    if errors:
        raise ValueError(
            "unsafe pretraining data receipt:\n  - "
            + "\n  - ".join(errors)
        )
    return {**core, "contract_canonical_sha256": expected_contract_sha}


def load_and_validate_pretraining_data_contract(
    receipt_path: str | Path,
    data_spec: str | Path | Sequence[str | Path],
    *,
    tokenizer_bundle: dict[str, Any],
    tokenizer_algorithm: dict[str, Any],
) -> dict[str, Any]:
    """Bind a producer receipt to the bytes the trainer actually resolved."""

    path = Path(receipt_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"pretraining data receipt does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid pretraining data receipt JSON: {path}: {exc}"
        ) from exc
    validated = validate_pretraining_data_contract_payload(
        payload,
        tokenizer_bundle=tokenizer_bundle,
        tokenizer_algorithm=tokenizer_algorithm,
    )
    actual = build_pretraining_data_contract(
        data_spec,
        producer_kind=validated["producer_kind"],
        tokenizer_bundle=tokenizer_bundle,
        tokenizer_algorithm=tokenizer_algorithm,
    )
    if validated != actual:
        raise ValueError(
            "unsafe pretraining data receipt:\n  - resolved data shards "
            "differ from the producer receipt"
        )
    return validated


def default_pretraining_data_contract_path(
    output: str | Path,
    *,
    sharded_directory: bool = False,
) -> Path:
    path = Path(output)
    if sharded_directory:
        return path / "pretraining_data_receipt.json"
    return Path(f"{path}.receipt.json")


def write_pretraining_data_contract(
    path: str | Path,
    payload: dict[str, Any],
) -> None:
    """Atomically and durably publish a completed producer receipt."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = [
    "PRETRAINING_DATA_CONTRACT_KIND",
    "PRETRAINING_DATA_CONTRACT_VERSION",
    "PRETRAINING_PRODUCER_ALGORITHM_VERSION",
    "PRETRAINING_PRODUCER_GENERIC_BUILDER",
    "PRETRAINING_PRODUCER_OCR_ALIGN_TEXT",
    "build_pretraining_data_contract",
    "canonical_json_sha256",
    "default_pretraining_data_contract_path",
    "file_sha256",
    "load_and_validate_pretraining_data_contract",
    "pretraining_producer_algorithm_contract",
    "resolve_pretraining_data_files",
    "validate_pretraining_data_contract_payload",
    "write_pretraining_data_contract",
]
