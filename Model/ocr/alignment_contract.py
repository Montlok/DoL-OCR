# -*- coding: utf-8 -*-

"""Immutable contract for pre-tokenized OCR visual-alignment data."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from Model.ocr.tokenization import canonical_json_sha256

OCR_ALIGNMENT_DATA_SCHEMA_VERSION = 3
OCR_IMAGE_BINDING_MODE = "jsonl_sha256_size+lazy_stream_verification_v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SHA256_ANY_CASE_RE = re.compile(r"[0-9a-fA-F]{64}")
_PAYLOAD_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "data_layout",
        "data_file_count",
        "data_files",
        "data_sha256",
        "image_binding",
        "image_manifest_sha256",
        "image_reference_count",
        "image_total_size_bytes",
        "shard_completion",
        "ocr_tokenization_contract",
    }
)


def _direct_builder_completion() -> dict[str, Any]:
    entries: list[dict[str, str]] = []
    return {
        "mode": "direct_builder_v1",
        "sentinel_count": 0,
        "sentinels": entries,
        "sentinel_manifest_sha256": canonical_json_sha256(entries),
    }


def _validate_shard_completion_manifest(value: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, dict) or set(value) != {
        "mode",
        "sentinel_count",
        "sentinels",
        "sentinel_manifest_sha256",
    }:
        return ["shard_completion has an invalid schema"]
    mode = value["mode"]
    entries = value["sentinels"]
    if mode not in {"direct_builder_v1", "validated_shard_sentinels_v2"}:
        errors.append("shard_completion.mode is invalid")
    if not isinstance(entries, list):
        errors.append("shard_completion.sentinels must be a list")
        entries = []
    normalized: list[dict[str, str]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != {"name", "sha256"}:
            errors.append(
                f"shard_completion.sentinels[{index}] has an invalid schema"
            )
            continue
        name = entry["name"]
        digest = entry["sha256"]
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
        ):
            errors.append(
                f"shard_completion.sentinels[{index}].name is invalid"
            )
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            errors.append(
                f"shard_completion.sentinels[{index}].sha256 is invalid"
            )
        normalized.append(entry)
    names = [entry.get("name") for entry in normalized]
    if names != sorted(names) or len(names) != len(set(names)):
        errors.append("shard_completion sentinels must be sorted and unique")
    count = value["sentinel_count"]
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count != len(normalized)
    ):
        errors.append("shard_completion.sentinel_count differs from sentinels")
    if mode == "direct_builder_v1" and normalized:
        errors.append("direct_builder_v1 cannot carry shard sentinels")
    if mode == "validated_shard_sentinels_v2" and not normalized:
        errors.append("validated shard completion requires sentinels")
    if value["sentinel_manifest_sha256"] != canonical_json_sha256(normalized):
        errors.append("shard_completion sentinel aggregate is invalid")
    return errors
_DERIVED_KEYS = frozenset(
    {
        "contract_canonical_sha256",
        "tokenizer_manifest_sha256",
        "corpus_manifest_sha256",
        "ocr_target_encoding",
        "ocr_tokenization_contract_version",
    }
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_ocr_image_binding_fields(
    row: dict[str, Any],
    *,
    context: str = "OCR row",
) -> tuple[str, str, int]:
    """Validate and return one row's image reference, digest, and byte size.

    Native frozen-LM OCR currently supports one image per row.  Keeping the
    expected digest beside that reference means the JSONL receipt binds the
    image bytes without materializing a corpus-sized image manifest.
    """

    images = row.get("images")
    if (
        not isinstance(images, list)
        or len(images) != 1
        or not isinstance(images[0], str)
        or not images[0]
    ):
        raise ValueError(f"{context}: images must contain exactly one path string")
    digest = row.get("image_sha256")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise ValueError(
            f"{context}: image_sha256 must be a lowercase 64-character SHA-256"
        )
    size = row.get("image_size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(
            f"{context}: image_size_bytes must be a non-negative integer"
        )
    return images[0], digest, size


def read_verified_image_bytes(
    image_ref: str | Path,
    expected_sha256: str,
    *,
    expected_size_bytes: int | None = None,
    context: str = "OCR row",
) -> bytes:
    """Read one image once and fail closed on expected byte drift."""

    if (
        not isinstance(expected_sha256, str)
        or not _SHA256_ANY_CASE_RE.fullmatch(expected_sha256)
    ):
        raise ValueError(f"{context}: expected image SHA-256 is invalid")
    if not isinstance(image_ref, (str, Path)) or not str(image_ref):
        raise ValueError(f"{context}: image path is invalid")
    if (
        expected_size_bytes is not None
        and (
            isinstance(expected_size_bytes, bool)
            or not isinstance(expected_size_bytes, int)
            or expected_size_bytes < 0
        )
    ):
        raise ValueError(f"{context}: expected image byte size is invalid")
    image_path = Path(image_ref)
    try:
        image_bytes = image_path.read_bytes()
    except OSError as exc:
        raise ValueError(
            f"{context}: cannot read OCR image {image_path}: {exc}"
        ) from exc
    if (
        expected_size_bytes is not None
        and len(image_bytes) != expected_size_bytes
    ):
        raise ValueError(
            f"{context}: OCR image size mismatch for {image_path}: "
            f"expected {expected_size_bytes}, got {len(image_bytes)}"
        )
    actual_digest = hashlib.sha256(image_bytes).hexdigest()
    if actual_digest != expected_sha256.lower():
        raise ValueError(
            f"{context}: OCR image SHA-256 mismatch for {image_path}: "
            f"expected {expected_sha256.lower()}, got {actual_digest}"
        )
    return image_bytes


def read_verified_ocr_image_bytes(
    row: dict[str, Any],
    *,
    context: str = "OCR row",
) -> bytes:
    """Verify a native alignment row and return the same bytes for decoding."""

    image_ref, expected_digest, expected_size = (
        validate_ocr_image_binding_fields(row, context=context)
    )
    return read_verified_image_bytes(
        image_ref,
        expected_digest,
        expected_size_bytes=expected_size,
        context=context,
    )


def _scan_alignment_files(
    files: list[Path],
) -> tuple[list[dict[str, Any]], str, int, int]:
    """Hash JSONL bytes and summarize expected image bytes in one pass."""

    entries: list[dict[str, Any]] = []
    image_manifest = hashlib.sha256()
    image_count = 0
    image_total_size = 0
    for source in files:
        file_digest = hashlib.sha256()
        size_bytes = 0
        with source.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                file_digest.update(raw_line)
                size_bytes += len(raw_line)
                if not raw_line.strip():
                    continue
                try:
                    row = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"invalid OCR alignment JSONL at "
                        f"{source}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(
                        f"invalid OCR alignment JSONL at "
                        f"{source}:{line_number}: row must be an object"
                    )
                image_ref, digest, image_size = validate_ocr_image_binding_fields(
                    row,
                    context=f"{source}:{line_number}",
                )
                image_record = [
                    source.name,
                    line_number,
                    image_ref,
                    digest,
                    image_size,
                ]
                image_manifest.update(
                    json.dumps(
                        image_record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
                image_manifest.update(b"\n")
                image_count += 1
                image_total_size += image_size
        entries.append(
            {
                "name": source.name,
                "sha256": file_digest.hexdigest(),
                "size_bytes": size_bytes,
            }
        )
    if image_count == 0:
        raise ValueError("OCR alignment data contains no image-bound rows")
    return (
        entries,
        image_manifest.hexdigest(),
        image_count,
        image_total_size,
    )


def resolve_ocr_alignment_data_files(data_spec: str | Path) -> list[Path]:
    """Resolve one JSONL, a shard directory, or a glob exactly like training."""

    raw = str(data_spec)
    path = Path(raw)
    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
    elif any(character in raw for character in "*?["):
        files = sorted(
            Path(candidate)
            for candidate in glob.glob(raw, recursive=True)
        )
    elif path.is_file():
        files = [path]
    else:
        files = []
    if not files:
        raise FileNotFoundError(
            f"no OCR alignment JSONL files resolved from: {data_spec!r}"
        )
    resolved = [candidate.resolve() for candidate in files]
    if len(resolved) != len(set(resolved)):
        raise ValueError("OCR alignment data spec resolves duplicate files")
    names = [candidate.name for candidate in resolved]
    if len(names) != len(set(names)):
        raise ValueError(
            "OCR alignment shard basenames must be unique for a relocatable receipt"
        )
    return resolved


def default_ocr_alignment_contract_path(data_spec: str | Path) -> Path:
    raw = str(data_spec)
    path = Path(raw)
    if path.is_dir():
        return path / "ocr_data_contract.json"
    if any(character in raw for character in "*?["):
        raise ValueError(
            "a globbed --data requires an explicit --ocr-data-contract"
        )
    return path.parent / "ocr_data_contract.json"


def build_ocr_alignment_data_contract(
    data_path: str | Path,
    tokenization_contract: dict[str, Any],
    *,
    shard_completion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the exact JSONL shard set and pretrained LM representation."""

    files = resolve_ocr_alignment_data_files(data_path)
    (
        entries,
        image_manifest_sha256,
        image_reference_count,
        image_total_size_bytes,
    ) = _scan_alignment_files(files)
    completion = (
        _direct_builder_completion()
        if shard_completion is None
        else shard_completion
    )
    completion_errors = _validate_shard_completion_manifest(completion)
    if completion_errors:
        raise ValueError(
            "invalid OCR shard completion manifest:\n  - "
            + "\n  - ".join(completion_errors)
        )
    return {
        "schema_version": OCR_ALIGNMENT_DATA_SCHEMA_VERSION,
        "kind": "pretokenized_ocr_alignment",
        "data_layout": "single_jsonl" if len(entries) == 1 else "sharded_jsonl",
        "data_file_count": len(entries),
        "data_files": entries,
        "data_sha256": canonical_json_sha256(entries),
        "image_binding": OCR_IMAGE_BINDING_MODE,
        "image_manifest_sha256": image_manifest_sha256,
        "image_reference_count": image_reference_count,
        "image_total_size_bytes": image_total_size_bytes,
        "shard_completion": completion,
        "ocr_tokenization_contract": tokenization_contract,
    }


def validate_ocr_alignment_data_contract_payload(
    payload: dict[str, Any],
    tokenization_contract: dict[str, Any],
) -> dict[str, Any]:
    """Validate an embedded receipt without trusting summary-only metadata."""

    if not isinstance(payload, dict):
        raise ValueError("OCR alignment data contract must be an object")
    unknown = set(payload) - _PAYLOAD_KEYS - _DERIVED_KEYS
    errors: list[str] = []
    if unknown:
        errors.append("unknown contract fields: " + ", ".join(sorted(unknown)))
    core = {key: payload.get(key) for key in _PAYLOAD_KEYS}
    if core["schema_version"] != OCR_ALIGNMENT_DATA_SCHEMA_VERSION:
        errors.append("schema_version differs from runtime")
    if core["kind"] != "pretokenized_ocr_alignment":
        errors.append("kind must be 'pretokenized_ocr_alignment'")
    errors.extend(
        _validate_shard_completion_manifest(core["shard_completion"])
    )
    if core["image_binding"] != OCR_IMAGE_BINDING_MODE:
        errors.append("image_binding differs from runtime")
    if (
        not isinstance(core["image_manifest_sha256"], str)
        or not _SHA256_RE.fullmatch(core["image_manifest_sha256"])
    ):
        errors.append("image_manifest_sha256 is invalid")
    image_count = core["image_reference_count"]
    if (
        isinstance(image_count, bool)
        or not isinstance(image_count, int)
        or image_count < 1
    ):
        errors.append("image_reference_count must be a positive integer")
    image_total_size = core["image_total_size_bytes"]
    if (
        isinstance(image_total_size, bool)
        or not isinstance(image_total_size, int)
        or image_total_size < 0
    ):
        errors.append("image_total_size_bytes must be a non-negative integer")
    entries = core["data_files"]
    if not isinstance(entries, list) or not entries:
        errors.append("data_files must be a non-empty list")
        entries = []
    normalized_entries: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != {
            "name",
            "sha256",
            "size_bytes",
        }:
            errors.append(f"data_files[{index}] has an invalid schema")
            continue
        name = entry.get("name")
        digest = entry.get("sha256")
        size = entry.get("size_bytes")
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
        ):
            errors.append(f"data_files[{index}].name must be a basename")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            errors.append(f"data_files[{index}].sha256 is invalid")
        if not isinstance(size, int) or size < 0:
            errors.append(f"data_files[{index}].size_bytes is invalid")
        normalized_entries.append(entry)
    names = [entry.get("name") for entry in normalized_entries]
    if names != sorted(names) or len(names) != len(set(names)):
        errors.append("data_files must be sorted with unique basenames")
    expected_layout = (
        "single_jsonl" if len(normalized_entries) == 1 else "sharded_jsonl"
    )
    if core["data_layout"] != expected_layout:
        errors.append("data_layout differs from file count")
    if core["data_file_count"] != len(normalized_entries):
        errors.append("data_file_count differs from data_files")
    aggregate = canonical_json_sha256(normalized_entries)
    if core["data_sha256"] != aggregate:
        errors.append("data_sha256 differs from the shard manifest")
    if core["ocr_tokenization_contract"] != tokenization_contract:
        errors.append("tokenizer/LM representation differs from runtime")

    canonical = canonical_json_sha256(core)
    if (
        "contract_canonical_sha256" in payload
        and payload.get("contract_canonical_sha256") != canonical
    ):
        errors.append("contract_canonical_sha256 differs from the receipt")
    derived_expected = {
        "tokenizer_manifest_sha256": tokenization_contract.get(
            "tokenizer_manifest_canonical_sha256"
        ),
        "corpus_manifest_sha256": aggregate,
        "ocr_target_encoding": tokenization_contract.get("target_encoding"),
        "ocr_tokenization_contract_version": tokenization_contract.get(
            "tokenization_contract_version"
        ),
    }
    for key, expected in derived_expected.items():
        if key in payload and payload.get(key) != expected:
            errors.append(f"{key} differs from the full contract")
    if errors:
        raise ValueError(
            "unsafe OCR alignment data contract:\n  - "
            + "\n  - ".join(errors)
        )
    return {
        **core,
        "contract_canonical_sha256": canonical,
        **derived_expected,
    }


def load_and_validate_ocr_alignment_data_contract(
    contract_path: str | Path,
    data_path: str | Path,
    tokenization_contract: dict[str, Any],
) -> dict[str, Any]:
    """Load a builder receipt and bind every resolved shard to runtime bytes."""

    path = Path(contract_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"OCR data contract does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid OCR data contract JSON: {path}: {exc}") from exc
    validated = validate_ocr_alignment_data_contract_payload(
        payload,
        tokenization_contract,
    )
    actual = build_ocr_alignment_data_contract(
        data_path,
        tokenization_contract,
        shard_completion=validated["shard_completion"],
    )
    expected_core = {key: actual[key] for key in _PAYLOAD_KEYS}
    actual_core = {key: validated[key] for key in _PAYLOAD_KEYS}
    if actual_core != expected_core:
        raise ValueError(
            "unsafe OCR alignment data contract:\n  - resolved data shards "
            "differ from the builder receipt"
        )
    return validated


def write_json_atomically(
    path: str | Path,
    payload: dict[str, Any],
) -> None:
    """Durably publish one JSON object without exposing a partial file."""

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


def write_ocr_alignment_data_contract(
    path: str | Path,
    payload: dict[str, Any],
) -> None:
    """Durably publish one OCR data receipt."""

    write_json_atomically(path, payload)


__all__ = [
    "OCR_ALIGNMENT_DATA_SCHEMA_VERSION",
    "OCR_IMAGE_BINDING_MODE",
    "build_ocr_alignment_data_contract",
    "default_ocr_alignment_contract_path",
    "file_sha256",
    "load_and_validate_ocr_alignment_data_contract",
    "read_verified_image_bytes",
    "read_verified_ocr_image_bytes",
    "resolve_ocr_alignment_data_files",
    "validate_ocr_image_binding_fields",
    "write_json_atomically",
    "validate_ocr_alignment_data_contract_payload",
    "write_ocr_alignment_data_contract",
]
