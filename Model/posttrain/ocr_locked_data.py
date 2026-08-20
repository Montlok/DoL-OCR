# -*- coding: utf-8 -*-

"""Claim-gated, in-memory capture of a sealed AnyRes OCR benchmark."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

from Model.posttrain.ocr_locked_contract import (
    LOCKED_BUCKETS,
    LockedGoldenClaim,
    LockedGoldenPreclaim,
    validate_locked_golden_claim,
)


LOCKED_COMBINED_MANIFEST_KIND = "dol_ocr_anyres_locked_combined_manifest_v1"
_MANIFEST_KEYS = {
    "schema_version",
    "kind",
    "image_records",
    "text_records",
    "canonical_sha256",
}
_IMAGE_KEYS = {
    "id",
    "bucket",
    "asset_relpath",
    "asset_sha256",
    "asset_size_bytes",
    "reference",
    "reference_utf8_sha256",
    "reference_token_count",
}
_TEXT_KEYS = {
    "id",
    "reference",
    "reference_utf8_sha256",
    "reference_token_count",
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha(value: object, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{where} must be lowercase SHA-256")
    return value


def _identifier(value: object, where: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 0x20 for character in value)
    ):
        raise ValueError(f"{where} must be a safe non-empty identifier")
    return value


def _reference(value: object, where: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or "\ufffd" in value
    ):
        raise ValueError(f"{where} must be non-empty exact Unicode text")
    return value


def _relpath(value: object, where: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ValueError(f"{where} must be a normalized POSIX relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise ValueError(f"{where} must be a normalized POSIX relative path")
    return value


def _positive_int(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{where} must be a positive integer")
    return value


def _normalize_image(value: object, index: int) -> dict[str, Any]:
    where = f"image_records[{index}]"
    if not isinstance(value, Mapping) or set(value) != _IMAGE_KEYS:
        raise ValueError(f"{where} fields differ from contract")
    reference = _reference(value["reference"], f"{where}.reference")
    digest = hashlib.sha256(reference.encode("utf-8")).hexdigest()
    if _sha(value["reference_utf8_sha256"], f"{where}.reference_utf8_sha256") != digest:
        raise ValueError(f"{where} reference UTF-8 SHA-256 differs")
    bucket = value["bucket"]
    if bucket not in LOCKED_BUCKETS:
        raise ValueError(f"{where}.bucket is invalid")
    return {
        "id": _identifier(value["id"], f"{where}.id"),
        "bucket": bucket,
        "asset_relpath": _relpath(value["asset_relpath"], f"{where}.asset_relpath"),
        "asset_sha256": _sha(value["asset_sha256"], f"{where}.asset_sha256"),
        "asset_size_bytes": _positive_int(
            value["asset_size_bytes"], f"{where}.asset_size_bytes"
        ),
        "reference": reference,
        "reference_utf8_sha256": digest,
        "reference_token_count": _positive_int(
            value["reference_token_count"], f"{where}.reference_token_count"
        ),
    }


def _normalize_text(value: object, index: int) -> dict[str, Any]:
    where = f"text_records[{index}]"
    if not isinstance(value, Mapping) or set(value) != _TEXT_KEYS:
        raise ValueError(f"{where} fields differ from contract")
    reference = _reference(value["reference"], f"{where}.reference")
    digest = hashlib.sha256(reference.encode("utf-8")).hexdigest()
    if _sha(value["reference_utf8_sha256"], f"{where}.reference_utf8_sha256") != digest:
        raise ValueError(f"{where} reference UTF-8 SHA-256 differs")
    return {
        "id": _identifier(value["id"], f"{where}.id"),
        "reference": reference,
        "reference_utf8_sha256": digest,
        "reference_token_count": _positive_int(
            value["reference_token_count"], f"{where}.reference_token_count"
        ),
    }


def _manifest_commitments(value: Mapping[str, Any]) -> dict[str, str]:
    images = value["image_records"]
    texts = value["text_records"]
    image_identity = [
        {
            "id": row["id"],
            "bucket": row["bucket"],
            "asset_relpath": row["asset_relpath"],
            "asset_sha256": row["asset_sha256"],
            "asset_size_bytes": row["asset_size_bytes"],
            "reference_utf8_sha256": row["reference_utf8_sha256"],
            "reference_token_count": row["reference_token_count"],
        }
        for row in images
    ]
    text_identity = [
        {
            "id": row["id"],
            "reference_utf8_sha256": row["reference_utf8_sha256"],
            "reference_token_count": row["reference_token_count"],
        }
        for row in texts
    ]
    return {
        "image_dataset_contract_sha256": _canonical_sha(
            {"kind": "dol_locked_image_dataset_v1", "records": image_identity}
        ),
        "text_dataset_contract_sha256": _canonical_sha(
            {"kind": "dol_locked_text_dataset_v1", "records": text_identity}
        ),
        "image_identity_commitment_sha256": _canonical_sha(image_identity),
        "text_identity_commitment_sha256": _canonical_sha(text_identity),
        "membership_commitment_sha256": _canonical_sha(
            {"image": image_identity, "text": text_identity}
        ),
    }


def validate_locked_combined_manifest(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _MANIFEST_KEYS:
        raise ValueError("locked combined manifest fields differ from contract")
    normalized = {
        "schema_version": value["schema_version"],
        "kind": value["kind"],
        "image_records": [
            _normalize_image(row, index)
            for index, row in enumerate(value["image_records"])
        ] if isinstance(value["image_records"], list) else None,
        "text_records": [
            _normalize_text(row, index)
            for index, row in enumerate(value["text_records"])
        ] if isinstance(value["text_records"], list) else None,
        "canonical_sha256": value["canonical_sha256"],
    }
    if normalized["schema_version"] != 1 or normalized["kind"] != (
        LOCKED_COMBINED_MANIFEST_KIND
    ):
        raise ValueError("locked combined manifest kind differs")
    images = normalized["image_records"]
    texts = normalized["text_records"]
    if not images or not texts:
        raise ValueError("locked combined manifest image/text records must be non-empty")
    if images != sorted(images, key=lambda row: row["id"]) or texts != sorted(
        texts, key=lambda row: row["id"]
    ):
        raise ValueError("locked combined manifest records must be sorted by id")
    identifiers = [row["id"] for row in [*images, *texts]]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("locked combined manifest IDs must be globally unique")
    asset_paths = [row["asset_relpath"] for row in images]
    if len(asset_paths) != len(set(asset_paths)):
        raise ValueError("locked combined manifest assets must be unique")
    counts = {bucket: 0 for bucket in LOCKED_BUCKETS}
    for row in images:
        counts[row["bucket"]] += 1
    if any(count < 2 for count in counts.values()):
        raise ValueError("locked combined manifest needs two images in every bucket")
    if len(texts) != len(images):
        raise ValueError("locked image/text sample counts must match")
    digest = _sha(normalized["canonical_sha256"], "manifest.canonical_sha256")
    unhashed = {key: item for key, item in normalized.items() if key != "canonical_sha256"}
    if _canonical_sha(unhashed) != digest:
        raise ValueError("locked combined manifest canonical SHA-256 differs")
    return normalized


def build_locked_combined_manifest(
    *,
    image_records: Sequence[Mapping[str, Any]],
    text_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "kind": LOCKED_COMBINED_MANIFEST_KIND,
        "image_records": [dict(row) for row in image_records],
        "text_records": [dict(row) for row in text_records],
    }
    payload["canonical_sha256"] = _canonical_sha(payload)
    return validate_locked_combined_manifest(payload)


def locked_combined_manifest_commitments(value: Mapping[str, Any]) -> dict[str, str]:
    return _manifest_commitments(validate_locked_combined_manifest(value))


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate locked manifest JSON key {key!r}")
        result[key] = value
    return result


def _open_dir_component(parent_fd: int, name: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=parent_fd)
    info = os.fstat(descriptor)
    if not stat.S_ISDIR(info.st_mode):
        os.close(descriptor)
        raise ValueError("sealed path component is not a real directory")
    return descriptor


def _read_sealed_file(
    root_fd: int,
    relpath: str,
    *,
    expected_sha256: str,
    expected_size: int,
) -> bytes:
    parts = PurePosixPath(_relpath(relpath, "sealed relpath")).parts
    current = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            next_fd = _open_dir_component(current, part)
            os.close(current)
            current = next_fd
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(parts[-1], flags, dir_fd=current)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
                raise ValueError("sealed file type or size differs from receipt")
            chunks: list[bytes] = []
            remaining = expected_size
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError("sealed file ended before its declared size")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise ValueError("sealed file exceeds its declared size")
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    finally:
        os.close(current)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError("sealed file changed while it was read")
    payload = b"".join(chunks)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("sealed file SHA-256 differs from receipt")
    return payload


@dataclass(frozen=True, slots=True)
class LockedImageExample:
    id: str
    bucket: str
    asset_bytes: bytes
    asset_sha256: str
    reference: str
    reference_token_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class LockedTextExample:
    id: str
    reference: str
    reference_token_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class LockedBenchmarkBatchSource:
    build_receipt_sha256: str
    manifest_canonical_sha256: str
    image_dataset_contract_sha256: str
    text_dataset_contract_sha256: str
    images: tuple[LockedImageExample, ...]
    texts: tuple[LockedTextExample, ...]


def _encode_exact(
    text: str,
    *,
    encode_reference: Callable[[str], Sequence[int]],
    decode_ids: Callable[[Sequence[int]], str],
    expected_count: int,
) -> tuple[int, ...]:
    values = tuple(int(value) for value in encode_reference(text))
    if len(values) != expected_count or decode_ids(values) != text:
        raise ValueError("locked native token count or exact roundtrip differs")
    if any(value < 0 for value in values):
        raise ValueError("locked native encoder produced a negative token ID")
    return values


def load_locked_benchmark_after_claim(
    sealed_root: str | Path,
    preclaim: LockedGoldenPreclaim,
    claim: LockedGoldenClaim,
    *,
    encode_reference: Callable[[str], Sequence[int]],
    decode_ids: Callable[[Sequence[int]], str],
) -> LockedBenchmarkBatchSource:
    if type(preclaim) is not LockedGoldenPreclaim:
        raise TypeError("validated LockedGoldenPreclaim is required")
    marker = validate_locked_golden_claim(
        claim,
        expected_build_receipt_sha256=preclaim.build_receipt_sha256,
    )
    if marker["claim_payload"].get("locked_golden_anchor_sha256") != (
        preclaim.anchor_canonical_sha256
    ):
        raise ValueError("claim payload is bound to another locked anchor")
    receipt = preclaim.build_receipt
    root = Path(sealed_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("sealed_root must be a real directory")
    root_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    root_flags |= getattr(os, "O_NOFOLLOW", 0)
    root_fd = os.open(root, root_flags)
    try:
        manifest_entry = receipt["manifest_file"]
        manifest_bytes = _read_sealed_file(
            root_fd,
            manifest_entry["relpath"],
            expected_sha256=manifest_entry["sha256"],
            expected_size=manifest_entry["size_bytes"],
        )
        try:
            manifest_value = json.loads(
                manifest_bytes.decode("utf-8"),
                object_pairs_hook=_object_pairs,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("sealed combined manifest is not strict UTF-8 JSON") from exc
        manifest = validate_locked_combined_manifest(manifest_value)
        declared = {row["relpath"]: row for row in receipt["asset_files"]}
        captured = {
            relpath: _read_sealed_file(
                root_fd,
                relpath,
                expected_sha256=entry["sha256"],
                expected_size=entry["size_bytes"],
            )
            for relpath, entry in declared.items()
        }
    finally:
        os.close(root_fd)
    commitments = _manifest_commitments(manifest)
    for key, digest in commitments.items():
        if receipt[key] != digest:
            raise ValueError(f"sealed manifest commitment differs for {key}")
    counts = {bucket: 0 for bucket in LOCKED_BUCKETS}
    image_examples: list[LockedImageExample] = []
    for row in manifest["image_records"]:
        entry = declared.get(row["asset_relpath"])
        if entry is None or entry["sha256"] != row["asset_sha256"] or entry[
            "size_bytes"
        ] != row["asset_size_bytes"]:
            raise ValueError("sealed image asset differs from build receipt")
        counts[row["bucket"]] += 1
        image_examples.append(
            LockedImageExample(
                id=row["id"],
                bucket=row["bucket"],
                asset_bytes=captured[row["asset_relpath"]],
                asset_sha256=row["asset_sha256"],
                reference=row["reference"],
                reference_token_ids=_encode_exact(
                    row["reference"],
                    encode_reference=encode_reference,
                    decode_ids=decode_ids,
                    expected_count=row["reference_token_count"],
                ),
            )
        )
    if counts != receipt["bucket_counts"]:
        raise ValueError("sealed manifest bucket counts differ from build receipt")
    text_examples = tuple(
        LockedTextExample(
            id=row["id"],
            reference=row["reference"],
            reference_token_ids=_encode_exact(
                row["reference"],
                encode_reference=encode_reference,
                decode_ids=decode_ids,
                expected_count=row["reference_token_count"],
            ),
        )
        for row in manifest["text_records"]
    )
    lengths = [len(row.reference_token_ids) for row in text_examples]
    stats = receipt["token_stats"]
    if (
        stats["sample_count"] != len(text_examples)
        or stats["minimum_reference_tokens"] != min(lengths)
        or stats["maximum_reference_tokens"] != max(lengths)
        or stats["total_reference_tokens"] != sum(lengths)
    ):
        raise ValueError("sealed text token statistics differ from build receipt")
    return LockedBenchmarkBatchSource(
        build_receipt_sha256=preclaim.build_receipt_sha256,
        manifest_canonical_sha256=manifest["canonical_sha256"],
        image_dataset_contract_sha256=commitments[
            "image_dataset_contract_sha256"
        ],
        text_dataset_contract_sha256=commitments[
            "text_dataset_contract_sha256"
        ],
        images=tuple(image_examples),
        texts=text_examples,
    )


__all__ = [
    "LOCKED_COMBINED_MANIFEST_KIND",
    "LockedBenchmarkBatchSource",
    "LockedImageExample",
    "LockedTextExample",
    "build_locked_combined_manifest",
    "load_locked_benchmark_after_claim",
    "locked_combined_manifest_commitments",
    "validate_locked_combined_manifest",
]
