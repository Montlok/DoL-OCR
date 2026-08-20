# -*- coding: utf-8 -*-

"""Completion receipts for streamed OCR shard producers.

This module owns only the integrity primitives shared by shard builders:
constant-memory manifests, exact-byte output hashing, sentinel construction,
and fail-closed validation.  Corpus routing and rebuild policy remain with the
producer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from Model.ocr.alignment_contract import file_sha256
from Model.ocr.tokenization import canonical_json_sha256

SHARD_SENTINEL_SCHEMA_VERSION = 2
SHARD_SENTINEL_KIND = "native_ocr_shard_completion"
_SHA256_HEX_LENGTH = 64
_SENTINEL_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "mode",
        "shard_index",
        "tokenization_contract_sha256",
        "producer_algorithm",
        "build_parameters",
        "build_parameters_sha256",
        "source_identity",
        "source_manifest_sha256",
        "source_record_count",
        "outputs",
        "images",
        "counters",
        "sentinel_canonical_sha256",
    }
)


def update_canonical_manifest(digest, record: Any) -> None:
    """Append one canonical JSON record to a streaming digest."""

    digest.update(
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\n")


@dataclass
class ImageBindingAccumulator:
    """Constant-memory aggregate of images referenced by one shard."""

    digest: Any = field(default_factory=hashlib.sha256, repr=False)
    reference_count: int = 0
    total_size_bytes: int = 0

    def add(self, image_ref: str, image_sha256: str, size_bytes: int) -> None:
        update_canonical_manifest(
            self.digest,
            [image_ref, image_sha256, int(size_bytes)],
        )
        self.reference_count += 1
        self.total_size_bytes += int(size_bytes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.digest.hexdigest(),
            "reference_count": self.reference_count,
            "total_size_bytes": self.total_size_bytes,
        }


class DigestingTextWriter:
    """Track exact UTF-8 output bytes while writing them once."""

    def __init__(self, handle, path: Path):
        self._handle = handle
        self.path = path
        self._digest = hashlib.sha256()
        self._size_bytes = 0

    def write(self, text: str) -> int:
        encoded = text.encode("utf-8")
        self._digest.update(encoded)
        self._size_bytes += len(encoded)
        return self._handle.write(text)

    def artifact(self, out_dir: Path) -> dict[str, Any]:
        return {
            "path": self.path.relative_to(out_dir).as_posix(),
            "sha256": self._digest.hexdigest(),
            "size_bytes": self._size_bytes,
        }


def validate_sha256(value: str, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_HEX_LENGTH
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
    return value


def producer_algorithm_contract(
    source_paths: tuple[tuple[str, Path], ...],
) -> dict[str, Any]:
    """Fingerprint the complete producer implementation surface."""

    entries = [
        {"path": name, "sha256": file_sha256(path)}
        for name, path in source_paths
    ]
    return {
        "version": 1,
        "source_files": entries,
        "aggregate_sha256": canonical_json_sha256(entries),
    }


def build_shard_sentinel(
    *,
    mode: str,
    shard_index: int,
    tokenization_contract_sha256: str,
    producer_algorithm: dict[str, Any],
    build_parameters: dict[str, Any],
    source_identity: dict[str, Any],
    source_manifest_sha256: str,
    source_record_count: int,
    output_artifacts: list[dict[str, Any]],
    image_manifest: ImageBindingAccumulator,
    counters: dict[str, Any],
) -> dict[str, Any]:
    """Build a self-authenticating completion receipt."""

    validate_sha256(
        tokenization_contract_sha256,
        field_name="tokenization_contract_sha256",
    )
    validate_sha256(
        source_manifest_sha256,
        field_name="source_manifest_sha256",
    )
    core = {
        "schema_version": SHARD_SENTINEL_SCHEMA_VERSION,
        "kind": SHARD_SENTINEL_KIND,
        "mode": mode,
        "shard_index": shard_index,
        "tokenization_contract_sha256": tokenization_contract_sha256,
        "producer_algorithm": producer_algorithm,
        "build_parameters": build_parameters,
        "build_parameters_sha256": canonical_json_sha256(build_parameters),
        "source_identity": source_identity,
        "source_manifest_sha256": source_manifest_sha256,
        "source_record_count": int(source_record_count),
        "outputs": output_artifacts,
        "images": image_manifest.as_dict(),
        "counters": counters,
    }
    return {
        **core,
        "sentinel_canonical_sha256": canonical_json_sha256(core),
    }


def _validate_output_artifacts(artifacts: Any, out_dir: Path) -> None:
    if not isinstance(artifacts, list) or len(artifacts) != 4:
        raise ValueError("sentinel outputs must contain four shard files")
    seen: set[str] = set()
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict) or set(artifact) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ValueError(f"sentinel outputs[{index}] has an invalid schema")
        relative = artifact["path"]
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative in seen
        ):
            raise ValueError(f"sentinel outputs[{index}].path is unsafe")
        seen.add(relative)
        expected_digest = validate_sha256(
            artifact["sha256"],
            field_name=f"outputs[{index}].sha256",
        )
        expected_size = artifact["size_bytes"]
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
        ):
            raise ValueError(f"sentinel outputs[{index}].size_bytes is invalid")
        path = out_dir / relative
        try:
            actual_size = path.stat().st_size
        except OSError as exc:
            raise ValueError(
                f"sentinel output is unavailable: {path}: {exc}"
            ) from exc
        if actual_size != expected_size:
            raise ValueError(f"sentinel output size changed: {path}")
        if file_sha256(path) != expected_digest:
            raise ValueError(f"sentinel output digest changed: {path}")


def _validate_producer_algorithm(payload: Any) -> None:
    if not isinstance(payload, dict) or set(payload) != {
        "version",
        "source_files",
        "aggregate_sha256",
    }:
        raise ValueError("sentinel producer_algorithm schema is invalid")
    files = payload["source_files"]
    if not isinstance(files, list) or not files:
        raise ValueError("sentinel producer source_files must be non-empty")
    for index, entry in enumerate(files):
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise ValueError(
                f"sentinel producer source_files[{index}] schema is invalid"
            )
        if not isinstance(entry["path"], str) or not entry["path"]:
            raise ValueError(
                f"sentinel producer source_files[{index}].path is invalid"
            )
        validate_sha256(
            entry["sha256"],
            field_name=f"producer source_files[{index}].sha256",
        )
    if payload["aggregate_sha256"] != canonical_json_sha256(files):
        raise ValueError("sentinel producer aggregate digest is invalid")


def validate_shard_sentinel_payload(
    sentinel_path: Path,
    out_dir: Path,
    *,
    counter_keys: frozenset[str],
) -> dict[str, Any]:
    """Validate a completion receipt without assuming one launch config."""

    payload = json.loads(sentinel_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != _SENTINEL_KEYS:
        raise ValueError("sentinel schema is incomplete or unknown")
    core = {
        key: value
        for key, value in payload.items()
        if key != "sentinel_canonical_sha256"
    }
    if (
        payload["schema_version"] != SHARD_SENTINEL_SCHEMA_VERSION
        or payload["kind"] != SHARD_SENTINEL_KIND
    ):
        raise ValueError("sentinel kind/schema differs from runtime")
    if payload["mode"] not in {"tar", "hanshi"}:
        raise ValueError("sentinel mode is invalid")
    payload_shard = payload["shard_index"]
    if (
        isinstance(payload_shard, bool)
        or not isinstance(payload_shard, int)
        or payload_shard < 0
    ):
        raise ValueError("sentinel shard_index is invalid")
    if payload["sentinel_canonical_sha256"] != canonical_json_sha256(core):
        raise ValueError("sentinel canonical digest is invalid")
    validate_sha256(
        payload["tokenization_contract_sha256"],
        field_name="tokenization_contract_sha256",
    )
    _validate_producer_algorithm(payload["producer_algorithm"])
    build_parameters = payload["build_parameters"]
    if not isinstance(build_parameters, dict) or not build_parameters:
        raise ValueError("sentinel build_parameters is invalid")
    if payload["build_parameters_sha256"] != canonical_json_sha256(
        build_parameters
    ):
        raise ValueError("sentinel build parameter digest is invalid")
    if not isinstance(payload["source_identity"], dict):
        raise ValueError("sentinel source_identity is invalid")
    validate_sha256(
        payload["source_manifest_sha256"],
        field_name="source_manifest_sha256",
    )
    source_count = payload["source_record_count"]
    if (
        isinstance(source_count, bool)
        or not isinstance(source_count, int)
        or source_count < 0
    ):
        raise ValueError("sentinel source_record_count is invalid")
    counters = payload["counters"]
    if not isinstance(counters, dict) or set(counters) != counter_keys:
        raise ValueError("sentinel counters schema is invalid")
    for key, value in counters.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"sentinel counter {key!r} is invalid")
    if source_count != counters.get("n_samples_seen"):
        raise ValueError("sentinel source record count differs from counters")
    images = payload["images"]
    if not isinstance(images, dict) or set(images) != {
        "sha256",
        "reference_count",
        "total_size_bytes",
    }:
        raise ValueError("sentinel image manifest schema is invalid")
    validate_sha256(images["sha256"], field_name="images.sha256")
    image_count = images["reference_count"]
    image_size = images["total_size_bytes"]
    if (
        isinstance(image_count, bool)
        or not isinstance(image_count, int)
        or image_count < 0
        or image_count
        != counters.get("n_align_written", -1)
        + counters.get("n_val_written", -1)
    ):
        raise ValueError("sentinel image count differs from emitted rows")
    if (
        isinstance(image_size, bool)
        or not isinstance(image_size, int)
        or image_size < 0
    ):
        raise ValueError("sentinel image byte total is invalid")
    _validate_output_artifacts(payload["outputs"], out_dir)
    return payload


def validate_shard_sentinel(
    sentinel_path: Path,
    out_dir: Path,
    *,
    mode: str,
    shard_index: int,
    tokenization_contract_sha256: str,
    producer_algorithm: dict[str, Any],
    build_parameters: dict[str, Any],
    source_identity: dict[str, Any],
    counter_keys: frozenset[str],
) -> dict[str, Any]:
    """Validate one completion receipt against the current shard build."""

    payload = validate_shard_sentinel_payload(
        sentinel_path,
        out_dir,
        counter_keys=counter_keys,
    )
    if payload["mode"] != mode or payload["shard_index"] != shard_index:
        raise ValueError("sentinel identity differs from this build")
    validate_sha256(
        tokenization_contract_sha256,
        field_name="tokenization_contract_sha256",
    )
    if payload["tokenization_contract_sha256"] != tokenization_contract_sha256:
        raise ValueError("sentinel tokenizer contract changed")
    if payload["producer_algorithm"] != producer_algorithm:
        raise ValueError("sentinel producer algorithm changed")
    if payload["build_parameters"] != build_parameters:
        raise ValueError("sentinel build parameters changed")
    if payload["source_identity"] != source_identity:
        raise ValueError("sentinel source identity changed")
    return payload["counters"]


__all__ = [
    "DigestingTextWriter",
    "ImageBindingAccumulator",
    "SHARD_SENTINEL_KIND",
    "SHARD_SENTINEL_SCHEMA_VERSION",
    "build_shard_sentinel",
    "producer_algorithm_contract",
    "update_canonical_manifest",
    "validate_sha256",
    "validate_shard_sentinel",
    "validate_shard_sentinel_payload",
]
