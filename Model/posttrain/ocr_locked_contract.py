# -*- coding: utf-8 -*-

"""Sealed AnyRes OCR locked-benchmark contracts and one-shot claims.

This module deliberately stops at the boundary immediately before the locked
manifest and image files may be opened.  Pre-claim code reads only the public,
label-free anchor and the sealed build receipt itself through stable file
descriptors.  Paths named by the receipt are data, not instructions, until an
evaluator has durably consumed the receipt-keyed one-shot claim.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


LOCKED_CONTRACT_SCHEMA_VERSION = 1
LOCKED_GOLDEN_ANCHOR_KIND = "dol_ocr_anyres_locked_golden_anchor_v1"
LOCKED_BUILD_RECEIPT_KIND = "dol_ocr_anyres_locked_build_receipt_v1"
LOCKED_GOLDEN_CLAIM_KIND = "dol_ocr_anyres_locked_golden_claim_v1"
LOCKED_INCOMPLETE_STUB_KIND = (
    "dol_ocr_anyres_locked_evaluation_incomplete_v1"
)
LOCKED_IDENTITY_COMMITMENT_SCHEME = (
    "sha256_domain_separated_blinded_identity_rows_v1"
)

LOCKED_BUCKETS = (
    "print",
    "handwritten_good",
    "handwritten_medium",
    "handwritten_poor",
)

_CONTAMINATION_INPUTS = (
    "source_pretraining_corpus",
    "anyres_train",
    "anyres_sft_validation",
    "anyres_kl_selection",
    "anyres_formal_monitor",
    "text_replay_train",
    "text_replay_sft_validation",
    "text_replay_kl_selection",
    "text_replay_formal_monitor",
)
_CONTAMINATION_IDENTITIES = (
    "sample_id",
    "source_document_id",
    "source_capture_id",
    "writer_id",
    "canonical_pixel_sha256",
    "near_duplicate_cluster",
    "reference_text_sha256",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MAX_ANCHOR_BYTES = 1024 * 1024
_MAX_BUILD_RECEIPT_BYTES = 128 * 1024 * 1024

_FILE_KEYS = frozenset({"relpath", "sha256", "size_bytes"})
_BUCKET_KEYS = frozenset(LOCKED_BUCKETS)
_TOKEN_STATS_KEYS = frozenset(
    {
        "sample_count",
        "minimum_reference_tokens",
        "maximum_reference_tokens",
        "total_reference_tokens",
        "recommended_max_new_tokens",
    }
)
_SCOPE_KEYS = frozenset(
    {"covered_inputs", "modalities", "identity_dimensions", "relation"}
)
_BUILD_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "manifest_file",
        "asset_files",
        "image_dataset_contract_sha256",
        "text_dataset_contract_sha256",
        "preprocess_contract_sha256",
        "tokenizer_contract_sha256",
        "identity_commitment_scheme",
        "image_identity_commitment_sha256",
        "text_identity_commitment_sha256",
        "membership_commitment_sha256",
        "cross_exclusion_receipt_sha256",
        "contamination_scope",
        "bucket_counts",
        "token_stats",
        "builder_source_sha256",
        "canonical_sha256",
    }
)
_ANCHOR_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "build_receipt_sha256",
        "identity_commitment_scheme",
        "image_identity_commitment_sha256",
        "text_identity_commitment_sha256",
        "membership_commitment_sha256",
        "cross_exclusion_receipt_sha256",
        "preprocess_contract_sha256",
        "tokenizer_contract_sha256",
        "contamination_scope",
        "bucket_counts",
        "canonical_sha256",
    }
)
_ANCHOR_RECEIPT_BINDINGS = (
    "identity_commitment_scheme",
    "image_identity_commitment_sha256",
    "text_identity_commitment_sha256",
    "membership_commitment_sha256",
    "cross_exclusion_receipt_sha256",
    "preprocess_contract_sha256",
    "tokenizer_contract_sha256",
    "contamination_scope",
    "bucket_counts",
)
_CLAIM_MARKER_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "build_receipt_sha256",
        "claim_payload",
        "claim_payload_sha256",
        "canonical_sha256",
    }
)
_INCOMPLETE_STUB_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "status",
        "build_receipt_sha256",
        "claim_marker_sha256",
        "claim_payload_sha256",
        "stub_payload",
        "canonical_sha256",
    }
)


def _canonical_json_bytes(value: object, *, newline: bool = False) -> bytes:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if newline:
        rendered += "\n"
    return rendered.encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _strict_object_pairs(where: str):
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{where}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    return hook


def _json_clone(value: object, *, where: str) -> Any:
    try:
        encoded = _canonical_json_bytes(value)
        return json.loads(
            encoded,
            object_pairs_hook=_strict_object_pairs(where),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{where} must be finite canonical JSON") from exc


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    where: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"{where} has an invalid exact schema; "
            f"missing={missing}, unknown={unknown}"
        )


def _require_sha256(value: object, *, where: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{where} must be a lowercase SHA-256")
    return value


def _normalized_posix_relpath(value: object, *, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} must be a non-empty POSIX relative path")
    if "\x00" in value or "\\" in value:
        raise ValueError(f"{where} is not a safe normalized POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{where} is not a safe normalized POSIX path")
    if path.as_posix() != value:
        raise ValueError(f"{where} is not a normalized POSIX path")
    return value


def _validate_file_entry(value: object, *, where: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be an object")
    _require_exact_keys(value, _FILE_KEYS, where=where)
    relpath = _normalized_posix_relpath(value["relpath"], where=f"{where}.relpath")
    digest = _require_sha256(value["sha256"], where=f"{where}.sha256")
    size = value["size_bytes"]
    if type(size) is not int or size <= 0:
        raise ValueError(f"{where}.size_bytes must be a positive integer")
    return {"relpath": relpath, "sha256": digest, "size_bytes": size}


def _expected_contamination_scope() -> dict[str, Any]:
    return {
        "covered_inputs": list(_CONTAMINATION_INPUTS),
        "modalities": ["image", "text"],
        "identity_dimensions": list(_CONTAMINATION_IDENTITIES),
        "relation": "locked_golden_disjoint_from_all_covered_inputs_v1",
    }


def _validate_contamination_scope(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("contamination_scope must be an object")
    _require_exact_keys(value, _SCOPE_KEYS, where="contamination_scope")
    normalized = _json_clone(value, where="contamination_scope")
    if normalized != _expected_contamination_scope():
        raise ValueError(
            "contamination_scope must cover every training, tuning, public "
            "evaluation, and source-pretraining identity surface"
        )
    return normalized


def _validate_bucket_counts(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError("bucket_counts must be an object")
    _require_exact_keys(value, _BUCKET_KEYS, where="bucket_counts")
    normalized: dict[str, int] = {}
    for bucket in LOCKED_BUCKETS:
        count = value[bucket]
        if type(count) is not int or count < 2:
            raise ValueError(
                f"bucket_counts.{bucket} must contain at least two samples"
            )
        normalized[bucket] = count
    return normalized


def _validate_token_stats(
    value: object,
    *,
    sample_count: int,
) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError("token_stats must be an object")
    _require_exact_keys(value, _TOKEN_STATS_KEYS, where="token_stats")
    normalized: dict[str, int] = {}
    for key in _TOKEN_STATS_KEYS:
        item = value[key]
        if type(item) is not int or item <= 0:
            raise ValueError(f"token_stats.{key} must be a positive integer")
        normalized[key] = item
    if normalized["sample_count"] != sample_count:
        raise ValueError("token_stats.sample_count differs from bucket_counts")
    minimum = normalized["minimum_reference_tokens"]
    maximum = normalized["maximum_reference_tokens"]
    total = normalized["total_reference_tokens"]
    if minimum > maximum:
        raise ValueError("token_stats minimum exceeds maximum")
    if not sample_count * minimum <= total <= sample_count * maximum:
        raise ValueError("token_stats total is inconsistent with its range")
    if normalized["recommended_max_new_tokens"] < maximum + 1:
        raise ValueError(
            "recommended_max_new_tokens must include the longest reference "
            "and its terminal token"
        )
    return {key: normalized[key] for key in sorted(normalized)}


def _validate_self_hash(
    normalized: Mapping[str, Any],
    *,
    where: str,
) -> None:
    expected = _require_sha256(
        normalized["canonical_sha256"],
        where=f"{where}.canonical_sha256",
    )
    unhashed = {
        key: value
        for key, value in normalized.items()
        if key != "canonical_sha256"
    }
    if _canonical_sha256(unhashed) != expected:
        raise ValueError(f"{where}.canonical_sha256 is invalid")


def validate_locked_build_receipt(value: object) -> dict[str, Any]:
    """Validate the sealed receipt without opening any receipt-named path."""

    if not isinstance(value, Mapping):
        raise ValueError("locked build receipt must be an object")
    _require_exact_keys(value, _BUILD_RECEIPT_KEYS, where="locked build receipt")
    normalized = _json_clone(value, where="locked build receipt")
    if normalized["schema_version"] != LOCKED_CONTRACT_SCHEMA_VERSION:
        raise ValueError("locked build receipt schema_version is unsupported")
    if normalized["kind"] != LOCKED_BUILD_RECEIPT_KIND:
        raise ValueError("locked build receipt kind is invalid")

    normalized["manifest_file"] = _validate_file_entry(
        normalized["manifest_file"],
        where="locked build receipt.manifest_file",
    )
    asset_values = normalized["asset_files"]
    if not isinstance(asset_values, list) or not asset_values:
        raise ValueError("locked build receipt.asset_files must be a non-empty list")
    assets = [
        _validate_file_entry(item, where=f"locked build receipt.asset_files[{index}]")
        for index, item in enumerate(asset_values)
    ]
    asset_paths = [str(item["relpath"]) for item in assets]
    if asset_paths != sorted(asset_paths) or len(asset_paths) != len(set(asset_paths)):
        raise ValueError(
            "locked build receipt.asset_files must have unique sorted relpaths"
        )
    if normalized["manifest_file"]["relpath"] in set(asset_paths):
        raise ValueError("locked manifest and asset relpaths must be distinct")
    normalized["asset_files"] = assets

    for key in (
        "image_dataset_contract_sha256",
        "text_dataset_contract_sha256",
        "preprocess_contract_sha256",
        "tokenizer_contract_sha256",
        "image_identity_commitment_sha256",
        "text_identity_commitment_sha256",
        "membership_commitment_sha256",
        "cross_exclusion_receipt_sha256",
        "builder_source_sha256",
    ):
        normalized[key] = _require_sha256(
            normalized[key],
            where=f"locked build receipt.{key}",
        )
    if (
        normalized["identity_commitment_scheme"]
        != LOCKED_IDENTITY_COMMITMENT_SCHEME
    ):
        raise ValueError("locked build receipt identity commitment scheme differs")
    normalized["contamination_scope"] = _validate_contamination_scope(
        normalized["contamination_scope"]
    )
    normalized["bucket_counts"] = _validate_bucket_counts(
        normalized["bucket_counts"]
    )
    sample_count = sum(normalized["bucket_counts"].values())
    normalized["token_stats"] = _validate_token_stats(
        normalized["token_stats"],
        sample_count=sample_count,
    )
    _validate_self_hash(normalized, where="locked build receipt")
    return normalized


def build_locked_golden_anchor(
    *,
    build_receipt_sha256: str,
    build_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the only label-free locked-golden object visible to training."""

    receipt_sha = _require_sha256(
        build_receipt_sha256,
        where="build_receipt_sha256",
    )
    receipt = validate_locked_build_receipt(build_receipt)
    payload = {
        "schema_version": LOCKED_CONTRACT_SCHEMA_VERSION,
        "kind": LOCKED_GOLDEN_ANCHOR_KIND,
        "build_receipt_sha256": receipt_sha,
        **{
            key: _json_clone(receipt[key], where=f"locked anchor.{key}")
            for key in _ANCHOR_RECEIPT_BINDINGS
        },
    }
    payload["canonical_sha256"] = _canonical_sha256(payload)
    return validate_locked_golden_anchor(payload)


def validate_locked_golden_anchor(value: object) -> dict[str, Any]:
    """Validate the exact, label-free public locked-golden anchor schema."""

    if not isinstance(value, Mapping):
        raise ValueError("locked golden anchor must be an object")
    _require_exact_keys(value, _ANCHOR_KEYS, where="locked golden anchor")
    normalized = _json_clone(value, where="locked golden anchor")
    if normalized["schema_version"] != LOCKED_CONTRACT_SCHEMA_VERSION:
        raise ValueError("locked golden anchor schema_version is unsupported")
    if normalized["kind"] != LOCKED_GOLDEN_ANCHOR_KIND:
        raise ValueError("locked golden anchor kind is invalid")
    normalized["build_receipt_sha256"] = _require_sha256(
        normalized["build_receipt_sha256"],
        where="locked golden anchor.build_receipt_sha256",
    )
    for key in (
        "image_identity_commitment_sha256",
        "text_identity_commitment_sha256",
        "membership_commitment_sha256",
        "cross_exclusion_receipt_sha256",
        "preprocess_contract_sha256",
        "tokenizer_contract_sha256",
    ):
        normalized[key] = _require_sha256(
            normalized[key],
            where=f"locked golden anchor.{key}",
        )
    if (
        normalized["identity_commitment_scheme"]
        != LOCKED_IDENTITY_COMMITMENT_SCHEME
    ):
        raise ValueError("locked golden anchor identity commitment scheme differs")
    normalized["contamination_scope"] = _validate_contamination_scope(
        normalized["contamination_scope"]
    )
    normalized["bucket_counts"] = _validate_bucket_counts(
        normalized["bucket_counts"]
    )
    _validate_self_hash(normalized, where="locked golden anchor")
    return normalized


def _stable_read_regular(path: str | Path, *, where: str, limit: int) -> tuple[bytes, os.stat_result]:
    source = os.fspath(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    elif Path(source).is_symlink():  # pragma: no cover - modern targets have O_NOFOLLOW
        raise ValueError(f"{where} must not be a symlink")
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise ValueError(f"{where} cannot be opened safely") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{where} must be a regular file")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, limit + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise ValueError(f"{where} exceeds its byte limit")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_identity != after_identity or total != after.st_size:
            raise ValueError(f"{where} changed while it was read")
        return b"".join(chunks), after
    finally:
        os.close(descriptor)


def _stable_read_json(
    path: str | Path,
    *,
    where: str,
    limit: int,
) -> tuple[dict[str, Any], bytes, os.stat_result]:
    raw, identity = _stable_read_regular(path, where=where, limit=limit)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object_pairs(where),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{where} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{where} must contain a JSON object")
    return value, raw, identity


_PRECLAIM_FACTORY_TOKEN = object()


@dataclass(frozen=True, slots=True, init=False)
class LockedGoldenPreclaim:
    """Validated pre-claim material; properties return defensive JSON copies."""

    build_receipt_sha256: str
    anchor_canonical_sha256: str
    build_receipt_canonical_sha256: str
    anchor_file_sha256: str
    _anchor_json: bytes
    _build_receipt_json: bytes

    def __init__(
        self,
        *,
        factory_token: object,
        build_receipt_sha256: str,
        anchor_canonical_sha256: str,
        build_receipt_canonical_sha256: str,
        anchor_file_sha256: str,
        anchor_json: bytes,
        build_receipt_json: bytes,
    ) -> None:
        if factory_token is not _PRECLAIM_FACTORY_TOKEN:
            raise TypeError(
                "LockedGoldenPreclaim can only be created by stable preclaim"
            )
        object.__setattr__(self, "build_receipt_sha256", build_receipt_sha256)
        object.__setattr__(self, "anchor_canonical_sha256", anchor_canonical_sha256)
        object.__setattr__(
            self,
            "build_receipt_canonical_sha256",
            build_receipt_canonical_sha256,
        )
        object.__setattr__(self, "anchor_file_sha256", anchor_file_sha256)
        object.__setattr__(self, "_anchor_json", bytes(anchor_json))
        object.__setattr__(self, "_build_receipt_json", bytes(build_receipt_json))

    @property
    def anchor(self) -> dict[str, Any]:
        return json.loads(self._anchor_json)

    @property
    def build_receipt(self) -> dict[str, Any]:
        return json.loads(self._build_receipt_json)


def preclaim_locked_benchmark(
    anchor_path: str | Path,
    build_receipt_path: str | Path,
) -> LockedGoldenPreclaim:
    """Validate only anchor/receipt bytes, never a receipt-referenced path."""

    anchor_value, anchor_raw, _ = _stable_read_json(
        anchor_path,
        where="locked golden anchor file",
        limit=_MAX_ANCHOR_BYTES,
    )
    receipt_value, receipt_raw, _ = _stable_read_json(
        build_receipt_path,
        where="locked build receipt file",
        limit=_MAX_BUILD_RECEIPT_BYTES,
    )
    anchor = validate_locked_golden_anchor(anchor_value)
    receipt = validate_locked_build_receipt(receipt_value)
    receipt_file_sha = hashlib.sha256(receipt_raw).hexdigest()
    if anchor["build_receipt_sha256"] != receipt_file_sha:
        raise ValueError("locked golden anchor build receipt SHA-256 differs")
    for key in _ANCHOR_RECEIPT_BINDINGS:
        if anchor[key] != receipt[key]:
            raise ValueError(
                f"locked golden anchor differs from build receipt for {key}"
            )
    return LockedGoldenPreclaim(
        factory_token=_PRECLAIM_FACTORY_TOKEN,
        build_receipt_sha256=receipt_file_sha,
        anchor_canonical_sha256=anchor["canonical_sha256"],
        build_receipt_canonical_sha256=receipt["canonical_sha256"],
        anchor_file_sha256=hashlib.sha256(anchor_raw).hexdigest(),
        anchor_json=anchor_raw,
        build_receipt_json=receipt_raw,
    )


def validate_locked_golden_preclaim(
    preclaim: LockedGoldenPreclaim,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate a preclaim object and revalidate its captured bytes."""

    if type(preclaim) is not LockedGoldenPreclaim:
        raise TypeError("a genuine LockedGoldenPreclaim is required")
    try:
        anchor_value = json.loads(
            preclaim._anchor_json.decode("utf-8"),
            object_pairs_hook=_strict_object_pairs("locked golden anchor file"),
        )
        receipt_value = json.loads(
            preclaim._build_receipt_json.decode("utf-8"),
            object_pairs_hook=_strict_object_pairs("locked build receipt file"),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("captured locked preclaim JSON is invalid") from exc
    anchor = validate_locked_golden_anchor(anchor_value)
    receipt = validate_locked_build_receipt(receipt_value)
    receipt_file_sha = hashlib.sha256(preclaim._build_receipt_json).hexdigest()
    anchor_file_sha = hashlib.sha256(preclaim._anchor_json).hexdigest()
    if receipt_file_sha != preclaim.build_receipt_sha256:
        raise ValueError("captured locked build receipt bytes changed")
    if anchor_file_sha != preclaim.anchor_file_sha256:
        raise ValueError("captured locked golden anchor bytes changed")
    if anchor["canonical_sha256"] != preclaim.anchor_canonical_sha256:
        raise ValueError("captured locked golden anchor identity changed")
    if (
        receipt["canonical_sha256"]
        != preclaim.build_receipt_canonical_sha256
    ):
        raise ValueError("captured locked build receipt identity changed")
    if anchor["build_receipt_sha256"] != receipt_file_sha:
        raise ValueError("locked golden anchor build receipt SHA-256 differs")
    for key in _ANCHOR_RECEIPT_BINDINGS:
        if anchor[key] != receipt[key]:
            raise ValueError(
                f"locked golden anchor differs from build receipt for {key}"
            )
    return anchor, receipt


def _open_real_directory(path: Path, *, where: str) -> tuple[int, os.stat_result]:
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise ValueError(f"{where} is not an accessible directory") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise ValueError(f"{where} must be a real directory, not a symlink")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{where} cannot be opened without following links") from exc
    after = os.fstat(descriptor)
    if not stat.S_ISDIR(after.st_mode) or (
        before.st_dev,
        before.st_ino,
    ) != (after.st_dev, after.st_ino):
        os.close(descriptor)
        raise ValueError(f"{where} changed while it was opened")
    return descriptor, after


def _prepare_ledger_directory(path: str | Path) -> tuple[Path, int, os.stat_result]:
    ledger = Path(path).absolute()
    created = False
    try:
        current = os.lstat(ledger)
    except FileNotFoundError:
        try:
            os.mkdir(ledger, mode=0o700)
            created = True
        except FileExistsError:
            pass
        except OSError as exc:
            raise ValueError("locked benchmark ledger cannot be created safely") from exc
    else:
        if stat.S_ISLNK(current.st_mode) or not stat.S_ISDIR(current.st_mode):
            raise ValueError(
                "locked benchmark ledger must be a real directory, not a symlink"
            )
    descriptor, identity = _open_real_directory(
        ledger,
        where="locked benchmark ledger",
    )
    if created:
        os.fchmod(descriptor, 0o700)
    return ledger, descriptor, identity


_CLAIM_FACTORY_TOKEN = object()


@dataclass(frozen=True, slots=True, init=False)
class LockedGoldenClaim:
    """Unforgeable-by-API capability bound to one durable marker identity."""

    build_receipt_sha256: str
    marker_sha256: str
    claim_payload_sha256: str
    _ledger_path: str
    _marker_name: str
    _ledger_device: int
    _ledger_inode: int
    _marker_device: int
    _marker_inode: int
    _marker_size: int
    _marker_mtime_ns: int
    _marker_ctime_ns: int

    def __init__(
        self,
        *,
        factory_token: object,
        build_receipt_sha256: str,
        marker_sha256: str,
        claim_payload_sha256: str,
        ledger_path: str,
        marker_name: str,
        ledger_device: int,
        ledger_inode: int,
        marker_device: int,
        marker_inode: int,
        marker_size: int,
        marker_mtime_ns: int,
        marker_ctime_ns: int,
    ) -> None:
        if factory_token is not _CLAIM_FACTORY_TOKEN:
            raise TypeError("LockedGoldenClaim can only be created by the one-shot claimer")
        object.__setattr__(self, "build_receipt_sha256", build_receipt_sha256)
        object.__setattr__(self, "marker_sha256", marker_sha256)
        object.__setattr__(self, "claim_payload_sha256", claim_payload_sha256)
        object.__setattr__(self, "_ledger_path", ledger_path)
        object.__setattr__(self, "_marker_name", marker_name)
        object.__setattr__(self, "_ledger_device", ledger_device)
        object.__setattr__(self, "_ledger_inode", ledger_inode)
        object.__setattr__(self, "_marker_device", marker_device)
        object.__setattr__(self, "_marker_inode", marker_inode)
        object.__setattr__(self, "_marker_size", marker_size)
        object.__setattr__(self, "_marker_mtime_ns", marker_mtime_ns)
        object.__setattr__(self, "_marker_ctime_ns", marker_ctime_ns)

    @property
    def marker_path(self) -> Path:
        return Path(self._ledger_path) / self._marker_name


def _validate_claim_marker(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("locked claim marker must be an object")
    _require_exact_keys(value, _CLAIM_MARKER_KEYS, where="locked claim marker")
    normalized = _json_clone(value, where="locked claim marker")
    if normalized["schema_version"] != LOCKED_CONTRACT_SCHEMA_VERSION:
        raise ValueError("locked claim marker schema_version is unsupported")
    if normalized["kind"] != LOCKED_GOLDEN_CLAIM_KIND:
        raise ValueError("locked claim marker kind is invalid")
    normalized["build_receipt_sha256"] = _require_sha256(
        normalized["build_receipt_sha256"],
        where="locked claim marker.build_receipt_sha256",
    )
    if not isinstance(normalized["claim_payload"], dict):
        raise ValueError("locked claim marker.claim_payload must be an object")
    payload_sha = _require_sha256(
        normalized["claim_payload_sha256"],
        where="locked claim marker.claim_payload_sha256",
    )
    if payload_sha != _canonical_sha256(normalized["claim_payload"]):
        raise ValueError("locked claim marker claim_payload SHA-256 is invalid")
    _validate_self_hash(normalized, where="locked claim marker")
    return normalized


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:  # pragma: no cover - POSIX write either advances or raises
            raise OSError("short write while publishing locked benchmark state")
        offset += written


def claim_locked_benchmark_once(
    ledger_dir: str | Path,
    build_receipt_sha: str,
    claim_payload: Mapping[str, Any],
) -> LockedGoldenClaim:
    """Durably consume a build-receipt keyed locked benchmark exactly once."""

    receipt_sha = _require_sha256(
        build_receipt_sha,
        where="build_receipt_sha",
    )
    if not isinstance(claim_payload, Mapping):
        raise ValueError("claim_payload must be a JSON object")
    normalized_payload = _json_clone(claim_payload, where="claim_payload")
    if not isinstance(normalized_payload, dict):
        raise ValueError("claim_payload must be a JSON object")
    marker = {
        "schema_version": LOCKED_CONTRACT_SCHEMA_VERSION,
        "kind": LOCKED_GOLDEN_CLAIM_KIND,
        "build_receipt_sha256": receipt_sha,
        "claim_payload": normalized_payload,
        "claim_payload_sha256": _canonical_sha256(normalized_payload),
    }
    marker["canonical_sha256"] = _canonical_sha256(marker)
    marker_bytes = _canonical_json_bytes(marker, newline=True)
    marker_sha = hashlib.sha256(marker_bytes).hexdigest()
    marker_name = f"{receipt_sha}.json"

    ledger, directory_fd, directory_identity = _prepare_ledger_directory(ledger_dir)
    marker_fd: int | None = None
    marker_identity: os.stat_result | None = None
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        try:
            marker_fd = os.open(
                marker_name,
                flags,
                0o600,
                dir_fd=directory_fd,
            )
        except FileExistsError as exc:
            raise FileExistsError(
                "locked benchmark was already claimed for build receipt "
                f"{receipt_sha}"
            ) from exc
        os.fchmod(marker_fd, 0o600)
        _write_all(marker_fd, marker_bytes)
        os.fsync(marker_fd)
        marker_identity = os.fstat(marker_fd)
        if not stat.S_ISREG(marker_identity.st_mode):  # pragma: no cover - O_EXCL regular
            raise ValueError("locked benchmark claim marker is not a regular file")
        os.close(marker_fd)
        marker_fd = None
        os.fsync(directory_fd)
    finally:
        if marker_fd is not None:
            os.close(marker_fd)
        os.close(directory_fd)
    assert marker_identity is not None
    return LockedGoldenClaim(
        factory_token=_CLAIM_FACTORY_TOKEN,
        build_receipt_sha256=receipt_sha,
        marker_sha256=marker_sha,
        claim_payload_sha256=marker["claim_payload_sha256"],
        ledger_path=str(ledger),
        marker_name=marker_name,
        ledger_device=directory_identity.st_dev,
        ledger_inode=directory_identity.st_ino,
        marker_device=marker_identity.st_dev,
        marker_inode=marker_identity.st_ino,
        marker_size=marker_identity.st_size,
        marker_mtime_ns=marker_identity.st_mtime_ns,
        marker_ctime_ns=marker_identity.st_ctime_ns,
    )


def validate_locked_golden_claim(
    claim: LockedGoldenClaim,
    *,
    expected_build_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Re-open and authenticate the marker behind a claim capability."""

    if type(claim) is not LockedGoldenClaim:
        raise TypeError("a genuine LockedGoldenClaim capability is required")
    if expected_build_receipt_sha256 is not None:
        expected = _require_sha256(
            expected_build_receipt_sha256,
            where="expected_build_receipt_sha256",
        )
        if expected != claim.build_receipt_sha256:
            raise ValueError("claim capability build receipt differs")
    ledger_fd, ledger_identity = _open_real_directory(
        Path(claim._ledger_path),
        where="locked benchmark ledger",
    )
    try:
        if (ledger_identity.st_dev, ledger_identity.st_ino) != (
            claim._ledger_device,
            claim._ledger_inode,
        ):
            raise ValueError("locked benchmark ledger identity changed")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            marker_fd = os.open(claim._marker_name, flags, dir_fd=ledger_fd)
        except OSError as exc:
            raise ValueError("locked benchmark claim marker cannot be opened safely") from exc
        try:
            before = os.fstat(marker_fd)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("locked benchmark claim marker is not regular")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(marker_fd, 1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_ANCHOR_BYTES:
                    raise ValueError("locked benchmark claim marker is too large")
                chunks.append(chunk)
            after = os.fstat(marker_fd)
        finally:
            os.close(marker_fd)
    finally:
        os.close(ledger_fd)
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
        raise ValueError("locked benchmark claim marker changed while read")
    if (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) != (
        claim._marker_device,
        claim._marker_inode,
        claim._marker_size,
        claim._marker_mtime_ns,
        claim._marker_ctime_ns,
    ):
        raise ValueError("locked benchmark claim marker identity changed")
    raw = b"".join(chunks)
    if hashlib.sha256(raw).hexdigest() != claim.marker_sha256:
        raise ValueError("locked benchmark claim marker bytes changed")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object_pairs("locked claim marker"),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("locked benchmark claim marker is invalid JSON") from exc
    marker = _validate_claim_marker(value)
    if marker["build_receipt_sha256"] != claim.build_receipt_sha256:
        raise ValueError("locked benchmark claim marker receipt differs")
    if marker["claim_payload_sha256"] != claim.claim_payload_sha256:
        raise ValueError("locked benchmark claim marker payload differs")
    return marker


def write_claimed_evaluation_incomplete_stub(
    destination: str | Path,
    claim: LockedGoldenClaim,
    stub_payload: Mapping[str, Any],
) -> Path:
    """Atomically publish a durable incomplete report after a valid claim."""

    marker = validate_locked_golden_claim(claim)
    if not isinstance(stub_payload, Mapping):
        raise ValueError("stub_payload must be a JSON object")
    normalized_payload = _json_clone(stub_payload, where="stub_payload")
    if not isinstance(normalized_payload, dict):
        raise ValueError("stub_payload must be a JSON object")
    payload = {
        "schema_version": LOCKED_CONTRACT_SCHEMA_VERSION,
        "kind": LOCKED_INCOMPLETE_STUB_KIND,
        "status": "claimed_evaluation_incomplete",
        "build_receipt_sha256": claim.build_receipt_sha256,
        "claim_marker_sha256": claim.marker_sha256,
        "claim_payload_sha256": marker["claim_payload_sha256"],
        "stub_payload": normalized_payload,
    }
    payload["canonical_sha256"] = _canonical_sha256(payload)
    _require_exact_keys(payload, _INCOMPLETE_STUB_KEYS, where="incomplete stub")
    rendered = _canonical_json_bytes(payload, newline=True)

    target = Path(destination).absolute()
    if target.name in {"", ".", ".."}:
        raise ValueError("incomplete stub destination must name a file")
    parent_fd, _ = _open_real_directory(
        target.parent,
        where="incomplete stub parent",
    )
    temporary_name = f".{target.name}.{secrets.token_hex(12)}.tmp"
    temp_fd: int | None = None
    linked = False
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        temp_fd = os.open(temporary_name, flags, 0o600, dir_fd=parent_fd)
        os.fchmod(temp_fd, 0o600)
        _write_all(temp_fd, rendered)
        os.fsync(temp_fd)
        os.close(temp_fd)
        temp_fd = None
        try:
            os.link(
                temporary_name,
                target.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite incomplete report stub: {target}"
            ) from exc
        linked = True
        os.fsync(parent_fd)
    finally:
        if temp_fd is not None:
            os.close(temp_fd)
        try:
            os.unlink(temporary_name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        if linked:
            os.fsync(parent_fd)
        os.close(parent_fd)
    return target


__all__ = [
    "LOCKED_BUCKETS",
    "LOCKED_BUILD_RECEIPT_KIND",
    "LOCKED_CONTRACT_SCHEMA_VERSION",
    "LOCKED_GOLDEN_ANCHOR_KIND",
    "LOCKED_GOLDEN_CLAIM_KIND",
    "LOCKED_IDENTITY_COMMITMENT_SCHEME",
    "LOCKED_INCOMPLETE_STUB_KIND",
    "LockedGoldenClaim",
    "LockedGoldenPreclaim",
    "build_locked_golden_anchor",
    "claim_locked_benchmark_once",
    "preclaim_locked_benchmark",
    "validate_locked_build_receipt",
    "validate_locked_golden_anchor",
    "validate_locked_golden_claim",
    "validate_locked_golden_preclaim",
    "write_claimed_evaluation_incomplete_stub",
]
