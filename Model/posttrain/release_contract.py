# -*- coding: utf-8 -*-

"""Fail-closed admission of reviewed streaming-v2 OCR releases.

The current alignment-v3 checkpoint contract intentionally rejects older
artifacts.  This module does not relax that contract.  Instead, it admits one
older release format through an external, code-reviewed lock that binds the
repository revision, release manifests, every published file, tokenizer
identity, and the training metadata required by OCR post-training.

The validation order is deliberate: no PyTorch payload is deserialized until
the external lock, ``BACKUP_MANIFEST.json``, ``SHA256SUMS``, and every file
declared by those manifests have passed byte-level verification.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import torch

from Model.ocr.tokenization import (
    OCR_NATIVE_TARGET_ENCODING,
    OCR_TOKENIZATION_CONTRACT_VERSION,
    tokenizer_vocab_sha256,
)
from Model.ocr.position_contract import (
    OCR_POSITION_CONTRACT_METADATA_VERSION,
    resolve_checkpoint_ocr_position_contract,
)
from Model.ocr.visual_input_contract import (
    DOL_OCR_LINE_LETTERBOX_224_V1,
    OCR_VISUAL_INPUT_CONTRACT_METADATA_KEY,
    OCR_VISUAL_INPUT_CONTRACT_VERSION_METADATA_KEY,
    ocr_visual_input_contract_version,
    resolve_checkpoint_ocr_visual_input_contract,
    validate_ocr_visual_input_contract,
)
from Tokenizer.unified.bundle import TokenizerBundle
from Tokenizer.unified.contract import tokenizer_bundle_contract


REVIEWED_OCR_RELEASE_LOCK_SCHEMA_VERSION = 1
REVIEWED_OCR_RELEASE_LOCK_KIND = "reviewed_streaming_v2_ocr_release"
STREAMING_V2_RELEASE_SOURCE_CONTRACT_KIND = "streaming_v2_release_v1"
NATIVE_V2_TO_V3_COMPATIBILITY_RULE = (
    "native_v2_to_strict_native_v3_same_bundle_v1"
)
VISUAL_SOURCE_CONTRACT_ALIGNMENT_V3 = "alignment-v3"
VISUAL_SOURCE_CONTRACT_STREAMING_V2_RELEASE = "streaming-v2-release"
MONTLOK_DOL_1_2_OCR_REPOSITORY_ID = "Montlok/DoL-1.2-OCR"
MONTLOK_DOL_1_2_OCR_REVISION = (
    "bee908ab2a9376f6224dff514564ebb0ae99a643"
)

# This is the trust root, not data read from the reviewed lock.  Each entry
# binds an immutable Hub source identity to the exact bytes of the repository-
# reviewed lock.  Tests may temporarily extend this mapping for synthetic
# fixtures; production callers cannot make an arbitrary external JSON file
# trusted merely by copying the repository/revision strings into it.
TRUSTED_REVIEWED_OCR_RELEASE_LOCKS: dict[tuple[str, str], str] = {
    (
        MONTLOK_DOL_1_2_OCR_REPOSITORY_ID,
        MONTLOK_DOL_1_2_OCR_REVISION,
    ): "01a291e921dab65d89a1bfd07570c8089a2427a7add1e28d7db30dced46a6044",
}

_REVIEWED_LEGACY_VISUAL_INPUT_BINDINGS = {
    (
        MONTLOK_DOL_1_2_OCR_REPOSITORY_ID,
        MONTLOK_DOL_1_2_OCR_REVISION,
    ): DOL_OCR_LINE_LETTERBOX_224_V1,
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IMMUTABLE_REVISION_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SHA256SUM_RE = re.compile(r"^([0-9a-f]{64})  ([^\r\n]+)$")
_WDS_SHARD_RE = re.compile(r"^shard-([0-9]{5})\.tar$")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _json_object_from_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"invalid {label}: {exc}") from exc
    return _json_object_from_bytes(payload, label=label)


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_sha256(value: object, field: str) -> str:
    digest = _require_string(value, field)
    if _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{field} must be a lowercase SHA256 digest")
    return digest


def _require_immutable_revision(value: object, field: str) -> str:
    revision = _require_string(value, field)
    if _IMMUTABLE_REVISION_RE.fullmatch(revision) is None:
        raise ValueError(f"{field} must be an immutable 40- or 64-hex revision")
    return revision


def _require_int(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _require_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


def _resolve_reviewed_release_visual_input_contract(
    metadata: Mapping[str, Any],
    *,
    repository_id: str,
    revision: str,
) -> str:
    bound_contract = _REVIEWED_LEGACY_VISUAL_INPUT_BINDINGS.get(
        (repository_id, revision)
    )
    if bound_contract is None:
        return resolve_checkpoint_ocr_visual_input_contract(metadata)

    has_contract = OCR_VISUAL_INPUT_CONTRACT_METADATA_KEY in metadata
    has_version = OCR_VISUAL_INPUT_CONTRACT_VERSION_METADATA_KEY in metadata
    if not has_contract and not has_version:
        return bound_contract
    return resolve_checkpoint_ocr_visual_input_contract(
        metadata,
        requested=bound_contract,
    )


def _require_exact_keys(
    value: Mapping[str, Any], field: str, expected: set[str]
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{field} keys differ from contract: missing={missing} extra={extra}"
        )


def _safe_relative_path(value: object, field: str) -> PurePosixPath:
    raw = _require_string(value, field)
    if "\\" in raw or "\x00" in raw:
        raise ValueError(f"{field} contains an unsafe path separator")
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or raw != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{field} must be a normalized relative path")
    return path


def _resolve_release_file(root: Path, relative: PurePosixPath, field: str) -> Path:
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{field} may not traverse a symbolic link")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"{field} does not resolve inside the release: {relative}") from exc
    if not resolved.is_file():
        raise ValueError(f"{field} is not a regular file: {relative}")
    return resolved


def _resolve_release_directory(
    root: Path, relative: PurePosixPath, field: str
) -> Path:
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{field} may not traverse a symbolic link")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"{field} does not resolve inside the release: {relative}") from exc
    if not resolved.is_dir():
        raise ValueError(f"{field} is not a directory: {relative}")
    return resolved


def _parse_backup_manifest(
    payload: Mapping[str, Any],
) -> tuple[dict[str, tuple[int, str]], str, int]:
    if payload.get("schema_version") != 1:
        raise ValueError("BACKUP_MANIFEST.json schema_version must be 1")
    release_name = _require_string(payload.get("release_name"), "release_name")
    parameter_count = _require_int(
        payload.get("parameter_count"), "parameter_count", minimum=1
    )
    declared: dict[str, tuple[int, str]] = {}

    def add_file(raw_name: object, raw_record: object, field: str) -> None:
        relative = _safe_relative_path(raw_name, field)
        name = relative.as_posix()
        if name in declared:
            raise ValueError(f"duplicate release file declaration: {name}")
        record = _require_mapping(raw_record, f"{field} record")
        size = _require_int(record.get("size"), f"{field}.size")
        digest = _require_sha256(record.get("sha256"), f"{field}.sha256")
        declared[name] = (size, digest)

    for name, record in _require_mapping(
        payload.get("files"), "BACKUP_MANIFEST.json files"
    ).items():
        add_file(name, record, f"files[{name!r}]")

    tokenizer = _require_mapping(
        payload.get("tokenizer"), "BACKUP_MANIFEST.json tokenizer"
    )
    tokenizer_root = _safe_relative_path(
        tokenizer.get("path"), "BACKUP_MANIFEST.json tokenizer.path"
    )
    for name, record in _require_mapping(
        tokenizer.get("files"), "BACKUP_MANIFEST.json tokenizer.files"
    ).items():
        child = _safe_relative_path(name, f"tokenizer.files[{name!r}]")
        add_file(
            (tokenizer_root / child).as_posix(),
            record,
            f"tokenizer.files[{name!r}]",
        )
    if not declared:
        raise ValueError("BACKUP_MANIFEST.json declares no files")
    return declared, release_name, parameter_count


def _parse_sha256sums(payload: bytes) -> dict[str, str]:
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise ValueError("SHA256SUMS is not UTF-8") from exc
    declared: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            continue
        match = _SHA256SUM_RE.fullmatch(line)
        if match is None:
            raise ValueError(f"invalid SHA256SUMS line {line_number}")
        digest, raw_name = match.groups()
        name = _safe_relative_path(
            raw_name, f"SHA256SUMS line {line_number} path"
        ).as_posix()
        if name in declared:
            raise ValueError(f"duplicate SHA256SUMS path: {name}")
        declared[name] = digest
    if not declared:
        raise ValueError("SHA256SUMS declares no files")
    return declared


def _verify_lock_source(
    lock: Mapping[str, Any],
    *,
    expected_repository_id: str | None,
    expected_revision: str | None,
) -> tuple[str, str]:
    if lock.get("schema_version") != REVIEWED_OCR_RELEASE_LOCK_SCHEMA_VERSION:
        raise ValueError("unsupported reviewed release lock schema_version")
    if lock.get("kind") != REVIEWED_OCR_RELEASE_LOCK_KIND:
        raise ValueError("unsupported reviewed release lock kind")
    source = _require_mapping(lock.get("source"), "lock.source")
    _require_exact_keys(source, "lock.source", {"repository_id", "revision"})
    repository_id = _require_string(source.get("repository_id"), "source.repository_id")
    revision = _require_immutable_revision(source.get("revision"), "source.revision")
    if expected_repository_id is not None and repository_id != expected_repository_id:
        raise ValueError(
            "reviewed release repository mismatch: "
            f"lock={repository_id!r} expected={expected_repository_id!r}"
        )
    if expected_revision is not None and revision != expected_revision:
        raise ValueError(
            "reviewed release revision mismatch: "
            f"lock={revision!r} expected={expected_revision!r}"
        )
    return repository_id, revision


def _validate_corpus_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(
        payload,
        "metadata.streaming.corpus_manifest",
        {
            "version",
            "wds",
            "hanshi",
            "excluded_shard_ids",
            "split",
            "mix_schedule",
            "seed",
        },
    )
    if payload.get("version") != 1:
        raise ValueError("streaming corpus manifest version must be 1")
    _require_int(payload.get("seed"), "corpus manifest seed")

    wds = _require_mapping(payload.get("wds"), "corpus manifest wds")
    _require_exact_keys(wds, "corpus manifest wds", {"shards"})
    raw_shards = wds.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ValueError("corpus manifest wds.shards must be a non-empty list")
    shard_ids: list[int] = []
    for index, raw_shard in enumerate(raw_shards):
        field = f"corpus manifest wds.shards[{index}]"
        shard = _require_mapping(raw_shard, field)
        _require_exact_keys(shard, field, {"id", "name", "size"})
        shard_id = _require_int(shard.get("id"), f"{field}.id")
        name = _require_string(shard.get("name"), f"{field}.name")
        match = _WDS_SHARD_RE.fullmatch(name)
        if match is None or int(match.group(1)) != shard_id:
            raise ValueError(f"{field}.name does not match its shard id")
        _require_int(shard.get("size"), f"{field}.size", minimum=1)
        shard_ids.append(shard_id)
    if shard_ids != sorted(set(shard_ids)):
        raise ValueError("corpus manifest WDS shard ids must be unique and sorted")

    hanshi = _require_mapping(payload.get("hanshi"), "corpus manifest hanshi")
    _require_exact_keys(
        hanshi, "corpus manifest hanshi", {"meta_name", "meta_size", "pages_name"}
    )
    for key in ("meta_name", "pages_name"):
        name = _safe_relative_path(hanshi.get(key), f"corpus manifest hanshi.{key}")
        if len(name.parts) != 1:
            raise ValueError(f"corpus manifest hanshi.{key} must be a basename")
    hanshi_meta_size = _require_int(
        hanshi.get("meta_size"), "corpus manifest hanshi.meta_size", minimum=1
    )

    raw_excluded = payload.get("excluded_shard_ids")
    if not isinstance(raw_excluded, list):
        raise ValueError("corpus manifest excluded_shard_ids must be a list")
    excluded = [
        _require_int(value, f"excluded_shard_ids[{index}]")
        for index, value in enumerate(raw_excluded)
    ]
    if excluded != sorted(set(excluded)):
        raise ValueError("excluded_shard_ids must be unique and sorted")
    if set(excluded) & set(shard_ids):
        raise ValueError("excluded WDS shard ids may not appear in wds.shards")

    split = _require_mapping(payload.get("split"), "corpus manifest split")
    _require_exact_keys(split, "corpus manifest split", {"train", "val", "test", "group_key"})
    if split.get("group_key") != "src_doc":
        raise ValueError("corpus manifest split.group_key must be 'src_doc'")
    ranges: dict[str, list[Any]] = {}
    for key in ("train", "val", "test"):
        value = split.get(key)
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"corpus manifest split.{key} must have two bounds")
        ranges[key] = value
    train_start = _require_int(ranges["train"][0], "split.train[0]")
    train_end = _require_int(ranges["train"][1], "split.train[1]", minimum=1)
    val_start = _require_int(ranges["val"][0], "split.val[0]", minimum=1)
    val_end = _require_int(ranges["val"][1], "split.val[1]", minimum=1)
    test_start = _require_int(ranges["test"][0], "split.test[0]", minimum=1)
    if ranges["test"][1] is not None:
        raise ValueError("corpus manifest split.test[1] must be null")
    if train_start != 0 or not (train_end == val_start < val_end == test_start):
        raise ValueError("corpus manifest split ranges are not contiguous and ordered")

    schedule = payload.get("mix_schedule")
    if (
        not isinstance(schedule, list)
        or not schedule
        or any(source not in {"wds", "hanshi"} for source in schedule)
        or set(schedule) != {"wds", "hanshi"}
    ):
        raise ValueError(
            "corpus manifest mix_schedule must be a non-empty WDS/Hanshi schedule"
        )
    return {
        "wds_shard_count": len(shard_ids),
        "hanshi_meta_size": hanshi_meta_size,
        "mix_schedule": list(schedule),
    }


def _validate_streaming_cursor(
    payload: Mapping[str, Any],
    *,
    corpus_info: Mapping[str, Any],
    corpus_complete: bool,
) -> int:
    _require_exact_keys(
        payload,
        "metadata.streaming.corpus_cursor",
        {"version", "mix_position", "wds", "hanshi", "exhausted", "counts"},
    )
    if payload.get("version") != 1:
        raise ValueError("streaming cursor version must be 1")
    mix_position = _require_int(payload.get("mix_position"), "cursor.mix_position")

    wds = _require_mapping(payload.get("wds"), "cursor.wds")
    _require_exact_keys(wds, "cursor.wds", {"shard_position", "sample_position"})
    shard_position = _require_int(wds.get("shard_position"), "cursor.wds.shard_position")
    _require_int(wds.get("sample_position"), "cursor.wds.sample_position")

    hanshi = _require_mapping(payload.get("hanshi"), "cursor.hanshi")
    _require_exact_keys(hanshi, "cursor.hanshi", {"byte_offset", "line_number"})
    byte_offset = _require_int(hanshi.get("byte_offset"), "cursor.hanshi.byte_offset")
    _require_int(hanshi.get("line_number"), "cursor.hanshi.line_number")

    exhausted = _require_mapping(payload.get("exhausted"), "cursor.exhausted")
    _require_exact_keys(exhausted, "cursor.exhausted", {"wds", "hanshi"})
    wds_exhausted = _require_bool(exhausted.get("wds"), "cursor.exhausted.wds")
    hanshi_exhausted = _require_bool(
        exhausted.get("hanshi"), "cursor.exhausted.hanshi"
    )
    if corpus_complete != (wds_exhausted and hanshi_exhausted):
        raise ValueError(
            "streaming corpus_complete must equal both source exhausted flags"
        )
    shard_count = int(corpus_info["wds_shard_count"])
    if shard_position > shard_count or (
        not wds_exhausted and shard_position >= shard_count
    ):
        raise ValueError("cursor WDS shard_position is outside corpus manifest")
    if byte_offset > int(corpus_info["hanshi_meta_size"]):
        raise ValueError("cursor Hanshi byte_offset exceeds manifest metadata size")

    counts = _require_mapping(payload.get("counts"), "cursor.counts")
    _require_exact_keys(counts, "cursor.counts", {"total", "wds", "hanshi"})
    total = _require_int(counts.get("total"), "cursor.counts.total", minimum=1)
    wds_count = _require_int(counts.get("wds"), "cursor.counts.wds")
    hanshi_count = _require_int(counts.get("hanshi"), "cursor.counts.hanshi")
    if wds_count + hanshi_count != total:
        raise ValueError("streaming cursor sample counts are inconsistent")
    if mix_position < total:
        raise ValueError("cursor.mix_position may not trail consumed sample count")
    if not wds_exhausted and not hanshi_exhausted:
        if mix_position != total:
            raise ValueError(
                "non-exhausted cursor.mix_position must equal consumed sample count"
            )
        schedule = list(corpus_info["mix_schedule"])
        cycles, remainder = divmod(total, len(schedule))
        expected = {
            source: cycles * schedule.count(source)
            + schedule[:remainder].count(source)
            for source in ("wds", "hanshi")
        }
        if wds_count != expected["wds"] or hanshi_count != expected["hanshi"]:
            raise ValueError("streaming cursor counts do not match mix_schedule")
    return total


def validate_reviewed_streaming_v2_ocr_release(
    release_dir: str | Path,
    release_lock: str | Path,
    *,
    runtime_native_tokenization_contract: Mapping[str, Any],
    expected_repository_id: str | None = None,
    expected_revision: str | None = None,
) -> dict[str, Any]:
    """Validate a reviewed streaming-v2 release and return metadata + lineage.

    ``release_lock`` must live outside the release directory.  In production it
    should be a version-controlled file reviewed together with this validator.
    Optional source arguments let a caller additionally pin the expected Hub
    repository and immutable revision instead of trusting runtime input.
    """

    lock_input = Path(release_lock)
    if lock_input.is_symlink():
        raise ValueError("release_lock must be a regular non-symlink file")
    try:
        root = Path(release_dir).resolve(strict=True)
        lock_path = lock_input.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise ValueError(f"release or lock path cannot be resolved: {exc}") from exc
    if not root.is_dir():
        raise ValueError("release_dir must be a directory")
    if not lock_path.is_file() or lock_path.is_symlink():
        raise ValueError("release_lock must be a regular non-symlink file")
    try:
        lock_path.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("release_lock must be external to the release directory")

    try:
        lock_raw = lock_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"reviewed release lock cannot be read: {exc}") from exc
    lock_raw_sha = _sha256_bytes(lock_raw)
    lock = _json_object_from_bytes(lock_raw, label="reviewed release lock")
    _require_exact_keys(
        lock,
        "reviewed release lock",
        {"schema_version", "kind", "source", "release", "tokenizer", "contracts", "streaming"},
    )
    repository_id, revision = _verify_lock_source(
        lock,
        expected_repository_id=expected_repository_id,
        expected_revision=expected_revision,
    )
    trusted_lock_sha = TRUSTED_REVIEWED_OCR_RELEASE_LOCKS.get(
        (repository_id, revision)
    )
    if trusted_lock_sha is None or lock_raw_sha != trusted_lock_sha:
        raise ValueError(
            "reviewed release lock is not anchored by the production trust registry"
        )
    release = _require_mapping(lock.get("release"), "lock.release")
    _require_exact_keys(
        release,
        "lock.release",
        {
            "name",
            "parameter_count",
            "backup_manifest_path",
            "backup_manifest_raw_sha256",
            "sha256sums_path",
            "sha256sums_raw_sha256",
            "model_path",
            "model_raw_sha256",
            "metadata_path",
            "metadata_raw_sha256",
        },
    )
    backup_relative = _safe_relative_path(
        release.get("backup_manifest_path"), "release.backup_manifest_path"
    )
    sums_relative = _safe_relative_path(
        release.get("sha256sums_path"), "release.sha256sums_path"
    )
    if backup_relative == sums_relative:
        raise ValueError("backup manifest and SHA256SUMS paths must be distinct")
    backup_path = _resolve_release_file(root, backup_relative, "backup manifest")
    sums_path = _resolve_release_file(root, sums_relative, "SHA256SUMS")
    backup_raw = backup_path.read_bytes()
    sums_raw = sums_path.read_bytes()
    expected_backup_sha = _require_sha256(
        release.get("backup_manifest_raw_sha256"),
        "release.backup_manifest_raw_sha256",
    )
    expected_sums_sha = _require_sha256(
        release.get("sha256sums_raw_sha256"),
        "release.sha256sums_raw_sha256",
    )
    if _sha256_bytes(backup_raw) != expected_backup_sha:
        raise ValueError("BACKUP_MANIFEST.json differs from reviewed lock")
    if _sha256_bytes(sums_raw) != expected_sums_sha:
        raise ValueError("SHA256SUMS differs from reviewed lock")

    backup = _json_object_from_bytes(backup_raw, label="BACKUP_MANIFEST.json")
    backup_files, release_name, parameter_count = _parse_backup_manifest(backup)
    sums_files = _parse_sha256sums(sums_raw)
    if set(backup_files) != set(sums_files):
        raise ValueError(
            "BACKUP_MANIFEST.json and SHA256SUMS declare different file sets"
        )
    for name, (expected_size, expected_digest) in sorted(backup_files.items()):
        if sums_files[name] != expected_digest:
            raise ValueError(f"manifest digest disagreement for {name}")
        path = _resolve_release_file(root, PurePosixPath(name), f"release file {name}")
        if path.stat().st_size != expected_size:
            raise ValueError(f"release file size mismatch for {name}")
        if _file_sha256(path) != expected_digest:
            raise ValueError(f"release file SHA256 mismatch for {name}")

    if release_name != _require_string(release.get("name"), "release.name"):
        raise ValueError("release name differs from reviewed lock")
    if parameter_count != _require_int(
        release.get("parameter_count"), "release.parameter_count", minimum=1
    ):
        raise ValueError("parameter_count differs from reviewed lock")
    model_relative = _safe_relative_path(release.get("model_path"), "release.model_path")
    metadata_relative = _safe_relative_path(
        release.get("metadata_path"), "release.metadata_path"
    )
    if model_relative == metadata_relative:
        raise ValueError("model_path and metadata_path must be distinct")
    model_name = model_relative.as_posix()
    metadata_name = metadata_relative.as_posix()
    if model_name not in backup_files or metadata_name not in backup_files:
        raise ValueError("model or metadata is absent from release manifests")
    model_sha = _require_sha256(
        release.get("model_raw_sha256"), "release.model_raw_sha256"
    )
    metadata_sha = _require_sha256(
        release.get("metadata_raw_sha256"), "release.metadata_raw_sha256"
    )
    if backup_files[model_name][1] != model_sha:
        raise ValueError("model SHA256 differs from reviewed lock")
    if backup_files[metadata_name][1] != metadata_sha:
        raise ValueError("metadata SHA256 differs from reviewed lock")

    tokenizer = _require_mapping(lock.get("tokenizer"), "lock.tokenizer")
    _require_exact_keys(
        tokenizer,
        "lock.tokenizer",
        {
            "root",
            "manifest_path",
            "manifest_raw_sha256",
            "manifest_canonical_sha256",
            "bundle_files_canonical_sha256",
            "vocab_path",
            "vocab_file_raw_sha256",
            "token_id_map_sha256",
        },
    )
    tokenizer_root = _safe_relative_path(tokenizer.get("root"), "tokenizer.root")
    tokenizer_prefix = tokenizer_root.as_posix() + "/"
    tokenizer_entries = {
        name[len(tokenizer_prefix) :]: (size, digest)
        for name, (size, digest) in backup_files.items()
        if name.startswith(tokenizer_prefix)
    }
    if not tokenizer_entries:
        raise ValueError("release manifest declares no tokenizer files")
    manifest_relative = _safe_relative_path(
        tokenizer.get("manifest_path"), "tokenizer.manifest_path"
    )
    vocab_relative = _safe_relative_path(
        tokenizer.get("vocab_path"), "tokenizer.vocab_path"
    )
    manifest_name = manifest_relative.as_posix()
    vocab_name = vocab_relative.as_posix()
    if not manifest_name.startswith(tokenizer_prefix) or not vocab_name.startswith(
        tokenizer_prefix
    ):
        raise ValueError("tokenizer manifest and vocab must live under tokenizer.root")
    manifest_child = manifest_name[len(tokenizer_prefix) :]
    vocab_child = vocab_name[len(tokenizer_prefix) :]
    if manifest_child not in tokenizer_entries or vocab_child not in tokenizer_entries:
        raise ValueError("tokenizer manifest or vocab is absent from release manifests")
    manifest_path = _resolve_release_file(root, manifest_relative, "tokenizer manifest")
    manifest_raw = manifest_path.read_bytes()
    manifest_raw_sha = _sha256_bytes(manifest_raw)
    if manifest_raw_sha != _require_sha256(
        tokenizer.get("manifest_raw_sha256"), "tokenizer.manifest_raw_sha256"
    ):
        raise ValueError("raw tokenizer manifest SHA256 differs from reviewed lock")
    manifest_payload = _json_object_from_bytes(
        manifest_raw, label="tokenizer manifest"
    )
    manifest_canonical_sha = _canonical_json_sha256(manifest_payload)
    if manifest_canonical_sha != _require_sha256(
        tokenizer.get("manifest_canonical_sha256"),
        "tokenizer.manifest_canonical_sha256",
    ):
        raise ValueError("canonical tokenizer manifest SHA256 differs from reviewed lock")
    internal_files = _require_mapping(
        manifest_payload.get("files"), "tokenizer manifest files"
    )
    internal_digests: dict[str, str] = {}
    for raw_name, raw_digest in internal_files.items():
        child = _safe_relative_path(raw_name, f"tokenizer manifest files[{raw_name!r}]")
        child_name = child.as_posix()
        if child_name in internal_digests:
            raise ValueError(f"duplicate tokenizer manifest path: {child_name}")
        internal_digests[child_name] = _require_sha256(
            raw_digest, f"tokenizer manifest files[{raw_name!r}]"
        )
    expected_internal = set(tokenizer_entries) - {manifest_child}
    if set(internal_digests) != expected_internal:
        raise ValueError("tokenizer manifest and release bundle declare different files")
    for child_name, digest in internal_digests.items():
        if tokenizer_entries[child_name][1] != digest:
            raise ValueError(f"tokenizer manifest digest mismatch for {child_name}")

    vocab_file_raw_sha = tokenizer_entries[vocab_child][1]
    if vocab_file_raw_sha != _require_sha256(
        tokenizer.get("vocab_file_raw_sha256"),
        "tokenizer.vocab_file_raw_sha256",
    ):
        raise ValueError("raw tokenizer vocab file SHA256 differs from reviewed lock")

    # Reuse the same semantic identities persisted by native-v3 OCR data.  Do
    # not invent a second bundle-row canonicalization or confuse the raw
    # vocab.json bytes with the token-to-id mapping produced by the tokenizer.
    tokenizer_dir = _resolve_release_directory(
        root, tokenizer_root, "tokenizer root"
    )
    bundle_contract = tokenizer_bundle_contract(tokenizer_dir)
    bundle_files_canonical_sha = _require_sha256(
        bundle_contract.get("files_canonical_sha256"),
        "computed tokenizer bundle files_canonical_sha256",
    )
    if bundle_files_canonical_sha != _require_sha256(
        tokenizer.get("bundle_files_canonical_sha256"),
        "tokenizer.bundle_files_canonical_sha256",
    ):
        raise ValueError("tokenizer bundle files contract differs from reviewed lock")
    loaded_bundle = TokenizerBundle.from_dir(str(tokenizer_dir))
    token_id_map_sha = tokenizer_vocab_sha256(loaded_bundle.tokenizer)
    if token_id_map_sha != _require_sha256(
        tokenizer.get("token_id_map_sha256"), "tokenizer.token_id_map_sha256"
    ):
        raise ValueError("tokenizer token-id map differs from reviewed lock")

    runtime_contract = _require_mapping(
        runtime_native_tokenization_contract,
        "runtime_native_tokenization_contract",
    )
    if runtime_contract.get("target_encoding") != OCR_NATIVE_TARGET_ENCODING:
        raise ValueError("runtime OCR tokenization contract must be strict native")
    if (
        runtime_contract.get("tokenization_contract_version")
        != OCR_TOKENIZATION_CONTRACT_VERSION
    ):
        raise ValueError("runtime OCR tokenization contract must be native-v3")
    if (
        runtime_contract.get("tokenizer_manifest_canonical_sha256")
        != manifest_canonical_sha
    ):
        raise ValueError("runtime tokenizer manifest differs from source release")
    runtime_bundle = _require_mapping(
        runtime_contract.get("tokenizer_bundle"),
        "runtime_native_tokenization_contract.tokenizer_bundle",
    )
    if runtime_bundle != bundle_contract:
        raise ValueError("runtime tokenizer bundle differs from source release")
    if runtime_contract.get("tokenizer_vocab_sha256") != token_id_map_sha:
        raise ValueError("runtime tokenizer token-id map differs from source release")

    # Deserialization is intentionally below every byte-level package check.
    metadata_path = _resolve_release_file(root, metadata_relative, "metadata")
    metadata_raw = metadata_path.read_bytes()
    if _sha256_bytes(metadata_raw) != metadata_sha:
        raise ValueError("metadata changed after release verification")
    try:
        payload = torch.load(
            io.BytesIO(metadata_raw), map_location="cpu", weights_only=True
        )
    except Exception as exc:
        raise ValueError(f"metadata cannot be loaded safely: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise ValueError("metadata payload must contain a metadata object")
    metadata = dict(payload["metadata"])
    visual_input_contract = _resolve_reviewed_release_visual_input_contract(
        metadata,
        repository_id=repository_id,
        revision=revision,
    )
    visual_input_contract_version = ocr_visual_input_contract_version(
        visual_input_contract
    )

    contracts = _require_mapping(lock.get("contracts"), "lock.contracts")
    _require_exact_keys(
        contracts,
        "lock.contracts",
        {
            "phase",
            "freeze_rdt",
            "frozen_vision",
            "final",
            "stop_reason",
            "ocr_position_contract",
            "ocr_position_contract_version",
            "ocr_target_encoding",
            "ocr_tokenization_contract_version",
            "checkpoint_step",
            "tokenizer_id_extent",
            "model_vocab_capacity",
        },
    )
    expected_contract = {
        "ocr_position_contract": "boundary_v1",
        "ocr_position_contract_version": 1,
        "ocr_target_encoding": "native",
        "ocr_tokenization_contract_version": 2,
        "phase": "vlm_align",
        "freeze_rdt": True,
        "frozen_vision": False,
        "final": True,
        "stop_reason": "loss_plateau",
    }
    for field, fixed_value in expected_contract.items():
        if contracts.get(field) != fixed_value:
            raise ValueError(
                f"reviewed lock {field} must be fixed to {fixed_value!r}"
            )
        if metadata.get(field) != fixed_value:
            raise ValueError(
                f"release metadata {field} must be {fixed_value!r}, "
                f"got {metadata.get(field)!r}"
            )
    expected_step = _require_int(
        contracts.get("checkpoint_step"), "contracts.checkpoint_step", minimum=1
    )
    if payload.get("step") != expected_step:
        raise ValueError("checkpoint step differs from reviewed lock")
    expected_tokenizer_extent = _require_int(
        contracts.get("tokenizer_id_extent"),
        "contracts.tokenizer_id_extent",
        minimum=1,
    )
    actual_vocab_extent = max(
        int(value) for value in loaded_bundle.tokenizer.vocab.values()
    ) + 1
    if expected_tokenizer_extent != actual_vocab_extent:
        raise ValueError("reviewed tokenizer extent differs from its token-id map")
    model_vocab_capacity = _require_int(
        contracts.get("model_vocab_capacity"),
        "contracts.model_vocab_capacity",
        minimum=actual_vocab_extent,
    )
    rdt_config = _require_mapping(metadata.get("rdt_config"), "metadata.rdt_config")
    if rdt_config.get("vocab_size") != model_vocab_capacity:
        raise ValueError("RDT vocab_size differs from reviewed model capacity")

    streaming_lock = _require_mapping(lock.get("streaming"), "lock.streaming")
    _require_exact_keys(
        streaming_lock,
        "lock.streaming",
        {
            "corpus_manifest_canonical_sha256",
            "tokenizer_manifest_canonical_sha256",
            "cursor_canonical_sha256",
            "samples_exposed",
            "corpus_complete",
        },
    )
    streaming = _require_mapping(metadata.get("streaming"), "metadata.streaming")
    _require_exact_keys(
        streaming,
        "metadata.streaming",
        {
            "corpus_manifest",
            "corpus_manifest_sha256",
            "tokenizer_manifest_sha256",
            "ocr_target_encoding",
            "ocr_tokenization_contract_version",
            "corpus_cursor",
            "corpus_complete",
        },
    )
    if streaming.get("ocr_target_encoding") != "native":
        raise ValueError("streaming evidence target encoding must be native")
    if streaming.get("ocr_tokenization_contract_version") != 2:
        raise ValueError("streaming evidence tokenization contract must be v2")
    corpus_manifest = _require_mapping(
        streaming.get("corpus_manifest"), "metadata.streaming.corpus_manifest"
    )
    corpus_info = _validate_corpus_manifest(corpus_manifest)
    corpus_manifest_sha = _canonical_json_sha256(corpus_manifest)
    expected_corpus_manifest_sha = _require_sha256(
        streaming_lock.get("corpus_manifest_canonical_sha256"),
        "streaming.corpus_manifest_canonical_sha256",
    )
    if (
        corpus_manifest_sha != expected_corpus_manifest_sha
        or streaming.get("corpus_manifest_sha256") != corpus_manifest_sha
    ):
        raise ValueError("streaming corpus manifest evidence is inconsistent")
    if streaming.get("tokenizer_manifest_sha256") != manifest_canonical_sha:
        raise ValueError(
            "streaming tokenizer hash must equal canonical tokenizer manifest hash"
        )
    if streaming_lock.get("tokenizer_manifest_canonical_sha256") != (
        manifest_canonical_sha
    ):
        raise ValueError("streaming tokenizer hash differs from reviewed lock")
    expected_complete = _require_bool(
        streaming_lock.get("corpus_complete"), "streaming.corpus_complete"
    )
    actual_complete = _require_bool(
        streaming.get("corpus_complete"), "metadata.streaming.corpus_complete"
    )
    if actual_complete is not expected_complete:
        raise ValueError("streaming corpus_complete differs from reviewed lock")
    if metadata.get("stop_reason") == "loss_plateau" and actual_complete:
        raise ValueError("loss-plateau release may not claim a complete corpus pass")
    cursor = _require_mapping(streaming.get("corpus_cursor"), "streaming.corpus_cursor")
    cursor_sha = _canonical_json_sha256(cursor)
    if cursor_sha != _require_sha256(
        streaming_lock.get("cursor_canonical_sha256"),
        "streaming.cursor_canonical_sha256",
    ):
        raise ValueError("streaming cursor differs from reviewed lock")
    total = _validate_streaming_cursor(
        cursor,
        corpus_info=corpus_info,
        corpus_complete=actual_complete,
    )
    if total != _require_int(
        streaming_lock.get("samples_exposed"),
        "streaming.samples_exposed",
        minimum=1,
    ):
        raise ValueError("streaming sample exposure differs from reviewed lock")

    lineage = {
        "source_contract_kind": STREAMING_V2_RELEASE_SOURCE_CONTRACT_KIND,
        "source_checkpoint": str(root.absolute()),
        "source_checkpoint_step": expected_step,
        "source_parameter_count": parameter_count,
        "source_model_vocab_capacity": model_vocab_capacity,
        "tokenizer_id_extent": actual_vocab_extent,
        "source_repository_id": repository_id,
        "source_revision": revision,
        "reviewed_release_lock_sha256": lock_raw_sha,
        "release_name": release_name,
        "backup_manifest_raw_sha256": expected_backup_sha,
        "sha256sums_raw_sha256": expected_sums_sha,
        "source_checkpoint_model_sha256": model_sha,
        "source_checkpoint_metadata_sha256": metadata_sha,
        "tokenizer_manifest_raw_sha256": manifest_raw_sha,
        "tokenizer_manifest_canonical_sha256": manifest_canonical_sha,
        "tokenizer_bundle_files_canonical_sha256": bundle_files_canonical_sha,
        "tokenizer_vocab_file_raw_sha256": vocab_file_raw_sha,
        "tokenizer_token_id_map_sha256": token_id_map_sha,
        "visual_corpus_manifest_sha256": corpus_manifest_sha,
        "visual_streaming_cursor_canonical_sha256": cursor_sha,
        "visual_samples_exposed": total,
        "visual_corpus_complete": actual_complete,
        "ocr_visual_input_contract": visual_input_contract,
        "ocr_visual_input_contract_version": visual_input_contract_version,
        "ocr_position_contract": "boundary_v1",
        "ocr_position_contract_version": 1,
        "ocr_target_encoding": "native",
        "source_ocr_tokenization_contract_version": 2,
        "ocr_tokenization_contract_version": OCR_TOKENIZATION_CONTRACT_VERSION,
        "tokenization_compatibility_rule": NATIVE_V2_TO_V3_COMPATIBILITY_RULE,
        "visual_stop_reason": "loss_plateau",
        "visual_final": True,
    }
    return {"metadata": metadata, "lineage": lineage}


def admit_visual_ocr_source(
    source_contract_kind: str = VISUAL_SOURCE_CONTRACT_ALIGNMENT_V3,
    *,
    checkpoint_dir: str | Path,
    metadata: dict[str, Any] | None,
    runtime_native_tokenization_contract: Mapping[str, Any],
    tokenizer_vocab_extent: int,
    require_terminal: bool = True,
    requested_visual_input_contract: str | None = None,
    expected_lineage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Single fail-closed dispatcher for visual OCR source admission.

    Alignment-v3 remains the default and is never silently retried as an older
    contract.  The historical streaming-v2 release is explicit and always uses
    the repository-owned reviewed lock; callers cannot supply a replacement
    lock path.  ``metadata`` must be absent for v2 so callers cannot deserialize
    its pickle before this module verifies the entire byte package.
    """

    runtime_contract = _require_mapping(
        runtime_native_tokenization_contract,
        "runtime_native_tokenization_contract",
    )
    if source_contract_kind == VISUAL_SOURCE_CONTRACT_ALIGNMENT_V3:
        if not isinstance(metadata, dict):
            raise ValueError("alignment-v3 admission requires checkpoint metadata")
        visual_input_contract = resolve_checkpoint_ocr_visual_input_contract(
            metadata,
            requested=requested_visual_input_contract,
        )
        from Model.posttrain.checkpointing import (
            validate_visual_ocr_source_contract,
        )

        lineage = dict(
            validate_visual_ocr_source_contract(
                metadata,
                checkpoint_dir,
                dict(runtime_contract),
                tokenizer_vocab_extent=tokenizer_vocab_extent,
                require_terminal=require_terminal,
            )
        )
        lineage["source_contract_kind"] = VISUAL_SOURCE_CONTRACT_ALIGNMENT_V3
        lineage["ocr_position_contract"] = (
            resolve_checkpoint_ocr_position_contract(metadata, None)
        )
        lineage["ocr_position_contract_version"] = (
            OCR_POSITION_CONTRACT_METADATA_VERSION
        )
        lineage["ocr_visual_input_contract"] = visual_input_contract
        lineage["ocr_visual_input_contract_version"] = (
            ocr_visual_input_contract_version(visual_input_contract)
        )
        result = {"metadata": dict(metadata), "lineage": lineage}
    elif source_contract_kind == VISUAL_SOURCE_CONTRACT_STREAMING_V2_RELEASE:
        if metadata is not None:
            raise ValueError(
                "streaming-v2-release metadata must not be loaded before admission"
            )
        fixed_lock = (
            Path(__file__).resolve().parent
            / "release_locks"
            / "dol_1_2_ocr.json"
        )
        result = validate_reviewed_streaming_v2_ocr_release(
            checkpoint_dir,
            fixed_lock,
            runtime_native_tokenization_contract=runtime_contract,
            expected_repository_id=MONTLOK_DOL_1_2_OCR_REPOSITORY_ID,
            expected_revision=MONTLOK_DOL_1_2_OCR_REVISION,
        )
        saved_visual_input_contract = validate_ocr_visual_input_contract(
            result["lineage"].get("ocr_visual_input_contract")
        )
        if requested_visual_input_contract is not None:
            requested = validate_ocr_visual_input_contract(
                requested_visual_input_contract
            )
            if requested != saved_visual_input_contract:
                raise ValueError(
                    "OCR visual input contract mismatch: "
                    f"checkpoint={saved_visual_input_contract!r} "
                    f"requested={requested!r}"
                )
        _require_mapping(
            result["metadata"].get("rdt_config"), "admitted metadata.rdt_config"
        )
        if result["lineage"].get("tokenizer_id_extent") != tokenizer_vocab_extent:
            raise ValueError(
                "admitted source tokenizer extent differs from runtime tokenizer extent"
            )
    else:
        raise ValueError(
            "unsupported visual OCR source contract kind: "
            f"{source_contract_kind!r}"
        )

    lineage = dict(result["lineage"])
    if expected_lineage is not None and lineage != dict(expected_lineage):
        raise ValueError("admitted visual source lineage differs from expected lineage")
    return {"metadata": dict(result["metadata"]), "lineage": lineage}


__all__ = [
    "MONTLOK_DOL_1_2_OCR_REPOSITORY_ID",
    "MONTLOK_DOL_1_2_OCR_REVISION",
    "NATIVE_V2_TO_V3_COMPATIBILITY_RULE",
    "REVIEWED_OCR_RELEASE_LOCK_KIND",
    "REVIEWED_OCR_RELEASE_LOCK_SCHEMA_VERSION",
    "STREAMING_V2_RELEASE_SOURCE_CONTRACT_KIND",
    "TRUSTED_REVIEWED_OCR_RELEASE_LOCKS",
    "VISUAL_SOURCE_CONTRACT_ALIGNMENT_V3",
    "VISUAL_SOURCE_CONTRACT_STREAMING_V2_RELEASE",
    "admit_visual_ocr_source",
    "validate_reviewed_streaming_v2_ocr_release",
]
