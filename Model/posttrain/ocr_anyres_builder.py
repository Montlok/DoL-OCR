# -*- coding: utf-8 -*-

"""Prepare human review packs and finalize approved anyres OCR manifest v2 data.

Review preparation proposes but never approves labels or reading order.
Finalization admits only reviews bound to exact source pixels and native plans.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from Model.config import EOS_ID
from Model.ocr.anyres_preprocess_contract import (
    validate_anyres_preprocess_contract,
)
from Model.ocr.cut_qa import analyze_cut_qa
from Model.ocr.near_duplicate import (
    PHASH_CONTRACT,
    cluster_perceptual_hashes,
    perceptual_hash,
)
from Model.ocr.tokenization import canonicalize_native_ocr_text
from Model.omvt.native_planner import plan_native_macro_windows
from Model.posttrain.ocr_anyres_data import (
    AnyresOCRDataset,
    rgba_pixel_sha256,
)
from Model.posttrain.ocr_anyres_manifest import (
    ANYRES_PUBLIC_SPLITS,
    ANYRES_VALIDATION_SPLITS,
    ANYRES_SCHEMA_VERSION,
    ANYRES_VISUAL_CONTRACT,
    canonical_json_sha256,
    quota_counts,
    validate_anyres_assets_views_samples,
)


SOURCE_SCHEMA_VERSION = 1
REVIEW_SCHEMA_VERSION = 1
SPLIT_POLICY_SCHEMA_VERSION = 1
SPLIT_LOCK_SCHEMA_VERSION = 1
BUILDER_RECEIPT_SCHEMA_VERSION = 1
SPLIT_POLICY_KIND = "dol_ocr_anyres_split_policy_v1"
SPLIT_LOCK_KIND = "dol_ocr_anyres_split_lock_v1"
BUILD_RECEIPT_KIND = "dol_ocr_anyres_finalize_receipt_v1"
READY_KIND = "dol_ocr_anyres_ready_v1"
CUT_ADJUDICATION_KIND = "dol_ocr_anyres_cut_adjudication_v1"
REVIEW_ITEM_KIND = "dol_ocr_anyres_review_item_v1"
REVIEW_PACK_RECEIPT_KIND = "dol_ocr_anyres_review_pack_receipt_v1"
REVIEW_PACK_READY_KIND = "dol_ocr_anyres_review_pack_ready_v1"
ALLOWED_SPLITS = ANYRES_PUBLIC_SPLITS
MAX_PHASH_RECORDS = 20_000
MAX_RAW_BYTES = 512 * 1024 * 1024

_SOURCE_KEYS = frozenset(
    {
        "schema_version",
        "sample_id",
        "raw_relpath",
        "source_document_id",
        "source_capture_id",
        "style",
        "font_id",
        "writer_id",
        "difficulty",
    }
)
_REVIEW_KEYS = frozenset(
    {
        "schema_version",
        "sample_id",
        "decision",
        "review_id",
        "reviewer_id",
        "reviewed_at",
        "reference_raw",
        "writing_mode",
        "raw_sha256",
        "canonical_pixel_sha256",
        "native_plan_sha256",
        "reading_order_window_indices",
        "cut_adjudications",
    }
)
_POLICY_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "seed",
        "fractions",
        "phash_contract",
        "phash_max_hamming_distance",
    }
)
_LOCK_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "policy_canonical_sha256",
        "assignments",
        "phash_index",
    }
)
_WRITING_MODES = frozenset(
    {"vertical-lr", "vertical-rl", "horizontal-tb"}
)
_DIFFICULTIES = frozenset({"good", "medium", "poor"})
_SPLIT_IDENTITY_KINDS = frozenset(
    {
        "sample",
        "document",
        "capture",
        "writer",
        "pixel",
        "near_duplicate",
        "cumulative_near_duplicate",
    }
)
_SPLIT_IDENTITY_RE = re.compile(
    rf"^(?:{'|'.join(sorted(_SPLIT_IDENTITY_KINDS))}):[0-9a-f]{{64}}$"
)
_ASSET_ID_RE = re.compile(r"^asset-[0-9a-f]{64}$")


class _Quarantine(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class _Candidate:
    sample_id: str
    source: dict[str, Any]
    review: dict[str, Any]
    asset: dict[str, Any]
    views: list[dict[str, Any]]
    reference_raw: dict[str, Any]
    reference_model: dict[str, Any]
    reference_token_count: int
    writing_mode: str
    reading_order: list[str]
    phash: str
    adjudication_records: list[dict[str, Any]]


def _canonical_json_bytes(value: object, *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
    else:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ) + "\n"
    return text.encode("utf-8")


def _strict_object_pairs(where: str):
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{where}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    return hook


def _read_stable_regular(
    path: str | Path,
    *,
    where: str,
    max_bytes: int | None = None,
) -> tuple[bytes, str]:
    source = Path(path)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(source, flags)
    except (FileNotFoundError, OSError) as exc:
        raise ValueError(f"{where} cannot be opened safely: {source}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{where} must be a regular file")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if max_bytes is not None and total > max_bytes:
                raise _Quarantine("raw_file_bytes_exceed_builder_limit")
            chunks.append(chunk)
            digest.update(chunk)
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
            raise ValueError(f"{where} changed while being read and hashed")
        return b"".join(chunks), digest.hexdigest()
    finally:
        os.close(descriptor)


def _decode_strict_json(raw: bytes, *, where: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(text, object_pairs_hook=_strict_object_pairs(where))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{where} must be strict UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{where} must contain one JSON object")
    return value


def _load_strict_json(path: str | Path, *, where: str) -> tuple[dict[str, Any], bytes, str]:
    raw, digest = _read_stable_regular(path, where=where)
    return _decode_strict_json(raw, where=where), raw, digest


def _load_strict_jsonl(path: str | Path, *, where: str) -> tuple[list[dict[str, Any]], str]:
    raw, digest = _read_stable_regular(path, where=where)
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{where} must be strict UTF-8 JSONL") from exc
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise ValueError(f"{where}:{line_no}: blank lines are forbidden")
        try:
            value = json.loads(
                line,
                object_pairs_hook=_strict_object_pairs(f"{where}:{line_no}"),
            )
        except json.JSONDecodeError as exc:
            raise ValueError(f"{where}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{where}:{line_no}: row must be an object")
        value = dict(value)
        value["_input_line"] = line_no
        rows.append(value)
    if not rows:
        raise ValueError(f"{where} must not be empty")
    return rows, digest


def _identifier(value: object, where: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise _Quarantine(f"{where}_missing_or_invalid")
    if "\x00" in value:
        raise _Quarantine(f"{where}_contains_nul")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise _Quarantine(f"{where}_invalid_unicode") from exc
    return value


def _optional_identifier(value: object, where: str) -> str | None:
    if value is None:
        return None
    return _identifier(value, where)


def _relative_path(value: object, where: str) -> str:
    text = _identifier(value, where)
    if "\\" in text:
        raise _Quarantine(f"{where}_not_posix_relative")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or path.as_posix() != text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise _Quarantine(f"{where}_not_posix_relative")
    return text


def _lower_sha256(value: object, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise _Quarantine(f"{where}_invalid_sha256")
    return value


def _normalize_source(row: Mapping[str, Any]) -> dict[str, Any]:
    line_no = row.get("_input_line", "unknown")
    payload = {key: value for key, value in row.items() if key != "_input_line"}
    if set(payload) != _SOURCE_KEYS:
        raise _Quarantine("source_schema_fields_invalid")
    if payload.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise _Quarantine("source_schema_version_invalid")
    sample_id = _identifier(payload["sample_id"], "sample_id")
    style = payload["style"]
    font_id = _optional_identifier(payload["font_id"], "font_id")
    writer_id = _optional_identifier(payload["writer_id"], "writer_id")
    difficulty = payload["difficulty"]
    if style == "print":
        if font_id is None or writer_id is not None or difficulty is not None:
            raise _Quarantine("print_provenance_invalid")
    elif style == "handwritten":
        if font_id is not None or writer_id is None or difficulty not in _DIFFICULTIES:
            raise _Quarantine("handwritten_provenance_invalid")
    else:
        raise _Quarantine("style_invalid")
    return {
        "sample_id": sample_id,
        "raw_relpath": _relative_path(payload["raw_relpath"], "raw_relpath"),
        "source_document_id": _identifier(
            payload["source_document_id"], "source_document_id"
        ),
        "source_capture_id": _identifier(
            payload["source_capture_id"], "source_capture_id"
        ),
        "style": style,
        "font_id": font_id,
        "writer_id": writer_id,
        "difficulty": difficulty,
        "_input_line": int(line_no),
    }


def _normalize_review(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = {key: value for key, value in row.items() if key != "_input_line"}
    if set(payload) != _REVIEW_KEYS:
        raise _Quarantine("review_schema_fields_invalid")
    if payload.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise _Quarantine("review_schema_version_invalid")
    sample_id = _identifier(payload["sample_id"], "review_sample_id")
    if payload["decision"] != "approved":
        raise _Quarantine("human_review_not_approved")
    reviewed_at = _identifier(payload["reviewed_at"], "reviewed_at")
    try:
        timestamp = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _Quarantine("reviewed_at_invalid") from exc
    if timestamp.tzinfo is None:
        raise _Quarantine("reviewed_at_missing_timezone")
    reference = payload["reference_raw"]
    if not isinstance(reference, str) or not reference:
        raise _Quarantine("reference_raw_missing_or_invalid")
    if "\x00" in reference or "\ufffd" in reference:
        raise _Quarantine("reference_raw_lossy_or_contains_nul")
    try:
        reference.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise _Quarantine("reference_raw_invalid_unicode") from exc
    writing_mode = payload["writing_mode"]
    if writing_mode not in _WRITING_MODES:
        raise _Quarantine("writing_mode_invalid")
    reading_order = payload["reading_order_window_indices"]
    if not isinstance(reading_order, list) or any(type(value) is not int for value in reading_order):
        raise _Quarantine("reading_order_invalid")
    adjudications = payload["cut_adjudications"]
    if not isinstance(adjudications, dict):
        raise _Quarantine("cut_adjudications_invalid")
    normalized_adjudications: dict[int, dict[str, str]] = {}
    for raw_index, record in adjudications.items():
        if (
            not isinstance(raw_index, str)
            or not raw_index.isdigit()
            or str(int(raw_index)) != raw_index
        ):
            raise _Quarantine("cut_adjudication_index_invalid")
        if not isinstance(record, Mapping) or set(record) != {"decision", "reason"}:
            raise _Quarantine("cut_adjudication_record_invalid")
        if record.get("decision") != "manual_accepted":
            raise _Quarantine("cut_adjudication_decision_invalid")
        reason = _identifier(record.get("reason"), "cut_adjudication_reason")
        normalized_adjudications[int(raw_index)] = {
            "decision": "manual_accepted",
            "reason": reason,
        }
    return {
        "sample_id": sample_id,
        "review_id": _identifier(payload["review_id"], "review_id"),
        "reviewer_id": _identifier(payload["reviewer_id"], "reviewer_id"),
        "reviewed_at": reviewed_at,
        "reference_raw": reference,
        "writing_mode": writing_mode,
        "raw_sha256": _lower_sha256(payload["raw_sha256"], "review_raw"),
        "canonical_pixel_sha256": _lower_sha256(
            payload["canonical_pixel_sha256"], "review_canonical_pixel"
        ),
        "native_plan_sha256": _lower_sha256(
            payload["native_plan_sha256"], "native_plan"
        ),
        "reading_order_window_indices": list(reading_order),
        "cut_adjudications": normalized_adjudications,
    }


def validate_split_policy(value: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) != _POLICY_KEYS:
        raise ValueError("split policy fields differ from contract")
    if value.get("schema_version") != SPLIT_POLICY_SCHEMA_VERSION:
        raise ValueError("split policy schema_version must be 1")
    if value.get("kind") != SPLIT_POLICY_KIND:
        raise ValueError(f"split policy kind must be {SPLIT_POLICY_KIND}")
    seed = value["seed"]
    if type(seed) is not int:
        raise ValueError("split policy seed must be an integer")
    fractions = value["fractions"]
    if not isinstance(fractions, Mapping) or set(fractions) != set(ALLOWED_SPLITS):
        raise ValueError(
            "split policy fractions must define train, sft_validation, "
            "kl_selection, and formal_monitor"
        )
    normalized_fractions: dict[str, float] = {}
    for split in ALLOWED_SPLITS:
        fraction = fractions[split]
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
            raise ValueError("split policy fractions must be numbers")
        fraction = float(fraction)
        if not math.isfinite(fraction) or fraction <= 0.0:
            raise ValueError("split policy fractions must be finite and positive")
        normalized_fractions[split] = fraction
    if not math.isclose(sum(normalized_fractions.values()), 1.0, abs_tol=1e-12):
        raise ValueError("split policy fractions must sum to 1")
    threshold = value["phash_max_hamming_distance"]
    if type(threshold) is not int or not 0 <= threshold <= 64:
        raise ValueError("phash_max_hamming_distance must be an integer in [0,64]")
    if value.get("phash_contract") != PHASH_CONTRACT:
        raise ValueError(f"split policy phash_contract must be {PHASH_CONTRACT}")
    return {
        "schema_version": SPLIT_POLICY_SCHEMA_VERSION,
        "kind": SPLIT_POLICY_KIND,
        "seed": seed,
        "fractions": normalized_fractions,
        "phash_contract": PHASH_CONTRACT,
        "phash_max_hamming_distance": threshold,
    }


def validate_split_lock(
    value: Mapping[str, Any],
    *,
    policy_canonical_sha256: str,
) -> dict[str, Any]:
    if set(value) != _LOCK_KEYS:
        raise ValueError("split lock fields differ from contract")
    if value.get("schema_version") != SPLIT_LOCK_SCHEMA_VERSION:
        raise ValueError("split lock schema_version must be 1")
    if value.get("kind") != SPLIT_LOCK_KIND:
        raise ValueError(f"split lock kind must be {SPLIT_LOCK_KIND}")
    if value.get("policy_canonical_sha256") != policy_canonical_sha256:
        raise ValueError("split lock was created under another split policy")
    assignments = value.get("assignments")
    if not isinstance(assignments, Mapping):
        raise ValueError("split lock assignments must be an object")
    normalized: dict[str, str] = {}
    for key, split in assignments.items():
        if not isinstance(key, str) or not key or key != key.strip():
            raise ValueError("split lock identity keys must be stripped strings")
        if _SPLIT_IDENTITY_RE.fullmatch(key) is None:
            raise ValueError("split lock contains a malformed identity key")
        if split not in ALLOWED_SPLITS:
            raise ValueError("split lock contains an unknown split")
        normalized[key] = str(split)
    phash_index = value.get("phash_index")
    if not isinstance(phash_index, Mapping):
        raise ValueError("split lock phash_index must be an object")
    normalized_phash: dict[str, dict[str, str]] = {}
    for asset_id, record in phash_index.items():
        if not isinstance(asset_id, str) or not asset_id or asset_id != asset_id.strip():
            raise ValueError("split lock pHash asset ids must be stripped strings")
        if _ASSET_ID_RE.fullmatch(asset_id) is None:
            raise ValueError("split lock pHash asset ids must be builder asset ids")
        if not isinstance(record, Mapping) or set(record) != {"value", "split"}:
            raise ValueError("split lock pHash records have invalid fields")
        value = record.get("value")
        if (
            not isinstance(value, str)
            or len(value) != 16
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("split lock pHash values must be 16 lowercase hex")
        split = record.get("split")
        if split not in ALLOWED_SPLITS:
            raise ValueError("split lock pHash record contains an unknown split")
        normalized_phash[asset_id] = {"value": value, "split": str(split)}
    return {
        "schema_version": SPLIT_LOCK_SCHEMA_VERSION,
        "kind": SPLIT_LOCK_KIND,
        "policy_canonical_sha256": policy_canonical_sha256,
        "assignments": dict(sorted(normalized.items())),
        "phash_index": dict(sorted(normalized_phash.items())),
    }


def split_identity_key(kind: str, value: str) -> str:
    if not kind or not value:
        raise ValueError("split identity kind/value must not be empty")
    digest = hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"{kind}:{digest}"


def _resolve_source_file(root: Path, relpath: str) -> Path:
    candidate = root.joinpath(*PurePosixPath(relpath).parts)
    current = root
    for part in PurePosixPath(relpath).parts:
        current = current / part
        if current.is_symlink():
            raise _Quarantine("raw_path_traverses_symlink")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise _Quarantine("raw_path_missing_or_escapes_root") from exc
    if not resolved.is_file():
        raise _Quarantine("raw_path_not_regular_file")
    return resolved


def _rgba_image_sha256(image: Image.Image) -> str:
    rgba = image.convert("RGBA")
    width, height = rgba.size
    header = f"dol-ocr-rgba-v1\0{width}\0{height}\0".encode("ascii")
    return hashlib.sha256(header + rgba.tobytes()).hexdigest()


def _png_bytes(image: Image.Image) -> bytes:
    target = io.BytesIO()
    image.save(
        target,
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    return target.getvalue()


def _decode_canonical_rgb(
    raw: bytes,
    *,
    max_pixels: int,
) -> tuple[Image.Image, tuple[int, int]]:
    try:
        with Image.open(io.BytesIO(raw)) as image:
            raw_size = tuple(int(value) for value in image.size)
            if raw_size[0] * raw_size[1] > max_pixels:
                raise _Quarantine("raw_image_pixel_budget_exceeded")
            oriented = ImageOps.exif_transpose(image)
            if oriented.width * oriented.height > max_pixels:
                raise _Quarantine("canonical_image_pixel_budget_exceeded")
            oriented.load()
            canonical = oriented.convert("RGB")
            canonical.load()
            if _rgba_image_sha256(oriented) != _rgba_image_sha256(canonical):
                raise _Quarantine("rgb_conversion_is_not_pixel_lossless")
            return canonical, raw_size
    except _Quarantine:
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError) as exc:
        raise _Quarantine("raw_image_decode_failed") from exc


def _reference_payload(text: str) -> dict[str, Any]:
    encoded = text.encode("utf-8", errors="strict")
    return {
        "text": text,
        "utf8_sha256": hashlib.sha256(encoded).hexdigest(),
        "codepoints": [ord(character) for character in text],
    }


def _write_bytes(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, mode)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    payload = b"".join(_canonical_json_bytes(row) for row in rows)
    _write_bytes(path, payload)
    return hashlib.sha256(payload).hexdigest()


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    directories = [Path(current) for current, _dirs, _files in os.walk(root)]
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        _fsync_directory(directory)


def _atomic_rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(source_bytes, destination_bytes, 0x00000004)
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(-100, source_bytes, -100, destination_bytes, 0x00000001)
    else:
        raise RuntimeError(
            "atomic no-replace directory publication is unsupported on this platform"
        )
    if result != 0:
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(
                error,
                f"refusing to overwrite immutable dataset: {destination}",
                str(destination),
            )
        raise OSError(error, os.strerror(error), str(destination))


def _candidate_asset_id(sample_id: str, raw_sha256: str) -> str:
    digest = hashlib.sha256(
        f"dol-ocr-anyres-asset-v1\0{sample_id}\0{raw_sha256}".encode("utf-8")
    ).hexdigest()
    return f"asset-{digest}"


def _source_geometry(
    *,
    source: Mapping[str, Any],
    source_root: Path,
    preprocess: Mapping[str, Any],
) -> dict[str, Any]:
    sample_id = str(source["sample_id"])
    resolved = _resolve_source_file(source_root, str(source["raw_relpath"]))
    raw_bytes, raw_sha256 = _read_stable_regular(
        resolved,
        where=f"raw image {sample_id!r}",
        max_bytes=MAX_RAW_BYTES,
    )
    max_pixels = int(preprocess["budgets"]["decode"]["max_pixels_per_asset"])
    canonical, raw_size = _decode_canonical_rgb(raw_bytes, max_pixels=max_pixels)
    canonical_bytes = _png_bytes(canonical)
    canonical_sha256 = hashlib.sha256(canonical_bytes).hexdigest()
    pixel_sha256, canonical_size = rgba_pixel_sha256(
        canonical_bytes,
        max_pixels=max_pixels,
    )
    if canonical_size != canonical.size:
        raise RuntimeError("canonical PNG round-trip changed image size")
    try:
        plan = plan_native_macro_windows(
            height=canonical.height,
            width=canonical.width,
            patch_shapes=preprocess["patch_contract"]["shapes_hw"],
            max_raw_patch_tokens=int(
                preprocess["budgets"]["patch"]["max_raw_tokens_per_view"]
            ),
            max_windows=int(
                preprocess["budgets"]["window"]["max_windows_per_asset"]
            ),
            halo_px=int(
                preprocess["budgets"]["window"]["halo_pixels_per_side"]
            ),
        )
    except ValueError as exc:
        raise _Quarantine("native_plan_budget_exceeded") from exc
    asset_id = _candidate_asset_id(sample_id, raw_sha256)
    view_ids = [f"view-{asset_id[6:]}-{window.index:04d}" for window in plan.windows]
    view_boxes: dict[str, list[int]] = {}
    for view_id, window in zip(view_ids, plan.windows, strict=True):
        y0, x0, y1, x1 = window.asset_bbox_yxxy
        view_boxes[view_id] = [x0, y0, x1, y1]
    cut_qa = analyze_cut_qa(
        np.asarray(canonical.convert("L"), dtype=np.uint8),
        view_boxes,
        canonical_pixel_sha256=pixel_sha256,
        plan_sha256=plan.canonical_sha256,
    )
    return {
        "raw_bytes": raw_bytes,
        "raw_sha256": raw_sha256,
        "raw_size": raw_size,
        "canonical": canonical,
        "canonical_bytes": canonical_bytes,
        "canonical_sha256": canonical_sha256,
        "pixel_sha256": pixel_sha256,
        "plan": plan,
        "asset_id": asset_id,
        "view_ids": view_ids,
        "cut_qa": cut_qa,
        "evidence_by_id": {
            evidence.view_id: evidence for evidence in cut_qa.views
        },
    }


def _exact_window_png(canonical: Image.Image, window) -> dict[str, Any]:
    ay0, ax0, ay1, ax1 = window.asset_bbox_yxxy
    crop = canonical.crop((ax0, ay0, ax1, ay1))
    crop_bytes = _png_bytes(crop)
    pixel_sha256, size = rgba_pixel_sha256(crop_bytes)
    if size != crop.size:
        raise RuntimeError("window PNG round-trip changed image size")
    return {
        "bytes": crop_bytes,
        "sha256": hashlib.sha256(crop_bytes).hexdigest(),
        "pixel_sha256": pixel_sha256,
        "width": crop.width,
        "height": crop.height,
    }


def _process_candidate(
    *,
    source: dict[str, Any],
    review: dict[str, Any],
    source_root: Path,
    staging: Path,
    preprocess: Mapping[str, Any],
    native_encoder: Callable[[str], Sequence[int]],
    decode_reference: Callable[[Sequence[int]], str],
) -> _Candidate:
    sample_id = source["sample_id"]
    geometry = _source_geometry(
        source=source,
        source_root=source_root,
        preprocess=preprocess,
    )
    raw_bytes = geometry["raw_bytes"]
    raw_sha256 = geometry["raw_sha256"]
    raw_size = geometry["raw_size"]
    canonical = geometry["canonical"]
    canonical_bytes = geometry["canonical_bytes"]
    canonical_sha256 = geometry["canonical_sha256"]
    pixel_sha256 = geometry["pixel_sha256"]
    plan = geometry["plan"]
    asset_id = geometry["asset_id"]
    view_ids = geometry["view_ids"]
    cut_qa = geometry["cut_qa"]
    evidence_by_id = geometry["evidence_by_id"]
    if review["raw_sha256"] != raw_sha256:
        raise _Quarantine("human_review_raw_sha256_mismatch")
    if review["canonical_pixel_sha256"] != pixel_sha256:
        raise _Quarantine("human_review_canonical_pixel_sha256_mismatch")
    if review["native_plan_sha256"] != plan.canonical_sha256:
        raise _Quarantine("human_review_native_plan_mismatch")
    indices = review["reading_order_window_indices"]
    if (
        len(indices) != len(set(indices))
        or set(indices) != set(range(len(plan.windows)))
    ):
        raise _Quarantine("human_review_reading_order_not_exact")

    raw_reference = review["reference_raw"]
    model_reference = canonicalize_native_ocr_text(raw_reference)
    stats = getattr(native_encoder, "stats", None)
    if getattr(native_encoder, "mode", None) != "native" or not isinstance(stats, dict):
        raise ValueError("builder requires the strict native OCR target encoder")
    byte_fallback_before = int(stats.get("byte_fallback", 0))
    try:
        token_ids = [int(value) for value in native_encoder(raw_reference)]
    except ValueError as exc:
        raise _Quarantine("native_tokenizer_rejected_reference") from exc
    if int(stats.get("byte_fallback", 0)) != byte_fallback_before:
        raise ValueError("native OCR builder observed a byte fallback")
    if not token_ids:
        raise _Quarantine("native_reference_tokenization_empty")
    if EOS_ID in token_ids:
        raise _Quarantine("native_reference_contains_eos_token")
    if decode_reference(token_ids) != model_reference:
        raise _Quarantine("native_reference_roundtrip_mismatch")
    recommended = int(
        preprocess["budgets"]["output"]["recommended_max_new_tokens"]
    )
    if len(token_ids) + 1 > recommended:
        raise _Quarantine("reference_token_budget_exceeded")
    max_sequence = int(
        preprocess["budgets"]["context"]["max_sequence_tokens"]
    )
    visual_prefix = int(
        preprocess["budgets"]["context"]["global_visual_prefix_tokens"]
    )
    fixed_prompt = int(
        preprocess["budgets"]["context"]["global_prompt_fixed_tokens"]
    )
    if fixed_prompt + visual_prefix + len(token_ids) + 1 > max_sequence:
        raise _Quarantine("reference_context_budget_exceeded")

    suspect_indices = {
        index
        for index, view_id in enumerate(view_ids)
        if evidence_by_id[view_id].cut_suspect
    }
    adjudications = review["cut_adjudications"]
    if set(adjudications) - suspect_indices:
        raise _Quarantine("unexpected_cut_adjudication")
    if suspect_indices - set(adjudications):
        raise _Quarantine("cut_suspect_missing_human_adjudication")
    if cut_qa.uncovered_component_ids:
        raise _Quarantine("connected_component_uncovered")

    raw_relpath = f"raw/{asset_id}.source"
    canonical_relpath = f"canonical/{asset_id}.png"
    views: list[dict[str, Any]] = []
    adjudication_records: list[dict[str, Any]] = []
    for index, (view_id, window) in enumerate(zip(view_ids, plan.windows, strict=True)):
        ay0, ax0, ay1, ax1 = window.asset_bbox_yxxy
        crop = _exact_window_png(canonical, window)
        derived_relpath = f"derived/{view_id}.png"
        qa = evidence_by_id[view_id].manifest_qa()
        if index in suspect_indices:
            adjudication_record = {
                "schema_version": 1,
                "kind": CUT_ADJUDICATION_KIND,
                "sample_id": sample_id,
                "view_id": view_id,
                "window_index": index,
                "native_plan_sha256": plan.canonical_sha256,
                "asset_pixel_sha256": pixel_sha256,
                "asset_raw_sha256": raw_sha256,
                "cut_qa_coverage_proof_sha256": cut_qa.coverage_proof_sha256,
                "review_id": review["review_id"],
                "reviewer_id": review["reviewer_id"],
                "reviewed_at": review["reviewed_at"],
                "decision": "manual_accepted",
                "reason": adjudications[index]["reason"],
            }
            adjudication_sha256 = canonical_json_sha256(adjudication_record)
            adjudication_record["canonical_sha256"] = adjudication_sha256
            adjudication_records.append(adjudication_record)
            qa["review_status"] = "manual_accepted"
            qa["adjudication_sha256"] = adjudication_sha256
        views.append(
            {
                "schema_version": ANYRES_SCHEMA_VERSION,
                "view_id": view_id,
                "asset_id": asset_id,
                "kind": "page" if plan.identity else "tile",
                "derived_relpath": derived_relpath,
                "derived_width": crop["width"],
                "derived_height": crop["height"],
                "derived_sha256": crop["sha256"],
                "box_xyxy": [ax0, ay0, ax1, ay1],
                "transform": {
                    "contract": "native_macro_window_crop_v1",
                    "plan_sha256": plan.canonical_sha256,
                    "window_index": index,
                    "asset_bbox_yxxy": list(window.asset_bbox_yxxy),
                    "ownership_bbox_yxxy": list(window.ownership_bbox_yxxy),
                    "halo_tlbr": list(window.halo_tlbr),
                    "raw_patch_tokens": window.raw_patch_tokens,
                },
                "pixel_sha256": crop["pixel_sha256"],
                "qa": qa,
            }
        )
        _write_bytes(staging / derived_relpath, crop["bytes"])
    _write_bytes(staging / raw_relpath, raw_bytes)
    _write_bytes(staging / canonical_relpath, canonical_bytes)

    asset = {
        "schema_version": ANYRES_SCHEMA_VERSION,
        "asset_id": asset_id,
        "source_document_id": source["source_document_id"],
        "source_capture_id": source["source_capture_id"],
        "raw_relpath": raw_relpath,
        "canonical_relpath": canonical_relpath,
        "raw_sha256": raw_sha256,
        "canonical_sha256": canonical_sha256,
        "pixel_sha256": pixel_sha256,
        "raw_width": raw_size[0],
        "raw_height": raw_size[1],
        "canonical_width": canonical.width,
        "canonical_height": canonical.height,
        "native_plan": {
            "preprocess_contract_sha256": preprocess[
                "contract_canonical_sha256"
            ],
            "plan_sha256": plan.canonical_sha256,
            "payload": plan.canonical_payload(),
        },
        "cut_qa": {
            "coverage_proof_sha256": cut_qa.coverage_proof_sha256,
            "payload": json.loads(
                json.dumps(
                    cut_qa.proof_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            ),
        },
        "near_duplicate": None,
    }
    return _Candidate(
        sample_id=sample_id,
        source=source,
        review=review,
        asset=asset,
        views=views,
        reference_raw=_reference_payload(raw_reference),
        reference_model=_reference_payload(model_reference),
        reference_token_count=len(token_ids),
        writing_mode=review["writing_mode"],
        reading_order=[view_ids[index] for index in indices],
        phash=perceptual_hash(canonical),
        adjudication_records=adjudication_records,
    )


class _UnionFind:
    def __init__(self, ids: Sequence[str]) -> None:
        self.parent = {value: value for value in ids}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        while parent != self.parent[parent]:
            self.parent[parent] = self.parent[self.parent[parent]]
            parent = self.parent[parent]
        self.parent[value] = parent
        return parent

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            low, high = sorted((left_root, right_root))
            self.parent[high] = low


def _component_id(member_asset_ids: Sequence[str]) -> str:
    digest = hashlib.sha256(
        json.dumps(
            sorted(member_asset_ids),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return f"leakage:{digest}"


def _assign_components(
    candidates: Sequence[_Candidate],
    *,
    combined_phash_clusters,
    policy: Mapping[str, Any],
    existing_lock: Mapping[str, Any],
) -> tuple[dict[str, tuple[str, str]], dict[str, Any], list[dict[str, Any]]]:
    ids = [str(candidate.asset["asset_id"]) for candidate in candidates]
    union_find = _UnionFind(ids)
    by_relation: dict[tuple[str, str], list[str]] = defaultdict(list)
    for candidate in candidates:
        asset_id = str(candidate.asset["asset_id"])
        source = candidate.source
        by_relation[("document", source["source_document_id"])].append(asset_id)
        by_relation[("capture", source["source_capture_id"])].append(asset_id)
        if source["writer_id"] is not None:
            by_relation[("writer", source["writer_id"])].append(asset_id)
        by_relation[("pixel", candidate.asset["pixel_sha256"])].append(asset_id)
    for cluster in combined_phash_clusters:
        current_members = [member for member in cluster.member_ids if member in union_find.parent]
        if not current_members:
            continue
        by_relation[("near_duplicate", cluster.cluster_id)].extend(current_members)
    for members in by_relation.values():
        for member in members[1:]:
            union_find.union(members[0], member)

    components: dict[str, list[str]] = defaultdict(list)
    candidate_by_asset = {str(candidate.asset["asset_id"]): candidate for candidate in candidates}
    for asset_id in ids:
        components[union_find.find(asset_id)].append(asset_id)
    assignments = dict(existing_lock["assignments"])
    historical_phash = existing_lock["phash_index"]
    combined_cluster_by_current: dict[str, Any] = {}
    for cluster in combined_phash_clusters:
        for member in cluster.member_ids:
            if member in candidate_by_asset:
                combined_cluster_by_current[member] = cluster
    result: dict[str, tuple[str, str]] = {}
    component_receipts: list[dict[str, Any]] = []
    for member_ids in sorted((sorted(values) for values in components.values())):
        identity_keys: set[str] = set()
        historical_phash_splits: set[str] = set()
        for asset_id in member_ids:
            candidate = candidate_by_asset[asset_id]
            source = candidate.source
            identity_keys.add(split_identity_key("sample", candidate.sample_id))
            identity_keys.add(split_identity_key("document", source["source_document_id"]))
            identity_keys.add(split_identity_key("capture", source["source_capture_id"]))
            if source["writer_id"] is not None:
                identity_keys.add(split_identity_key("writer", source["writer_id"]))
            identity_keys.add(split_identity_key("pixel", candidate.asset["pixel_sha256"]))
            cluster_id = candidate.asset["near_duplicate"]["cluster_id"]
            identity_keys.add(split_identity_key("near_duplicate", cluster_id))
            combined_cluster = combined_cluster_by_current[asset_id]
            identity_keys.add(
                split_identity_key("cumulative_near_duplicate", combined_cluster.cluster_id)
            )
            for member in combined_cluster.member_ids:
                if member in historical_phash:
                    historical_phash_splits.add(
                        str(historical_phash[member]["split"])
                    )
        locked_splits = {
            assignments[key] for key in identity_keys if key in assignments
        } | historical_phash_splits
        if len(locked_splits) > 1:
            raise ValueError(
                "current document/capture/writer/exact-pixel/pHash component "
                f"cross-connects locked splits: members={member_ids}, "
                f"splits={sorted(locked_splits)}"
            )
        component_id = _component_id(member_ids)
        if locked_splits:
            split = next(iter(locked_splits))
            origin = "existing_lock"
        else:
            stable = hashlib.sha256(
                f"{policy['seed']}\0{component_id}".encode("utf-8")
            ).digest()
            unit = int.from_bytes(stable, "big") / float(1 << 256)
            cumulative = 0.0
            split = ALLOWED_SPLITS[-1]
            for candidate_split in ALLOWED_SPLITS:
                cumulative += float(policy["fractions"][candidate_split])
                if unit < cumulative:
                    split = candidate_split
                    break
            origin = "deterministic_policy"
        for key in identity_keys:
            prior = assignments.get(key)
            if prior is not None and prior != split:
                raise RuntimeError("split-lock mutation was attempted")
            assignments[key] = split
        for asset_id in member_ids:
            result[asset_id] = (split, component_id)
        component_receipts.append(
            {
                "component_id": component_id,
                "member_asset_ids": member_ids,
                "identity_keys": sorted(identity_keys),
                "split": split,
                "assignment_origin": origin,
            }
        )
    updated_lock = {
        "schema_version": SPLIT_LOCK_SCHEMA_VERSION,
        "kind": SPLIT_LOCK_KIND,
        "policy_canonical_sha256": existing_lock["policy_canonical_sha256"],
        "assignments": dict(sorted(assignments.items())),
        "phash_index": dict(sorted(historical_phash.items())),
    }
    for candidate in candidates:
        asset_id = str(candidate.asset["asset_id"])
        split = result[asset_id][0]
        prior = updated_lock["phash_index"].get(asset_id)
        current_record = {"value": candidate.phash, "split": split}
        if prior is not None and prior != current_record:
            raise ValueError(
                f"immutable pHash index changed for asset {asset_id!r}: "
                f"{prior!r} != {current_record!r}"
            )
        updated_lock["phash_index"][asset_id] = current_record
    updated_lock["phash_index"] = dict(sorted(updated_lock["phash_index"].items()))
    return result, updated_lock, component_receipts


def _quarantine_entry(
    sample_id: str,
    *,
    source_line: int | None,
    reason: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "sample_id": sample_id,
        "source_line": source_line,
        "reasons": [reason],
    }


def _reading_order_proposals(plan) -> dict[str, list[int]]:
    windows = list(plan.windows)
    return {
        "vertical-lr": [
            window.index
            for window in sorted(
                windows,
                key=lambda window: (
                    window.ownership_bbox_yxxy[1],
                    window.ownership_bbox_yxxy[0],
                    window.index,
                ),
            )
        ],
        "vertical-rl": [
            window.index
            for window in sorted(
                windows,
                key=lambda window: (
                    -window.ownership_bbox_yxxy[3],
                    window.ownership_bbox_yxxy[0],
                    window.index,
                ),
            )
        ],
        "horizontal-tb": [
            window.index
            for window in sorted(
                windows,
                key=lambda window: (
                    window.ownership_bbox_yxxy[0],
                    window.ownership_bbox_yxxy[1],
                    window.index,
                ),
            )
        ],
    }


def _prepare_review_source(
    *,
    source: dict[str, Any],
    source_root: Path,
    staging: Path,
    preprocess: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    sample_id = source["sample_id"]
    geometry = _source_geometry(
        source=source,
        source_root=source_root,
        preprocess=preprocess,
    )
    raw_bytes = geometry["raw_bytes"]
    raw_sha256 = geometry["raw_sha256"]
    raw_size = geometry["raw_size"]
    canonical = geometry["canonical"]
    canonical_bytes = geometry["canonical_bytes"]
    canonical_sha256 = geometry["canonical_sha256"]
    pixel_sha256 = geometry["pixel_sha256"]
    plan = geometry["plan"]
    asset_id = geometry["asset_id"]
    view_ids = geometry["view_ids"]
    cut_qa = geometry["cut_qa"]
    evidence_by_id = geometry["evidence_by_id"]
    raw_relpath = f"raw/{asset_id}.source"
    canonical_relpath = f"canonical/{asset_id}.png"
    _write_bytes(staging / raw_relpath, raw_bytes)
    _write_bytes(staging / canonical_relpath, canonical_bytes)
    window_rows: list[dict[str, Any]] = []
    suspect_indices: list[int] = []
    for view_id, window in zip(view_ids, plan.windows, strict=True):
        crop = _exact_window_png(canonical, window)
        preview_relpath = f"windows/{view_id}.png"
        _write_bytes(staging / preview_relpath, crop["bytes"])
        qa = evidence_by_id[view_id].manifest_qa()
        if bool(qa["cut_suspect"]):
            suspect_indices.append(window.index)
        window_rows.append(
            {
                "view_id": view_id,
                "window_index": window.index,
                "preview_relpath": preview_relpath,
                "preview_sha256": crop["sha256"],
                "preview_pixel_sha256": crop["pixel_sha256"],
                "width": crop["width"],
                "height": crop["height"],
                "asset_bbox_yxxy": list(window.asset_bbox_yxxy),
                "ownership_bbox_yxxy": list(window.ownership_bbox_yxxy),
                "halo_tlbr": list(window.halo_tlbr),
                "raw_patch_tokens": window.raw_patch_tokens,
                "cut_qa": qa,
            }
        )
    proposals = _reading_order_proposals(plan)
    cut_payload = json.loads(
        json.dumps(
            cut_qa.proof_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    item = {
        "schema_version": 1,
        "kind": REVIEW_ITEM_KIND,
        "sample_id": sample_id,
        "source_line": source["_input_line"],
        "provenance": {
            "source_document_id": source["source_document_id"],
            "source_capture_id": source["source_capture_id"],
            "style": source["style"],
            "font_id": source["font_id"],
            "writer_id": source["writer_id"],
            "difficulty": source["difficulty"],
        },
        "raw": {
            "relpath": raw_relpath,
            "sha256": raw_sha256,
            "width": raw_size[0],
            "height": raw_size[1],
        },
        "canonical": {
            "relpath": canonical_relpath,
            "sha256": canonical_sha256,
            "pixel_sha256": pixel_sha256,
            "width": canonical.width,
            "height": canonical.height,
        },
        "native_plan": {
            "preprocess_contract_sha256": preprocess[
                "contract_canonical_sha256"
            ],
            "plan_sha256": plan.canonical_sha256,
            "payload": plan.canonical_payload(),
        },
        "windows": window_rows,
        "reading_order_proposals": proposals,
        "cut_qa": {
            "coverage_proof_sha256": cut_qa.coverage_proof_sha256,
            "payload": cut_payload,
            "suspect_window_indices": suspect_indices,
            "uncovered_component_ids": list(cut_qa.uncovered_component_ids),
        },
        "admission_blockers": (
            ["connected_component_uncovered"]
            if cut_qa.uncovered_component_ids
            else []
        ),
    }
    template = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "sample_id": sample_id,
        "decision": "pending",
        "review_id": "",
        "reviewer_id": "",
        "reviewed_at": "",
        "reference_raw": "",
        "writing_mode": "vertical-lr",
        "raw_sha256": raw_sha256,
        "canonical_pixel_sha256": pixel_sha256,
        "native_plan_sha256": plan.canonical_sha256,
        "reading_order_window_indices": proposals["vertical-lr"],
        "cut_adjudications": {
            str(index): {
                "decision": "manual_accepted",
                "reason": "",
            }
            for index in suspect_indices
        },
    }
    return item, template


def _verify_review_pack_files(staging: Path, items: Sequence[Mapping[str, Any]]) -> None:
    for item in items:
        raw = item["raw"]
        _raw_bytes, raw_sha = _read_stable_regular(
            staging / str(raw["relpath"]),
            where=f"review raw {item['sample_id']!r}",
        )
        if raw_sha != raw["sha256"]:
            raise RuntimeError("review raw file changed after materialization")
        canonical = item["canonical"]
        canonical_bytes, canonical_sha = _read_stable_regular(
            staging / str(canonical["relpath"]),
            where=f"review canonical {item['sample_id']!r}",
        )
        if canonical_sha != canonical["sha256"]:
            raise RuntimeError("review canonical file changed after materialization")
        canonical_pixel, canonical_size = rgba_pixel_sha256(canonical_bytes)
        if canonical_pixel != canonical["pixel_sha256"] or canonical_size != (
            canonical["width"],
            canonical["height"],
        ):
            raise RuntimeError("review canonical pixels changed after materialization")
        with Image.open(io.BytesIO(canonical_bytes)) as image:
            image.load()
            canonical_rgba = image.convert("RGBA")
            for window in item["windows"]:
                preview_bytes, preview_sha = _read_stable_regular(
                    staging / str(window["preview_relpath"]),
                    where=f"review window {window['view_id']!r}",
                )
                if preview_sha != window["preview_sha256"]:
                    raise RuntimeError("review window changed after materialization")
                preview_pixel, preview_size = rgba_pixel_sha256(preview_bytes)
                if preview_pixel != window["preview_pixel_sha256"] or preview_size != (
                    window["width"],
                    window["height"],
                ):
                    raise RuntimeError("review window pixels changed after materialization")
                y0, x0, y1, x1 = window["asset_bbox_yxxy"]
                if _rgba_image_sha256(
                    canonical_rgba.crop((x0, y0, x1, y1))
                ) != preview_pixel:
                    raise RuntimeError("review window is not the exact canonical crop")


def prepare_anyres_review_pack(
    *,
    sources_path: str | Path,
    preprocess_contract_path: str | Path,
    source_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Prepare immutable pixels and evidence for human review; never approve."""

    output = Path(output_dir).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable review pack: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(source_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("source_root must be a directory")
    source_rows, sources_sha256 = _load_strict_jsonl(sources_path, where="sources")
    preprocess_raw, preprocess_bytes, preprocess_file_sha256 = _load_strict_json(
        preprocess_contract_path,
        where="preprocess contract",
    )
    preprocess = validate_anyres_preprocess_contract(preprocess_raw)
    _builder_source, builder_source_sha256 = _read_stable_regular(
        Path(__file__), where="anyres builder source"
    )
    seen_ids: set[str] = set()
    for row in source_rows:
        sample_id = row.get("sample_id")
        if isinstance(sample_id, str) and sample_id.strip():
            if sample_id in seen_ids:
                raise ValueError(f"duplicate source sample_id {sample_id!r}")
            seen_ids.add(sample_id)
    lock_path = output.parent / f".{output.name}.build.lock"
    lock_fd = None
    staging: Path | None = None
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
        )
        items: list[dict[str, Any]] = []
        templates: list[dict[str, Any]] = []
        quarantine: list[dict[str, Any]] = []
        for row_index, raw_source in enumerate(source_rows, start=1):
            raw_sample_id = raw_source.get("sample_id")
            sample_id = (
                raw_sample_id
                if isinstance(raw_sample_id, str) and raw_sample_id.strip()
                else f"source-line-{row_index}"
            )
            source_line = raw_source.get("_input_line")
            try:
                source = _normalize_source(raw_source)
                item, template = _prepare_review_source(
                    source=source,
                    source_root=root,
                    staging=staging,
                    preprocess=preprocess,
                )
                items.append(item)
                templates.append(template)
            except _Quarantine as exc:
                quarantine.append(
                    _quarantine_entry(
                        str(sample_id),
                        source_line=(
                            int(source_line) if type(source_line) is int else None
                        ),
                        reason=exc.reason,
                    )
                )
        if not items:
            raise ValueError("no sources passed review-pack preparation")
        artifacts = {
            "review_items.jsonl": _write_jsonl(
                staging / "review_items.jsonl",
                sorted(items, key=lambda row: row["sample_id"]),
            ),
            "approved_reviews.template.jsonl": _write_jsonl(
                staging / "approved_reviews.template.jsonl",
                sorted(templates, key=lambda row: row["sample_id"]),
            ),
            "quarantine.jsonl": _write_jsonl(
                staging / "quarantine.jsonl",
                sorted(quarantine, key=lambda row: row["sample_id"]),
            ),
        }
        _verify_review_pack_files(staging, items)
        _write_bytes(staging / "preprocess_contract.json", preprocess_bytes)
        if (
            validate_anyres_preprocess_contract(preprocess_raw)[
                "contract_canonical_sha256"
            ]
            != preprocess["contract_canonical_sha256"]
        ):
            raise RuntimeError("anyres preprocess implementation drifted during prepare")
        _builder_source_end, builder_source_end = _read_stable_regular(
            Path(__file__), where="anyres builder source"
        )
        if builder_source_end != builder_source_sha256:
            raise RuntimeError("anyres builder source drifted during prepare")
        receipt = {
            "schema_version": 1,
            "kind": REVIEW_PACK_RECEIPT_KIND,
            "inputs": {
                "sources_sha256": sources_sha256,
                "preprocess_file_sha256": preprocess_file_sha256,
            },
            "contracts": {
                "preprocess_contract_canonical_sha256": preprocess[
                    "contract_canonical_sha256"
                ],
                "builder_source_sha256": builder_source_sha256,
                "human_approval_generated": False,
            },
            "counts": {
                "sources": len(source_rows),
                "review_items": len(items),
                "quarantined": len(quarantine),
            },
            "artifacts_sha256": artifacts,
        }
        receipt_bytes = _canonical_json_bytes(receipt, pretty=True)
        _write_bytes(staging / "receipts" / "build.json", receipt_bytes)
        ready = {
            "schema_version": 1,
            "kind": REVIEW_PACK_READY_KIND,
            "build_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "preprocess_contract_canonical_sha256": preprocess[
                "contract_canonical_sha256"
            ],
            "artifacts_sha256": artifacts,
            "review_items": len(items),
            "human_approval_generated": False,
        }
        _write_bytes(staging / "READY", _canonical_json_bytes(ready, pretty=True))
        _fsync_tree(staging)
        _fsync_directory(output.parent)
        _atomic_rename_noreplace(staging, output)
        _fsync_directory(output.parent)
        staging = None
        return {
            "output": str(output),
            "ready": str(output / "READY"),
            "review_items": str(output / "review_items.jsonl"),
            "review_template": str(output / "approved_reviews.template.jsonl"),
            "counts": receipt["counts"],
        }
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
        if lock_fd is not None:
            os.close(lock_fd)
        if lock_path.exists():
            lock_path.unlink()
            _fsync_directory(output.parent)


def finalize_anyres_dataset(
    *,
    sources_path: str | Path,
    approved_reviews_path: str | Path,
    preprocess_contract_path: str | Path,
    split_policy_path: str | Path,
    split_lock_path: str | Path,
    source_root: str | Path,
    output_dir: str | Path,
    native_encoder: Callable[[str], Sequence[int]],
    decode_reference: Callable[[Sequence[int]], str],
    tokenizer_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one immutable dataset version and atomically publish it."""

    output = Path(output_dir).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable dataset: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(source_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("source_root must be a directory")

    source_rows, sources_sha256 = _load_strict_jsonl(sources_path, where="sources")
    review_rows, reviews_sha256 = _load_strict_jsonl(
        approved_reviews_path, where="approved reviews"
    )
    preprocess_raw, preprocess_bytes, preprocess_file_sha256 = _load_strict_json(
        preprocess_contract_path, where="preprocess contract"
    )
    preprocess = validate_anyres_preprocess_contract(preprocess_raw)
    policy_raw, _policy_bytes, policy_file_sha256 = _load_strict_json(
        split_policy_path, where="split policy"
    )
    policy = validate_split_policy(policy_raw)
    policy_sha256 = canonical_json_sha256(policy)
    lock_raw, _lock_bytes, lock_file_sha256 = _load_strict_json(
        split_lock_path, where="split lock"
    )
    existing_lock = validate_split_lock(
        lock_raw,
        policy_canonical_sha256=policy_sha256,
    )
    try:
        canonical_json_sha256(tokenizer_contract)
    except (TypeError, ValueError) as exc:
        raise ValueError("tokenizer_contract must be finite JSON") from exc
    stats = getattr(native_encoder, "stats", None)
    if getattr(native_encoder, "mode", None) != "native" or not isinstance(stats, dict):
        raise ValueError("finalize requires a strict native OCR encoder")
    if int(stats.get("byte_fallback", 0)) != 0:
        raise ValueError("native encoder already reports byte fallback use")
    _builder_source_bytes, builder_source_sha256 = _read_stable_regular(
        Path(__file__), where="anyres builder source"
    )

    review_by_sample: dict[str, Mapping[str, Any]] = {}
    seen_review_ids: set[str] = set()
    for row in review_rows:
        raw_sample_id = row.get("sample_id")
        if not isinstance(raw_sample_id, str) or not raw_sample_id.strip():
            raise ValueError("approved review row has no usable sample_id")
        if raw_sample_id in review_by_sample:
            raise ValueError(f"duplicate approved review for {raw_sample_id!r}")
        review_id = row.get("review_id")
        if isinstance(review_id, str) and review_id.strip():
            if review_id in seen_review_ids:
                raise ValueError(f"duplicate approved review_id {review_id!r}")
            seen_review_ids.add(review_id)
        review_by_sample[raw_sample_id] = row

    seen_source_ids: set[str] = set()
    for row in source_rows:
        raw_sample_id = row.get("sample_id")
        if isinstance(raw_sample_id, str) and raw_sample_id.strip():
            if raw_sample_id in seen_source_ids:
                raise ValueError(f"duplicate source sample_id {raw_sample_id!r}")
            seen_source_ids.add(raw_sample_id)
    orphan_reviews = sorted(set(review_by_sample) - seen_source_ids)
    if orphan_reviews:
        raise ValueError(
            "approved reviews reference unknown sources: " + ", ".join(orphan_reviews[:8])
        )

    lock_path = output.parent / f".{output.name}.build.lock"
    lock_fd = None
    staging: Path | None = None
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
        )
        quarantine: list[dict[str, Any]] = []
        candidates: list[_Candidate] = []
        for row_index, raw_source in enumerate(source_rows, start=1):
            raw_sample_id = raw_source.get("sample_id")
            sample_id = (
                raw_sample_id
                if isinstance(raw_sample_id, str) and raw_sample_id.strip()
                else f"source-line-{row_index}"
            )
            source_line = raw_source.get("_input_line")
            try:
                source = _normalize_source(raw_source)
                raw_review = review_by_sample.get(source["sample_id"])
                if raw_review is None:
                    raise _Quarantine("human_approved_review_missing")
                review = _normalize_review(raw_review)
                if review["sample_id"] != source["sample_id"]:
                    raise _Quarantine("review_source_sample_id_mismatch")
                candidate = _process_candidate(
                    source=source,
                    review=review,
                    source_root=root,
                    staging=staging,
                    preprocess=preprocess,
                    native_encoder=native_encoder,
                    decode_reference=decode_reference,
                )
                candidates.append(candidate)
            except _Quarantine as exc:
                quarantine.append(
                    _quarantine_entry(
                        str(sample_id),
                        source_line=(int(source_line) if type(source_line) is int else None),
                        reason=exc.reason,
                    )
                )
        if not candidates:
            raise ValueError("no samples passed finalize-only admission")
        if len(candidates) > MAX_PHASH_RECORDS:
            raise ValueError(
                f"pHash review set exceeds the exact global cap {MAX_PHASH_RECORDS}; "
                "a safe build requires a chunked all-pairs global index"
            )

        threshold = int(policy["phash_max_hamming_distance"])
        phash_clusters = cluster_perceptual_hashes(
            {str(candidate.asset["asset_id"]): candidate.phash for candidate in candidates},
            max_hamming_distance=threshold,
            max_records=MAX_PHASH_RECORDS,
        )
        cumulative_hashes = {
            asset_id: str(record["value"])
            for asset_id, record in existing_lock["phash_index"].items()
        }
        for candidate in candidates:
            asset_id = str(candidate.asset["asset_id"])
            prior = cumulative_hashes.get(asset_id)
            if prior is not None and prior != candidate.phash:
                raise ValueError(
                    f"immutable pHash index changed for asset {asset_id!r}"
                )
            cumulative_hashes[asset_id] = candidate.phash
        if len(cumulative_hashes) > MAX_PHASH_RECORDS:
            raise ValueError(
                f"cumulative pHash index exceeds the exact global cap "
                f"{MAX_PHASH_RECORDS}; a safe build requires a chunked "
                "all-pairs global index"
            )
        cumulative_phash_clusters = cluster_perceptual_hashes(
            cumulative_hashes,
            max_hamming_distance=threshold,
            max_records=MAX_PHASH_RECORDS,
        )
        for cluster in cumulative_phash_clusters:
            historical_splits = {
                str(existing_lock["phash_index"][member]["split"])
                for member in cluster.member_ids
                if member in existing_lock["phash_index"]
            }
            if len(historical_splits) > 1:
                raise ValueError(
                    "split lock pHash index already cross-connects locked splits: "
                    f"cluster={cluster.cluster_id}, splits={sorted(historical_splits)}"
                )
        cluster_by_asset = {
            asset_id: cluster.cluster_id
            for cluster in phash_clusters
            for asset_id in cluster.member_ids
        }
        for candidate in candidates:
            asset_id = str(candidate.asset["asset_id"])
            candidate.asset["near_duplicate"] = {
                "contract": PHASH_CONTRACT,
                "value": candidate.phash,
                "max_hamming_distance": threshold,
                "cluster_id": cluster_by_asset[asset_id],
            }

        component_assignments, updated_lock, component_receipts = _assign_components(
            candidates,
            combined_phash_clusters=cumulative_phash_clusters,
            policy=policy,
            existing_lock=existing_lock,
        )
        assets = [candidate.asset for candidate in candidates]
        views = [view for candidate in candidates for view in candidate.views]
        samples: list[dict[str, Any]] = []
        for candidate in candidates:
            asset_id = str(candidate.asset["asset_id"])
            split, leakage_cluster = component_assignments[asset_id]
            source = candidate.source
            near_cluster = candidate.asset["near_duplicate"]["cluster_id"]
            samples.append(
                {
                    "schema_version": ANYRES_SCHEMA_VERSION,
                    "sample_id": candidate.sample_id,
                    "split": split,
                    "visual_contract": ANYRES_VISUAL_CONTRACT,
                    "asset_id": asset_id,
                    "view_ids": [str(view["view_id"]) for view in candidate.views],
                    "preprocess_contract_sha256": preprocess[
                        "contract_canonical_sha256"
                    ],
                    "reference_token_count": candidate.reference_token_count,
                    "writing_mode": candidate.writing_mode,
                    "reading_order": candidate.reading_order,
                    "leakage_cluster": leakage_cluster,
                    "group_ids": {
                        "document_id": source["source_document_id"],
                        "capture_id": source["source_capture_id"],
                        "writer_id": source["writer_id"],
                        "near_duplicate_cluster": near_cluster,
                    },
                    "style": source["style"],
                    "font_id": source["font_id"],
                    "writer_id": source["writer_id"],
                    "difficulty": source["difficulty"],
                    "reference_raw": candidate.reference_raw,
                    "reference_model": candidate.reference_model,
                    "qa_state": "accepted",
                }
            )
        normalized, manifest_quarantine = validate_anyres_assets_views_samples(
            assets, views, samples
        )
        if manifest_quarantine:
            raise RuntimeError(
                "builder admitted rows that authoritative manifest validation quarantined"
            )
        if len(normalized["samples"]) != len(candidates):
            raise RuntimeError("authoritative manifest validation changed sample count")
        split_samples = {
            split: [
                row for row in normalized["samples"] if row["split"] == split
            ]
            for split in ALLOWED_SPLITS
        }
        empty_splits = [
            split for split, rows in split_samples.items() if not rows
        ]
        if empty_splits:
            raise ValueError(
                "split policy/lock produced empty public splits: "
                f"{empty_splits}"
            )
        split_quota = {
            split: quota_counts(rows)
            for split, rows in split_samples.items()
        }
        train_quota = split_quota["train"]
        if any(count <= 0 for count in train_quota.values()):
            raise ValueError(
                "READY gate requires every train quota bucket to be non-empty: "
                f"{train_quota}"
            )
        for split in ANYRES_VALIDATION_SPLITS:
            counts = split_quota[split]
            if any(count < 2 for count in counts.values()):
                raise ValueError(
                    f"READY gate requires at least two {split} samples in "
                    f"every quota bucket: {counts}"
                )
        recommended = int(
            preprocess["budgets"]["output"]["recommended_max_new_tokens"]
        )
        observed_recommended = max(
            int(row["reference_token_count"]) + 1 for row in normalized["samples"]
        )
        if observed_recommended != recommended:
            raise ValueError(
                "preprocess recommended_max_new_tokens must equal the accepted "
                f"maximum native target plus EOS: {recommended} != {observed_recommended}"
            )

        manifests = {
            "assets.jsonl": _write_jsonl(staging / "assets.jsonl", normalized["assets"]),
            "views.jsonl": _write_jsonl(staging / "views.jsonl", normalized["views"]),
            **{
                f"{split}.jsonl": _write_jsonl(
                    staging / f"{split}.jsonl",
                    split_samples[split],
                )
                for split in ALLOWED_SPLITS
            },
            "quarantine.jsonl": _write_jsonl(
                staging / "quarantine.jsonl", sorted(quarantine, key=lambda row: row["sample_id"])
            ),
        }
        adjudication_rows = sorted(
            (
                record
                for candidate in candidates
                for record in candidate.adjudication_records
            ),
            key=lambda row: (row["sample_id"], row["window_index"]),
        )
        adjudications_sha256 = _write_jsonl(
            staging / "receipts" / "cut_adjudications.jsonl",
            adjudication_rows,
        )
        split_policy_bytes = _canonical_json_bytes(policy, pretty=True)
        split_lock_bytes = _canonical_json_bytes(updated_lock, pretty=True)
        _write_bytes(staging / "preprocess_contract.json", preprocess_bytes)
        _write_bytes(staging / "split_policy.json", split_policy_bytes)
        _write_bytes(staging / "split_lock.json", split_lock_bytes)

        for split in ALLOWED_SPLITS:
            AnyresOCRDataset(
                root=staging,
                assets_manifest=staging / "assets.jsonl",
                views_manifest=staging / "views.jsonl",
                samples_manifest=staging / f"{split}.jsonl",
                split=split,
                expected_preprocess_contract_sha256=preprocess[
                    "contract_canonical_sha256"
                ],
                max_decode_pixels=int(
                    preprocess["budgets"]["decode"]["max_pixels_per_asset"]
                ),
                max_views_per_sample=int(
                    preprocess["budgets"]["window"]["max_windows_per_asset"]
                ),
                image_delivery="bytes",
            )
        if (
            validate_anyres_preprocess_contract(preprocess_raw)[
                "contract_canonical_sha256"
            ]
            != preprocess["contract_canonical_sha256"]
        ):
            raise RuntimeError("anyres preprocess implementation drifted during build")
        _builder_source_end, builder_source_sha256_end = _read_stable_regular(
            Path(__file__), where="anyres builder source"
        )
        if builder_source_sha256_end != builder_source_sha256:
            raise RuntimeError("anyres builder source drifted during build")

        receipt = {
            "schema_version": BUILDER_RECEIPT_SCHEMA_VERSION,
            "kind": BUILD_RECEIPT_KIND,
            "inputs": {
                "sources_sha256": sources_sha256,
                "approved_reviews_sha256": reviews_sha256,
                "preprocess_file_sha256": preprocess_file_sha256,
                "split_policy_file_sha256": policy_file_sha256,
                "split_lock_file_sha256": lock_file_sha256,
            },
            "contracts": {
                "preprocess_contract_canonical_sha256": preprocess[
                    "contract_canonical_sha256"
                ],
                "split_policy_canonical_sha256": policy_sha256,
                "tokenizer_contract": json.loads(
                    json.dumps(tokenizer_contract, ensure_ascii=False, allow_nan=False)
                ),
                "builder_source": {
                    "path": "Model/posttrain/ocr_anyres_builder.py",
                    "sha256": builder_source_sha256,
                },
                "pHash": {
                    "contract": PHASH_CONTRACT,
                    "max_records": MAX_PHASH_RECORDS,
                    "max_hamming_distance": threshold,
                },
            },
            "counts": {
                "source_rows": len(source_rows),
                "accepted": len(normalized["samples"]),
                **{
                    split: len(split_samples[split])
                    for split in ALLOWED_SPLITS
                },
                "quarantined": len(quarantine),
                "assets": len(normalized["assets"]),
                "views": len(normalized["views"]),
                "leakage_components": len(component_receipts),
                "cumulative_phash_records": len(updated_lock["phash_index"]),
            },
            "quota_counts": split_quota,
            "manifests_sha256": manifests,
            "cut_adjudications_sha256": adjudications_sha256,
            "split_lock_sha256": hashlib.sha256(split_lock_bytes).hexdigest(),
            "components": component_receipts,
        }
        receipt_bytes = _canonical_json_bytes(receipt, pretty=True)
        _write_bytes(staging / "receipts" / "build.json", receipt_bytes)
        ready = {
            "schema_version": 1,
            "kind": READY_KIND,
            "build_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "preprocess_contract_canonical_sha256": preprocess[
                "contract_canonical_sha256"
            ],
            "manifests_sha256": manifests,
            "accepted_samples": len(normalized["samples"]),
        }
        ready_bytes = _canonical_json_bytes(ready, pretty=True)
        _write_bytes(staging / "READY", ready_bytes)
        _fsync_tree(staging)
        _fsync_directory(output.parent)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"refusing to overwrite immutable dataset: {output}")
        _atomic_rename_noreplace(staging, output)
        _fsync_directory(output.parent)
        staging = None
        return {
            "output": str(output),
            "ready": str(output / "READY"),
            "counts": receipt["counts"],
            "build_receipt_sha256": ready["build_receipt_sha256"],
            "split_lock": str(output / "split_lock.json"),
        }
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
        if lock_fd is not None:
            os.close(lock_fd)
        if lock_path.exists():
            lock_path.unlink()
            _fsync_directory(output.parent)


__all__ = [
    "BUILD_RECEIPT_KIND",
    "MAX_PHASH_RECORDS",
    "READY_KIND",
    "REVIEW_ITEM_KIND",
    "REVIEW_PACK_READY_KIND",
    "SPLIT_LOCK_KIND",
    "SPLIT_POLICY_KIND",
    "finalize_anyres_dataset",
    "prepare_anyres_review_pack",
    "split_identity_key",
    "validate_split_lock",
    "validate_split_policy",
]
