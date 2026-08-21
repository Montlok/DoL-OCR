# -*- coding: utf-8 -*-

"""Fail-closed manifest contract for DoL OCR arbitrary-resolution data."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import PurePosixPath
from typing import Any

from Model.ocr.cut_qa import CUT_QA_CONTRACT
from Model.ocr.tokenization import canonicalize_native_ocr_text
from Model.ocr.near_duplicate import (
    PHASH_CONTRACT,
    cluster_perceptual_hashes,
    phash_hamming_distance,
)
from Model.omvt.native_planner import plan_native_macro_windows
from Model.omvt.patcher import PATCH_KINDS


ANYRES_SCHEMA_VERSION = 2
ANYRES_VISUAL_CONTRACT = "dol_ocr_anyres_v2"
ANYRES_QUOTA_60_10_20_10 = {
    "print": 60,
    "handwritten_good": 10,
    "handwritten_medium": 20,
    "handwritten_poor": 10,
}

_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")
_ASSET_KEYS = frozenset(
    {
        "schema_version",
        "asset_id",
        "source_document_id",
        "source_capture_id",
        "raw_relpath",
        "canonical_relpath",
        "raw_sha256",
        "canonical_sha256",
        "pixel_sha256",
        "raw_width",
        "raw_height",
        "canonical_width",
        "canonical_height",
        "native_plan",
        "cut_qa",
        "near_duplicate",
    }
)
_VIEW_KEYS = frozenset(
    {
        "schema_version",
        "view_id",
        "asset_id",
        "kind",
        "derived_relpath",
        "derived_width",
        "derived_height",
        "derived_sha256",
        "box_xyxy",
        "transform",
        "pixel_sha256",
        "qa",
    }
)
_VIEW_QA_KEYS = frozenset(
    {
        "cut_suspect",
        "edge_ink_fraction",
        "cc_crossing_count",
        "cc_uncovered_count",
        "coverage",
        "coverage_proof_sha256",
        "review_status",
        "adjudication_sha256",
    }
)
_EDGE_KEYS = frozenset({"top", "bottom", "left", "right"})
_NATIVE_PLAN_KEYS = frozenset(
    {"preprocess_contract_sha256", "plan_sha256", "payload"}
)
_PLAN_PAYLOAD_KEYS = frozenset(
    {
        "contract",
        "asset_hw",
        "patch_shapes",
        "max_raw_patch_tokens",
        "max_windows",
        "halo_px",
        "identity",
        "windows",
        "coverage_proof",
    }
)
_VIEW_TRANSFORM_KEYS = frozenset(
    {
        "contract",
        "plan_sha256",
        "window_index",
        "asset_bbox_yxxy",
        "ownership_bbox_yxxy",
        "halo_tlbr",
        "raw_patch_tokens",
    }
)
_VIEW_TRANSFORM_CONTRACT = "native_macro_window_crop_v1"
_CUT_QA_KEYS = frozenset({"coverage_proof_sha256", "payload"})
_CUT_QA_PAYLOAD_KEYS = frozenset(
    {
        "contract",
        "canonical_pixel_sha256",
        "plan_sha256",
        "parameters",
        "image_hw",
        "components",
        "view_boxes",
        "complete_owners",
        "uncovered",
    }
)
_CUT_QA_PARAMETER_KEYS = frozenset(
    {
        "ink_threshold",
        "min_component_pixels",
        "edge_band_px",
        "max_edge_ink_fraction",
    }
)
_NEAR_DUPLICATE_KEYS = frozenset(
    {"contract", "value", "max_hamming_distance", "cluster_id"}
)
_SAMPLE_KEYS = frozenset(
    {
        "schema_version",
        "sample_id",
        "split",
        "visual_contract",
        "asset_id",
        "view_ids",
        "preprocess_contract_sha256",
        "reference_token_count",
        "writing_mode",
        "reading_order",
        "leakage_cluster",
        "group_ids",
        "style",
        "font_id",
        "writer_id",
        "difficulty",
        "reference_raw",
        "reference_model",
        "qa_state",
    }
)
_GROUP_KEYS = frozenset(
    {"document_id", "capture_id", "writer_id", "near_duplicate_cluster"}
)
_REFERENCE_KEYS = frozenset({"text", "utf8_sha256", "codepoints"})
ANYRES_PUBLIC_SPLITS = (
    "train",
    "sft_validation",
    "kl_selection",
    "formal_monitor",
)
ANYRES_VALIDATION_SPLITS = ANYRES_PUBLIC_SPLITS[1:]
_SPLIT_ALIASES: dict[str, str] = {}
_ALLOWED_SPLITS = frozenset(ANYRES_PUBLIC_SPLITS)
_ALLOWED_DIFFICULTIES = frozenset({"good", "medium", "poor"})
_ALLOWED_VIEW_KINDS = frozenset({"page", "column", "region", "tile", "line"})
_ALLOWED_WRITING_MODES = frozenset(
    {"vertical-lr", "vertical-rl", "horizontal-tb"}
)


def canonical_json_sha256(value: object) -> str:
    """Hash the UTF-8 bytes of deterministic, non-NaN canonical JSON."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _object(value: object, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], where: str) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ValueError(f"{where} has invalid fields: " + ", ".join(details))


def _schema(value: Mapping[str, Any], where: str) -> None:
    if value.get("schema_version") != ANYRES_SCHEMA_VERSION:
        raise ValueError(f"{where}.schema_version must be 2")


def _identifier(value: object, where: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{where} must be a non-empty, unpadded string")
    return value


def _relative_path(value: object, where: str) -> str:
    value = _identifier(value, where)
    assert value is not None
    if "\\" in value or "\x00" in value:
        raise ValueError(f"{where} must be a normalized POSIX relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{where} must be a normalized POSIX relative path without '..'")
    return value


def _sha256(value: object, where: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{where} must be a 64-character SHA-256 hex digest")
    return value.lower()


def _positive_int(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{where} must be a positive integer")
    return value


def _nonnegative_int(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{where} must be a non-negative integer")
    return value


def _bool(value: object, where: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{where} must be boolean")
    return value


def _json_copy(value: object, where: str) -> Any:
    try:
        canonical_json_sha256(value)
        return json.loads(
            json.dumps(value, ensure_ascii=False, allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where} must be finite JSON data") from exc


def _reference(value: object, where: str) -> dict[str, Any]:
    obj = _object(value, where)
    _exact_keys(obj, _REFERENCE_KEYS, where)
    text = obj["text"]
    if not isinstance(text, str) or not text:
        raise ValueError(f"{where}.text must be a non-empty Unicode string")
    try:
        utf8 = text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{where}.text contains invalid Unicode surrogates") from exc
    expected_hash = _sha256(obj["utf8_sha256"], f"{where}.utf8_sha256")
    actual_hash = hashlib.sha256(utf8).hexdigest()
    if expected_hash != actual_hash:
        raise ValueError(f"{where}.utf8_sha256 does not match text bytes")
    codepoints = obj["codepoints"]
    if not isinstance(codepoints, Sequence) or isinstance(
        codepoints, (str, bytes, bytearray)
    ):
        raise ValueError(f"{where}.codepoints must be an array")
    normalized_codepoints: list[int] = []
    for index, codepoint in enumerate(codepoints):
        if (
            isinstance(codepoint, bool)
            or not isinstance(codepoint, int)
            or not 0 <= codepoint <= 0x10FFFF
            or 0xD800 <= codepoint <= 0xDFFF
        ):
            raise ValueError(f"{where}.codepoints[{index}] is not a Unicode scalar")
        normalized_codepoints.append(codepoint)
    if normalized_codepoints != [ord(char) for char in text]:
        raise ValueError(f"{where}.codepoints do not match text")
    return {
        "text": text,
        "utf8_sha256": actual_hash,
        "codepoints": normalized_codepoints,
    }


def _native_plan(
    value: object,
    *,
    where: str,
    canonical_width: int,
    canonical_height: int,
) -> dict[str, Any]:
    obj = _object(value, where)
    _exact_keys(obj, _NATIVE_PLAN_KEYS, where)
    preprocess_sha = _sha256(
        obj["preprocess_contract_sha256"],
        f"{where}.preprocess_contract_sha256",
    )
    payload = _object(obj["payload"], f"{where}.payload")
    _exact_keys(payload, _PLAN_PAYLOAD_KEYS, f"{where}.payload")
    asset_hw = payload["asset_hw"]
    if (
        not isinstance(asset_hw, list)
        or len(asset_hw) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in asset_hw
        )
    ):
        raise ValueError(f"{where}.payload.asset_hw must contain positive H/W")
    if asset_hw != [canonical_height, canonical_width]:
        raise ValueError(f"{where}.payload.asset_hw differs from canonical asset")
    patch_shapes = _object(
        payload["patch_shapes"], f"{where}.payload.patch_shapes"
    )
    _exact_keys(
        patch_shapes,
        frozenset(PATCH_KINDS),
        f"{where}.payload.patch_shapes",
    )
    normalized_shapes: dict[str, list[int]] = {}
    for kind in PATCH_KINDS:
        shape = patch_shapes[kind]
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0
                for item in shape
            )
        ):
            raise ValueError(
                f"{where}.payload.patch_shapes.{kind} must contain positive H/W"
            )
        normalized_shapes[kind] = [int(shape[0]), int(shape[1])]
    max_raw_patch_tokens = _positive_int(
        payload["max_raw_patch_tokens"],
        f"{where}.payload.max_raw_patch_tokens",
    )
    max_windows = _positive_int(
        payload["max_windows"], f"{where}.payload.max_windows"
    )
    halo_px = _nonnegative_int(payload["halo_px"], f"{where}.payload.halo_px")
    expected = plan_native_macro_windows(
        height=canonical_height,
        width=canonical_width,
        patch_shapes=normalized_shapes,
        max_raw_patch_tokens=max_raw_patch_tokens,
        max_windows=max_windows,
        halo_px=halo_px,
    )
    expected_payload = expected.canonical_payload()
    if _json_copy(payload, f"{where}.payload") != expected_payload:
        raise ValueError(f"{where}.payload is not the deterministic native plan")
    plan_sha = _sha256(obj["plan_sha256"], f"{where}.plan_sha256")
    if plan_sha != expected.canonical_sha256:
        raise ValueError(f"{where}.plan_sha256 differs from deterministic plan")
    return {
        "preprocess_contract_sha256": preprocess_sha,
        "plan_sha256": plan_sha,
        "payload": expected_payload,
    }


def _cut_qa(
    value: object,
    *,
    where: str,
    canonical_pixel_sha256: str,
    plan_sha256: str,
    canonical_width: int,
    canonical_height: int,
) -> dict[str, Any]:
    obj = _object(value, where)
    _exact_keys(obj, _CUT_QA_KEYS, where)
    payload = _object(obj["payload"], f"{where}.payload")
    _exact_keys(payload, _CUT_QA_PAYLOAD_KEYS, f"{where}.payload")
    if payload.get("contract") != CUT_QA_CONTRACT:
        raise ValueError(f"{where}.payload.contract must be {CUT_QA_CONTRACT}")
    if _sha256(
        payload["canonical_pixel_sha256"],
        f"{where}.payload.canonical_pixel_sha256",
    ) != canonical_pixel_sha256:
        raise ValueError(f"{where}.payload canonical pixels differ from asset")
    if _sha256(
        payload["plan_sha256"], f"{where}.payload.plan_sha256"
    ) != plan_sha256:
        raise ValueError(f"{where}.payload plan differs from asset")
    if payload.get("image_hw") != [canonical_height, canonical_width]:
        raise ValueError(f"{where}.payload.image_hw differs from canonical asset")
    parameters = _object(payload["parameters"], f"{where}.payload.parameters")
    _exact_keys(parameters, _CUT_QA_PARAMETER_KEYS, f"{where}.payload.parameters")
    ink_threshold = _nonnegative_int(
        parameters["ink_threshold"], f"{where}.payload.parameters.ink_threshold"
    )
    if ink_threshold > 255:
        raise ValueError(f"{where}.payload.parameters.ink_threshold exceeds 255")
    _positive_int(
        parameters["min_component_pixels"],
        f"{where}.payload.parameters.min_component_pixels",
    )
    _positive_int(
        parameters["edge_band_px"],
        f"{where}.payload.parameters.edge_band_px",
    )
    edge_limit = parameters["max_edge_ink_fraction"]
    if (
        isinstance(edge_limit, bool)
        or not isinstance(edge_limit, (int, float))
        or not math.isfinite(float(edge_limit))
        or not 0.0 <= float(edge_limit) <= 1.0
    ):
        raise ValueError(
            f"{where}.payload.parameters.max_edge_ink_fraction must be in [0,1]"
        )
    for field in ("components", "view_boxes", "complete_owners", "uncovered"):
        _json_copy(payload[field], f"{where}.payload.{field}")
    normalized_payload = _json_copy(payload, f"{where}.payload")
    proof_sha = _sha256(
        obj["coverage_proof_sha256"], f"{where}.coverage_proof_sha256"
    )
    if proof_sha != canonical_json_sha256(normalized_payload):
        raise ValueError(f"{where}.coverage_proof_sha256 differs from payload")
    return {"coverage_proof_sha256": proof_sha, "payload": normalized_payload}


def _near_duplicate(value: object, *, where: str) -> dict[str, Any]:
    obj = _object(value, where)
    _exact_keys(obj, _NEAR_DUPLICATE_KEYS, where)
    if obj.get("contract") != PHASH_CONTRACT:
        raise ValueError(f"{where}.contract must be {PHASH_CONTRACT}")
    phash = obj.get("value")
    phash_hamming_distance(phash, phash)
    max_distance = _nonnegative_int(
        obj["max_hamming_distance"], f"{where}.max_hamming_distance"
    )
    if max_distance > 64:
        raise ValueError(f"{where}.max_hamming_distance exceeds 64")
    cluster_id = _identifier(obj["cluster_id"], f"{where}.cluster_id")
    return {
        "contract": PHASH_CONTRACT,
        "value": phash,
        "max_hamming_distance": max_distance,
        "cluster_id": cluster_id,
    }


def _normalize_asset(value: object, index: int) -> dict[str, Any]:
    where = f"assets[{index}]"
    obj = _object(value, where)
    _exact_keys(obj, _ASSET_KEYS, where)
    _schema(obj, where)
    canonical_width = _positive_int(
        obj["canonical_width"], f"{where}.canonical_width"
    )
    canonical_height = _positive_int(
        obj["canonical_height"], f"{where}.canonical_height"
    )
    pixel_sha256 = _sha256(obj["pixel_sha256"], f"{where}.pixel_sha256")
    native_plan = _native_plan(
        obj["native_plan"],
        where=f"{where}.native_plan",
        canonical_width=canonical_width,
        canonical_height=canonical_height,
    )
    canonical_relpath = _relative_path(
        obj["canonical_relpath"], f"{where}.canonical_relpath"
    )
    if not canonical_relpath.lower().endswith(".png"):
        raise ValueError(f"{where}.canonical_relpath must be lossless PNG")
    return {
        "schema_version": ANYRES_SCHEMA_VERSION,
        "asset_id": _identifier(obj["asset_id"], f"{where}.asset_id"),
        "source_document_id": _identifier(
            obj["source_document_id"], f"{where}.source_document_id"
        ),
        "source_capture_id": _identifier(
            obj["source_capture_id"], f"{where}.source_capture_id"
        ),
        "raw_relpath": _relative_path(obj["raw_relpath"], f"{where}.raw_relpath"),
        "canonical_relpath": canonical_relpath,
        "raw_sha256": _sha256(obj["raw_sha256"], f"{where}.raw_sha256"),
        "canonical_sha256": _sha256(
            obj["canonical_sha256"], f"{where}.canonical_sha256"
        ),
        "pixel_sha256": pixel_sha256,
        "raw_width": _positive_int(obj["raw_width"], f"{where}.raw_width"),
        "raw_height": _positive_int(obj["raw_height"], f"{where}.raw_height"),
        "canonical_width": canonical_width,
        "canonical_height": canonical_height,
        "native_plan": native_plan,
        "cut_qa": _cut_qa(
            obj["cut_qa"],
            where=f"{where}.cut_qa",
            canonical_pixel_sha256=pixel_sha256,
            plan_sha256=native_plan["plan_sha256"],
            canonical_width=canonical_width,
            canonical_height=canonical_height,
        ),
        "near_duplicate": _near_duplicate(
            obj["near_duplicate"],
            where=f"{where}.near_duplicate",
        ),
    }


def _normalize_view(
    value: object,
    index: int,
    assets_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    where = f"views[{index}]"
    obj = _object(value, where)
    _exact_keys(obj, _VIEW_KEYS, where)
    _schema(obj, where)
    view_id = _identifier(obj["view_id"], f"{where}.view_id")
    asset_id = _identifier(obj["asset_id"], f"{where}.asset_id")
    kind = _identifier(obj["kind"], f"{where}.kind")
    if kind not in _ALLOWED_VIEW_KINDS:
        raise ValueError(
            f"{where}.kind must be page, column, region, tile, or line"
        )
    asset = assets_by_id.get(asset_id)
    if asset is None:
        raise ValueError(f"{where}.asset_id references an unknown asset")
    box = obj["box_xyxy"]
    if not isinstance(box, Sequence) or isinstance(box, (str, bytes)) or len(box) != 4:
        raise ValueError(f"{where}.box_xyxy must contain four integers")
    normalized_box: list[int] = []
    for coordinate in box:
        if isinstance(coordinate, bool) or not isinstance(coordinate, int):
            raise ValueError(f"{where}.box_xyxy must contain four integers")
        normalized_box.append(coordinate)
    x0, y0, x1, y1 = normalized_box
    if not (
        0 <= x0 < x1 <= int(asset["canonical_width"])
        and 0 <= y0 < y1 <= int(asset["canonical_height"])
    ):
        raise ValueError(f"{where}.box_xyxy is outside the canonical asset")
    transform = _object(obj["transform"], f"{where}.transform")
    _exact_keys(transform, _VIEW_TRANSFORM_KEYS, f"{where}.transform")
    if transform.get("contract") != _VIEW_TRANSFORM_CONTRACT:
        raise ValueError(
            f"{where}.transform.contract must be {_VIEW_TRANSFORM_CONTRACT}"
        )
    plan = asset["native_plan"]
    if _sha256(
        transform["plan_sha256"], f"{where}.transform.plan_sha256"
    ) != plan["plan_sha256"]:
        raise ValueError(f"{where}.transform.plan_sha256 differs from asset plan")
    window_index = _nonnegative_int(
        transform["window_index"], f"{where}.transform.window_index"
    )
    windows = plan["payload"]["windows"]
    if window_index >= len(windows):
        raise ValueError(f"{where}.transform.window_index is outside asset plan")
    expected_window = windows[window_index]
    normalized_transform = {
        "contract": _VIEW_TRANSFORM_CONTRACT,
        "plan_sha256": plan["plan_sha256"],
        "window_index": window_index,
        "asset_bbox_yxxy": list(expected_window["asset_bbox_yxxy"]),
        "ownership_bbox_yxxy": list(expected_window["ownership_bbox_yxxy"]),
        "halo_tlbr": list(expected_window["halo_tlbr"]),
        "raw_patch_tokens": int(expected_window["raw_patch_tokens"]),
    }
    if _json_copy(transform, f"{where}.transform") != normalized_transform:
        raise ValueError(f"{where}.transform differs from deterministic plan window")
    ay0, ax0, ay1, ax1 = normalized_transform["asset_bbox_yxxy"]
    if normalized_box != [ax0, ay0, ax1, ay1]:
        raise ValueError(f"{where}.box_xyxy differs from planned asset crop")
    derived_width = _positive_int(
        obj["derived_width"], f"{where}.derived_width"
    )
    derived_height = _positive_int(
        obj["derived_height"], f"{where}.derived_height"
    )
    if derived_width != x1 - x0 or derived_height != y1 - y0:
        raise ValueError(
            f"{where} derived dimensions must equal its no-resize canonical crop"
        )
    qa = _object(obj["qa"], f"{where}.qa")
    _exact_keys(qa, _VIEW_QA_KEYS, f"{where}.qa")
    cut_suspect = _bool(qa["cut_suspect"], f"{where}.qa.cut_suspect")
    edge_ink = _object(
        qa["edge_ink_fraction"], f"{where}.qa.edge_ink_fraction"
    )
    _exact_keys(edge_ink, _EDGE_KEYS, f"{where}.qa.edge_ink_fraction")
    normalized_edges: dict[str, float] = {}
    for edge in sorted(_EDGE_KEYS):
        value = edge_ink[edge]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"{where}.qa.edge_ink_fraction.{edge} must be in [0, 1]"
            )
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{where}.qa.edge_ink_fraction.{edge} must be in [0, 1]"
            )
        normalized_edges[edge] = value
    cc_crossing_count = _nonnegative_int(
        qa["cc_crossing_count"], f"{where}.qa.cc_crossing_count"
    )
    cc_uncovered_count = _nonnegative_int(
        qa["cc_uncovered_count"], f"{where}.qa.cc_uncovered_count"
    )
    if cc_uncovered_count > cc_crossing_count:
        raise ValueError(
            f"{where}.qa.cc_uncovered_count exceeds cc_crossing_count"
        )
    coverage = qa["coverage"]
    if isinstance(coverage, bool) or not isinstance(coverage, (int, float)):
        raise ValueError(f"{where}.qa.coverage must be a finite number in [0, 1]")
    coverage = float(coverage)
    if not math.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
        raise ValueError(f"{where}.qa.coverage must be a finite number in [0, 1]")
    coverage_proof_sha256 = _sha256(
        qa["coverage_proof_sha256"], f"{where}.qa.coverage_proof_sha256"
    )
    if coverage_proof_sha256 != asset["cut_qa"]["coverage_proof_sha256"]:
        raise ValueError(f"{where}.qa coverage proof differs from parent asset")
    review_status = _identifier(
        qa["review_status"], f"{where}.qa.review_status"
    )
    if review_status not in {"accepted", "manual_accepted", "quarantine"}:
        raise ValueError(
            f"{where}.qa.review_status must be accepted, manual_accepted, or quarantine"
        )
    adjudication = qa["adjudication_sha256"]
    if review_status == "manual_accepted":
        adjudication_sha256 = _sha256(
            adjudication, f"{where}.qa.adjudication_sha256"
        )
    else:
        if adjudication is not None:
            raise ValueError(
                f"{where}.qa.adjudication_sha256 is only valid for manual_accepted"
            )
        adjudication_sha256 = None
    reasons: list[str] = []
    if cut_suspect and review_status != "manual_accepted":
        reasons.append("cut_suspect")
    if cc_uncovered_count:
        reasons.append("cc_uncovered")
    if coverage < 1.0:
        reasons.append("incomplete_coverage")
    if review_status == "quarantine":
        reasons.append("review_quarantine")
    derived_relpath = _relative_path(
        obj["derived_relpath"], f"{where}.derived_relpath"
    )
    if not derived_relpath.lower().endswith(".png"):
        raise ValueError(f"{where}.derived_relpath must be lossless PNG")
    return (
        {
            "schema_version": ANYRES_SCHEMA_VERSION,
            "view_id": view_id,
            "asset_id": asset_id,
            "kind": kind,
            "derived_relpath": derived_relpath,
            "derived_width": derived_width,
            "derived_height": derived_height,
            "derived_sha256": _sha256(
                obj["derived_sha256"], f"{where}.derived_sha256"
            ),
            "box_xyxy": normalized_box,
            "transform": normalized_transform,
            "pixel_sha256": _sha256(
                obj["pixel_sha256"], f"{where}.pixel_sha256"
            ),
            "qa": {
                "cut_suspect": cut_suspect,
                "edge_ink_fraction": normalized_edges,
                "cc_crossing_count": cc_crossing_count,
                "cc_uncovered_count": cc_uncovered_count,
                "coverage": coverage,
                "coverage_proof_sha256": coverage_proof_sha256,
                "review_status": review_status,
                "adjudication_sha256": adjudication_sha256,
            },
        },
        reasons,
    )


def _normalize_sample(
    value: object,
    index: int,
    assets_by_id: Mapping[str, Mapping[str, Any]],
    views_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    where = f"samples[{index}]"
    obj = _object(value, where)
    _exact_keys(obj, _SAMPLE_KEYS, where)
    _schema(obj, where)
    sample_id = _identifier(obj["sample_id"], f"{where}.sample_id")
    split = _identifier(obj["split"], f"{where}.split")
    split = _SPLIT_ALIASES.get(split, split)
    if split not in _ALLOWED_SPLITS:
        raise ValueError(f"{where}.split is not supported")
    if obj["visual_contract"] != ANYRES_VISUAL_CONTRACT:
        raise ValueError(
            f"{where}.visual_contract must be {ANYRES_VISUAL_CONTRACT!r}"
        )
    asset_id = _identifier(obj["asset_id"], f"{where}.asset_id")
    asset = assets_by_id.get(asset_id)
    if asset is None:
        raise ValueError(f"{where}.asset_id references an unknown asset")
    if (
        obj["preprocess_contract_sha256"]
        != asset["native_plan"]["preprocess_contract_sha256"]
    ):
        raise ValueError(
            f"{where}.preprocess_contract_sha256 differs from asset native plan"
        )
    raw_view_ids = obj["view_ids"]
    if not isinstance(raw_view_ids, Sequence) or isinstance(
        raw_view_ids, (str, bytes, bytearray)
    ) or not raw_view_ids:
        raise ValueError(f"{where}.view_ids must be a non-empty array")
    view_ids: list[str] = []
    for view_index, raw_view_id in enumerate(raw_view_ids):
        view_id = _identifier(
            raw_view_id, f"{where}.view_ids[{view_index}]"
        )
        assert view_id is not None
        if view_id in view_ids:
            raise ValueError(f"{where}.view_ids must not contain duplicates")
        view = views_by_id.get(view_id)
        if view is None:
            raise ValueError(
                f"{where}.view_ids[{view_index}] references an unknown view"
            )
        if view["asset_id"] != asset_id:
            raise ValueError(f"{where}.asset_id and view parent differ")
        view_ids.append(view_id)
    expected_asset_views = sorted(
        str(candidate_id)
        for candidate_id, candidate in views_by_id.items()
        if candidate["asset_id"] == asset_id
    )
    if sorted(view_ids) != expected_asset_views:
        raise ValueError(
            f"{where}.view_ids must contain every planned view for its asset"
        )
    reading_order_raw = obj["reading_order"]
    if not isinstance(reading_order_raw, Sequence) or isinstance(
        reading_order_raw, (str, bytes, bytearray)
    ):
        raise ValueError(f"{where}.reading_order must be an array of view ids")
    reading_order = [
        _identifier(value, f"{where}.reading_order[{order_index}]")
        for order_index, value in enumerate(reading_order_raw)
    ]
    if len(reading_order) != len(set(reading_order)) or set(reading_order) != set(
        view_ids
    ):
        raise ValueError(
            f"{where}.reading_order must be an exact permutation of view_ids"
        )
    writing_mode = _identifier(obj["writing_mode"], f"{where}.writing_mode")
    if writing_mode not in _ALLOWED_WRITING_MODES:
        raise ValueError(
            f"{where}.writing_mode must be vertical-lr, vertical-rl, or horizontal-tb"
        )
    preprocess_contract_sha256 = _sha256(
        obj["preprocess_contract_sha256"],
        f"{where}.preprocess_contract_sha256",
    )
    reference_token_count = _positive_int(
        obj["reference_token_count"], f"{where}.reference_token_count"
    )
    group_ids = _object(obj["group_ids"], f"{where}.group_ids")
    _exact_keys(group_ids, _GROUP_KEYS, f"{where}.group_ids")
    document_id = _identifier(
        group_ids["document_id"], f"{where}.group_ids.document_id"
    )
    capture_id = _identifier(
        group_ids["capture_id"], f"{where}.group_ids.capture_id"
    )
    if document_id != asset["source_document_id"]:
        raise ValueError(f"{where}.group_ids.document_id differs from parent asset")
    if capture_id != asset["source_capture_id"]:
        raise ValueError(f"{where}.group_ids.capture_id differs from parent asset")
    near_duplicate = _identifier(
        group_ids["near_duplicate_cluster"],
        f"{where}.group_ids.near_duplicate_cluster",
    )
    if near_duplicate != asset["near_duplicate"]["cluster_id"]:
        raise ValueError(
            f"{where}.group_ids.near_duplicate_cluster differs from parent asset"
        )
    style = obj["style"]
    font_id = _identifier(obj["font_id"], f"{where}.font_id", nullable=True)
    writer_id = _identifier(obj["writer_id"], f"{where}.writer_id", nullable=True)
    group_writer_id = _identifier(
        group_ids["writer_id"], f"{where}.group_ids.writer_id", nullable=True
    )
    difficulty = obj["difficulty"]
    if style == "print":
        if font_id is None or writer_id is not None or group_writer_id is not None:
            raise ValueError(
                f"{where} print samples require font_id and forbid writer_id"
            )
        if difficulty is not None:
            raise ValueError(f"{where} print samples must set difficulty to null")
    elif style == "handwritten":
        if font_id is not None or writer_id is None or group_writer_id != writer_id:
            raise ValueError(
                f"{where} handwritten samples require matching writer_id and no font_id"
            )
        if difficulty not in _ALLOWED_DIFFICULTIES:
            raise ValueError(
                f"{where}.difficulty must be good, medium, or poor"
            )
    else:
        raise ValueError(f"{where}.style must be print or handwritten")
    qa_state = obj["qa_state"]
    if qa_state not in {"accepted", "quarantine"}:
        raise ValueError(f"{where}.qa_state must be accepted or quarantine")
    reference_raw = _reference(obj["reference_raw"], f"{where}.reference_raw")
    reference_model = _reference(
        obj["reference_model"], f"{where}.reference_model"
    )
    expected_model_text = canonicalize_native_ocr_text(reference_raw["text"])
    if reference_model["text"] != expected_model_text:
        raise ValueError(
            f"{where}.reference_model.text must equal the reviewed native "
            "canonicalization of reference_raw.text"
        )
    reasons = ["qa_state"] if qa_state == "quarantine" else []
    return (
        {
            "schema_version": ANYRES_SCHEMA_VERSION,
            "sample_id": sample_id,
            "split": split,
            "visual_contract": ANYRES_VISUAL_CONTRACT,
            "asset_id": asset_id,
            "view_ids": view_ids,
            "preprocess_contract_sha256": preprocess_contract_sha256,
            "reference_token_count": reference_token_count,
            "writing_mode": writing_mode,
            "reading_order": reading_order,
            "leakage_cluster": _identifier(
                obj["leakage_cluster"], f"{where}.leakage_cluster"
            ),
            "group_ids": {
                "document_id": document_id,
                "capture_id": capture_id,
                "writer_id": group_writer_id,
                "near_duplicate_cluster": near_duplicate,
            },
            "style": style,
            "font_id": font_id,
            "writer_id": writer_id,
            "difficulty": difficulty,
            "reference_raw": reference_raw,
            "reference_model": reference_model,
            "qa_state": qa_state,
        },
        reasons,
    )


def validate_no_cross_split_leakage(samples: Sequence[Mapping[str, Any]]) -> None:
    """Reject reuse of any identity/cluster group across dataset splits.

    ``near_duplicate_cluster`` is mandatory input.  This module intentionally
    does not generate perceptual hashes or infer near-duplicate clusters.
    """

    owners: dict[tuple[str, str], tuple[str, str]] = {}
    for index, sample in enumerate(samples):
        where = f"samples[{index}]"
        split = sample.get("split")
        sample_id = sample.get("sample_id")
        groups = _object(sample.get("group_ids"), f"{where}.group_ids")
        candidates = {
            "asset": sample.get("asset_id"),
            "document": groups.get("document_id"),
            "capture": groups.get("capture_id"),
            "writer": groups.get("writer_id"),
            "near_duplicate": groups.get("near_duplicate_cluster"),
            "leakage_cluster": sample.get("leakage_cluster"),
        }
        for kind, identity in candidates.items():
            if identity is None:
                continue
            key = (kind, str(identity))
            previous = owners.get(key)
            if previous is not None and previous[0] != split:
                raise ValueError(
                    f"cross-split leakage for {kind}={identity!r}: "
                    f"{previous[1]!r}/{previous[0]!r} vs "
                    f"{sample_id!r}/{split!r}"
                )
            owners[key] = (str(split), str(sample_id))


def _validate_asset_near_duplicates(
    assets: Sequence[Mapping[str, Any]],
) -> None:
    thresholds = {
        int(asset["near_duplicate"]["max_hamming_distance"])
        for asset in assets
    }
    if len(thresholds) != 1:
        raise ValueError("all assets must use one pHash distance threshold")
    threshold = next(iter(thresholds))
    hashes = {
        str(asset["asset_id"]): str(asset["near_duplicate"]["value"])
        for asset in assets
    }
    clusters = _cached_perceptual_clusters(tuple(sorted(hashes.items())), threshold)
    expected = {
        asset_id: cluster.cluster_id
        for cluster in clusters
        for asset_id in cluster.member_ids
    }
    for asset in assets:
        asset_id = str(asset["asset_id"])
        if asset["near_duplicate"]["cluster_id"] != expected[asset_id]:
            raise ValueError(
                f"asset {asset_id!r} near-duplicate cluster is not reproducible"
            )


@lru_cache(maxsize=4)
def _cached_perceptual_clusters(
    hashes: tuple[tuple[str, str], ...],
    max_hamming_distance: int,
):
    return cluster_perceptual_hashes(
        dict(hashes),
        max_hamming_distance=max_hamming_distance,
    )


def _validate_asset_content_no_cross_split(
    samples: Sequence[Mapping[str, Any]],
    assets_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    owners: dict[tuple[str, str], tuple[str, str]] = {}
    for sample in samples:
        asset = assets_by_id[str(sample["asset_id"])]
        for kind, digest in (
            ("raw_sha256", asset["raw_sha256"]),
            ("canonical_sha256", asset["canonical_sha256"]),
            ("pixel_sha256", asset["pixel_sha256"]),
        ):
            key = (kind, str(digest))
            previous = owners.get(key)
            if previous is not None and previous[0] != sample["split"]:
                raise ValueError(
                    f"cross-split leakage for {kind}={digest!r}: "
                    f"{previous[1]!r}/{previous[0]!r} vs "
                    f"{sample['sample_id']!r}/{sample['split']!r}"
                )
            owners[key] = (str(sample["split"]), str(sample["sample_id"]))


def quota_bucket(sample: Mapping[str, Any], *, where: str = "sample") -> str:
    """Return the deterministic pre-collation quota bucket for one sample."""

    style = sample.get("style")
    if style == "print":
        return "print"
    if style == "handwritten":
        bucket = f"handwritten_{sample.get('difficulty')}"
        if bucket in ANYRES_QUOTA_60_10_20_10:
            return bucket
        raise ValueError(f"{where}.difficulty is not good, medium, or poor")
    raise ValueError(f"{where}.style must be print or handwritten")


def quota_counts(samples: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count print and three handwritten-difficulty quota buckets."""

    counts = {key: 0 for key in ANYRES_QUOTA_60_10_20_10}
    for index, sample in enumerate(samples):
        counts[quota_bucket(sample, where=f"samples[{index}]")] += 1
    return counts


def validate_quota_60_10_20_10(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Require the exact 60/10/20/10 print/good/medium/poor distribution."""

    counts = quota_counts(samples)
    total = sum(counts.values())
    if total == 0:
        raise ValueError("quota validation requires at least one accepted sample")
    mismatches = [
        bucket
        for bucket, percent in ANYRES_QUOTA_60_10_20_10.items()
        if counts[bucket] * 100 != total * percent
    ]
    if mismatches:
        raise ValueError(
            "anyres quota must be exactly 60/10/20/10 "
            f"(print/handwritten good/medium/poor); counts={counts}"
        )
    return counts


def validate_anyres_assets_views_samples(
    assets: Sequence[Mapping[str, Any]],
    views: Sequence[Mapping[str, Any]],
    samples: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Normalize valid v2 rows and separate QA-quarantined views/samples."""

    normalized_assets = [_normalize_asset(value, i) for i, value in enumerate(assets)]
    assets_by_id: dict[str, dict[str, Any]] = {}
    for asset in normalized_assets:
        asset_id = str(asset["asset_id"])
        if asset_id in assets_by_id:
            raise ValueError(f"duplicate asset_id {asset_id!r}")
        assets_by_id[asset_id] = asset
    _validate_asset_near_duplicates(normalized_assets)

    all_views: list[dict[str, Any]] = []
    view_reasons: dict[str, list[str]] = {}
    views_by_id: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(views):
        view, reasons = _normalize_view(value, index, assets_by_id)
        view_id = str(view["view_id"])
        if view_id in views_by_id:
            raise ValueError(f"duplicate view_id {view_id!r}")
        views_by_id[view_id] = view
        all_views.append(view)
        if reasons:
            view_reasons[view_id] = sorted(set(reasons))

    for asset_id, asset in assets_by_id.items():
        expected = set(range(len(asset["native_plan"]["payload"]["windows"])))
        observed = [
            int(view["transform"]["window_index"])
            for view in all_views
            if view["asset_id"] == asset_id
        ]
        if len(observed) != len(set(observed)) or set(observed) != expected:
            raise ValueError(
                f"asset {asset_id!r} views must map one-to-one onto every "
                "deterministic plan window"
            )
        expected_boxes = {
            str(view["view_id"]): list(view["box_xyxy"])
            for view in all_views
            if view["asset_id"] == asset_id
        }
        if asset["cut_qa"]["payload"]["view_boxes"] != expected_boxes:
            raise ValueError(
                f"asset {asset_id!r} cut-QA view boxes differ from manifests"
            )

    all_samples: list[dict[str, Any]] = []
    sample_reasons: dict[str, list[str]] = {}
    seen_sample_ids: set[str] = set()
    for index, value in enumerate(samples):
        sample, reasons = _normalize_sample(
            value, index, assets_by_id, views_by_id
        )
        sample_id = str(sample["sample_id"])
        if sample_id in seen_sample_ids:
            raise ValueError(f"duplicate sample_id {sample_id!r}")
        seen_sample_ids.add(sample_id)
        for view_id in sample["view_ids"]:
            parent_reasons = view_reasons.get(str(view_id), [])
            reasons.extend(
                f"parent_view:{view_id}:{reason}" for reason in parent_reasons
            )
        all_samples.append(sample)
        if reasons:
            sample_reasons[sample_id] = sorted(set(reasons))

    validate_no_cross_split_leakage(all_samples)
    _validate_asset_content_no_cross_split(all_samples, assets_by_id)

    safe_views = [view for view in all_views if view["view_id"] not in view_reasons]
    safe_samples = [
        sample for sample in all_samples if sample["sample_id"] not in sample_reasons
    ]
    normalized = {
        "schema_version": ANYRES_SCHEMA_VERSION,
        "assets": sorted(normalized_assets, key=lambda row: str(row["asset_id"])),
        "views": sorted(safe_views, key=lambda row: str(row["view_id"])),
        "samples": sorted(safe_samples, key=lambda row: str(row["sample_id"])),
    }
    quarantine = [
        {"entity_type": "view", "entity_id": view_id, "reasons": reasons}
        for view_id, reasons in sorted(view_reasons.items())
    ] + [
        {"entity_type": "sample", "entity_id": sample_id, "reasons": reasons}
        for sample_id, reasons in sorted(sample_reasons.items())
    ]
    return normalized, quarantine


__all__ = [
    "ANYRES_PUBLIC_SPLITS",
    "ANYRES_VALIDATION_SPLITS",
    "ANYRES_QUOTA_60_10_20_10",
    "ANYRES_SCHEMA_VERSION",
    "ANYRES_VISUAL_CONTRACT",
    "canonical_json_sha256",
    "quota_bucket",
    "quota_counts",
    "validate_anyres_assets_views_samples",
    "validate_no_cross_split_leakage",
    "validate_quota_60_10_20_10",
]
