# -*- coding: utf-8 -*-
"""Immutable preprocessing and resource budgets for OMVT anyres OCR."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import operator
from pathlib import Path
import re
from typing import Any

from Model.ocr.visual_input_contract import (
    DOL_OCR_ANYRES_V2,
    ocr_visual_input_contract_version,
)
from Model.omvt.native_planner import COVERAGE_PROOF_METHOD, PLAN_CONTRACT
from Model.omvt.patcher import PATCH_KINDS


ANYRES_PREPROCESS_CONTRACT_SCHEMA_VERSION = 1
ANYRES_PREPROCESS_CONTRACT_KIND = "dol_ocr_anyres_preprocess_budget_v1"
ANYRES_PREPROCESS_CANONICALIZATION = "utf8_json_sorted_keys_compact_v1"
ANYRES_GLOBAL_VISUAL_PREFIX_TOKENS = 256
ANYRES_GLOBAL_PROMPT_FIXED_TOKENS = 3

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_PATHS = {
    "native_processor": "Tokenizer/multimodal/native_image_io.py",
    "native_patcher": "Model/omvt/native_patcher.py",
    "native_planner": "Model/omvt/native_planner.py",
    "native_tower": "Model/omvt/native_tower.py",
    "detail_compressor": "Model/omvt/native_compressor.py",
    "detail_bridge": "Model/layers/vision_cross_attention.py",
}
_TOP_LEVEL_KEYS = {
    "schema_version",
    "kind",
    "canonicalization",
    "visual_input_contract",
    "pixel_contract",
    "patch_contract",
    "implementation_sources",
    "planner_contract",
    "budgets",
    "contract_canonical_sha256",
}


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer")
    try:
        normalized = int(operator.index(value))
    except TypeError as exc:
        raise ValueError(f"{field} must be a positive integer") from exc
    if normalized <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return normalized


def _require_exact_keys(
    value: object,
    expected: set[str],
    field: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{field} keys differ from contract: "
            f"missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _current_implementation_sources() -> dict[str, dict[str, str]]:
    root = _source_root()
    sources: dict[str, dict[str, str]] = {}
    for role, relative in _SOURCE_PATHS.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"anyres implementation source is unsafe: {relative}")
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"anyres implementation source escapes repository: {relative}"
            ) from exc
        sources[role] = {"path": relative, "sha256": _file_sha256(resolved)}
    return sources


def _canonical_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key != "contract_canonical_sha256"
    }


def canonical_anyres_preprocess_contract_json(
    payload: Mapping[str, Any],
) -> str:
    if not isinstance(payload, Mapping):
        raise ValueError("anyres preprocess contract must be an object")
    return json.dumps(
        _canonical_payload(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def anyres_preprocess_contract_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        canonical_anyres_preprocess_contract_json(payload).encode("utf-8")
    ).hexdigest()


def _patch_shapes_from_config(omvt_cfg: object) -> dict[str, list[int]]:
    attributes = {
        "vertical": "vertical_patch",
        "horizontal": "horizontal_patch",
        "square": "square_patch",
        "layout": "layout_patch",
    }
    shapes: dict[str, list[int]] = {}
    for kind in PATCH_KINDS:
        shape = getattr(omvt_cfg, attributes[kind], None)
        if (
            not isinstance(shape, Sequence)
            or isinstance(shape, (str, bytes, bytearray))
            or len(shape) != 2
        ):
            raise ValueError(f"omvt_cfg.{attributes[kind]} must contain H and W")
        shapes[kind] = [
            _positive_int(shape[0], f"omvt_cfg.{attributes[kind]}[0]"),
            _positive_int(shape[1], f"omvt_cfg.{attributes[kind]}[1]"),
        ]
    return shapes


def _pixel_contract() -> dict[str, Any]:
    normalized_white = [
        (1.0 - mean) / std
        for mean, std in zip(_IMAGENET_MEAN, _IMAGENET_STD, strict=True)
    ]
    return {
        "orientation": "pil_imageops_exif_transpose_v1",
        "color_mode": "RGB",
        "in_channels": 3,
        "resize": "none",
        "crop": "none",
        "normalization": {
            "name": "imagenet_rgb_v1",
            "mean": list(_IMAGENET_MEAN),
            "std": list(_IMAGENET_STD),
        },
        "padding": {
            "color": "white",
            "rgb_u8": [255, 255, 255],
            "normalized_rgb": normalized_white,
            "pixel_valid_mask": False,
        },
    }


def build_anyres_preprocess_contract(
    omvt_cfg: object,
    max_decode_pixels: object,
    max_raw_patch_tokens_per_view: object,
    max_windows: object,
    halo_px: object,
    max_detail_tokens: object,
    source_tokens_per_detail_token: object,
    max_seq_len: object,
    recommended_max_new_tokens: object,
) -> dict[str, Any]:
    max_decode_pixels = _positive_int(max_decode_pixels, "max_decode_pixels")
    max_raw_patch_tokens_per_view = _positive_int(
        max_raw_patch_tokens_per_view,
        "max_raw_patch_tokens_per_view",
    )
    max_windows = _positive_int(max_windows, "max_windows")
    halo_px = _positive_int(halo_px, "halo_px")
    max_detail_tokens = _positive_int(max_detail_tokens, "max_detail_tokens")
    source_tokens_per_detail_token = _positive_int(
        source_tokens_per_detail_token,
        "source_tokens_per_detail_token",
    )
    max_seq_len = _positive_int(max_seq_len, "max_seq_len")
    recommended_max_new_tokens = _positive_int(
        recommended_max_new_tokens,
        "recommended_max_new_tokens",
    )
    in_channels = _positive_int(
        getattr(omvt_cfg, "in_channels", None),
        "omvt_cfg.in_channels",
    )
    if in_channels != 3:
        raise ValueError("dol_ocr_anyres_v2 requires RGB OMVT input")
    global_prefix = _positive_int(
        getattr(omvt_cfg, "compress_to", None),
        "omvt_cfg.compress_to",
    )
    if global_prefix != ANYRES_GLOBAL_VISUAL_PREFIX_TOKENS:
        raise ValueError(
            "dol_ocr_anyres_v2 requires exactly 256 global visual prefix tokens"
        )
    if (
        ANYRES_GLOBAL_PROMPT_FIXED_TOKENS
        + global_prefix
        + recommended_max_new_tokens
        > max_seq_len
    ):
        raise ValueError(
            "global OCR prompt plus recommended output exceeds max_seq_len"
        )

    effective_detail_per_view = min(
        max_detail_tokens,
        (
            max_raw_patch_tokens_per_view
            + source_tokens_per_detail_token
            - 1
        )
        // source_tokens_per_detail_token,
    )
    payload: dict[str, Any] = {
        "schema_version": ANYRES_PREPROCESS_CONTRACT_SCHEMA_VERSION,
        "kind": ANYRES_PREPROCESS_CONTRACT_KIND,
        "canonicalization": ANYRES_PREPROCESS_CANONICALIZATION,
        "visual_input_contract": {
            "name": DOL_OCR_ANYRES_V2,
            "version": ocr_visual_input_contract_version(DOL_OCR_ANYRES_V2),
        },
        "pixel_contract": _pixel_contract(),
        "patch_contract": {
            "stream_order": list(PATCH_KINDS),
            "shapes_hw": _patch_shapes_from_config(omvt_cfg),
            "bbox_coordinates": "per_sample_yxxy_half_open_v1",
            "edge_padding": "bottom_right_normalized_white_invalid_v1",
        },
        "implementation_sources": _current_implementation_sources(),
        "planner_contract": {
            "name": PLAN_CONTRACT,
            "coverage_proof_method": COVERAGE_PROOF_METHOD,
        },
        "budgets": {
            "decode": {"max_pixels_per_asset": max_decode_pixels},
            "window": {
                "max_windows_per_asset": max_windows,
                "halo_pixels_per_side": halo_px,
            },
            "patch": {
                "max_raw_tokens_per_view": max_raw_patch_tokens_per_view,
                "max_raw_tokens_per_asset": (
                    max_raw_patch_tokens_per_view * max_windows
                ),
            },
            "detail": {
                "max_tokens_per_view": max_detail_tokens,
                "source_tokens_per_detail_token": (
                    source_tokens_per_detail_token
                ),
                "max_effective_tokens_per_view": effective_detail_per_view,
                "max_effective_tokens_per_asset": (
                    effective_detail_per_view * max_windows
                ),
                "transport": "out_of_band_ragged_cross_attention_v1",
            },
            "context": {
                "max_sequence_tokens": max_seq_len,
                "global_visual_prefix_tokens": global_prefix,
                "global_prompt_fixed_tokens": ANYRES_GLOBAL_PROMPT_FIXED_TOKENS,
                "remaining_nonvisual_prompt_tokens_at_recommended_output": (
                    max_seq_len
                    - ANYRES_GLOBAL_PROMPT_FIXED_TOKENS
                    - global_prefix
                    - recommended_max_new_tokens
                ),
            },
            "output": {
                "recommended_max_new_tokens": recommended_max_new_tokens
            },
        },
    }
    payload["contract_canonical_sha256"] = (
        anyres_preprocess_contract_sha256(payload)
    )
    return payload


def validate_anyres_preprocess_contract(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    root = _require_exact_keys(payload, _TOP_LEVEL_KEYS, "anyres contract")
    if root.get("schema_version") != ANYRES_PREPROCESS_CONTRACT_SCHEMA_VERSION:
        raise ValueError("unsupported anyres preprocess schema_version")
    if root.get("kind") != ANYRES_PREPROCESS_CONTRACT_KIND:
        raise ValueError("unsupported anyres preprocess contract kind")
    if root.get("canonicalization") != ANYRES_PREPROCESS_CANONICALIZATION:
        raise ValueError("unsupported anyres preprocess canonicalization")

    visual = _require_exact_keys(
        root.get("visual_input_contract"),
        {"name", "version"},
        "visual_input_contract",
    )
    if visual.get("name") != DOL_OCR_ANYRES_V2 or visual.get("version") != 2:
        raise ValueError("anyres preprocess contract must bind dol_ocr_anyres_v2")

    if root.get("pixel_contract") != _pixel_contract():
        raise ValueError("pixel contract differs from native EXIF/RGB/ImageNet rules")

    patch = _require_exact_keys(
        root.get("patch_contract"),
        {"stream_order", "shapes_hw", "bbox_coordinates", "edge_padding"},
        "patch_contract",
    )
    if patch.get("stream_order") != list(PATCH_KINDS):
        raise ValueError("patch stream order differs from OMVT")
    if patch.get("bbox_coordinates") != "per_sample_yxxy_half_open_v1":
        raise ValueError("unsupported patch bbox coordinate contract")
    if patch.get("edge_padding") != "bottom_right_normalized_white_invalid_v1":
        raise ValueError("unsupported anyres edge-padding contract")
    shapes = _require_exact_keys(
        patch.get("shapes_hw"), set(PATCH_KINDS), "patch_contract.shapes_hw"
    )
    for kind in PATCH_KINDS:
        shape = shapes[kind]
        if not isinstance(shape, list) or len(shape) != 2:
            raise ValueError(f"{kind} patch shape must contain H and W")
        _positive_int(shape[0], f"{kind}.patch_h")
        _positive_int(shape[1], f"{kind}.patch_w")

    sources = _require_exact_keys(
        root.get("implementation_sources"),
        set(_SOURCE_PATHS),
        "implementation_sources",
    )
    current_sources = _current_implementation_sources()
    for role, relative in _SOURCE_PATHS.items():
        record = _require_exact_keys(
            sources[role], {"path", "sha256"}, f"implementation_sources.{role}"
        )
        digest = record.get("sha256")
        if record.get("path") != relative:
            raise ValueError(f"implementation source path mismatch for {role}")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"implementation source SHA256 is invalid for {role}")
        if record != current_sources[role]:
            raise ValueError(f"anyres implementation source drift for {role}")

    planner = _require_exact_keys(
        root.get("planner_contract"),
        {"name", "coverage_proof_method"},
        "planner_contract",
    )
    if planner != {
        "name": PLAN_CONTRACT,
        "coverage_proof_method": COVERAGE_PROOF_METHOD,
    }:
        raise ValueError("native macro-window planner contract mismatch")

    budgets = _require_exact_keys(
        root.get("budgets"),
        {"decode", "window", "patch", "detail", "context", "output"},
        "budgets",
    )
    decode = _require_exact_keys(
        budgets["decode"], {"max_pixels_per_asset"}, "budgets.decode"
    )
    window = _require_exact_keys(
        budgets["window"],
        {"max_windows_per_asset", "halo_pixels_per_side"},
        "budgets.window",
    )
    patch_budget = _require_exact_keys(
        budgets["patch"],
        {"max_raw_tokens_per_view", "max_raw_tokens_per_asset"},
        "budgets.patch",
    )
    detail = _require_exact_keys(
        budgets["detail"],
        {
            "max_tokens_per_view",
            "source_tokens_per_detail_token",
            "max_effective_tokens_per_view",
            "max_effective_tokens_per_asset",
            "transport",
        },
        "budgets.detail",
    )
    context = _require_exact_keys(
        budgets["context"],
        {
            "max_sequence_tokens",
            "global_visual_prefix_tokens",
            "global_prompt_fixed_tokens",
            "remaining_nonvisual_prompt_tokens_at_recommended_output",
        },
        "budgets.context",
    )
    output = _require_exact_keys(
        budgets["output"],
        {"recommended_max_new_tokens"},
        "budgets.output",
    )

    _positive_int(
        decode["max_pixels_per_asset"], "budgets.decode.max_pixels_per_asset"
    )
    max_windows = _positive_int(
        window["max_windows_per_asset"], "budgets.window.max_windows_per_asset"
    )
    _positive_int(
        window["halo_pixels_per_side"], "budgets.window.halo_pixels_per_side"
    )
    raw_per_view = _positive_int(
        patch_budget["max_raw_tokens_per_view"],
        "budgets.patch.max_raw_tokens_per_view",
    )
    raw_per_asset = _positive_int(
        patch_budget["max_raw_tokens_per_asset"],
        "budgets.patch.max_raw_tokens_per_asset",
    )
    if raw_per_asset != raw_per_view * max_windows:
        raise ValueError("aggregate raw patch-token budget is inconsistent")
    detail_cap = _positive_int(
        detail["max_tokens_per_view"], "budgets.detail.max_tokens_per_view"
    )
    ratio = _positive_int(
        detail["source_tokens_per_detail_token"],
        "budgets.detail.source_tokens_per_detail_token",
    )
    effective = min(detail_cap, (raw_per_view + ratio - 1) // ratio)
    effective_per_view = _positive_int(
        detail["max_effective_tokens_per_view"],
        "budgets.detail.max_effective_tokens_per_view",
    )
    if effective_per_view != effective:
        raise ValueError("effective detail-token budget is inconsistent")
    effective_per_asset = _positive_int(
        detail["max_effective_tokens_per_asset"],
        "budgets.detail.max_effective_tokens_per_asset",
    )
    if effective_per_asset != effective * max_windows:
        raise ValueError("aggregate detail-token budget is inconsistent")
    if detail.get("transport") != "out_of_band_ragged_cross_attention_v1":
        raise ValueError("unsupported detail-memory transport")

    max_seq_len = _positive_int(
        context["max_sequence_tokens"], "budgets.context.max_sequence_tokens"
    )
    global_prefix = _positive_int(
        context["global_visual_prefix_tokens"],
        "budgets.context.global_visual_prefix_tokens",
    )
    if global_prefix != ANYRES_GLOBAL_VISUAL_PREFIX_TOKENS:
        raise ValueError("anyres global visual prefix must contain 256 tokens")
    if context["global_prompt_fixed_tokens"] != ANYRES_GLOBAL_PROMPT_FIXED_TOKENS:
        raise ValueError("anyres global prompt fixed-token budget is inconsistent")
    output_tokens = _positive_int(
        output["recommended_max_new_tokens"],
        "budgets.output.recommended_max_new_tokens",
    )
    if (
        ANYRES_GLOBAL_PROMPT_FIXED_TOKENS + global_prefix + output_tokens
        > max_seq_len
    ):
        raise ValueError(
            "global OCR prompt plus recommended output exceeds max_seq_len"
        )
    remaining = (
        max_seq_len
        - ANYRES_GLOBAL_PROMPT_FIXED_TOKENS
        - global_prefix
        - output_tokens
    )
    if context.get(
        "remaining_nonvisual_prompt_tokens_at_recommended_output"
    ) != remaining:
        raise ValueError("derived causal context budget is inconsistent")

    expected_sha = root.get("contract_canonical_sha256")
    if not isinstance(expected_sha, str) or _SHA256_RE.fullmatch(expected_sha) is None:
        raise ValueError("contract_canonical_sha256 must be a lowercase SHA256")
    actual_sha = anyres_preprocess_contract_sha256(root)
    if actual_sha != expected_sha:
        raise ValueError("anyres preprocess contract canonical SHA256 mismatch")
    return json.loads(canonical_anyres_preprocess_contract_json(root)) | {
        "contract_canonical_sha256": expected_sha
    }


__all__ = [
    "ANYRES_GLOBAL_PROMPT_FIXED_TOKENS",
    "ANYRES_GLOBAL_VISUAL_PREFIX_TOKENS",
    "ANYRES_PREPROCESS_CANONICALIZATION",
    "ANYRES_PREPROCESS_CONTRACT_KIND",
    "ANYRES_PREPROCESS_CONTRACT_SCHEMA_VERSION",
    "anyres_preprocess_contract_sha256",
    "build_anyres_preprocess_contract",
    "canonical_anyres_preprocess_contract_json",
    "validate_anyres_preprocess_contract",
]
