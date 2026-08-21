# -*- coding: utf-8 -*-

"""Strict checkpoint reconstruction for post-training.

OCR alignment must reconstruct the exact RDT and OMVT geometry *before* loading
weights.  Loading a multimodal checkpoint into the default lazy vision module
with ``strict=False`` silently drops ``vision.omvt.*`` tensors, producing a text-
only policy while the trainer appears to run normally.  This module makes that
state impossible.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from Model.config import OMVTConfig, RDTConfig, rdt_config_from_dict
from Model.model import RDTForCausalLM
from Model.ocr.alignment_contract import (
    validate_ocr_alignment_data_contract_payload,
)
from Model.ocr.tokenization import (
    OCR_NATIVE_TARGET_ENCODING,
    OCR_TOKENIZATION_CONTRACT_VERSION,
    canonical_json_sha256,
)
from Model.ocr.visual_input_contract import (
    DOL_OCR_ANYRES_V2,
    OCR_VISUAL_INPUT_CONTRACT_METADATA_KEY,
    OCR_VISUAL_INPUT_CONTRACT_VERSION_METADATA_KEY,
    resolve_checkpoint_ocr_visual_input_contract,
)
from Model.omvt import OMVTInjector
from Model.omvt.native_migration import MIGRATION_CONTRACT
from Model.training.checkpoint import (
    load_checkpoint_metadata,
    resolve_checkpoint_dir,
)


OCR_GRPO_CONTRACT_VERSION = 4
ANYRES_NATIVE_DETAIL_CONFIG_METADATA_KEY = "native_detail_config"
ANYRES_CROSS_ATTENTION_CONFIG_METADATA_KEY = (
    "vision_cross_attention_config"
)
ANYRES_MIGRATION_RECEIPT_METADATA_KEY = "native_migration_receipt"
_VISUAL_TERMINAL_REASONS = frozenset({"loss_plateau", "max_steps", "completed"})
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")


@dataclass
class ReconstructedPolicy:
    model: RDTForCausalLM
    rdt_config: RDTConfig
    omvt_config: OMVTConfig | None
    metadata: dict[str, Any]
    checkpoint_dir: Path
    model_sha256: str
    metadata_sha256: str
    native_detail_config: dict[str, Any] | None = None
    vision_cross_attention_config: dict[str, Any] | None = None
    native_migration_receipt: dict[str, Any] | None = None


def _normalized_expected_sha256(
    expected_sha256: str | None,
    *,
    context: str,
) -> str | None:
    if expected_sha256 is None:
        return None
    if not isinstance(expected_sha256, str) or not _SHA256_RE.fullmatch(
        expected_sha256
    ):
        raise ValueError(f"{context} expected SHA-256 must be 64 hex characters")
    return expected_sha256.lower()


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    """Fields that must remain stable while one opened artifact is consumed."""

    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _open_regular_file_no_follow(path: Path):
    """Open ``path`` once, rejecting symlinks and non-regular artifacts."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    elif path.is_symlink():  # pragma: no cover - modern Linux/macOS use O_NOFOLLOW
        raise ValueError(f"checkpoint artifact must not be a symlink: {path}")
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if path.is_symlink():
            raise ValueError(
                f"checkpoint artifact must not be a symlink: {path}"
            ) from exc
        raise
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(
                f"checkpoint artifact must be a regular file: {path}"
            )
        return os.fdopen(descriptor, "rb"), before
    except BaseException:
        os.close(descriptor)
        raise


def _handle_sha256(handle) -> str:
    digest = hashlib.sha256()
    handle.seek(0)
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    handle.seek(0)
    return digest.hexdigest()


def _assert_stable_artifact(
    path: Path,
    *,
    before_stat: os.stat_result,
    before_sha256: str,
    after_stat: os.stat_result,
    after_sha256: str,
) -> None:
    if (
        before_sha256 != after_sha256
        or _stat_identity(before_stat) != _stat_identity(after_stat)
    ):
        raise ValueError(f"checkpoint artifact changed while being loaded: {path}")


def _verified_file_sha256(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> str:
    """Hash one stable, non-symlink regular-file descriptor."""

    expected = _normalized_expected_sha256(
        expected_sha256,
        context=str(path),
    )
    handle, before_stat = _open_regular_file_no_follow(path)
    with handle:
        before_sha256 = _handle_sha256(handle)
        if expected is not None and before_sha256 != expected:
            raise ValueError(
                f"checkpoint artifact SHA-256 mismatch: {path}: "
                f"expected={expected} actual={before_sha256}"
            )
        after_sha256 = _handle_sha256(handle)
        after_stat = os.fstat(handle.fileno())
    _assert_stable_artifact(
        path,
        before_stat=before_stat,
        before_sha256=before_sha256,
        after_stat=after_stat,
        after_sha256=after_sha256,
    )
    return before_sha256


def _verified_torch_load(
    path: Path,
    *,
    expected_sha256: str | None = None,
):
    """Hash-before, weights-only load, then hash/fstat the same descriptor."""

    expected = _normalized_expected_sha256(
        expected_sha256,
        context=str(path),
    )
    handle, before_stat = _open_regular_file_no_follow(path)
    with handle:
        before_sha256 = _handle_sha256(handle)
        if expected is not None and before_sha256 != expected:
            raise ValueError(
                f"checkpoint artifact SHA-256 mismatch: {path}: "
                f"expected={expected} actual={before_sha256}"
            )
        payload = torch.load(
            handle,
            map_location="cpu",
            weights_only=True,
        )
        after_sha256 = _handle_sha256(handle)
        after_stat = os.fstat(handle.fileno())
    _assert_stable_artifact(
        path,
        before_stat=before_stat,
        before_sha256=before_sha256,
        after_stat=after_stat,
        after_sha256=after_sha256,
    )
    return payload, before_sha256


def verified_file_sha256(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> str:
    """Public stable-descriptor SHA-256 verification for checkpoint members."""

    return _verified_file_sha256(
        Path(path),
        expected_sha256=expected_sha256,
    )


def load_verified_torch_payload(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[Any, str]:
    """Safely load one stable checkpoint member with ``weights_only=True``."""

    return _verified_torch_load(
        Path(path),
        expected_sha256=expected_sha256,
    )


def load_verified_checkpoint_metadata_envelope(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Load an exact ``{step, metadata}`` checkpoint envelope safely.

    The outer step is intentionally retained.  Consumers deciding whether a
    checkpoint is terminal/resumable must bind filesystem markers and progress
    state to the saved checkpoint step rather than silently discarding it.
    """

    source = Path(path)
    meta_path = source if source.name == "meta.pt" else (
        resolve_checkpoint_dir(source) / "meta.pt"
    )
    envelope, actual_sha256 = _verified_torch_load(
        meta_path,
        expected_sha256=expected_sha256,
    )
    if not isinstance(envelope, dict) or set(envelope) != {"step", "metadata"}:
        raise TypeError(
            f"checkpoint metadata envelope must contain exactly step/metadata: "
            f"{meta_path}"
        )
    step = envelope["step"]
    metadata = envelope["metadata"]
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise TypeError(f"checkpoint metadata step must be non-negative: {meta_path}")
    if not isinstance(metadata, dict):
        raise TypeError(f"checkpoint metadata payload must be a dict: {meta_path}")
    return {"step": step, "metadata": metadata}, actual_sha256


def load_verified_policy_metadata(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Atomically load policy metadata from a checkpoint dir or ``meta.pt``."""

    source = Path(path)
    if source.name == "meta.pt":
        meta_path = source
    else:
        meta_path = resolve_checkpoint_dir(source) / "meta.pt"
    if not meta_path.exists() and not meta_path.is_symlink():
        if expected_sha256 is not None:
            raise FileNotFoundError(f"checkpoint metadata not found: {meta_path}")
        return {}, ""
    meta, actual_sha256 = _verified_torch_load(
        meta_path,
        expected_sha256=expected_sha256,
    )
    if not isinstance(meta, dict):
        raise TypeError(f"checkpoint metadata must be a dict: {meta_path}")
    payload = meta.get("metadata", meta)
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint metadata payload must be a dict: {meta_path}")
    return payload, actual_sha256


def validate_visual_ocr_source_contract(
    metadata: dict[str, Any],
    checkpoint_dir: str | Path,
    tokenization_contract: dict[str, Any],
    *,
    tokenizer_vocab_extent: int,
    require_terminal: bool = True,
) -> dict[str, Any]:
    """Validate that a visual checkpoint was trained in the current LM space.

    Tensor shapes cannot detect a tokenizer whose ids were permuted, nor can
    they prove that a frozen language model was supervised with its native
    representation.  This gate binds both before OCR-GRPO allocates a model.

    Returns lineage fields that must be copied into every policy/reference
    checkpoint.
    """

    checkpoint_dir = resolve_checkpoint_dir(checkpoint_dir)
    errors: list[str] = []
    if not (checkpoint_dir / "COMPLETE").is_file():
        errors.append("checkpoint has no COMPLETE marker")
    if metadata.get("phase") != "vlm_align":
        errors.append(f"phase must be 'vlm_align', got {metadata.get('phase')!r}")
    if metadata.get("freeze_rdt") is not True:
        errors.append("freeze_rdt must be true")
    if metadata.get("frozen_vision") is not False:
        errors.append("frozen_vision must be explicitly false")
    if require_terminal and metadata.get("final") is not True:
        errors.append(
            "visual checkpoint must be the final terminal artifact, not a "
            "periodic COMPLETE checkpoint"
        )
    stop_reason = metadata.get("stop_reason")
    if require_terminal and stop_reason not in _VISUAL_TERMINAL_REASONS:
        errors.append(
            "visual checkpoint stop_reason must be terminal "
            f"{sorted(_VISUAL_TERMINAL_REASONS)}, got {stop_reason!r}"
        )

    expected_mode = tokenization_contract.get("target_encoding")
    expected_version = tokenization_contract.get("tokenization_contract_version")
    if expected_mode != OCR_NATIVE_TARGET_ENCODING:
        errors.append("current OCR tokenization contract is not strict native")
    if expected_version != OCR_TOKENIZATION_CONTRACT_VERSION:
        errors.append(
            "current OCR tokenization contract version differs from runtime"
        )
    if metadata.get("ocr_target_encoding") != expected_mode:
        errors.append(
            "checkpoint ocr_target_encoding differs from current strict-native "
            f"contract: {metadata.get('ocr_target_encoding')!r}"
        )
    if metadata.get("ocr_tokenization_contract_version") != expected_version:
        errors.append(
            "checkpoint ocr_tokenization_contract_version differs from current "
            f"contract: {metadata.get('ocr_tokenization_contract_version')!r}"
        )

    data_lineage = metadata.get("ocr_data_contract")
    if not isinstance(data_lineage, dict):
        errors.append(
            "checkpoint has no schema-versioned OCR alignment data contract"
        )
        data_lineage = {}
    try:
        data_lineage = validate_ocr_alignment_data_contract_payload(
            data_lineage,
            tokenization_contract,
        )
    except ValueError as exc:
        errors.append(str(exc))

    expected_bundle = tokenization_contract.get("tokenizer_bundle")
    if not isinstance(expected_bundle, dict) or not expected_bundle:
        errors.append("current tokenizer contract has no tokenizer_bundle")
        expected_bundle = {}
    source_bundle = metadata.get("source_rdt_tokenizer_bundle")
    source_algorithm = metadata.get("source_rdt_tokenizer_algorithm")
    resolved_source_metadata: dict[str, Any] = {}
    if (
        not isinstance(source_bundle, dict)
        or not source_bundle
        or not isinstance(source_algorithm, dict)
        or not source_algorithm
    ):
        source_path = metadata.get("source_rdt_checkpoint")
        if not isinstance(source_path, str) or not source_path:
            errors.append("visual checkpoint has no source RDT tokenizer lineage")
        else:
            try:
                resolved_source_metadata = load_checkpoint_metadata(
                    resolve_checkpoint_dir(source_path)
                )
            except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
                errors.append(
                    "cannot resolve source RDT checkpoint tokenizer lineage: "
                    f"{exc}"
                )
    if not isinstance(source_bundle, dict) or not source_bundle:
        source_bundle = resolved_source_metadata.get("tokenizer_bundle")
    if not isinstance(source_bundle, dict) or not source_bundle:
        errors.append(
            "source RDT checkpoint has no tokenizer bundle contract"
        )
        source_bundle = {}
    if source_bundle != expected_bundle:
        errors.append(
            "frozen source RDT tokenizer bundle contract differs from the "
            "current OCR tokenizer bundle"
        )
    expected_algorithm = tokenization_contract.get(
        "pretraining_tokenizer_algorithm"
    )
    if not isinstance(expected_algorithm, dict) or not expected_algorithm:
        errors.append("current tokenizer contract has no algorithm fingerprint")
        expected_algorithm = {}
    if not isinstance(source_algorithm, dict) or not source_algorithm:
        source_algorithm = resolved_source_metadata.get("tokenizer_algorithm")
    if not isinstance(source_algorithm, dict) or not source_algorithm:
        errors.append("source RDT checkpoint has no tokenizer algorithm fingerprint")
        source_algorithm = {}
    if source_algorithm != expected_algorithm:
        errors.append(
            "frozen source RDT tokenizer algorithm differs from the current "
            "OCR runtime"
        )

    raw_rdt = metadata.get("rdt_config")
    if not isinstance(raw_rdt, dict):
        errors.append("checkpoint has no rdt_config metadata")
    elif raw_rdt.get("vocab_size") != tokenizer_vocab_extent:
        errors.append(
            "checkpoint vocab_size differs from tokenizer id extent: "
            f"{raw_rdt.get('vocab_size')!r} != {tokenizer_vocab_extent}"
        )

    model_path = checkpoint_dir / "model.pt"
    metadata_path = checkpoint_dir / "meta.pt"
    model_sha256 = ""
    metadata_sha256 = ""
    try:
        model_sha256 = _verified_file_sha256(model_path)
    except (FileNotFoundError, OSError, ValueError) as exc:
        errors.append(f"checkpoint model artifact is unsafe: {exc}")
    try:
        metadata_sha256 = _verified_file_sha256(metadata_path)
    except (FileNotFoundError, OSError, ValueError) as exc:
        errors.append(f"checkpoint metadata artifact is unsafe: {exc}")
    if errors:
        raise ValueError(
            "unsafe visual OCR source checkpoint:\n  - " + "\n  - ".join(errors)
        )

    return {
        "source_checkpoint": str(checkpoint_dir.absolute()),
        "source_checkpoint_model_sha256": model_sha256,
        "source_checkpoint_metadata_sha256": metadata_sha256,
        "visual_corpus_manifest_sha256": data_lineage.get(
            "corpus_manifest_sha256", ""
        ),
        "visual_ocr_data_contract_sha256": data_lineage.get(
            "contract_canonical_sha256", ""
        ),
        "source_rdt_tokenizer_bundle": dict(source_bundle),
        "source_rdt_tokenizer_algorithm": dict(source_algorithm),
        "visual_stop_reason": metadata.get("stop_reason", ""),
        "visual_final": bool(metadata.get("final", False)),
    }


def _rdt_config_from_metadata(
    metadata: dict[str, Any],
    fallback: RDTConfig | None,
) -> RDTConfig:
    raw = metadata.get("rdt_config")
    if raw is None:
        if fallback is None:
            raise ValueError(
                "checkpoint has no rdt_config metadata; exact post-training "
                "reconstruction requires an explicit fallback config"
            )
        return fallback
    if not isinstance(raw, dict):
        raise TypeError("checkpoint rdt_config metadata must be a dict")
    return rdt_config_from_dict(raw)


def _omvt_config_from_metadata(metadata: dict[str, Any]) -> OMVTConfig | None:
    raw = metadata.get("omvt_config")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TypeError("checkpoint omvt_config metadata must be a dict")
    values = dict(raw)
    for key in ("vertical_patch", "horizontal_patch", "square_patch", "layout_patch"):
        if key in values:
            values[key] = tuple(values[key])
    return OMVTConfig(**values)


def _visual_input_contract_from_metadata(
    metadata: Mapping[str, Any],
) -> str | None:
    """Resolve an explicit visual contract without inventing one for legacy v1."""

    has_name = OCR_VISUAL_INPUT_CONTRACT_METADATA_KEY in metadata
    has_version = OCR_VISUAL_INPUT_CONTRACT_VERSION_METADATA_KEY in metadata
    if not has_name and not has_version:
        return None
    return resolve_checkpoint_ocr_visual_input_contract(metadata)


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return int(value)


def _strict_mapping(
    value: object,
    *,
    field: str,
    keys: set[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    payload = dict(value)
    if set(payload) != keys:
        missing = sorted(keys - set(payload))
        extra = sorted(set(payload) - keys)
        raise ValueError(
            f"{field} fields differ from the v2 contract: "
            f"missing={missing} extra={extra}"
        )
    return payload


def _validate_native_migration_receipt(
    value: object,
    *,
    native_config: Mapping[str, Any],
) -> dict[str, Any]:
    wrapper = _strict_mapping(
        value,
        field=ANYRES_MIGRATION_RECEIPT_METADATA_KEY,
        keys={"payload", "canonical_sha256"},
    )
    payload = _strict_mapping(
        wrapper["payload"],
        field="native_migration_receipt.payload",
        keys={
            "contract",
            "initialized_from_legacy",
            "source_class",
            "target_class",
            "source_schema_sha256",
            "target_schema_sha256",
            "max_detail_tokens_per_sample",
            "source_tokens_per_detail_token",
            "mappings",
        },
    )
    digest = wrapper["canonical_sha256"]
    if (
        not isinstance(digest, str)
        or not _SHA256_RE.fullmatch(digest)
        or digest != digest.lower()
    ):
        raise ValueError(
            "native_migration_receipt.canonical_sha256 must be lowercase SHA-256"
        )
    try:
        actual_digest = canonical_json_sha256(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "native_migration_receipt.payload is not canonical JSON"
        ) from exc
    if digest != actual_digest:
        raise ValueError("native_migration_receipt canonical SHA-256 mismatch")
    if payload["contract"] != MIGRATION_CONTRACT:
        raise ValueError("native_migration_receipt contract differs from runtime")

    initialized = payload["initialized_from_legacy"]
    if type(initialized) is not bool:
        raise ValueError(
            "native_migration_receipt initialized_from_legacy must be bool"
        )
    source_class = payload["source_class"]
    source_schema = payload["source_schema_sha256"]
    if initialized:
        if not isinstance(source_class, str) or not source_class:
            raise ValueError(
                "legacy migration receipt must name its source class"
            )
        if not isinstance(source_schema, str) or not _SHA256_RE.fullmatch(
            source_schema
        ):
            raise ValueError(
                "legacy migration receipt must bind its source schema"
            )
    elif source_class is not None or source_schema is not None:
        raise ValueError(
            "fresh migration receipt must use null source class/schema"
        )
    if not isinstance(payload["target_class"], str) or not payload["target_class"]:
        raise ValueError("native_migration_receipt target_class is invalid")
    target_schema = payload["target_schema_sha256"]
    if not isinstance(target_schema, str) or not _SHA256_RE.fullmatch(target_schema):
        raise ValueError("native_migration_receipt target schema is invalid")
    if payload["max_detail_tokens_per_sample"] != native_config[
        "max_detail_tokens"
    ]:
        raise ValueError(
            "native_migration_receipt max_detail_tokens differs from config"
        )
    if payload["source_tokens_per_detail_token"] != native_config[
        "source_tokens_per_detail_token"
    ]:
        raise ValueError(
            "native_migration_receipt source-token ratio differs from config"
        )

    mappings = payload["mappings"]
    if not isinstance(mappings, list) or not mappings:
        raise ValueError("native_migration_receipt mappings must be non-empty")
    allowed_actions = {
        "copy_exact",
        "copy_prefix_repeat_extension",
        "fresh_initialization",
        "zero_new_geometry",
        "zero_new_validity",
        "identity_new_layer_norm_affine",
        "zero_new_sample_context",
    }
    target_schema_rows: list[dict[str, Any]] = []
    seen_targets: set[str] = set()
    for index, raw_mapping in enumerate(mappings):
        if not isinstance(raw_mapping, Mapping):
            raise ValueError(
                f"native_migration_receipt mappings[{index}] must be an object"
            )
        mapping = dict(raw_mapping)
        required = {"target", "action", "shape", "dtype"}
        allowed = required | {"source"}
        if not required.issubset(mapping) or not set(mapping).issubset(allowed):
            raise ValueError(
                f"native_migration_receipt mappings[{index}] has invalid fields"
            )
        target = mapping["target"]
        action = mapping["action"]
        shape = mapping["shape"]
        dtype = mapping["dtype"]
        if not isinstance(target, str) or not target or target in seen_targets:
            raise ValueError(
                "native_migration_receipt mapping targets must be unique strings"
            )
        if action not in allowed_actions:
            raise ValueError(
                f"native_migration_receipt mapping action is unsupported: {action!r}"
            )
        if not isinstance(shape, list) or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            for dimension in shape
        ):
            raise ValueError(
                f"native_migration_receipt mapping {target!r} has invalid shape"
            )
        if not isinstance(dtype, str) or not dtype:
            raise ValueError(
                f"native_migration_receipt mapping {target!r} has invalid dtype"
            )
        source = mapping.get("source")
        needs_source = action in {"copy_exact", "copy_prefix_repeat_extension"}
        if needs_source and (not isinstance(source, str) or not source):
            raise ValueError(
                f"native_migration_receipt mapping {target!r} has invalid source"
            )
        if not needs_source and source is not None:
            raise ValueError(
                f"native_migration_receipt mapping {target!r} has invalid source"
            )
        seen_targets.add(target)
        target_schema_rows.append(
            {"name": target, "shape": list(shape), "dtype": dtype}
        )
    calculated_target_schema = canonical_json_sha256(
        sorted(target_schema_rows, key=lambda row: row["name"])
    )
    if calculated_target_schema != target_schema:
        raise ValueError(
            "native_migration_receipt target schema differs from its mappings"
        )
    return {"payload": payload, "canonical_sha256": digest}


def _validate_anyres_v2_metadata(
    metadata: Mapping[str, Any],
    *,
    rdt_cfg: RDTConfig,
    omvt_cfg: OMVTConfig | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if omvt_cfg is None:
        raise ValueError("anyres-v2 checkpoint requires omvt_config metadata")
    native = _strict_mapping(
        metadata.get(ANYRES_NATIVE_DETAIL_CONFIG_METADATA_KEY),
        field=ANYRES_NATIVE_DETAIL_CONFIG_METADATA_KEY,
        keys={"max_detail_tokens", "source_tokens_per_detail_token"},
    )
    native = {
        "max_detail_tokens": _positive_int(
            native["max_detail_tokens"],
            field="native_detail_config.max_detail_tokens",
        ),
        "source_tokens_per_detail_token": _positive_int(
            native["source_tokens_per_detail_token"],
            field="native_detail_config.source_tokens_per_detail_token",
        ),
    }
    cross = _strict_mapping(
        metadata.get(ANYRES_CROSS_ATTENTION_CONFIG_METADATA_KEY),
        field=ANYRES_CROSS_ATTENTION_CONFIG_METADATA_KEY,
        keys={"memory_dim", "n_heads", "dropout"},
    )
    memory_dim = _positive_int(
        cross["memory_dim"],
        field="vision_cross_attention_config.memory_dim",
    )
    n_heads = _positive_int(
        cross["n_heads"],
        field="vision_cross_attention_config.n_heads",
    )
    dropout = cross["dropout"]
    if (
        isinstance(dropout, bool)
        or not isinstance(dropout, (int, float))
        or not 0.0 <= float(dropout) < 1.0
    ):
        raise ValueError("vision_cross_attention_config.dropout must be in [0, 1)")
    if memory_dim != int(omvt_cfg.d_vision):
        raise ValueError(
            "vision cross-attention memory_dim must equal OMVT d_vision"
        )
    if int(rdt_cfg.d_model) % n_heads != 0:
        raise ValueError(
            "vision cross-attention n_heads must divide RDT d_model"
        )
    cross = {
        "memory_dim": memory_dim,
        "n_heads": n_heads,
        "dropout": float(dropout),
    }
    receipt = _validate_native_migration_receipt(
        metadata.get(ANYRES_MIGRATION_RECEIPT_METADATA_KEY),
        native_config=native,
    )
    return native, cross, receipt


def reconstruct_policy_from_checkpoint(
    path: str | Path,
    *,
    fallback_rdt_config: RDTConfig | None = None,
    require_vision: bool = False,
    metadata_override: dict[str, Any] | None = None,
    expected_metadata_sha256: str | None = None,
    expected_model_sha256: str | None = None,
) -> ReconstructedPolicy:
    """Build and strictly load an RDT/OMVT policy from verified artifacts.

    ``metadata_override`` lets a caller validate metadata once, then reuse that
    exact in-memory payload without deserializing ``meta.pt`` again.  The file
    is still hashed from a stable descriptor so ``expected_metadata_sha256``
    remains an artifact binding, not a trust in the path name.
    """

    checkpoint_dir = resolve_checkpoint_dir(path)
    if metadata_override is None:
        metadata, metadata_sha256 = load_verified_policy_metadata(
            checkpoint_dir,
            expected_sha256=expected_metadata_sha256,
        )
    else:
        if not isinstance(metadata_override, dict):
            raise TypeError("metadata_override must be a dict")
        meta_path = checkpoint_dir / "meta.pt"
        if meta_path.exists() or meta_path.is_symlink():
            metadata_sha256 = _verified_file_sha256(
                meta_path,
                expected_sha256=expected_metadata_sha256,
            )
        elif expected_metadata_sha256 is not None:
            raise FileNotFoundError(f"checkpoint metadata not found: {meta_path}")
        else:
            metadata_sha256 = ""
        metadata = dict(metadata_override)
    if require_vision and "rdt_config" not in metadata:
        raise ValueError(
            "image-conditioned OCR post-training requires checkpoint "
            "rdt_config metadata; tensor shapes alone do not recover behavioral "
            "settings such as recurrent depth and context length"
        )
    rdt_cfg = _rdt_config_from_metadata(metadata, fallback_rdt_config)
    omvt_cfg = _omvt_config_from_metadata(metadata)
    visual_input_contract = _visual_input_contract_from_metadata(metadata)
    anyres_metadata_keys = {
        ANYRES_NATIVE_DETAIL_CONFIG_METADATA_KEY,
        ANYRES_CROSS_ATTENTION_CONFIG_METADATA_KEY,
        ANYRES_MIGRATION_RECEIPT_METADATA_KEY,
    }
    if visual_input_contract != DOL_OCR_ANYRES_V2 and (
        anyres_metadata_keys & set(metadata)
    ):
        raise ValueError(
            "native-detail metadata conflicts with the checkpoint visual "
            "input contract"
        )
    native_detail_config = None
    cross_attention_config = None
    native_migration_receipt = None
    if visual_input_contract == DOL_OCR_ANYRES_V2:
        (
            native_detail_config,
            cross_attention_config,
            native_migration_receipt,
        ) = _validate_anyres_v2_metadata(
            metadata,
            rdt_cfg=rdt_cfg,
            omvt_cfg=omvt_cfg,
        )

    state, model_sha256 = _verified_torch_load(
        checkpoint_dir / "model.pt",
        expected_sha256=expected_model_sha256,
    )
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint model state must be a dict: {checkpoint_dir}")
    has_omvt_weights = any(str(key).startswith("vision.omvt.") for key in state)
    has_native_detail_weights = any(
        str(key).startswith("vision.native_detail_tower.") for key in state
    )
    has_cross_attention_weights = any(
        str(key).startswith("vision_cross_attention.") for key in state
    )
    if visual_input_contract == DOL_OCR_ANYRES_V2:
        if not has_omvt_weights:
            raise ValueError(
                "anyres-v2 checkpoint must retain the legacy OMVT state"
            )
        if not has_native_detail_weights or not has_cross_attention_weights:
            raise ValueError(
                "anyres-v2 checkpoint is missing native-detail or cross-attention "
                "weights"
            )
    elif has_native_detail_weights or has_cross_attention_weights:
        raise ValueError(
            "checkpoint state contains anyres-v2 modules but metadata does not "
            "declare dol_ocr_anyres_v2"
        )
    if has_omvt_weights and omvt_cfg is None:
        raise ValueError(
            "checkpoint contains vision.omvt weights but has no omvt_config metadata; "
            "refusing to guess tower geometry"
        )
    if require_vision and not has_omvt_weights:
        raise ValueError(
            "image-conditioned OCR post-training requires vision.omvt weights"
        )

    model = RDTForCausalLM(rdt_cfg)
    if has_omvt_weights:
        assert omvt_cfg is not None
        model.vision._omvt_cfg = omvt_cfg
        model.vision.omvt = OMVTInjector(rdt_cfg, omvt_cfg)
    if visual_input_contract == DOL_OCR_ANYRES_V2:
        assert omvt_cfg is not None
        assert native_detail_config is not None
        assert cross_attention_config is not None
        assert native_migration_receipt is not None
        reconstruction_receipt = model.vision.install_native_detail_tower(
            omvt_cfg,
            max_detail_tokens=native_detail_config["max_detail_tokens"],
            ratio=native_detail_config["source_tokens_per_detail_token"],
            initialize_from_legacy=False,
        )
        model.install_vision_cross_attention(**cross_attention_config)
        saved_receipt_payload = native_migration_receipt["payload"]
        receipt_mismatches = []
        for field in (
            "target_class",
            "target_schema_sha256",
            "max_detail_tokens_per_sample",
            "source_tokens_per_detail_token",
        ):
            if saved_receipt_payload[field] != reconstruction_receipt.canonical_payload()[
                field
            ]:
                receipt_mismatches.append(field)
        if receipt_mismatches:
            raise ValueError(
                "native_migration_receipt target geometry differs from runtime: "
                + ", ".join(receipt_mismatches)
            )

    # strict=True is the safety property of this loader.  A missing projector,
    # silently skipped tower, wrong tokenizer geometry, or backend mismatch must
    # stop before an optimizer is allocated.
    model.load_state_dict(state, strict=True)
    return ReconstructedPolicy(
        model=model,
        rdt_config=rdt_cfg,
        omvt_config=omvt_cfg,
        metadata=metadata,
        checkpoint_dir=checkpoint_dir,
        model_sha256=model_sha256,
        metadata_sha256=metadata_sha256,
        native_detail_config=native_detail_config,
        vision_cross_attention_config=cross_attention_config,
        native_migration_receipt=native_migration_receipt,
    )


__all__ = [
    "ANYRES_CROSS_ATTENTION_CONFIG_METADATA_KEY",
    "ANYRES_MIGRATION_RECEIPT_METADATA_KEY",
    "ANYRES_NATIVE_DETAIL_CONFIG_METADATA_KEY",
    "OCR_GRPO_CONTRACT_VERSION",
    "ReconstructedPolicy",
    "load_verified_checkpoint_metadata_envelope",
    "load_verified_policy_metadata",
    "load_verified_torch_payload",
    "reconstruct_policy_from_checkpoint",
    "validate_visual_ocr_source_contract",
    "verified_file_sha256",
]
