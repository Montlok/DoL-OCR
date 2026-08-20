# -*- coding: utf-8 -*-
"""Admission and derived-checkpoint contract for anyres OCR SFT."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import re
from typing import Any

from Model.config import OMVTConfig, RDTConfig
from Model.model import RDTForCausalLM
from Model.ocr.anyres_preprocess_contract import (
    validate_anyres_preprocess_contract,
)
from Model.ocr.position_contract import (
    BOUNDARY_V1,
    OCR_POSITION_CONTRACT_METADATA_VERSION,
)
from Model.ocr.tokenization import (
    OCR_NATIVE_TARGET_ENCODING,
    OCR_TOKENIZATION_CONTRACT_VERSION,
    canonical_json_sha256,
)
from Model.ocr.visual_input_contract import (
    DOL_OCR_ANYRES_V2,
    DOL_OCR_LINE_LETTERBOX_224_V1,
    ocr_visual_input_contract_version,
)
from Model.posttrain.checkpointing import (
    load_verified_policy_metadata,
    reconstruct_policy_from_checkpoint,
)
from Model.posttrain.ocr_anyres_grpo import AnyresGRPOAdmission
from Model.posttrain.ocr_joint_eval import (
    DEPLOYMENT_BUCKET_WEIGHTS,
    dual_baseline_joint_eligibility,
    joint_eval_eligibility,
)
from Model.posttrain.release_contract import (
    MONTLOK_DOL_1_2_OCR_REPOSITORY_ID,
    MONTLOK_DOL_1_2_OCR_REVISION,
    STREAMING_V2_RELEASE_SOURCE_CONTRACT_KIND,
    VISUAL_SOURCE_CONTRACT_STREAMING_V2_RELEASE,
    admit_visual_ocr_source,
)
from Model.posttrain.text_replay import validate_text_replay_partition_contract


OCR_JOINT_RUN_CONTRACT_SCHEMA_VERSION = 1
OCR_JOINT_RUN_CONTRACT_KIND = "dol_ocr_anyres_joint_sft_run_v1"
OCR_JOINT_RUN_CONTRACT_METADATA_KEY = "ocr_joint_run_contract"
OCR_JOINT_RUN_CONTRACT_SHA256_METADATA_KEY = (
    "ocr_joint_run_contract_sha256"
)
PARENT_ANYRES_CHECKPOINT_METADATA_KEY = "parent_anyres_checkpoint"
VISUAL_STAGE_RESULT_KIND = "dol_ocr_anyres_visual_stage_result_v1"
JOINT_STAGE_RESULT_KIND = "dol_ocr_anyres_joint_stage_result_v1"
OCR_JOINT_STAGES = ("visual", "joint")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUN_KEYS = {
    "schema_version",
    "kind",
    "metadata_contract",
    "trainability_contract",
    "contract_canonical_sha256",
}
_METADATA_CONTRACT_KEYS = {
    "phase",
    "stage",
    PARENT_ANYRES_CHECKPOINT_METADATA_KEY,
    "source_visual_lineage",
    "source_max_seq_len",
    "target_max_seq_len",
    "rdt_config",
    "omvt_config",
    "tokenizer_contract",
    "tokenizer_vocab_extent",
    "ocr_target_encoding",
    "ocr_tokenization_contract_version",
    "ocr_visual_input_contract",
    "ocr_visual_input_contract_version",
    "anyres_preprocess_contract",
    "anyres_preprocess_contract_sha256",
    "native_detail_config",
    "vision_cross_attention_config",
    "native_migration_receipt",
    "ocr_position_contract",
    "ocr_position_contract_version",
    "reverse_loss_enabled",
}
_PARENT_CHECKPOINT_KEYS = {
    "path",
    "model_sha256",
    "metadata_sha256",
    "visual_run_contract_sha256",
    "visual_stage_result_canonical_sha256",
    "best_validation_sha256",
}
_VISUAL_STAGE_RESULT_KEYS = {
    "schema_version",
    "kind",
    "run_contract_sha256",
    "final_checkpoint",
    "best_eligible_checkpoint",
    "baseline_validation_sha256",
    "final_validation_sha256",
    "best_validation_sha256",
    "last_eligibility",
    "promotion_allowed",
    "completed_cycles",
    "stop_reason",
    "frozen_parameter_contract",
    "runtime_source_receipt",
    "runtime_environment",
    "canonical_sha256",
}
_JOINT_STAGE_RESULT_KEYS = {
    "schema_version",
    "kind",
    "run_contract_sha256",
    "parent_anyres_checkpoint",
    "final_checkpoint",
    "best_eligible_checkpoint",
    "baseline_validation_sha256",
    "historical_visual_validation_sha256",
    "final_validation_sha256",
    "best_validation_sha256",
    "last_eligibility",
    "grpo_promotion_allowed",
    "completed_cycles",
    "stop_reason",
    "optimizer_contract_sha256",
    "quota_sampler_state_sha256",
    "text_replay_cursor_sha256",
    "runtime_source_receipt",
    "runtime_environment",
    "canonical_sha256",
}


@dataclass
class PreparedAnyresPolicy:
    policy: RDTForCausalLM
    run_contract: dict[str, Any]
    metadata_template: dict[str, Any]
    checkpoint_dir: Path
    rdt_config: RDTConfig
    omvt_config: OMVTConfig
    model_sha256: str | None = None
    metadata_sha256: str | None = None


@dataclass
class PreparedAnyresGRPO:
    policy: RDTForCausalLM
    reference: RDTForCausalLM | None
    admission: AnyresGRPOAdmission
    joint_stage_result: dict[str, Any]
    checkpoint_dir: Path
    metadata: dict[str, Any]
    model_sha256: str
    metadata_sha256: str


def _exact_mapping(
    value: object,
    *,
    field: str,
    keys: set[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    result = dict(value)
    if set(result) != keys:
        raise ValueError(
            f"{field} keys differ from contract: "
            f"missing={sorted(keys - set(result))} "
            f"extra={sorted(set(result) - keys)}"
        )
    return result


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return int(value)


def _stage(value: object) -> str:
    if not isinstance(value, str) or value not in OCR_JOINT_STAGES:
        raise ValueError(f"stage must be one of {OCR_JOINT_STAGES}")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be lowercase SHA-256")
    return value


def _checkpoint_identity(value: object, field: str) -> dict[str, str]:
    identity = _exact_mapping(
        value,
        field=field,
        keys={"path", "model_sha256", "metadata_sha256"},
    )
    path = identity["path"]
    if not isinstance(path, str) or not path:
        raise ValueError(f"{field}.path must be a non-empty string")
    resolved_path = str(Path(path).resolve())
    if path != resolved_path:
        raise ValueError(f"{field}.path must be absolute and canonical")
    return {
        "path": resolved_path,
        "model_sha256": _sha256(
            identity["model_sha256"], f"{field}.model_sha256"
        ),
        "metadata_sha256": _sha256(
            identity["metadata_sha256"], f"{field}.metadata_sha256"
        ),
    }


def _embedded_contract_sha256(value: object, field: str) -> str:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    payload = dict(value)
    digest = _sha256(payload.pop("contract_sha256", None), f"{field}.contract_sha256")
    if canonical_json_sha256(payload) != digest:
        raise ValueError(f"{field}.contract_sha256 is not canonical")
    return digest


def _validate_parent_anyres_checkpoint(
    value: object,
    *,
    stage: str,
) -> dict[str, str] | None:
    if stage == "visual":
        if value is not None:
            raise ValueError("visual stage parent_anyres_checkpoint must be null")
        return None
    parent = _exact_mapping(
        value,
        field=PARENT_ANYRES_CHECKPOINT_METADATA_KEY,
        keys=_PARENT_CHECKPOINT_KEYS,
    )
    path = parent["path"]
    if not isinstance(path, str) or not path:
        raise ValueError("parent_anyres_checkpoint.path must be non-empty")
    resolved_path = str(Path(path).resolve())
    if path != resolved_path:
        raise ValueError("parent_anyres_checkpoint.path must be absolute and canonical")
    return {
        "path": resolved_path,
        "model_sha256": _sha256(
            parent["model_sha256"],
            "parent_anyres_checkpoint.model_sha256",
        ),
        "metadata_sha256": _sha256(
            parent["metadata_sha256"],
            "parent_anyres_checkpoint.metadata_sha256",
        ),
        "visual_run_contract_sha256": _sha256(
            parent["visual_run_contract_sha256"],
            "parent_anyres_checkpoint.visual_run_contract_sha256",
        ),
        "visual_stage_result_canonical_sha256": _sha256(
            parent["visual_stage_result_canonical_sha256"],
            "parent_anyres_checkpoint.visual_stage_result_canonical_sha256",
        ),
        "best_validation_sha256": _sha256(
            parent["best_validation_sha256"],
            "parent_anyres_checkpoint.best_validation_sha256",
        ),
    }


def _validate_visual_stage_result(value: object) -> dict[str, Any]:
    result = _exact_mapping(
        value,
        field="visual_stage_result",
        keys=_VISUAL_STAGE_RESULT_KEYS,
    )
    if result["schema_version"] != 1:
        raise ValueError("unsupported visual stage result schema_version")
    if result["kind"] != VISUAL_STAGE_RESULT_KIND:
        raise ValueError("unsupported visual stage result kind")
    digest = _sha256(result["canonical_sha256"], "visual stage result hash")
    unhashed = {
        key: deepcopy(item)
        for key, item in result.items()
        if key != "canonical_sha256"
    }
    if canonical_json_sha256(unhashed) != digest:
        raise ValueError("visual stage result canonical SHA256 mismatch")
    _sha256(result["run_contract_sha256"], "visual result run contract hash")
    _checkpoint_identity(result["final_checkpoint"], "final_checkpoint")
    best = result["best_eligible_checkpoint"]
    if result["promotion_allowed"] is not True or best is None:
        raise ValueError("visual stage result does not allow joint promotion")
    _checkpoint_identity(best, "best_eligible_checkpoint")
    for field in (
        "baseline_validation_sha256",
        "final_validation_sha256",
        "best_validation_sha256",
    ):
        _sha256(result[field], f"visual stage result {field}")
    _positive_int(result["completed_cycles"], "completed_cycles")
    if not isinstance(result["stop_reason"], str) or not result["stop_reason"]:
        raise ValueError("visual stage result stop_reason must be non-empty")
    eligibility = result["last_eligibility"]
    if eligibility is not None and not isinstance(eligibility, Mapping):
        raise ValueError("visual stage result last_eligibility must be an object/null")
    frozen = _exact_mapping(
        result["frozen_parameter_contract"],
        field="visual stage frozen_parameter_contract",
        keys={"language_sha256", "legacy_omvt_sha256"},
    )
    _sha256(frozen["language_sha256"], "visual stage language SHA256")
    _sha256(frozen["legacy_omvt_sha256"], "visual stage legacy OMVT SHA256")
    if not isinstance(result["runtime_source_receipt"], Mapping):
        raise ValueError("visual stage runtime_source_receipt must be an object")
    if not isinstance(result["runtime_environment"], Mapping):
        raise ValueError("visual stage runtime_environment must be an object")
    return deepcopy(result)


def _validate_joint_stage_result(value: object) -> dict[str, Any]:
    result = _exact_mapping(
        value,
        field="joint_stage_result",
        keys=_JOINT_STAGE_RESULT_KEYS,
    )
    if result["schema_version"] != 1:
        raise ValueError("unsupported joint stage result schema_version")
    if result["kind"] != JOINT_STAGE_RESULT_KIND:
        raise ValueError("unsupported joint stage result kind")
    digest = _sha256(result["canonical_sha256"], "joint stage result hash")
    unhashed = {
        key: deepcopy(item)
        for key, item in result.items()
        if key != "canonical_sha256"
    }
    if canonical_json_sha256(unhashed) != digest:
        raise ValueError("joint stage result canonical SHA256 mismatch")
    _sha256(result["run_contract_sha256"], "joint result run contract hash")
    _validate_parent_anyres_checkpoint(
        result["parent_anyres_checkpoint"],
        stage="joint",
    )
    _checkpoint_identity(result["final_checkpoint"], "final_checkpoint")
    best = result["best_eligible_checkpoint"]
    promotion = result["grpo_promotion_allowed"]
    if type(promotion) is not bool:
        raise ValueError("grpo_promotion_allowed must be bool")
    if promotion:
        if best is None or result["best_validation_sha256"] is None:
            raise ValueError("GRPO promotion requires an eligible best checkpoint")
        _checkpoint_identity(best, "best_eligible_checkpoint")
        _sha256(result["best_validation_sha256"], "best_validation_sha256")
    elif best is not None or result["best_validation_sha256"] is not None:
        raise ValueError("non-promotable joint result must not publish a best")
    for field in (
        "baseline_validation_sha256",
        "historical_visual_validation_sha256",
        "final_validation_sha256",
        "optimizer_contract_sha256",
        "quota_sampler_state_sha256",
        "text_replay_cursor_sha256",
    ):
        _sha256(result[field], f"joint stage result {field}")
    _positive_int(result["completed_cycles"], "completed_cycles")
    if not isinstance(result["stop_reason"], str) or not result["stop_reason"]:
        raise ValueError("joint stage result stop_reason must be non-empty")
    eligibility = result["last_eligibility"]
    if eligibility is not None and not isinstance(eligibility, Mapping):
        raise ValueError("joint stage result last_eligibility must be object/null")
    if not isinstance(result["runtime_source_receipt"], Mapping):
        raise ValueError("joint stage runtime_source_receipt must be an object")
    if not isinstance(result["runtime_environment"], Mapping):
        raise ValueError("joint stage runtime_environment must be an object")
    return deepcopy(result)


def _validated_source_lineage(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("source visual lineage must be an object")
    lineage = dict(value)
    required = {
        "source_contract_kind": STREAMING_V2_RELEASE_SOURCE_CONTRACT_KIND,
        "source_repository_id": MONTLOK_DOL_1_2_OCR_REPOSITORY_ID,
        "source_revision": MONTLOK_DOL_1_2_OCR_REVISION,
        "ocr_visual_input_contract": DOL_OCR_LINE_LETTERBOX_224_V1,
        "ocr_visual_input_contract_version": 1,
        "ocr_position_contract": BOUNDARY_V1,
        "ocr_position_contract_version": (
            OCR_POSITION_CONTRACT_METADATA_VERSION
        ),
    }
    for field, expected in required.items():
        if lineage.get(field) != expected:
            raise ValueError(
                f"official source lineage mismatch for {field}: "
                f"{lineage.get(field)!r} != {expected!r}"
            )
    for field in (
        "source_checkpoint_model_sha256",
        "source_checkpoint_metadata_sha256",
    ):
        digest = lineage.get(field)
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"official source lineage has invalid {field}")
    if not isinstance(lineage.get("source_checkpoint"), str):
        raise ValueError("official source lineage has no source_checkpoint")
    return lineage


def _normalize_native_and_cross_configs(
    native_detail_config: object,
    vision_cross_attention_config: object,
    *,
    preprocess: Mapping[str, Any],
    rdt_cfg: RDTConfig,
    omvt_cfg: OMVTConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    native = _exact_mapping(
        native_detail_config,
        field="native_detail_config",
        keys={"max_detail_tokens", "source_tokens_per_detail_token"},
    )
    native = {
        "max_detail_tokens": _positive_int(
            native["max_detail_tokens"], "native_detail_config.max_detail_tokens"
        ),
        "source_tokens_per_detail_token": _positive_int(
            native["source_tokens_per_detail_token"],
            "native_detail_config.source_tokens_per_detail_token",
        ),
    }
    detail_budget = preprocess["budgets"]["detail"]
    if native["max_detail_tokens"] != detail_budget["max_tokens_per_view"]:
        raise ValueError("native max_detail_tokens differs from preprocess contract")
    if (
        native["source_tokens_per_detail_token"]
        != detail_budget["source_tokens_per_detail_token"]
    ):
        raise ValueError("native source-token ratio differs from preprocess contract")

    cross = _exact_mapping(
        vision_cross_attention_config,
        field="vision_cross_attention_config",
        keys={"memory_dim", "n_heads", "dropout"},
    )
    memory_dim = _positive_int(
        cross["memory_dim"], "vision_cross_attention_config.memory_dim"
    )
    n_heads = _positive_int(
        cross["n_heads"], "vision_cross_attention_config.n_heads"
    )
    dropout = cross["dropout"]
    if (
        isinstance(dropout, bool)
        or not isinstance(dropout, (int, float))
        or not 0.0 <= float(dropout) < 1.0
    ):
        raise ValueError("vision_cross_attention_config.dropout must be in [0, 1)")
    if memory_dim != int(omvt_cfg.d_vision):
        raise ValueError("cross-attention memory_dim must equal OMVT d_vision")
    if int(rdt_cfg.d_model) % n_heads != 0:
        raise ValueError("cross-attention n_heads must divide RDT d_model")
    return native, {
        "memory_dim": memory_dim,
        "n_heads": n_heads,
        "dropout": float(dropout),
    }


def _set_stage_trainability(
    model: RDTForCausalLM,
    stage: str,
) -> dict[str, Any]:
    stage = _stage(stage)
    model.reverse_loss_enabled = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    native_vision = (
        model.vision.native_detail_tower,
        model.vision_cross_attention,
    )
    for module in native_vision:
        if module is None:
            raise RuntimeError("anyres policy is missing an active visual module")
        for parameter in module.parameters():
            parameter.requires_grad_(True)

    policy = "native_detail_and_bridge_only_v1"
    if stage == "joint":
        policy = "joint_language_and_active_visual_v1"
        if model.vision.omvt is None:
            raise RuntimeError("joint stage requires installed legacy OMVT")
        for parameter in model.vision.omvt.parameters():
            parameter.requires_grad_(True)
        language_modules = (
            model.embed,
            model.prelude,
            model.recurrent,
            model.coda,
            model.final_norm,
            model.lm_head,
            model.reverse_head,
        )
        for module in language_modules:
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)

    trainable_names = sorted(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "stage": stage,
        "policy": policy,
        "trainable_parameter_names_sha256": canonical_json_sha256(trainable_names),
        "trainable_parameter_count": trainable_count,
        "frozen_parameter_count": total_count - trainable_count,
        "total_parameter_count": total_count,
    }


def _receipt_wrapper(receipt: object) -> dict[str, Any]:
    payload = receipt.canonical_payload()
    digest = receipt.canonical_sha256
    if digest != canonical_json_sha256(payload):
        raise RuntimeError("native migration receipt is not canonical")
    return {"payload": payload, "canonical_sha256": digest}


def _run_contract(
    metadata_contract: Mapping[str, Any],
    trainability_contract: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": OCR_JOINT_RUN_CONTRACT_SCHEMA_VERSION,
        "kind": OCR_JOINT_RUN_CONTRACT_KIND,
        "metadata_contract": deepcopy(dict(metadata_contract)),
        "trainability_contract": deepcopy(dict(trainability_contract)),
    }
    payload["contract_canonical_sha256"] = canonical_json_sha256(payload)
    return payload


def validate_ocr_joint_run_contract(value: object) -> dict[str, Any]:
    run = _exact_mapping(value, field="ocr_joint_run_contract", keys=_RUN_KEYS)
    if run["schema_version"] != OCR_JOINT_RUN_CONTRACT_SCHEMA_VERSION:
        raise ValueError("unsupported OCR joint run contract schema_version")
    if run["kind"] != OCR_JOINT_RUN_CONTRACT_KIND:
        raise ValueError("unsupported OCR joint run contract kind")
    digest = run["contract_canonical_sha256"]
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise ValueError("OCR joint run contract SHA256 is invalid")
    unhashed = {key: deepcopy(item) for key, item in run.items() if key != "contract_canonical_sha256"}
    if canonical_json_sha256(unhashed) != digest:
        raise ValueError("OCR joint run contract canonical SHA256 mismatch")

    metadata = _exact_mapping(
        run["metadata_contract"],
        field="ocr_joint_run_contract.metadata_contract",
        keys=_METADATA_CONTRACT_KEYS,
    )
    stage = _stage(metadata["stage"])
    _validate_parent_anyres_checkpoint(
        metadata[PARENT_ANYRES_CHECKPOINT_METADATA_KEY],
        stage=stage,
    )
    if metadata["phase"] != "ocr_anyres_sft":
        raise ValueError("OCR joint metadata phase must be ocr_anyres_sft")
    lineage = _validated_source_lineage(metadata["source_visual_lineage"])
    del lineage
    preprocess = validate_anyres_preprocess_contract(
        metadata["anyres_preprocess_contract"]
    )
    if (
        metadata["anyres_preprocess_contract_sha256"]
        != preprocess["contract_canonical_sha256"]
    ):
        raise ValueError("preprocess contract SHA differs from embedded payload")
    if metadata["ocr_visual_input_contract"] != DOL_OCR_ANYRES_V2 or (
        metadata["ocr_visual_input_contract_version"]
        != ocr_visual_input_contract_version(DOL_OCR_ANYRES_V2)
    ):
        raise ValueError("derived policy does not declare anyres-v2")
    if metadata["ocr_position_contract"] != BOUNDARY_V1 or (
        metadata["ocr_position_contract_version"]
        != OCR_POSITION_CONTRACT_METADATA_VERSION
    ):
        raise ValueError("derived policy does not declare boundary_v1 positions")
    if metadata["reverse_loss_enabled"] is not False:
        raise ValueError("OCR joint SFT must disable reverse-LM loss")
    source_max = _positive_int(metadata["source_max_seq_len"], "source_max_seq_len")
    target_max = _positive_int(metadata["target_max_seq_len"], "target_max_seq_len")
    if target_max <= source_max:
        raise ValueError("target max_seq_len must explicitly extend the source")
    if preprocess["budgets"]["context"]["max_sequence_tokens"] != target_max:
        raise ValueError("preprocess max_seq_len differs from derived policy")
    rdt_values = dict(metadata["rdt_config"])
    omvt_values = dict(metadata["omvt_config"])
    for key in ("vertical_patch", "horizontal_patch", "square_patch", "layout_patch"):
        if key in omvt_values:
            omvt_values[key] = tuple(omvt_values[key])
    rdt_cfg = RDTConfig(**rdt_values)
    omvt_cfg = OMVTConfig(**omvt_values)
    if rdt_cfg.max_seq_len != target_max:
        raise ValueError("target RDT config max_seq_len differs from contract")
    _normalize_native_and_cross_configs(
        metadata["native_detail_config"],
        metadata["vision_cross_attention_config"],
        preprocess=preprocess,
        rdt_cfg=rdt_cfg,
        omvt_cfg=omvt_cfg,
    )
    wrapper = _exact_mapping(
        metadata["native_migration_receipt"],
        field="native_migration_receipt",
        keys={"payload", "canonical_sha256"},
    )
    if wrapper["canonical_sha256"] != canonical_json_sha256(wrapper["payload"]):
        raise ValueError("native migration receipt SHA256 mismatch")
    if metadata["ocr_target_encoding"] != OCR_NATIVE_TARGET_ENCODING or (
        metadata["ocr_tokenization_contract_version"]
        != OCR_TOKENIZATION_CONTRACT_VERSION
    ):
        raise ValueError("derived policy tokenizer mode is not strict native")
    if not isinstance(metadata["tokenizer_contract"], Mapping):
        raise ValueError("tokenizer_contract must be an object")
    _positive_int(metadata["tokenizer_vocab_extent"], "tokenizer_vocab_extent")

    trainability = _exact_mapping(
        run["trainability_contract"],
        field="ocr_joint_run_contract.trainability_contract",
        keys={
            "stage",
            "policy",
            "trainable_parameter_names_sha256",
            "trainable_parameter_count",
            "frozen_parameter_count",
            "total_parameter_count",
        },
    )
    if trainability["stage"] != stage:
        raise ValueError("trainability stage differs from metadata stage")
    expected_policy = (
        "native_detail_and_bridge_only_v1"
        if stage == "visual"
        else "joint_language_and_active_visual_v1"
    )
    if trainability["policy"] != expected_policy:
        raise ValueError("trainability policy differs from metadata stage")
    name_digest = trainability["trainable_parameter_names_sha256"]
    if not isinstance(name_digest, str) or _SHA256_RE.fullmatch(name_digest) is None:
        raise ValueError("trainable parameter-name SHA256 is invalid")
    trainable = _positive_int(
        trainability["trainable_parameter_count"], "trainable_parameter_count"
    )
    frozen = trainability["frozen_parameter_count"]
    total = _positive_int(trainability["total_parameter_count"], "total_parameter_count")
    if isinstance(frozen, bool) or not isinstance(frozen, int) or frozen < 0:
        raise ValueError("frozen_parameter_count must be a non-negative integer")
    if trainable + frozen != total:
        raise ValueError("trainability parameter counts are inconsistent")
    return deepcopy(run)


def admit_and_prepare_anyres_policy(
    source_checkpoint: str | Path,
    source_contract_kind: str,
    tokenizer_contract: Mapping[str, Any],
    tokenizer_vocab_extent: int,
    preprocess_contract: Mapping[str, Any],
    stage: str,
    target_max_seq_len: int,
    native_detail_config: Mapping[str, Any],
    vision_cross_attention_config: Mapping[str, Any],
) -> PreparedAnyresPolicy:
    if source_contract_kind != VISUAL_SOURCE_CONTRACT_STREAMING_V2_RELEASE:
        raise ValueError("anyres derivation requires the reviewed streaming-v2 release")
    stage = _stage(stage)
    if stage != "visual":
        raise ValueError(
            "fresh anyres admission must start at visual stage; use "
            "promote_visual_policy_to_joint for joint stage"
        )
    tokenizer_vocab_extent = _positive_int(
        tokenizer_vocab_extent, "tokenizer_vocab_extent"
    )
    if not isinstance(tokenizer_contract, Mapping):
        raise ValueError("tokenizer_contract must be an object")
    tokenizer_contract = deepcopy(dict(tokenizer_contract))
    preprocess = validate_anyres_preprocess_contract(preprocess_contract)
    target_max_seq_len = _positive_int(target_max_seq_len, "target_max_seq_len")
    if preprocess["budgets"]["context"]["max_sequence_tokens"] != target_max_seq_len:
        raise ValueError("target max_seq_len differs from preprocess contract")

    admitted = admit_visual_ocr_source(
        source_contract_kind,
        checkpoint_dir=source_checkpoint,
        metadata=None,
        runtime_native_tokenization_contract=tokenizer_contract,
        tokenizer_vocab_extent=tokenizer_vocab_extent,
        requested_visual_input_contract=DOL_OCR_LINE_LETTERBOX_224_V1,
    )
    lineage = _validated_source_lineage(admitted["lineage"])
    restored = reconstruct_policy_from_checkpoint(
        source_checkpoint,
        require_vision=True,
        metadata_override=admitted["metadata"],
        expected_metadata_sha256=lineage["source_checkpoint_metadata_sha256"],
        expected_model_sha256=lineage["source_checkpoint_model_sha256"],
    )
    if restored.metadata != admitted["metadata"]:
        raise ValueError("atomic source reconstruction metadata differs from admission")
    if restored.model_sha256 != lineage["source_checkpoint_model_sha256"] or (
        restored.metadata_sha256
        != lineage["source_checkpoint_metadata_sha256"]
    ):
        raise ValueError("atomic source reconstruction hashes differ from lineage")
    if restored.omvt_config is None or restored.model.vision.omvt is None:
        raise ValueError("official source reconstruction has no legacy OMVT")
    if restored.model.vision.native_detail_tower is not None or (
        restored.model.vision_cross_attention is not None
    ):
        raise ValueError("official source unexpectedly contains anyres-v2 modules")

    source_max_seq_len = int(restored.rdt_config.max_seq_len)
    if target_max_seq_len <= source_max_seq_len:
        raise ValueError("target max_seq_len must explicitly extend the source")
    restored.rdt_config.max_seq_len = target_max_seq_len
    restored.model.cfg.max_seq_len = target_max_seq_len
    native, cross = _normalize_native_and_cross_configs(
        native_detail_config,
        vision_cross_attention_config,
        preprocess=preprocess,
        rdt_cfg=restored.rdt_config,
        omvt_cfg=restored.omvt_config,
    )
    receipt = restored.model.vision.install_native_detail_tower(
        restored.omvt_config,
        max_detail_tokens=native["max_detail_tokens"],
        ratio=native["source_tokens_per_detail_token"],
        initialize_from_legacy=True,
    )
    bridge = restored.model.install_vision_cross_attention(**cross)
    if int(bridge.output_projection.weight.count_nonzero().item()) != 0:
        raise RuntimeError("new anyres bridge must start as an exact zero residual")
    receipt_wrapper = _receipt_wrapper(receipt)
    trainability = _set_stage_trainability(restored.model, stage)

    metadata_contract = {
        "phase": "ocr_anyres_sft",
        "stage": stage,
        PARENT_ANYRES_CHECKPOINT_METADATA_KEY: None,
        "source_visual_lineage": deepcopy(lineage),
        "source_max_seq_len": source_max_seq_len,
        "target_max_seq_len": target_max_seq_len,
        "rdt_config": asdict(restored.rdt_config),
        "omvt_config": asdict(restored.omvt_config),
        "tokenizer_contract": tokenizer_contract,
        "tokenizer_vocab_extent": tokenizer_vocab_extent,
        "ocr_target_encoding": OCR_NATIVE_TARGET_ENCODING,
        "ocr_tokenization_contract_version": OCR_TOKENIZATION_CONTRACT_VERSION,
        "ocr_visual_input_contract": DOL_OCR_ANYRES_V2,
        "ocr_visual_input_contract_version": (
            ocr_visual_input_contract_version(DOL_OCR_ANYRES_V2)
        ),
        "anyres_preprocess_contract": deepcopy(preprocess),
        "anyres_preprocess_contract_sha256": preprocess[
            "contract_canonical_sha256"
        ],
        "native_detail_config": native,
        "vision_cross_attention_config": cross,
        "native_migration_receipt": receipt_wrapper,
        "ocr_position_contract": BOUNDARY_V1,
        "ocr_position_contract_version": OCR_POSITION_CONTRACT_METADATA_VERSION,
        "reverse_loss_enabled": False,
    }
    run_contract = _run_contract(metadata_contract, trainability)
    validate_ocr_joint_run_contract(run_contract)
    metadata_template = deepcopy(metadata_contract)
    metadata_template[OCR_JOINT_RUN_CONTRACT_METADATA_KEY] = deepcopy(run_contract)
    metadata_template[OCR_JOINT_RUN_CONTRACT_SHA256_METADATA_KEY] = run_contract[
        "contract_canonical_sha256"
    ]
    return PreparedAnyresPolicy(
        policy=restored.model,
        run_contract=run_contract,
        metadata_template=metadata_template,
        checkpoint_dir=restored.checkpoint_dir,
        rdt_config=restored.rdt_config,
        omvt_config=restored.omvt_config,
    )


def promote_visual_policy_to_joint(
    checkpoint: str | Path,
    expected_model_sha256: str,
    expected_metadata_sha256: str,
    visual_stage_result: Mapping[str, Any],
) -> PreparedAnyresPolicy:
    """Promote only the immutable best eligible visual checkpoint to joint SFT."""

    expected_model_sha256 = _sha256(
        expected_model_sha256,
        "expected visual checkpoint model SHA256",
    )
    expected_metadata_sha256 = _sha256(
        expected_metadata_sha256,
        "expected visual checkpoint metadata SHA256",
    )
    result = _validate_visual_stage_result(visual_stage_result)
    best_identity = _checkpoint_identity(
        result["best_eligible_checkpoint"],
        "best_eligible_checkpoint",
    )
    requested_path = str(Path(checkpoint).resolve())
    if best_identity != {
        "path": requested_path,
        "model_sha256": expected_model_sha256,
        "metadata_sha256": expected_metadata_sha256,
    }:
        raise ValueError(
            "promotion checkpoint identity is not VISUAL_STAGE_RESULT best"
        )

    metadata, metadata_sha256 = load_verified_policy_metadata(
        checkpoint,
        expected_sha256=expected_metadata_sha256,
    )
    saved_visual_run = validate_ocr_joint_run_contract(
        metadata.get(OCR_JOINT_RUN_CONTRACT_METADATA_KEY)
    )
    visual_run_sha256 = saved_visual_run["contract_canonical_sha256"]
    if saved_visual_run["metadata_contract"]["stage"] != "visual":
        raise ValueError("promotion source run contract stage must be visual")
    if result["run_contract_sha256"] != visual_run_sha256:
        raise ValueError("visual stage result run contract SHA256 differs")
    if metadata.get(OCR_JOINT_RUN_CONTRACT_SHA256_METADATA_KEY) != visual_run_sha256:
        raise ValueError("visual checkpoint run contract SHA metadata differs")
    for field, expected_value in saved_visual_run["metadata_contract"].items():
        if metadata.get(field) != expected_value:
            raise ValueError(f"visual checkpoint metadata contract drift for {field}")
    if metadata.get("training_stage") != "visual":
        raise ValueError("promotion checkpoint training_stage must be visual")
    if metadata.get("final") is not False:
        raise ValueError(
            "promotion requires the best eligible checkpoint, not final checkpoint"
        )
    for field in (
        "frozen_parameter_contract",
        "runtime_source_receipt",
        "runtime_environment",
    ):
        if metadata.get(field) != result[field]:
            raise ValueError(
                f"visual stage result {field} differs from best checkpoint metadata"
            )

    best_validation = metadata.get("last_validation")
    if not isinstance(best_validation, Mapping):
        raise ValueError("best visual checkpoint has no last_validation report")
    best_validation = deepcopy(dict(best_validation))
    best_validation_sha256 = canonical_json_sha256(best_validation)
    if best_validation_sha256 != result["best_validation_sha256"]:
        raise ValueError("best visual validation SHA256 differs from stage result")
    baseline_validation = metadata.get("baseline_validation")
    if not isinstance(baseline_validation, Mapping) or canonical_json_sha256(
        baseline_validation
    ) != result["baseline_validation_sha256"]:
        raise ValueError("visual baseline validation differs from stage result")
    baseline_bucket_cer = {
        bucket: float(
            baseline_validation["real"]["buckets"][bucket][
                "raw_grapheme_cer"
            ]
        )
        for bucket in DEPLOYMENT_BUCKET_WEIGHTS
    }
    recomputed_eligibility = joint_eval_eligibility(
        best_validation,
        baseline_bucket_cer=baseline_bucket_cer,
        baseline_text_token_nll=float(
            baseline_validation["text_replay"]["token_nll"]
        ),
        min_relative_cer_improvement=0.5,
    )
    eligibility = best_validation.get("eligibility")
    if eligibility != recomputed_eligibility:
        raise ValueError("best visual eligibility differs from recomputation")
    if recomputed_eligibility.get("eligible") is not True:
        raise ValueError("best visual validation is not promotion eligible")

    restored = reconstruct_policy_from_checkpoint(
        checkpoint,
        require_vision=True,
        metadata_override=metadata,
        expected_metadata_sha256=expected_metadata_sha256,
        expected_model_sha256=expected_model_sha256,
    )
    if restored.metadata != metadata:
        raise ValueError("promoted checkpoint metadata changed during reconstruction")
    if restored.model_sha256 != expected_model_sha256 or (
        restored.metadata_sha256 != expected_metadata_sha256
    ):
        raise ValueError("promoted checkpoint artifact hashes differ from expected")
    if restored.native_detail_config is None or (
        restored.vision_cross_attention_config is None
    ):
        raise ValueError("promotion source did not strictly reconstruct as anyres-v2")
    if restored.omvt_config is None:
        raise ValueError("promotion source has no OMVT configuration")

    parent = {
        "path": str(restored.checkpoint_dir.resolve()),
        "model_sha256": restored.model_sha256,
        "metadata_sha256": restored.metadata_sha256,
        "visual_run_contract_sha256": visual_run_sha256,
        "visual_stage_result_canonical_sha256": result["canonical_sha256"],
        "best_validation_sha256": best_validation_sha256,
    }
    joint_metadata_contract = deepcopy(saved_visual_run["metadata_contract"])
    joint_metadata_contract["stage"] = "joint"
    joint_metadata_contract[PARENT_ANYRES_CHECKPOINT_METADATA_KEY] = parent
    trainability = _set_stage_trainability(restored.model, "joint")
    joint_run = _run_contract(joint_metadata_contract, trainability)
    validate_ocr_joint_run_contract(joint_run)
    metadata_template = deepcopy(joint_metadata_contract)
    metadata_template[OCR_JOINT_RUN_CONTRACT_METADATA_KEY] = deepcopy(joint_run)
    metadata_template[OCR_JOINT_RUN_CONTRACT_SHA256_METADATA_KEY] = joint_run[
        "contract_canonical_sha256"
    ]
    return PreparedAnyresPolicy(
        policy=restored.model,
        run_contract=joint_run,
        metadata_template=metadata_template,
        checkpoint_dir=restored.checkpoint_dir,
        rdt_config=restored.rdt_config,
        omvt_config=restored.omvt_config,
        model_sha256=restored.model_sha256,
        metadata_sha256=metadata_sha256,
    )


def admit_joint_policy_for_grpo(
    checkpoint: str | Path,
    expected_model_sha256: str,
    expected_metadata_sha256: str,
    joint_stage_result: Mapping[str, Any],
    *,
    reward_contract_sha256: str,
    kl_coef: float,
) -> PreparedAnyresGRPO:
    """Admit only the hash-bound eligible joint best for single-process GRPO."""

    expected_model_sha256 = _sha256(
        expected_model_sha256,
        "expected joint checkpoint model SHA256",
    )
    expected_metadata_sha256 = _sha256(
        expected_metadata_sha256,
        "expected joint checkpoint metadata SHA256",
    )
    reward_contract_sha256 = _sha256(
        reward_contract_sha256,
        "reward contract SHA256",
    )
    if (
        isinstance(kl_coef, bool)
        or not isinstance(kl_coef, (int, float))
        or not math.isfinite(float(kl_coef))
        or float(kl_coef) < 0
    ):
        raise ValueError("kl_coef must be finite and non-negative")
    result = _validate_joint_stage_result(joint_stage_result)
    if result["grpo_promotion_allowed"] is not True:
        raise ValueError("joint stage result does not allow GRPO promotion")
    best_identity = _checkpoint_identity(
        result["best_eligible_checkpoint"],
        "best_eligible_checkpoint",
    )
    requested_identity = {
        "path": str(Path(checkpoint).resolve()),
        "model_sha256": expected_model_sha256,
        "metadata_sha256": expected_metadata_sha256,
    }
    if best_identity != requested_identity:
        raise ValueError("GRPO checkpoint identity is not JOINT_STAGE_RESULT best")

    metadata, metadata_sha256 = load_verified_policy_metadata(
        checkpoint,
        expected_sha256=expected_metadata_sha256,
    )
    saved_run = validate_ocr_joint_run_contract(
        metadata.get(OCR_JOINT_RUN_CONTRACT_METADATA_KEY)
    )
    run_sha = saved_run["contract_canonical_sha256"]
    if saved_run["metadata_contract"]["stage"] != "joint":
        raise ValueError("GRPO source run contract stage must be joint")
    if result["run_contract_sha256"] != run_sha or metadata.get(
        OCR_JOINT_RUN_CONTRACT_SHA256_METADATA_KEY
    ) != run_sha:
        raise ValueError("joint run contract SHA differs from stage result")
    for field, expected_value in saved_run["metadata_contract"].items():
        if metadata.get(field) != expected_value:
            raise ValueError(f"joint checkpoint metadata contract drift for {field}")
    if metadata.get("training_stage") != "joint" or metadata.get("final") is not False:
        raise ValueError("GRPO requires a non-final joint best checkpoint")
    if metadata.get(PARENT_ANYRES_CHECKPOINT_METADATA_KEY) != result[
        "parent_anyres_checkpoint"
    ]:
        raise ValueError("joint parent checkpoint differs from stage result")
    for field in ("runtime_source_receipt", "runtime_environment"):
        if metadata.get(field) != result[field]:
            raise ValueError(f"joint stage result {field} differs from best metadata")

    validation = metadata.get("last_validation")
    if not isinstance(validation, Mapping):
        raise ValueError("joint best checkpoint has no last_validation")
    validation = deepcopy(dict(validation))
    if canonical_json_sha256(validation) != result["best_validation_sha256"]:
        raise ValueError("joint best validation SHA differs from stage result")
    runtime_baseline = metadata.get("joint_runtime_baseline")
    historical_baseline = metadata.get("historical_visual_best_validation")
    if not isinstance(runtime_baseline, Mapping) or canonical_json_sha256(
        runtime_baseline
    ) != result["baseline_validation_sha256"]:
        raise ValueError("joint runtime baseline differs from stage result")
    if not isinstance(historical_baseline, Mapping) or canonical_json_sha256(
        historical_baseline
    ) != result["historical_visual_validation_sha256"]:
        raise ValueError("historical visual validation differs from stage result")
    recomputed_eligibility = dual_baseline_joint_eligibility(
        validation,
        runtime_baseline=runtime_baseline,
        historical_visual_baseline=historical_baseline,
    )
    eligibility = validation.get("eligibility")
    if eligibility != recomputed_eligibility:
        raise ValueError("joint best eligibility differs from recomputation")
    if recomputed_eligibility.get("eligible") is not True:
        raise ValueError("joint best validation is not GRPO eligible")
    optimizer_contract = metadata.get("optimizer_contract")
    if not isinstance(optimizer_contract, Mapping) or (
        optimizer_contract.get("canonical_sha256")
        != result["optimizer_contract_sha256"]
    ):
        raise ValueError("joint optimizer contract differs from stage result")
    sampler_state = metadata.get("quota_sampler_state")
    if canonical_json_sha256(sampler_state) != result["quota_sampler_state_sha256"]:
        raise ValueError("joint quota sampler state differs from stage result")
    text_cursor = metadata.get("text_replay_cursor")
    if not isinstance(text_cursor, Mapping) or text_cursor.get(
        "canonical_sha256"
    ) != result["text_replay_cursor_sha256"]:
        raise ValueError("joint text replay cursor differs from stage result")
    for field in (
        "dataset_admission_report",
        "train_dataset_contract",
        "validation_dataset_contract",
        "sft_validation_dataset_contract",
        "kl_selection_dataset_contract",
        "formal_monitor_dataset_contract",
        "text_replay_train_contract",
        "text_replay_sft_validation_contract",
        "text_replay_kl_selection_contract",
        "text_replay_formal_monitor_contract",
        "text_replay_validation_contract",
        "text_replay_partition_contract",
    ):
        if not isinstance(metadata.get(field), Mapping):
            raise ValueError(f"joint best metadata has no {field}")
    validation_alias = dict(metadata["validation_dataset_contract"])
    sft_validation_contract = dict(
        metadata["sft_validation_dataset_contract"]
    )
    if validation_alias != sft_validation_contract:
        raise ValueError(
            "validation_dataset_contract is only an SFT validation alias and "
            "must equal sft_validation_dataset_contract"
        )
    sft_validation_dataset_contract_sha256 = _embedded_contract_sha256(
        sft_validation_contract,
        "sft_validation_dataset_contract",
    )
    if _embedded_contract_sha256(
        validation_alias,
        "validation_dataset_contract",
    ) != sft_validation_dataset_contract_sha256:
        raise ValueError(
            "validation_dataset_contract alias SHA differs from SFT validation"
        )
    kl_selection_dataset_contract_sha256 = _embedded_contract_sha256(
        metadata["kl_selection_dataset_contract"],
        "kl_selection_dataset_contract",
    )
    formal_monitor_dataset_contract_sha256 = _embedded_contract_sha256(
        metadata["formal_monitor_dataset_contract"],
        "formal_monitor_dataset_contract",
    )
    train_dataset_contract_sha256 = _embedded_contract_sha256(
        metadata["train_dataset_contract"],
        "train_dataset_contract",
    )
    if len(
        {
            train_dataset_contract_sha256,
            sft_validation_dataset_contract_sha256,
            kl_selection_dataset_contract_sha256,
            formal_monitor_dataset_contract_sha256,
        }
    ) != 4:
        raise ValueError(
            "image train, SFT validation, KL selection, and formal monitor "
            "contracts must be distinct"
        )

    text_validation_alias = dict(metadata["text_replay_validation_contract"])
    text_sft_validation_contract = dict(
        metadata["text_replay_sft_validation_contract"]
    )
    if text_validation_alias != text_sft_validation_contract:
        raise ValueError(
            "text_replay_validation_contract is only an SFT validation alias "
            "and must equal text_replay_sft_validation_contract"
        )
    text_train_contract_sha256 = _embedded_contract_sha256(
        metadata["text_replay_train_contract"],
        "text_replay_train_contract",
    )
    text_sft_validation_contract_sha256 = _embedded_contract_sha256(
        text_sft_validation_contract,
        "text_replay_sft_validation_contract",
    )
    if _embedded_contract_sha256(
        text_validation_alias,
        "text_replay_validation_contract",
    ) != text_sft_validation_contract_sha256:
        raise ValueError("text validation alias SHA differs from SFT validation")
    text_kl_selection_contract_sha256 = _embedded_contract_sha256(
        metadata["text_replay_kl_selection_contract"],
        "text_replay_kl_selection_contract",
    )
    text_formal_monitor_contract_sha256 = _embedded_contract_sha256(
        metadata["text_replay_formal_monitor_contract"],
        "text_replay_formal_monitor_contract",
    )
    text_contract_sha256s = {
        "train": text_train_contract_sha256,
        "sft_validation": text_sft_validation_contract_sha256,
        "kl_selection": text_kl_selection_contract_sha256,
        "formal_monitor": text_formal_monitor_contract_sha256,
    }
    if len(set(text_contract_sha256s.values())) != 4:
        raise ValueError(
            "text train, SFT validation, KL selection, and formal monitor "
            "dataset contracts must be distinct"
        )
    text_partition_contract = validate_text_replay_partition_contract(
        metadata["text_replay_partition_contract"]
    )
    if text_partition_contract.get("split_contract_sha256") != (
        text_contract_sha256s
    ):
        raise ValueError("joint best text replay partition split contracts differ")
    for report_name, report in (
        ("joint best", validation),
        ("joint runtime baseline", runtime_baseline),
        ("historical visual baseline", historical_baseline),
    ):
        if report.get("image_dataset_contract_sha256") != (
            sft_validation_dataset_contract_sha256
        ):
            raise ValueError(
                f"{report_name} image dataset contract is not SFT validation"
            )
        text_report = report.get("text_replay")
        if not isinstance(text_report, Mapping) or text_report.get(
            "dataset_contract_sha256"
        ) != text_sft_validation_contract_sha256:
            raise ValueError(
                f"{report_name} text dataset contract is not SFT validation"
            )

    restored = reconstruct_policy_from_checkpoint(
        checkpoint,
        require_vision=True,
        metadata_override=metadata,
        expected_metadata_sha256=expected_metadata_sha256,
        expected_model_sha256=expected_model_sha256,
    )
    if restored.metadata != metadata or restored.metadata_sha256 != metadata_sha256:
        raise ValueError("joint best metadata changed during reconstruction")
    if restored.model_sha256 != expected_model_sha256:
        raise ValueError("joint best model bytes changed during reconstruction")
    trainability = _set_stage_trainability(restored.model, "joint")
    if trainability != saved_run["trainability_contract"]:
        raise ValueError("joint GRPO trainability differs from saved run")

    reference = None
    if float(kl_coef) > 0:
        reference_restored = reconstruct_policy_from_checkpoint(
            checkpoint,
            require_vision=True,
            metadata_override=metadata,
            expected_metadata_sha256=expected_metadata_sha256,
            expected_model_sha256=expected_model_sha256,
        )
        reference = reference_restored.model
        reference.reverse_loss_enabled = False
        reference.requires_grad_(False)
        reference.eval()
        if reference is restored.model:
            raise RuntimeError("GRPO reference must be an independent model object")

    metadata_contract = saved_run["metadata_contract"]
    tokenizer_contract_sha256 = canonical_json_sha256(
        metadata_contract["tokenizer_contract"]
    )
    admission = AnyresGRPOAdmission(
        policy_checkpoint_sha256=expected_model_sha256,
        policy_metadata_sha256=expected_metadata_sha256,
        reference_checkpoint_sha256=expected_model_sha256,
        reference_metadata_sha256=expected_metadata_sha256,
        joint_stage_result_sha256=result["canonical_sha256"],
        tokenizer_contract_sha256=tokenizer_contract_sha256,
        visual_contract_sha256=canonical_json_sha256(
            {
                "name": metadata_contract["ocr_visual_input_contract"],
                "version": metadata_contract[
                    "ocr_visual_input_contract_version"
                ],
            }
        ),
        preprocess_contract_sha256=metadata_contract[
            "anyres_preprocess_contract_sha256"
        ],
        native_migration_receipt_sha256=metadata_contract[
            "native_migration_receipt"
        ]["canonical_sha256"],
        trainability_contract_sha256=canonical_json_sha256(trainability),
        trainable_parameter_names_sha256=trainability[
            "trainable_parameter_names_sha256"
        ],
        reward_contract_sha256=reward_contract_sha256,
        dataset_admission_report_sha256=canonical_json_sha256(
            metadata.get("dataset_admission_report")
        ),
        train_dataset_contract_sha256=train_dataset_contract_sha256,
        sft_validation_dataset_contract_sha256=(
            sft_validation_dataset_contract_sha256
        ),
        kl_selection_dataset_contract_sha256=(
            kl_selection_dataset_contract_sha256
        ),
        formal_monitor_dataset_contract_sha256=(
            formal_monitor_dataset_contract_sha256
        ),
        text_replay_train_contract_sha256=text_train_contract_sha256,
        text_replay_sft_validation_contract_sha256=(
            text_sft_validation_contract_sha256
        ),
        text_replay_kl_selection_contract_sha256=(
            text_kl_selection_contract_sha256
        ),
        text_replay_formal_monitor_contract_sha256=(
            text_formal_monitor_contract_sha256
        ),
        runtime_source_receipt_sha256=_sha256(
            metadata["runtime_source_receipt"]["canonical_sha256"],
            "joint runtime source receipt SHA256",
        ),
        recommended_max_new_tokens=int(
            metadata_contract["anyres_preprocess_contract"]["budgets"][
                "output"
            ]["recommended_max_new_tokens"]
        ),
    )
    return PreparedAnyresGRPO(
        policy=restored.model,
        reference=reference,
        admission=admission,
        joint_stage_result=result,
        checkpoint_dir=restored.checkpoint_dir,
        metadata=metadata,
        model_sha256=restored.model_sha256,
        metadata_sha256=restored.metadata_sha256,
    )


def resume_anyres_policy(
    checkpoint: str | Path,
    expected_run_contract: Mapping[str, Any],
    expected_model_sha256: str,
    expected_metadata_sha256: str,
) -> PreparedAnyresPolicy:
    expected = validate_ocr_joint_run_contract(expected_run_contract)
    metadata, metadata_sha256 = load_verified_policy_metadata(
        checkpoint,
        expected_sha256=expected_metadata_sha256,
    )
    saved_run = validate_ocr_joint_run_contract(
        metadata.get(OCR_JOINT_RUN_CONTRACT_METADATA_KEY)
    )
    if saved_run != expected:
        raise ValueError("resume OCR joint run contract differs from expected contract")
    if metadata.get(OCR_JOINT_RUN_CONTRACT_SHA256_METADATA_KEY) != expected[
        "contract_canonical_sha256"
    ]:
        raise ValueError("resume OCR joint run contract SHA metadata differs")
    for field, expected_value in expected["metadata_contract"].items():
        if metadata.get(field) != expected_value:
            raise ValueError(f"resume metadata contract drift for {field}")

    restored = reconstruct_policy_from_checkpoint(
        checkpoint,
        require_vision=True,
        metadata_override=metadata,
        expected_metadata_sha256=expected_metadata_sha256,
        expected_model_sha256=expected_model_sha256,
    )
    if restored.native_detail_config is None or (
        restored.vision_cross_attention_config is None
    ):
        raise ValueError("resume checkpoint did not reconstruct as anyres-v2")
    trainability = _set_stage_trainability(
        restored.model,
        expected["metadata_contract"]["stage"],
    )
    if trainability != expected["trainability_contract"]:
        raise ValueError("resume trainability contract differs from saved policy")
    assert restored.omvt_config is not None
    return PreparedAnyresPolicy(
        policy=restored.model,
        run_contract=saved_run,
        metadata_template=deepcopy(metadata),
        checkpoint_dir=restored.checkpoint_dir,
        rdt_config=restored.rdt_config,
        omvt_config=restored.omvt_config,
        model_sha256=restored.model_sha256,
        metadata_sha256=metadata_sha256,
    )


__all__ = [
    "JOINT_STAGE_RESULT_KIND",
    "OCR_JOINT_RUN_CONTRACT_KIND",
    "OCR_JOINT_RUN_CONTRACT_METADATA_KEY",
    "OCR_JOINT_RUN_CONTRACT_SCHEMA_VERSION",
    "OCR_JOINT_RUN_CONTRACT_SHA256_METADATA_KEY",
    "OCR_JOINT_STAGES",
    "PARENT_ANYRES_CHECKPOINT_METADATA_KEY",
    "PreparedAnyresPolicy",
    "PreparedAnyresGRPO",
    "VISUAL_STAGE_RESULT_KIND",
    "admit_and_prepare_anyres_policy",
    "admit_joint_policy_for_grpo",
    "promote_visual_policy_to_joint",
    "resume_anyres_policy",
    "validate_ocr_joint_run_contract",
]
