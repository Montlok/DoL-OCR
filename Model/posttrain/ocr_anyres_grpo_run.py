# -*- coding: utf-8 -*-

"""Immutable run, progress, pilot, and KL-selection contracts for anyres GRPO."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from Model.ocr.tokenization import canonical_json_sha256
from Model.posttrain.ocr_anyres_grpo import AnyresGRPOAdmission
from Model.posttrain.ocr_anyres_grpo_protocol import (
    validate_fixed_kl_pilot_protocol,
)
from Model.posttrain.ocr_anyres_grpo_trainer import AnyresGRPOKLAblationTrial
from Model.posttrain.ocr_anyres_grpo_formal_protocol import (
    validate_formal_grpo_protocol,
)


ANYRES_GRPO_RUN_KIND = "dol_ocr_anyres_grpo_run_v1"
ANYRES_GRPO_PROGRESS_KIND = "dol_ocr_anyres_grpo_progress_v1"
ANYRES_GRPO_PILOT_RESULT_KIND = "dol_ocr_anyres_grpo_kl_pilot_result_v1"
ANYRES_GRPO_KL_SELECTION_KIND = "dol_ocr_anyres_grpo_kl_selection_v1"
ANYRES_GRPO_KL_SELECTION_RULE = "paired_stratified_bootstrap_95ci_v1"

_DEPLOYMENT_BUCKET_WEIGHTS = {
    "print": 0.6,
    "handwritten_good": 0.1,
    "handwritten_medium": 0.2,
    "handwritten_poor": 0.1,
}
_SELECTION_RECORD_KEYS = {
    "sample_id",
    "bucket",
    "grapheme_edits",
    "reference_graphemes",
}
_BOOTSTRAP_RESAMPLES = 2_000
_BOOTSTRAP_SEED = 42
_EQUIVALENCE_FRACTION_OF_BASELINE = 0.005

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_CHECKPOINT_KEYS = {"path", "model_sha256", "metadata_sha256"}
_SCHEDULE_KEYS = {
    "max_optimizer_steps",
    "max_rollout_attempts",
    "global_batch_size",
    "text_batch_size",
    "eval_every",
    "save_every",
    "early_stop_patience",
    "text_weight",
    "grad_clip",
    "loss_chunk_size",
    "weight_decay",
    "warmup_steps",
    "scheduler_contract_sha256",
    "world_size",
    "dist_mode",
    "max_consecutive_no_update",
    "max_consecutive_optimizer_skips",
    "ocr_prompt_schedule_sha256",
    "text_batch_schedule_sha256",
    "rollout_seed_schedule_sha256",
}
_RUN_KEYS = {
    "schema_version",
    "kind",
    "mode",
    "parent_joint_checkpoint",
    "parent_joint_stage_result_sha256",
    "admission",
    "admission_sha256",
    "dataset_ready_admission_sha256",
    "train_dataset_contract_sha256",
    "sft_validation_dataset_contract_sha256",
    "kl_selection_dataset_contract_sha256",
    "formal_monitor_dataset_contract_sha256",
    "text_replay_train_contract_sha256",
    "text_replay_sft_validation_contract_sha256",
    "text_replay_kl_selection_contract_sha256",
    "text_replay_formal_monitor_contract_sha256",
    "reward_contract_sha256",
    "source_closure_sha256",
    "runtime_environment_sha256",
    "baseline_validation_sha256",
    "pilot_protocol",
    "pilot_protocol_sha256",
    "formal_protocol",
    "formal_protocol_sha256",
    "kl_trial",
    "kl_trial_sha256",
    "trial_comparison_contract_sha256",
    "formal_selection_receipt",
    "formal_selection_receipt_sha256",
    "grpo_config",
    "optimizer_contract_sha256",
    "schedule",
    "seed",
    "precision",
    "canonical_sha256",
}
_PROGRESS_KEYS = {
    "schema_version",
    "kind",
    "run_contract_sha256",
    "optimizer_steps",
    "rollout_attempts",
    "no_update_count",
    "consecutive_no_update",
    "optimizer_skip_count",
    "consecutive_optimizer_skips",
    "sampler_state",
    "text_cursor_state",
    "text_ce_steps",
    "eval_count",
    "bad_eval_count",
    "best_validation_eligible",
    "best_validation_step",
    "best_checkpoint",
    "best_validation_sha256",
    "last_validation_sha256",
    "stop_reason",
    "terminal",
    "canonical_sha256",
}
_PILOT_RESULT_KEYS = {
    "schema_version",
    "kind",
    "run_contract_sha256",
    "parent_joint_checkpoint",
    "parent_joint_stage_result_sha256",
    "admission_sha256",
    "kl_trial",
    "kl_trial_sha256",
    "optimizer_steps",
    "rollout_attempts",
    "no_update_count",
    "optimizer_skip_count",
    "selection_rollout_attempt",
    "terminal_checkpoint",
    "selection_checkpoint",
    "baseline_validation",
    "baseline_validation_sha256",
    "selection_validation",
    "selection_validation_sha256",
    "image_dataset_contract_sha256",
    "text_replay_dataset_contract_sha256",
    "paired_sample_contract_sha256",
    "hard_eligible",
    "trial_comparison_contract_sha256",
    "source_closure_sha256",
    "dataset_ready_admission_sha256",
    "completed",
    "canonical_sha256",
}
_SELECTION_KEYS = {
    "schema_version",
    "kind",
    "selection_rule",
    "selection_rule_sha256",
    "candidate_result_sha256",
    "selected_result_sha256",
    "selected_kl_coef",
    "selected_metrics",
    "status",
    "formal_run_allowed",
    "candidate_eligibility",
    "paired_comparisons",
    "statistically_equivalent_kl_coefs",
    "paired_sample_contract_sha256",
    "image_dataset_contract_sha256",
    "text_replay_dataset_contract_sha256",
    "trial_comparison_contract_sha256",
    "parent_joint_checkpoint",
    "formal_init_checkpoint",
    "parent_joint_stage_result_sha256",
    "admission_sha256",
    "dataset_ready_admission_sha256",
    "source_closure_sha256",
    "canonical_sha256",
}


def _sha(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be lowercase SHA-256")
    return value


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _finite(value: object, field: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{field} must be finite and >= {minimum}")
    return result


def _exact(value: object, keys: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{field} fields differ from contract")
    return dict(value)


def _checkpoint(value: object, field: str) -> dict[str, str]:
    row = _exact(value, _CHECKPOINT_KEYS, field)
    path = row["path"]
    if not isinstance(path, str) or path != str(Path(path).resolve()) or not path:
        raise ValueError(f"{field}.path must be canonical absolute")
    return {
        "path": path,
        "model_sha256": _sha(row["model_sha256"], f"{field}.model_sha256"),
        "metadata_sha256": _sha(
            row["metadata_sha256"], f"{field}.metadata_sha256"
        ),
    }


def _trial_comparison_payload(run: Mapping[str, Any]) -> dict[str, Any]:
    trial = dict(run["kl_trial"])
    for key in ("selected_index", "selected_kl_coef"):
        trial.pop(key, None)
    config = dict(run["grpo_config"])
    config.pop("kl_coef", None)
    return {
        "schema_version": 1,
        "kind": "dol_ocr_anyres_grpo_trial_comparison_v1",
        "parent_joint_checkpoint": deepcopy(run["parent_joint_checkpoint"]),
        "parent_joint_stage_result_sha256": run[
            "parent_joint_stage_result_sha256"
        ],
        "admission_sha256": run["admission_sha256"],
        "dataset_ready_admission_sha256": run[
            "dataset_ready_admission_sha256"
        ],
        "train_dataset_contract_sha256": run[
            "train_dataset_contract_sha256"
        ],
        "sft_validation_dataset_contract_sha256": run[
            "sft_validation_dataset_contract_sha256"
        ],
        "kl_selection_dataset_contract_sha256": run[
            "kl_selection_dataset_contract_sha256"
        ],
        "formal_monitor_dataset_contract_sha256": run[
            "formal_monitor_dataset_contract_sha256"
        ],
        "text_replay_train_contract_sha256": run[
            "text_replay_train_contract_sha256"
        ],
        "text_replay_sft_validation_contract_sha256": run[
            "text_replay_sft_validation_contract_sha256"
        ],
        "text_replay_kl_selection_contract_sha256": run[
            "text_replay_kl_selection_contract_sha256"
        ],
        "text_replay_formal_monitor_contract_sha256": run[
            "text_replay_formal_monitor_contract_sha256"
        ],
        "reward_contract_sha256": run["reward_contract_sha256"],
        "source_closure_sha256": run["source_closure_sha256"],
        "runtime_environment_sha256": run["runtime_environment_sha256"],
        "baseline_validation_sha256": run["baseline_validation_sha256"],
        "pilot_protocol": deepcopy(run["pilot_protocol"]),
        "pilot_protocol_sha256": run["pilot_protocol_sha256"],
        "kl_trial_without_selection": trial,
        "grpo_config_without_kl": config,
        "optimizer_contract_sha256": run["optimizer_contract_sha256"],
        "schedule": deepcopy(run["schedule"]),
        "seed": run["seed"],
        "precision": run["precision"],
    }


def build_anyres_grpo_run_contract(
    *,
    mode: str,
    parent_joint_checkpoint: Mapping[str, Any],
    parent_joint_stage_result_sha256: str,
    admission: AnyresGRPOAdmission,
    dataset_ready_admission_sha256: str,
    reward_contract_sha256: str,
    source_closure_sha256: str,
    runtime_environment_sha256: str,
    baseline_validation_sha256: str,
    pilot_protocol: Mapping[str, Any] | None,
    kl_trial: AnyresGRPOKLAblationTrial,
    formal_selection_receipt: Mapping[str, Any] | None,
    grpo_config: Mapping[str, Any],
    optimizer_contract_sha256: str,
    schedule: Mapping[str, Any],
    seed: int,
    precision: str,
    formal_protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if mode not in {"pilot", "formal"}:
        raise ValueError("GRPO mode must be pilot or formal")
    if mode == "pilot" and formal_selection_receipt is not None:
        raise ValueError("pilot run must not consume a formal selection receipt")
    if mode == "formal" and formal_selection_receipt is None:
        raise ValueError("formal run requires a KL selection receipt")
    if formal_selection_receipt is not None and not isinstance(
        formal_selection_receipt, Mapping
    ):
        raise TypeError("formal_selection_receipt must be a mapping")
    if pilot_protocol is not None and not isinstance(pilot_protocol, Mapping):
        raise TypeError("pilot_protocol must be a mapping")
    if formal_protocol is not None and not isinstance(formal_protocol, Mapping):
        raise TypeError("formal_protocol must be a mapping")
    _nonnegative_int(seed, "seed")
    if precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError("precision must be fp32, bf16, or fp16")
    trial_payload = kl_trial.canonical_payload
    payload = {
        "schema_version": 1,
        "kind": ANYRES_GRPO_RUN_KIND,
        "mode": mode,
        "parent_joint_checkpoint": _checkpoint(
            parent_joint_checkpoint, "parent_joint_checkpoint"
        ),
        "parent_joint_stage_result_sha256": _sha(
            parent_joint_stage_result_sha256,
            "parent_joint_stage_result_sha256",
        ),
        "admission": admission.canonical_payload,
        "admission_sha256": admission.canonical_sha256,
        "dataset_ready_admission_sha256": _sha(
            dataset_ready_admission_sha256,
            "dataset_ready_admission_sha256",
        ),
        "train_dataset_contract_sha256": admission.train_dataset_contract_sha256,
        "sft_validation_dataset_contract_sha256": (
            admission.sft_validation_dataset_contract_sha256
        ),
        "kl_selection_dataset_contract_sha256": (
            admission.kl_selection_dataset_contract_sha256
        ),
        "formal_monitor_dataset_contract_sha256": (
            admission.formal_monitor_dataset_contract_sha256
        ),
        "text_replay_train_contract_sha256": (
            admission.text_replay_train_contract_sha256
        ),
        "text_replay_sft_validation_contract_sha256": (
            admission.text_replay_sft_validation_contract_sha256
        ),
        "text_replay_kl_selection_contract_sha256": (
            admission.text_replay_kl_selection_contract_sha256
        ),
        "text_replay_formal_monitor_contract_sha256": (
            admission.text_replay_formal_monitor_contract_sha256
        ),
        "reward_contract_sha256": _sha(
            reward_contract_sha256, "reward_contract_sha256"
        ),
        "source_closure_sha256": _sha(
            source_closure_sha256, "source_closure_sha256"
        ),
        "runtime_environment_sha256": _sha(
            runtime_environment_sha256, "runtime_environment_sha256"
        ),
        "baseline_validation_sha256": _sha(
            baseline_validation_sha256, "baseline_validation_sha256"
        ),
        "pilot_protocol": (
            None if pilot_protocol is None else deepcopy(dict(pilot_protocol))
        ),
        "pilot_protocol_sha256": (
            None
            if pilot_protocol is None
            else _sha(
                pilot_protocol.get("canonical_sha256"),
                "pilot_protocol.canonical_sha256",
            )
        ),
        "formal_protocol": (
            None if formal_protocol is None else deepcopy(dict(formal_protocol))
        ),
        "formal_protocol_sha256": (
            None
            if formal_protocol is None
            else _sha(
                formal_protocol.get("canonical_sha256"),
                "formal_protocol.canonical_sha256",
            )
        ),
        "kl_trial": trial_payload,
        "kl_trial_sha256": kl_trial.canonical_sha256,
        "trial_comparison_contract_sha256": None,
        "formal_selection_receipt": (
            None
            if formal_selection_receipt is None
            else deepcopy(dict(formal_selection_receipt))
        ),
        "formal_selection_receipt_sha256": (
            None
            if formal_selection_receipt is None
            else _sha(
                formal_selection_receipt.get("canonical_sha256"),
                "formal_selection_receipt.canonical_sha256",
            )
        ),
        "grpo_config": deepcopy(dict(grpo_config)),
        "optimizer_contract_sha256": _sha(
            optimizer_contract_sha256, "optimizer_contract_sha256"
        ),
        "schedule": deepcopy(dict(schedule)),
        "seed": seed,
        "precision": precision,
    }
    payload["trial_comparison_contract_sha256"] = (
        None
        if mode == "formal"
        else canonical_json_sha256(_trial_comparison_payload(payload))
    )
    payload["canonical_sha256"] = canonical_json_sha256(payload)
    return validate_anyres_grpo_run_contract(payload)


def validate_anyres_grpo_run_contract(value: object) -> dict[str, Any]:
    run = _exact(value, _RUN_KEYS, "anyres GRPO run")
    if run["schema_version"] != 1 or run["kind"] != ANYRES_GRPO_RUN_KIND:
        raise ValueError("unsupported anyres GRPO run contract")
    digest = _sha(run["canonical_sha256"], "run canonical_sha256")
    unhashed = {key: deepcopy(item) for key, item in run.items() if key != "canonical_sha256"}
    if canonical_json_sha256(unhashed) != digest:
        raise ValueError("run contract canonical SHA-256 mismatch")
    mode = run["mode"]
    if mode not in {"pilot", "formal"}:
        raise ValueError("run mode is invalid")
    _checkpoint(run["parent_joint_checkpoint"], "parent_joint_checkpoint")
    for field in (
        "parent_joint_stage_result_sha256",
        "admission_sha256",
        "dataset_ready_admission_sha256",
        "train_dataset_contract_sha256",
        "sft_validation_dataset_contract_sha256",
        "kl_selection_dataset_contract_sha256",
        "formal_monitor_dataset_contract_sha256",
        "text_replay_train_contract_sha256",
        "text_replay_sft_validation_contract_sha256",
        "text_replay_kl_selection_contract_sha256",
        "text_replay_formal_monitor_contract_sha256",
        "reward_contract_sha256",
        "source_closure_sha256",
        "runtime_environment_sha256",
        "baseline_validation_sha256",
        "kl_trial_sha256",
        "optimizer_contract_sha256",
    ):
        _sha(run[field], field)
    admission_payload = dict(run["admission"])
    if admission_payload.pop("schema_version", None) != 1 or (
        admission_payload.pop("kind", None)
        != "dol_ocr_anyres_grpo_admission_v1"
    ):
        raise ValueError("embedded GRPO admission kind differs")
    admission = AnyresGRPOAdmission(**admission_payload)
    if admission.canonical_sha256 != run["admission_sha256"]:
        raise ValueError("embedded GRPO admission hash differs")
    if admission.policy_checkpoint_sha256 != run["parent_joint_checkpoint"]["model_sha256"]:
        raise ValueError("GRPO admission policy differs from parent checkpoint")
    if admission.policy_metadata_sha256 != run["parent_joint_checkpoint"]["metadata_sha256"]:
        raise ValueError("GRPO admission metadata differs from parent checkpoint")
    if admission.reference_checkpoint_sha256 != run["parent_joint_checkpoint"][
        "model_sha256"
    ] or admission.reference_metadata_sha256 != run["parent_joint_checkpoint"][
        "metadata_sha256"
    ]:
        raise ValueError("GRPO immutable reference differs from parent checkpoint")
    if admission.joint_stage_result_sha256 != run[
        "parent_joint_stage_result_sha256"
    ]:
        raise ValueError("GRPO admission joint stage result differs from run")
    if admission.dataset_admission_report_sha256 != run[
        "dataset_ready_admission_sha256"
    ]:
        raise ValueError("GRPO admission dataset report differs from run")
    for field in (
        "train_dataset_contract_sha256",
        "sft_validation_dataset_contract_sha256",
        "kl_selection_dataset_contract_sha256",
        "formal_monitor_dataset_contract_sha256",
        "text_replay_train_contract_sha256",
        "text_replay_sft_validation_contract_sha256",
        "text_replay_kl_selection_contract_sha256",
        "text_replay_formal_monitor_contract_sha256",
    ):
        if getattr(admission, field) != run[field]:
            raise ValueError(f"run {field} differs from admission")
    if admission.reward_contract_sha256 != run["reward_contract_sha256"]:
        raise ValueError("run reward contract differs from admission")
    trial = AnyresGRPOKLAblationTrial(
        candidate_kl_coefs=tuple(run["kl_trial"]["candidate_kl_coefs"]),
        selected_index=int(run["kl_trial"]["selected_index"]),
        trial_steps=int(run["kl_trial"]["trial_steps"]),
        experiment_id=str(run["kl_trial"]["experiment_id"]),
    )
    if trial.canonical_payload != run["kl_trial"] or trial.canonical_sha256 != run[
        "kl_trial_sha256"
    ]:
        raise ValueError("KL trial contract differs")
    comparison_sha = run["trial_comparison_contract_sha256"]
    if run["mode"] == "pilot":
        protocol = validate_fixed_kl_pilot_protocol(run["pilot_protocol"])
        if run["pilot_protocol_sha256"] != run["pilot_protocol"][
            "canonical_sha256"
        ]:
            raise ValueError("pilot protocol SHA-256 differs")
        bindings = {
            "global_batch_size": protocol["global_batch_size"],
            "text_batch_size": protocol["text_batch_size"],
            "world_size": protocol["world_size"],
            "ocr_prompt_schedule_sha256": protocol[
                "ocr_prompt_schedule_sha256"
            ],
            "text_batch_schedule_sha256": protocol[
                "text_batch_schedule_sha256"
            ],
            "rollout_seed_schedule_sha256": protocol[
                "rollout_seed_schedule_sha256"
            ],
        }
        if any(run["schedule"][field] != value for field, value in bindings.items()):
            raise ValueError("pilot run schedule differs from embedded protocol")
        if run["seed"] != protocol["base_seed"]:
            raise ValueError("pilot run seed differs from embedded protocol")
        if run["train_dataset_contract_sha256"] != protocol[
            "ocr_dataset_contract_sha256"
        ] or run["text_replay_train_contract_sha256"] != protocol[
            "text_dataset_contract_sha256"
        ]:
            raise ValueError("pilot train datasets differ from embedded protocol")
        if comparison_sha != canonical_json_sha256(_trial_comparison_payload(run)):
            raise ValueError("pilot comparison contract SHA-256 differs")
        if run["formal_protocol"] is not None or run[
            "formal_protocol_sha256"
        ] is not None:
            raise ValueError("pilot run must not declare a formal protocol")
    elif any(
        item is not None
        for item in (
            comparison_sha,
            run["pilot_protocol"],
            run["pilot_protocol_sha256"],
        )
    ):
        raise ValueError("formal run must not declare a pilot comparison contract")
    config = run["grpo_config"]
    if not isinstance(config, Mapping):
        raise ValueError("grpo_config must be an object")
    if float(config.get("kl_coef", float("nan"))) != trial.selected_kl_coef:
        raise ValueError("grpo_config KL differs from selected trial")
    if int(config.get("max_new_tokens", -1)) != admission.recommended_max_new_tokens:
        raise ValueError("grpo_config max_new_tokens differs from admission")
    if float(config.get("min_reward_spread", float("nan"))) != float(
        admission.minimum_reward_spread
    ):
        raise ValueError("grpo_config min_reward_spread differs from admission")
    if config.get("clip_eps") is not None or config.get("advantage_mode") != "centered":
        raise ValueError("anyres GRPO must be unclipped centered on-policy")
    schedule = _exact(run["schedule"], _SCHEDULE_KEYS, "schedule")
    max_optimizer_steps = _positive_int(
        schedule.get("max_optimizer_steps"), "max_optimizer_steps"
    )
    max_rollout_attempts = _positive_int(
        schedule.get("max_rollout_attempts"), "max_rollout_attempts"
    )
    if max_rollout_attempts < max_optimizer_steps:
        raise ValueError("max_rollout_attempts must cover max_optimizer_steps")
    for field in (
        "global_batch_size",
        "text_batch_size",
        "eval_every",
        "save_every",
        "loss_chunk_size",
        "max_consecutive_no_update",
        "max_consecutive_optimizer_skips",
    ):
        _positive_int(schedule.get(field), field)
    _positive_int(schedule.get("early_stop_patience"), "early_stop_patience")
    _nonnegative_int(schedule.get("warmup_steps"), "warmup_steps")
    for field in ("text_weight", "grad_clip", "weight_decay"):
        _finite(schedule.get(field), field, minimum=0.0)
    if float(schedule["grad_clip"]) <= 0:
        raise ValueError("grad_clip must be positive")
    for field in (
        "scheduler_contract_sha256",
        "ocr_prompt_schedule_sha256",
        "text_batch_schedule_sha256",
        "rollout_seed_schedule_sha256",
    ):
        _sha(schedule.get(field), field)
    if schedule.get("world_size") != 1 or schedule.get("dist_mode") != "single":
        raise ValueError("anyres GRPO owner currently requires single/world_size=1")
    if int(schedule["global_batch_size"]) % 20:
        raise ValueError("global_batch_size must be divisible by 20")
    if mode == "pilot" and max_rollout_attempts != trial.trial_steps:
        raise ValueError("pilot rollout-attempt budget differs from KL trial")
    if mode == "pilot" and (
        trial.trial_steps != 200
        or max_optimizer_steps != 200
        or schedule["eval_every"] != 50
        or schedule["save_every"] != 50
    ):
        raise ValueError(
            "pilot protocol requires 200 attempts/updates with 50-attempt cadence"
        )
    selection = run["formal_selection_receipt"]
    selection_sha = run["formal_selection_receipt_sha256"]
    if mode == "pilot":
        if selection is not None or selection_sha is not None:
            raise ValueError("pilot run must not contain a formal selection receipt")
    else:
        formal_protocol = validate_formal_grpo_protocol(run["formal_protocol"])
        if run["formal_protocol_sha256"] != run["formal_protocol"][
            "canonical_sha256"
        ]:
            raise ValueError("formal protocol SHA-256 differs")
        formal_bindings = {
            "global_batch_size": formal_protocol["global_batch_size"],
            "text_batch_size": formal_protocol["text_batch_size"],
            "world_size": formal_protocol["world_size"],
            "max_rollout_attempts": formal_protocol["max_rollout_attempts"],
            "max_optimizer_steps": formal_protocol["max_optimizer_steps"],
            "eval_every": formal_protocol["eval_every"],
            "save_every": formal_protocol["save_every"],
            "early_stop_patience": formal_protocol["early_stop_patience"],
            "ocr_prompt_schedule_sha256": formal_protocol[
                "ocr_prompt_schedule_sha256"
            ],
            "text_batch_schedule_sha256": formal_protocol[
                "text_batch_schedule_sha256"
            ],
            "rollout_seed_schedule_sha256": formal_protocol[
                "rollout_seed_schedule_sha256"
            ],
        }
        if any(run["schedule"][field] != expected for field, expected in formal_bindings.items()):
            raise ValueError("formal run schedule differs from embedded protocol")
        if run["seed"] != formal_protocol["base_seed"]:
            raise ValueError("formal run seed differs from embedded protocol")
        if run["train_dataset_contract_sha256"] != formal_protocol[
            "ocr_dataset_contract_sha256"
        ] or run["text_replay_train_contract_sha256"] != formal_protocol[
            "text_dataset_contract_sha256"
        ]:
            raise ValueError("formal train datasets differ from embedded protocol")
        receipt = validate_kl_selection_receipt(selection)
        if receipt["canonical_sha256"] != selection_sha:
            raise ValueError("formal selection receipt SHA-256 differs")
        if receipt["status"] != "selected" or receipt[
            "formal_run_allowed"
        ] is not True:
            raise ValueError("formal selection receipt does not authorize training")
        if float(receipt["selected_kl_coef"]) != trial.selected_kl_coef:
            raise ValueError("formal selected KL differs from run trial")
        bindings = {
            "parent_joint_checkpoint": receipt["formal_init_checkpoint"],
            "parent_joint_stage_result_sha256": receipt[
                "parent_joint_stage_result_sha256"
            ],
            "admission_sha256": receipt["admission_sha256"],
            "dataset_ready_admission_sha256": receipt[
                "dataset_ready_admission_sha256"
            ],
            "source_closure_sha256": receipt["source_closure_sha256"],
            "kl_selection_dataset_contract_sha256": receipt[
                "image_dataset_contract_sha256"
            ],
            "text_replay_kl_selection_contract_sha256": receipt[
                "text_replay_dataset_contract_sha256"
            ],
        }
        if any(run[field] != expected for field, expected in bindings.items()):
            raise ValueError("formal run differs from KL selection receipt")
    _nonnegative_int(run["seed"], "seed")
    if run["precision"] not in {"fp32", "bf16", "fp16"}:
        raise ValueError("run precision is invalid")
    return deepcopy(run)


def build_anyres_grpo_progress(
    run_contract: Mapping[str, Any],
    *,
    optimizer_steps: int,
    rollout_attempts: int,
    no_update_count: int,
    consecutive_no_update: int,
    optimizer_skip_count: int,
    consecutive_optimizer_skips: int,
    sampler_state: Mapping[str, Any],
    text_cursor_state: Mapping[str, Any],
    text_ce_steps: int,
    eval_count: int,
    bad_eval_count: int,
    best_validation_eligible: bool,
    best_validation_step: int,
    best_checkpoint: str | None,
    best_validation_sha256: str | None,
    last_validation_sha256: str | None,
    stop_reason: str,
    terminal: bool,
) -> dict[str, Any]:
    run = validate_anyres_grpo_run_contract(run_contract)
    payload = {
        "schema_version": 1,
        "kind": ANYRES_GRPO_PROGRESS_KIND,
        "run_contract_sha256": run["canonical_sha256"],
        "optimizer_steps": optimizer_steps,
        "rollout_attempts": rollout_attempts,
        "no_update_count": no_update_count,
        "consecutive_no_update": consecutive_no_update,
        "optimizer_skip_count": optimizer_skip_count,
        "consecutive_optimizer_skips": consecutive_optimizer_skips,
        "sampler_state": deepcopy(dict(sampler_state)),
        "text_cursor_state": deepcopy(dict(text_cursor_state)),
        "text_ce_steps": text_ce_steps,
        "eval_count": eval_count,
        "bad_eval_count": bad_eval_count,
        "best_validation_eligible": best_validation_eligible,
        "best_validation_step": best_validation_step,
        "best_checkpoint": best_checkpoint,
        "best_validation_sha256": best_validation_sha256,
        "last_validation_sha256": last_validation_sha256,
        "stop_reason": stop_reason,
        "terminal": terminal,
    }
    payload["canonical_sha256"] = canonical_json_sha256(payload)
    return validate_anyres_grpo_progress(payload, run_contract=run)


def build_text_schedule_cursor_state(
    *,
    dataset_contract_sha256: str,
    batch_size: int,
    schedule_cursor: int,
    ce_steps: int,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "kind": "dol_text_replay_schedule_cursor_v1",
        "dataset_contract_sha256": _sha(
            dataset_contract_sha256, "text dataset contract SHA256"
        ),
        "batch_size": _positive_int(batch_size, "text batch_size"),
        "schedule_cursor": _nonnegative_int(
            schedule_cursor, "text schedule cursor"
        ),
        "ce_steps": _nonnegative_int(ce_steps, "text CE steps"),
        "wrap": "deterministic_modulo_dataset_v1",
    }
    payload["canonical_sha256"] = canonical_json_sha256(payload)
    return payload


def _validate_text_cursor_state(
    value: object,
    *,
    expected_dataset_contract_sha256: str,
    expected_batch_size: int,
) -> dict[str, Any]:
    keys = {
        "schema_version",
        "kind",
        "dataset_contract_sha256",
        "batch_size",
        "schedule_cursor",
        "ce_steps",
        "wrap",
        "canonical_sha256",
    }
    state = _exact(value, keys, "text schedule cursor")
    digest = _sha(state["canonical_sha256"], "text cursor canonical_sha256")
    unhashed = {
        key: item for key, item in state.items() if key != "canonical_sha256"
    }
    if canonical_json_sha256(unhashed) != digest:
        raise ValueError("text cursor canonical SHA-256 mismatch")
    if state["schema_version"] != 1 or state["kind"] != (
        "dol_text_replay_schedule_cursor_v1"
    ) or state["wrap"] != "deterministic_modulo_dataset_v1":
        raise ValueError("text cursor contract differs")
    if state["dataset_contract_sha256"] != expected_dataset_contract_sha256:
        raise ValueError("text cursor dataset contract differs")
    if state["batch_size"] != expected_batch_size:
        raise ValueError("text cursor batch size differs")
    _nonnegative_int(state["schedule_cursor"], "text schedule cursor")
    _nonnegative_int(state["ce_steps"], "text CE steps")
    return state


def validate_anyres_grpo_progress(
    value: object,
    *,
    run_contract: Mapping[str, Any],
) -> dict[str, Any]:
    run = validate_anyres_grpo_run_contract(run_contract)
    progress = _exact(value, _PROGRESS_KEYS, "anyres GRPO progress")
    if progress["schema_version"] != 1 or progress["kind"] != ANYRES_GRPO_PROGRESS_KIND:
        raise ValueError("unsupported anyres GRPO progress")
    digest = _sha(progress["canonical_sha256"], "progress canonical_sha256")
    unhashed = {
        key: deepcopy(item)
        for key, item in progress.items()
        if key != "canonical_sha256"
    }
    if canonical_json_sha256(unhashed) != digest:
        raise ValueError("progress canonical SHA-256 mismatch")
    if progress["run_contract_sha256"] != run["canonical_sha256"]:
        raise ValueError("progress belongs to another GRPO run")
    optimizer_steps = _nonnegative_int(progress["optimizer_steps"], "optimizer_steps")
    rollout_attempts = _nonnegative_int(progress["rollout_attempts"], "rollout_attempts")
    if optimizer_steps > rollout_attempts:
        raise ValueError("optimizer_steps cannot exceed rollout_attempts")
    no_updates = _nonnegative_int(progress["no_update_count"], "no_update_count")
    skips = _nonnegative_int(progress["optimizer_skip_count"], "optimizer_skip_count")
    if optimizer_steps + no_updates + skips != rollout_attempts:
        raise ValueError("rollout attempt accounting is inconsistent")
    if optimizer_steps > int(run["schedule"]["max_optimizer_steps"]):
        raise ValueError("optimizer_steps exceed run budget")
    if rollout_attempts > int(run["schedule"]["max_rollout_attempts"]):
        raise ValueError("rollout_attempts exceed run budget")
    consecutive_no_update = _nonnegative_int(
        progress["consecutive_no_update"], "consecutive_no_update"
    )
    consecutive_optimizer_skips = _nonnegative_int(
        progress["consecutive_optimizer_skips"], "consecutive_optimizer_skips"
    )
    if consecutive_no_update > no_updates:
        raise ValueError("consecutive_no_update exceeds no_update_count")
    if consecutive_optimizer_skips > skips:
        raise ValueError("consecutive_optimizer_skips exceeds optimizer_skip_count")
    sampler = progress["sampler_state"]
    if not isinstance(sampler, Mapping) or sampler.get("pending_global_batch") is not None:
        raise ValueError("checkpoint progress requires a committed sampler state")
    expected_draws = rollout_attempts * int(run["schedule"]["global_batch_size"])
    if sampler.get("draw_counter") != expected_draws:
        raise ValueError("sampler draw_counter differs from rollout attempts")
    cursor = _validate_text_cursor_state(
        progress["text_cursor_state"],
        expected_dataset_contract_sha256=run[
            "text_replay_train_contract_sha256"
        ],
        expected_batch_size=int(run["schedule"]["text_batch_size"]),
    )
    text_ce_steps = _nonnegative_int(progress["text_ce_steps"], "text_ce_steps")
    if cursor["schedule_cursor"] != rollout_attempts:
        raise ValueError("text schedule cursor must equal rollout attempts")
    if cursor["ce_steps"] != optimizer_steps or text_ce_steps != optimizer_steps:
        raise ValueError("text CE steps must equal optimizer steps")
    eval_count = _nonnegative_int(progress["eval_count"], "eval_count")
    bad_eval_count = _nonnegative_int(progress["bad_eval_count"], "bad_eval_count")
    if bad_eval_count > eval_count:
        raise ValueError("bad_eval_count exceeds eval_count")
    if type(progress["best_validation_eligible"]) is not bool:
        raise ValueError("best_validation_eligible must be bool")
    best_step = _nonnegative_int(progress["best_validation_step"], "best_validation_step")
    if best_step > optimizer_steps:
        raise ValueError("best_validation_step exceeds optimizer_steps")
    if progress["best_validation_eligible"]:
        if best_step <= 0 or not isinstance(progress["best_checkpoint"], str):
            raise ValueError("eligible best checkpoint identity is incomplete")
        _sha(progress["best_validation_sha256"], "best_validation_sha256")
    elif any(
        item is not None
        for item in (
            progress["best_checkpoint"],
            progress["best_validation_sha256"],
        )
    ) or best_step != 0:
        raise ValueError("non-eligible progress must not publish a best")
    if progress["last_validation_sha256"] is not None:
        _sha(progress["last_validation_sha256"], "last_validation_sha256")
    if not isinstance(progress["stop_reason"], str) or not progress["stop_reason"]:
        raise ValueError("stop_reason must be non-empty")
    if type(progress["terminal"]) is not bool:
        raise ValueError("terminal must be bool")
    if progress["terminal"] != (progress["stop_reason"] != "running"):
        raise ValueError("terminal flag differs from stop_reason")
    return deepcopy(progress)


def build_kl_pilot_result(
    *,
    run_contract: Mapping[str, Any],
    progress: Mapping[str, Any],
    terminal_checkpoint: Mapping[str, Any],
    selection_checkpoint: Mapping[str, Any] | None,
    baseline_validation: Mapping[str, Any],
    selection_validation: Mapping[str, Any],
) -> dict[str, Any]:
    run = validate_anyres_grpo_run_contract(run_contract)
    if run["mode"] != "pilot":
        raise ValueError("pilot result requires a pilot run contract")
    state = validate_anyres_grpo_progress(progress, run_contract=run)
    if not state["terminal"] or state["rollout_attempts"] != run["schedule"][
        "max_rollout_attempts"
    ]:
        raise ValueError("KL pilot did not complete its rollout-attempt budget")
    if not isinstance(baseline_validation, Mapping) or not isinstance(
        selection_validation, Mapping
    ):
        raise ValueError("pilot baseline/selection validation must be objects")
    if canonical_json_sha256(baseline_validation) != run[
        "baseline_validation_sha256"
    ]:
        raise ValueError("pilot baseline validation differs from run contract")
    expected_image_contract = run["kl_selection_dataset_contract_sha256"]
    expected_text_contract = run[
        "text_replay_kl_selection_contract_sha256"
    ]
    for name, report in (
        ("baseline", baseline_validation),
        ("selection", selection_validation),
    ):
        if report.get("image_dataset_contract_sha256") != expected_image_contract:
            raise ValueError(f"pilot {name} image dataset contract differs")
        text_report = report.get("text_replay")
        if not isinstance(text_report, Mapping) or text_report.get(
            "dataset_contract_sha256"
        ) != expected_text_contract:
            raise ValueError(f"pilot {name} text replay contract differs")
    eligibility = selection_validation.get("eligibility")
    hard_eligible = bool(
        isinstance(eligibility, Mapping) and eligibility.get("eligible") is True
    )
    selection_validation_sha = canonical_json_sha256(selection_validation)
    paired_records = _validate_paired_records(baseline_validation)
    _paired_rows(baseline_validation, selection_validation)
    terminal_identity = _checkpoint(terminal_checkpoint, "terminal_checkpoint")
    if hard_eligible:
        if selection_checkpoint is None:
            raise ValueError("eligible pilot must publish its cycle-200 checkpoint")
        if state["best_validation_sha256"] != selection_validation_sha:
            raise ValueError("pilot progress and selection validation differ")
        selection_identity = _checkpoint(
            selection_checkpoint, "selection_checkpoint"
        )
        if selection_identity != terminal_identity:
            raise ValueError("pilot selection must be the cycle-200 terminal checkpoint")
    else:
        if selection_checkpoint is not None or state[
            "best_validation_eligible"
        ]:
            raise ValueError("ineligible pilot must not publish an eligible best")
        selection_identity = None
    payload = {
        "schema_version": 1,
        "kind": ANYRES_GRPO_PILOT_RESULT_KIND,
        "run_contract_sha256": run["canonical_sha256"],
        "parent_joint_checkpoint": deepcopy(run["parent_joint_checkpoint"]),
        "parent_joint_stage_result_sha256": run[
            "parent_joint_stage_result_sha256"
        ],
        "admission_sha256": run["admission_sha256"],
        "kl_trial": deepcopy(run["kl_trial"]),
        "kl_trial_sha256": run["kl_trial_sha256"],
        "trial_comparison_contract_sha256": run[
            "trial_comparison_contract_sha256"
        ],
        "optimizer_steps": state["optimizer_steps"],
        "rollout_attempts": state["rollout_attempts"],
        "no_update_count": state["no_update_count"],
        "optimizer_skip_count": state["optimizer_skip_count"],
        "selection_rollout_attempt": state["rollout_attempts"],
        "terminal_checkpoint": terminal_identity,
        "selection_checkpoint": selection_identity,
        "baseline_validation": deepcopy(dict(baseline_validation)),
        "baseline_validation_sha256": canonical_json_sha256(
            baseline_validation
        ),
        "selection_validation": deepcopy(dict(selection_validation)),
        "selection_validation_sha256": selection_validation_sha,
        "image_dataset_contract_sha256": expected_image_contract,
        "text_replay_dataset_contract_sha256": expected_text_contract,
        "paired_sample_contract_sha256": _paired_sample_contract_sha256(
            paired_records
        ),
        "hard_eligible": hard_eligible,
        "source_closure_sha256": run["source_closure_sha256"],
        "dataset_ready_admission_sha256": run[
            "dataset_ready_admission_sha256"
        ],
        "completed": True,
    }
    payload["canonical_sha256"] = canonical_json_sha256(payload)
    return validate_kl_pilot_result(payload)


def validate_kl_pilot_result(value: object) -> dict[str, Any]:
    result = _exact(value, _PILOT_RESULT_KEYS, "KL pilot result")
    if result["schema_version"] != 1 or result["kind"] != (
        ANYRES_GRPO_PILOT_RESULT_KIND
    ):
        raise ValueError("unsupported KL pilot result")
    digest = _sha(result["canonical_sha256"], "pilot result canonical_sha256")
    unhashed = {
        key: deepcopy(item)
        for key, item in result.items()
        if key != "canonical_sha256"
    }
    if canonical_json_sha256(unhashed) != digest:
        raise ValueError("pilot result canonical SHA-256 mismatch")
    _checkpoint(result["parent_joint_checkpoint"], "parent_joint_checkpoint")
    _checkpoint(result["terminal_checkpoint"], "terminal_checkpoint")
    if result["selection_checkpoint"] is not None:
        selection_identity = _checkpoint(
            result["selection_checkpoint"], "selection_checkpoint"
        )
        if selection_identity != result["terminal_checkpoint"]:
            raise ValueError("pilot selection must be the cycle-200 terminal checkpoint")
    for field in (
        "run_contract_sha256",
        "parent_joint_stage_result_sha256",
        "admission_sha256",
        "kl_trial_sha256",
        "trial_comparison_contract_sha256",
        "baseline_validation_sha256",
        "selection_validation_sha256",
        "image_dataset_contract_sha256",
        "text_replay_dataset_contract_sha256",
        "paired_sample_contract_sha256",
        "source_closure_sha256",
        "dataset_ready_admission_sha256",
    ):
        _sha(result[field], field)
    trial = AnyresGRPOKLAblationTrial(
        candidate_kl_coefs=tuple(result["kl_trial"]["candidate_kl_coefs"]),
        selected_index=int(result["kl_trial"]["selected_index"]),
        trial_steps=int(result["kl_trial"]["trial_steps"]),
        experiment_id=str(result["kl_trial"]["experiment_id"]),
    )
    if trial.canonical_payload != result["kl_trial"] or trial.canonical_sha256 != result[
        "kl_trial_sha256"
    ]:
        raise ValueError("pilot KL trial contract differs")
    if (
        trial.candidate_kl_coefs != (0.04, 0.01, 0.0)
        or trial.trial_steps != 200
        or trial.experiment_id != "ocr_anyres_kl_ablation_v1"
    ):
        raise ValueError("pilot result differs from the fixed 200-attempt KL protocol")
    optimizer_steps = _nonnegative_int(result["optimizer_steps"], "optimizer_steps")
    rollout_attempts = _positive_int(result["rollout_attempts"], "rollout_attempts")
    if rollout_attempts != trial.trial_steps or optimizer_steps > rollout_attempts:
        raise ValueError("pilot attempt/update counts differ from trial budget")
    no_updates = _nonnegative_int(result["no_update_count"], "no_update_count")
    skips = _nonnegative_int(result["optimizer_skip_count"], "optimizer_skip_count")
    if optimizer_steps + no_updates + skips != rollout_attempts:
        raise ValueError("pilot result rollout-attempt accounting differs")
    if result["selection_rollout_attempt"] != rollout_attempts:
        raise ValueError("pilot selection was not performed at terminal attempt")
    if result["completed"] is not True:
        raise ValueError("pilot result is not complete")
    baseline = result["baseline_validation"]
    validation = result["selection_validation"]
    if not isinstance(baseline, Mapping) or canonical_json_sha256(baseline) != result[
        "baseline_validation_sha256"
    ]:
        raise ValueError("pilot baseline validation hash differs")
    if not isinstance(validation, Mapping) or canonical_json_sha256(validation) != result[
        "selection_validation_sha256"
    ]:
        raise ValueError("pilot selection validation hash differs")
    eligibility = validation.get("eligibility")
    hard_eligible = bool(
        isinstance(eligibility, Mapping) and eligibility.get("eligible") is True
    )
    if result["hard_eligible"] is not hard_eligible:
        raise ValueError("pilot hard eligibility differs from validation")
    if hard_eligible != (result["selection_checkpoint"] is not None):
        raise ValueError("pilot eligible checkpoint presence differs")
    for name, report in (("baseline", baseline), ("selection", validation)):
        if report.get("image_dataset_contract_sha256") != result[
            "image_dataset_contract_sha256"
        ]:
            raise ValueError(f"pilot {name} image dataset contract differs")
        text_report = report.get("text_replay")
        if not isinstance(text_report, Mapping) or text_report.get(
            "dataset_contract_sha256"
        ) != result["text_replay_dataset_contract_sha256"]:
            raise ValueError(f"pilot {name} text replay contract differs")
    _pilot_metrics(validation)
    baseline_records = _validate_paired_records(baseline)
    _validate_paired_records(validation)
    _paired_rows(baseline, validation)
    if _paired_sample_contract_sha256(baseline_records) != result[
        "paired_sample_contract_sha256"
    ]:
        raise ValueError("pilot paired sample contract differs")
    return deepcopy(result)


def _pilot_metrics(validation: Mapping[str, Any]) -> dict[str, float]:
    metrics = {
        "deployment_weighted_cer": _finite(
            validation.get("deployment_weighted_cer"),
            "deployment_weighted_cer",
            minimum=0.0,
        ),
        "worst_bucket_cer": _finite(
            validation.get("worst_bucket", {}).get("raw_grapheme_cer"),
            "worst_bucket_cer",
            minimum=0.0,
        ),
        "text_token_nll": _finite(
            validation.get("text_replay", {}).get("token_nll"),
            "text_token_nll",
            minimum=0.0,
        ),
        "raw_line_exact": _finite(
            validation.get("real", {}).get("overall", {}).get("raw_line_exact"),
            "raw_line_exact",
            minimum=0.0,
        ),
    }
    if metrics["raw_line_exact"] > 1.0:
        raise ValueError("raw_line_exact must be <= 1")
    return metrics


def _validate_paired_records(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = report.get("selection_records")
    if not isinstance(records, list) or not records:
        raise ValueError("validation selection_records must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    counts = {bucket: 0 for bucket in _DEPLOYMENT_BUCKET_WEIGHTS}
    for index, value in enumerate(records):
        row = _exact(value, _SELECTION_RECORD_KEYS, f"selection_records[{index}]")
        sample_id = row["sample_id"]
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or sample_id != sample_id.strip()
            or sample_id in seen
        ):
            raise ValueError("validation sample_id is empty, unsafe, or duplicated")
        bucket = row["bucket"]
        if bucket not in _DEPLOYMENT_BUCKET_WEIGHTS:
            raise ValueError("validation quota bucket is invalid")
        seen.add(sample_id)
        counts[bucket] += 1
        normalized.append(
            {
                "sample_id": sample_id,
                "bucket": bucket,
                "grapheme_edits": _nonnegative_int(
                    row["grapheme_edits"], "grapheme_edits"
                ),
                "reference_graphemes": _positive_int(
                    row["reference_graphemes"], "reference_graphemes"
                ),
            }
        )
    if [row["sample_id"] for row in normalized] != sorted(seen):
        raise ValueError("validation selection_records must be sorted by sample_id")
    if any(count < 2 for count in counts.values()):
        raise ValueError(
            "validation selection_records require two samples per quota bucket"
        )
    if report.get("deployment_weights") != _DEPLOYMENT_BUCKET_WEIGHTS:
        raise ValueError("validation deployment weights differ from selection contract")
    reported = _finite(
        report.get("deployment_weighted_cer"),
        "deployment_weighted_cer",
        minimum=0.0,
    )
    computed = _weighted_cer(normalized)
    if not math.isclose(reported, computed, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("validation weighted CER differs from per-sample records")
    return normalized


def _weighted_cer(records: Sequence[Mapping[str, Any]]) -> float:
    edits = {bucket: 0 for bucket in _DEPLOYMENT_BUCKET_WEIGHTS}
    references = {bucket: 0 for bucket in _DEPLOYMENT_BUCKET_WEIGHTS}
    for row in records:
        bucket = str(row["bucket"])
        edits[bucket] += int(row["grapheme_edits"])
        references[bucket] += int(row["reference_graphemes"])
    if any(value <= 0 for value in references.values()):
        raise ValueError("weighted CER requires references in every quota bucket")
    return sum(
        _DEPLOYMENT_BUCKET_WEIGHTS[bucket]
        * (edits[bucket] / references[bucket])
        for bucket in _DEPLOYMENT_BUCKET_WEIGHTS
    )


def _paired_sample_contract_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    return canonical_json_sha256(
        {
            "schema_version": 1,
            "kind": "dol_ocr_anyres_paired_validation_samples_v1",
            "samples": [
                {
                    "sample_id": row["sample_id"],
                    "bucket": row["bucket"],
                    "reference_graphemes": row["reference_graphemes"],
                }
                for row in records
            ],
        }
    )


def _paired_rows(
    left_report: Mapping[str, Any],
    right_report: Mapping[str, Any],
) -> dict[str, list[tuple[int, int, int]]]:
    left = _validate_paired_records(left_report)
    right = _validate_paired_records(right_report)
    if len(left) != len(right):
        raise ValueError("paired validations contain different sample counts")
    paired = {bucket: [] for bucket in _DEPLOYMENT_BUCKET_WEIGHTS}
    for left_row, right_row in zip(left, right, strict=True):
        identity = (
            left_row["sample_id"],
            left_row["bucket"],
            left_row["reference_graphemes"],
        )
        if identity != (
            right_row["sample_id"],
            right_row["bucket"],
            right_row["reference_graphemes"],
        ):
            raise ValueError("paired validations differ in sample identity or reference")
        paired[str(left_row["bucket"])].append(
            (
                int(left_row["grapheme_edits"]),
                int(right_row["grapheme_edits"]),
                int(left_row["reference_graphemes"]),
            )
        )
    return paired


def _paired_bootstrap_delta(
    left_report: Mapping[str, Any],
    right_report: Mapping[str, Any],
) -> dict[str, Any]:
    paired = _paired_rows(left_report, right_report)
    observed = _weighted_cer(_validate_paired_records(left_report)) - _weighted_cer(
        _validate_paired_records(right_report)
    )
    deltas: list[float] = []
    mask = (1 << 64) - 1
    for resample in range(_BOOTSTRAP_RESAMPLES):
        weighted_delta = 0.0
        for bucket, weight in _DEPLOYMENT_BUCKET_WEIGHTS.items():
            rows = paired[bucket]
            state = int(
                canonical_json_sha256(
                    {
                        "schema_version": 1,
                        "seed": _BOOTSTRAP_SEED,
                        "resample": resample,
                        "bucket": bucket,
                    }
                )[:16],
                16,
            )
            left_edits = 0
            right_edits = 0
            references = 0
            for _ in rows:
                state = (
                    6_364_136_223_846_793_005 * state
                    + 1_442_695_040_888_963_407
                ) & mask
                left_value, right_value, reference_value = rows[state % len(rows)]
                left_edits += left_value
                right_edits += right_value
                references += reference_value
            weighted_delta += weight * (
                left_edits / references - right_edits / references
            )
        deltas.append(weighted_delta)
    ordered = sorted(deltas)
    low = _linear_quantile(ordered, 0.025)
    high = _linear_quantile(ordered, 0.975)
    return {
        "observed_delta": observed,
        "bootstrap_mean_delta": sum(deltas) / len(deltas),
        "ci95_low": low,
        "ci95_high": high,
        "sample_count": sum(len(rows) for rows in paired.values()),
        "bucket_sample_counts": {
            bucket: len(paired[bucket]) for bucket in _DEPLOYMENT_BUCKET_WEIGHTS
        },
        "resamples": _BOOTSTRAP_RESAMPLES,
        "seed": _BOOTSTRAP_SEED,
        "ci_level": 0.95,
        "quantile_method": "sorted_linear_interpolation_at_r_minus_1_times_p_v1",
        "estimator": (
            "paired_stratified_deployment_weighted_bucket_ratio_of_sums_v1"
        ),
    }


def paired_stratified_validation_delta(
    left_report: Mapping[str, Any],
    right_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the fixed paired weighted-CER estimate ``left - right``."""

    return _paired_bootstrap_delta(left_report, right_report)


def _linear_quantile(values: Sequence[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("linear quantile input is invalid")
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(values[lower])
    fraction = position - lower
    return float(values[lower] * (1.0 - fraction) + values[upper] * fraction)


def _comparison(
    *,
    kind: str,
    candidate: Mapping[str, Any],
    estimate: Mapping[str, Any],
    reference: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "kind": kind,
        "candidate_result_sha256": candidate["canonical_sha256"],
        "candidate_kl_coef": float(candidate["kl_trial"]["selected_kl_coef"]),
        "reference_result_sha256": (
            None if reference is None else reference["canonical_sha256"]
        ),
        "reference_kl_coef": (
            None
            if reference is None
            else float(reference["kl_trial"]["selected_kl_coef"])
        ),
        "estimate": deepcopy(dict(estimate)),
    }
    payload["canonical_sha256"] = canonical_json_sha256(payload)
    return payload


def _validate_comparison(value: object) -> dict[str, Any]:
    keys = {
        "schema_version",
        "kind",
        "candidate_result_sha256",
        "candidate_kl_coef",
        "reference_result_sha256",
        "reference_kl_coef",
        "estimate",
        "canonical_sha256",
    }
    row = _exact(value, keys, "paired comparison")
    if row["schema_version"] != 1 or row["kind"] not in {
        "baseline_minus_candidate",
        "candidate_minus_point_best",
    }:
        raise ValueError("paired comparison kind differs")
    digest = _sha(row["canonical_sha256"], "paired comparison canonical_sha256")
    unhashed = {key: item for key, item in row.items() if key != "canonical_sha256"}
    if canonical_json_sha256(unhashed) != digest:
        raise ValueError("paired comparison canonical SHA-256 mismatch")
    _sha(row["candidate_result_sha256"], "candidate_result_sha256")
    if float(row["candidate_kl_coef"]) not in {0.04, 0.01, 0.0}:
        raise ValueError("paired comparison candidate KL is invalid")
    if row["kind"] == "baseline_minus_candidate":
        if row["reference_result_sha256"] is not None or row[
            "reference_kl_coef"
        ] is not None:
            raise ValueError("baseline comparison must not name a pilot reference")
    else:
        _sha(row["reference_result_sha256"], "reference_result_sha256")
        if float(row["reference_kl_coef"]) not in {0.04, 0.01, 0.0}:
            raise ValueError("paired comparison reference KL is invalid")
    estimate = _exact(
        row["estimate"],
        {
            "observed_delta",
            "bootstrap_mean_delta",
            "ci95_low",
            "ci95_high",
            "sample_count",
            "bucket_sample_counts",
            "resamples",
            "seed",
            "ci_level",
            "quantile_method",
            "estimator",
        },
        "paired bootstrap estimate",
    )
    for field in (
        "observed_delta",
        "bootstrap_mean_delta",
        "ci95_low",
        "ci95_high",
    ):
        _finite(estimate[field], field)
    if float(estimate["ci95_low"]) > float(estimate["ci95_high"]):
        raise ValueError("paired bootstrap confidence interval is inverted")
    _positive_int(estimate["sample_count"], "paired sample_count")
    counts = _exact(
        estimate["bucket_sample_counts"],
        set(_DEPLOYMENT_BUCKET_WEIGHTS),
        "paired bucket counts",
    )
    if sum(_positive_int(value, "paired bucket count") for value in counts.values()) != int(
        estimate["sample_count"]
    ):
        raise ValueError("paired bootstrap bucket counts differ from sample_count")
    if estimate["resamples"] != _BOOTSTRAP_RESAMPLES or estimate[
        "seed"
    ] != _BOOTSTRAP_SEED or estimate["ci_level"] != 0.95 or estimate[
        "quantile_method"
    ] != "sorted_linear_interpolation_at_r_minus_1_times_p_v1" or estimate[
        "estimator"
    ] != (
        "paired_stratified_deployment_weighted_bucket_ratio_of_sums_v1"
    ):
        raise ValueError("paired bootstrap protocol differs")
    return deepcopy(row)


def _selection_rule() -> dict[str, Any]:
    return {
        "name": ANYRES_GRPO_KL_SELECTION_RULE,
        "requires_all_candidates_completed": True,
        "candidate_kl_coefs": [0.04, 0.01, 0.0],
        "pilot_budget": {"unit": "rollout_attempts", "count": 200},
        "candidate_admission": {
            "hard_eligibility": True,
            "paired_baseline_improvement_ci95_low": "strictly_greater_than_zero",
        },
        "point_metric": "deployment_weighted_cer:ascending",
        "bootstrap": {
            "strata": list(_DEPLOYMENT_BUCKET_WEIGHTS),
            "weights": dict(_DEPLOYMENT_BUCKET_WEIGHTS),
            "resamples": _BOOTSTRAP_RESAMPLES,
            "seed": _BOOTSTRAP_SEED,
            "generator": "sha256_seeded_lcg64_v1",
            "quantile_method": (
                "sorted_linear_interpolation_at_r_minus_1_times_p_v1"
            ),
        },
        "equivalence": {
            "observed_delta_at_most_fraction_of_baseline": (
                _EQUIVALENCE_FRACTION_OF_BASELINE
            ),
            "candidate_minus_best_ci95_low": "less_than_or_equal_to_zero",
        },
        "tie_break": "highest_kl_among_statistically_equivalent_candidates",
        "zero_admitted_candidates": "no_go",
        "formal_restart": "parent_joint_checkpoint_not_pilot_checkpoint",
        "golden_data": "forbidden",
    }


def select_kl_pilot_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(results) != 3:
        raise ValueError("KL selection requires exactly three pilot results")
    candidates = [validate_kl_pilot_result(result) for result in results]
    candidate_kl = [float(row["kl_trial"]["selected_kl_coef"]) for row in candidates]
    if sorted(candidate_kl, reverse=True) != [0.04, 0.01, 0.0] or len(
        set(candidate_kl)
    ) != 3:
        raise ValueError("KL pilot results must cover exactly 0.04, 0.01, and 0")
    invariants = (
        "parent_joint_checkpoint",
        "parent_joint_stage_result_sha256",
        "admission_sha256",
        "baseline_validation_sha256",
        "paired_sample_contract_sha256",
        "image_dataset_contract_sha256",
        "text_replay_dataset_contract_sha256",
        "trial_comparison_contract_sha256",
        "source_closure_sha256",
        "dataset_ready_admission_sha256",
    )
    anchor = candidates[0]
    for candidate in candidates[1:]:
        for field in invariants:
            if candidate[field] != anchor[field]:
                raise ValueError(f"KL pilot invariant differs for {field}")
    if len({row["run_contract_sha256"] for row in candidates}) != 3:
        raise ValueError("KL pilots must come from distinct run contracts")
    if len({row["terminal_checkpoint"]["path"] for row in candidates}) != 3:
        raise ValueError("KL pilots must use isolated output roots")

    comparisons: list[dict[str, Any]] = []
    eligibility_rows: list[dict[str, Any]] = []
    admitted: list[dict[str, Any]] = []
    for candidate in sorted(
        candidates,
        key=lambda row: -float(row["kl_trial"]["selected_kl_coef"]),
    ):
        estimate = _paired_bootstrap_delta(
            candidate["baseline_validation"],
            candidate["selection_validation"],
        )
        comparison = _comparison(
            kind="baseline_minus_candidate",
            candidate=candidate,
            estimate=estimate,
            reference=None,
        )
        comparisons.append(comparison)
        significant = float(estimate["ci95_low"]) > 0.0
        reasons: list[str] = []
        if not candidate["hard_eligible"]:
            reasons.append("hard_eligibility_failed")
        if not significant:
            reasons.append("paired_baseline_improvement_not_significant")
        is_admitted = bool(candidate["hard_eligible"] and significant)
        eligibility_rows.append(
            {
                "result_sha256": candidate["canonical_sha256"],
                "kl_coef": float(candidate["kl_trial"]["selected_kl_coef"]),
                "hard_eligible": bool(candidate["hard_eligible"]),
                "paired_baseline_improvement_significant": significant,
                "admitted": is_admitted,
                "reasons": reasons,
                "baseline_comparison_sha256": comparison["canonical_sha256"],
            }
        )
        if is_admitted:
            admitted.append(candidate)

    selected: dict[str, Any] | None = None
    equivalent_kl: list[float] = []
    if admitted:
        point_best = min(
            admitted,
            key=lambda row: (
                float(row["selection_validation"]["deployment_weighted_cer"]),
                -float(row["kl_trial"]["selected_kl_coef"]),
            ),
        )
        baseline_cer = float(anchor["baseline_validation"]["deployment_weighted_cer"])
        margin = _EQUIVALENCE_FRACTION_OF_BASELINE * baseline_cer
        equivalent: list[dict[str, Any]] = []
        for candidate in admitted:
            estimate = _paired_bootstrap_delta(
                candidate["selection_validation"],
                point_best["selection_validation"],
            )
            comparisons.append(
                _comparison(
                    kind="candidate_minus_point_best",
                    candidate=candidate,
                    estimate=estimate,
                    reference=point_best,
                )
            )
            if float(estimate["observed_delta"]) <= margin and float(
                estimate["ci95_low"]
            ) <= 0.0:
                equivalent.append(candidate)
        selected = max(
            equivalent,
            key=lambda row: float(row["kl_trial"]["selected_kl_coef"]),
        )
        equivalent_kl = sorted(
            (float(row["kl_trial"]["selected_kl_coef"]) for row in equivalent),
            reverse=True,
        )

    rule = _selection_rule()
    payload = {
        "schema_version": 1,
        "kind": ANYRES_GRPO_KL_SELECTION_KIND,
        "selection_rule": rule,
        "selection_rule_sha256": canonical_json_sha256(rule),
        "candidate_result_sha256": sorted(
            row["canonical_sha256"] for row in candidates
        ),
        "selected_result_sha256": (
            None if selected is None else selected["canonical_sha256"]
        ),
        "selected_kl_coef": (
            None if selected is None else float(selected["kl_trial"]["selected_kl_coef"])
        ),
        "selected_metrics": (
            None if selected is None else _pilot_metrics(selected["selection_validation"])
        ),
        "status": "no_candidate_admitted" if selected is None else "selected",
        "formal_run_allowed": selected is not None,
        "candidate_eligibility": eligibility_rows,
        "paired_comparisons": sorted(
            comparisons,
            key=lambda row: (
                row["kind"],
                -float(row["candidate_kl_coef"]),
            ),
        ),
        "statistically_equivalent_kl_coefs": equivalent_kl,
        "paired_sample_contract_sha256": anchor[
            "paired_sample_contract_sha256"
        ],
        "image_dataset_contract_sha256": anchor[
            "image_dataset_contract_sha256"
        ],
        "text_replay_dataset_contract_sha256": anchor[
            "text_replay_dataset_contract_sha256"
        ],
        "trial_comparison_contract_sha256": anchor[
            "trial_comparison_contract_sha256"
        ],
        "parent_joint_checkpoint": deepcopy(anchor["parent_joint_checkpoint"]),
        "formal_init_checkpoint": (
            None if selected is None else deepcopy(anchor["parent_joint_checkpoint"])
        ),
        "parent_joint_stage_result_sha256": anchor[
            "parent_joint_stage_result_sha256"
        ],
        "admission_sha256": anchor["admission_sha256"],
        "dataset_ready_admission_sha256": anchor[
            "dataset_ready_admission_sha256"
        ],
        "source_closure_sha256": anchor["source_closure_sha256"],
    }
    payload["canonical_sha256"] = canonical_json_sha256(payload)
    return validate_kl_selection_receipt(payload)


def validate_kl_selection_receipt(value: object) -> dict[str, Any]:
    receipt = _exact(value, _SELECTION_KEYS, "KL selection receipt")
    if receipt["schema_version"] != 1 or receipt["kind"] != (
        ANYRES_GRPO_KL_SELECTION_KIND
    ):
        raise ValueError("unsupported KL selection receipt")
    digest = _sha(receipt["canonical_sha256"], "selection canonical_sha256")
    unhashed = {
        key: deepcopy(item)
        for key, item in receipt.items()
        if key != "canonical_sha256"
    }
    if canonical_json_sha256(unhashed) != digest:
        raise ValueError("KL selection receipt canonical SHA-256 mismatch")
    rule = receipt["selection_rule"]
    if rule != _selection_rule() or canonical_json_sha256(rule) != receipt[
        "selection_rule_sha256"
    ]:
        raise ValueError("KL selection rule differs")
    candidates = receipt["candidate_result_sha256"]
    if not isinstance(candidates, list) or len(candidates) != 3 or candidates != sorted(
        set(candidates)
    ):
        raise ValueError("selection candidate result hashes are invalid")
    for digest_value in candidates:
        _sha(digest_value, "candidate_result_sha256")
    rows = receipt["candidate_eligibility"]
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("candidate eligibility must contain three rows")
    row_keys = {
        "result_sha256",
        "kl_coef",
        "hard_eligible",
        "paired_baseline_improvement_significant",
        "admitted",
        "reasons",
        "baseline_comparison_sha256",
    }
    admitted_kl: set[float] = set()
    row_result_sha: set[str] = set()
    result_to_kl: dict[str, float] = {}
    ordered_row_kl: list[float] = []
    for value_row in rows:
        row = _exact(value_row, row_keys, "candidate eligibility")
        result_sha = _sha(row["result_sha256"], "candidate result SHA256")
        row_result_sha.add(result_sha)
        coef = float(row["kl_coef"])
        if coef not in {0.04, 0.01, 0.0}:
            raise ValueError("candidate eligibility KL is invalid")
        result_to_kl[result_sha] = coef
        ordered_row_kl.append(coef)
        if any(type(row[field]) is not bool for field in (
            "hard_eligible",
            "paired_baseline_improvement_significant",
            "admitted",
        )):
            raise ValueError("candidate eligibility flags must be bool")
        expected_admitted = bool(
            row["hard_eligible"]
            and row["paired_baseline_improvement_significant"]
        )
        if row["admitted"] is not expected_admitted:
            raise ValueError("candidate admitted flag differs from gates")
        expected_reasons = []
        if not row["hard_eligible"]:
            expected_reasons.append("hard_eligibility_failed")
        if not row["paired_baseline_improvement_significant"]:
            expected_reasons.append("paired_baseline_improvement_not_significant")
        if row["reasons"] != expected_reasons:
            raise ValueError("candidate eligibility reasons differ")
        _sha(row["baseline_comparison_sha256"], "baseline comparison SHA256")
        if row["admitted"]:
            admitted_kl.add(coef)
    if row_result_sha != set(candidates):
        raise ValueError("candidate eligibility results differ from candidates")
    if ordered_row_kl != [0.04, 0.01, 0.0]:
        raise ValueError("candidate eligibility rows must be ordered by descending KL")
    comparisons = receipt["paired_comparisons"]
    if not isinstance(comparisons, list) or len(comparisons) < 3:
        raise ValueError("paired comparisons are incomplete")
    validated_comparisons = [_validate_comparison(row) for row in comparisons]
    comparison_shas = [row["canonical_sha256"] for row in validated_comparisons]
    if len(comparison_shas) != len(set(comparison_shas)):
        raise ValueError("paired comparisons contain duplicates")
    baseline_shas = {
        row["canonical_sha256"]
        for row in validated_comparisons
        if row["kind"] == "baseline_minus_candidate"
    }
    if baseline_shas != {
        row["baseline_comparison_sha256"] for row in rows
    }:
        raise ValueError("candidate baseline comparison references differ")
    equivalent = receipt["statistically_equivalent_kl_coefs"]
    if not isinstance(equivalent, list) or equivalent != sorted(
        set(float(value) for value in equivalent), reverse=True
    ) or any(float(value) not in admitted_kl for value in equivalent):
        raise ValueError("statistically equivalent KL set is invalid")
    status = receipt["status"]
    if type(receipt["formal_run_allowed"]) is not bool:
        raise ValueError("formal_run_allowed must be bool")
    if status == "selected":
        if receipt["formal_run_allowed"] is not True:
            raise ValueError("selected receipt must allow formal training")
        if receipt["selected_result_sha256"] not in candidates:
            raise ValueError("selected result is absent from candidates")
        selected_kl = float(receipt["selected_kl_coef"])
        if selected_kl not in admitted_kl or not equivalent or selected_kl != max(
            equivalent
        ):
            raise ValueError("selected KL differs from conservative tie-break")
        if result_to_kl[receipt["selected_result_sha256"]] != selected_kl:
            raise ValueError("selected result and KL coefficient differ")
        metrics = receipt["selected_metrics"]
        if not isinstance(metrics, Mapping) or set(metrics) != {
            "deployment_weighted_cer",
            "worst_bucket_cer",
            "text_token_nll",
            "raw_line_exact",
        }:
            raise ValueError("selected metrics differ from contract")
        for field, metric in metrics.items():
            value_metric = _finite(metric, field, minimum=0.0)
            if field == "raw_line_exact" and value_metric > 1.0:
                raise ValueError("selected raw_line_exact exceeds one")
        parent = _checkpoint(
            receipt["parent_joint_checkpoint"], "parent_joint_checkpoint"
        )
        if _checkpoint(
            receipt["formal_init_checkpoint"], "formal_init_checkpoint"
        ) != parent:
            raise ValueError("formal run must restart from the parent joint checkpoint")
    elif status == "no_candidate_admitted":
        if receipt["formal_run_allowed"] is not False or admitted_kl:
            raise ValueError("no-candidate receipt conflicts with candidate gates")
        if any(
            receipt[field] is not None
            for field in (
                "selected_result_sha256",
                "selected_kl_coef",
                "selected_metrics",
                "formal_init_checkpoint",
            )
        ) or equivalent:
            raise ValueError("no-candidate receipt must not authorize a formal run")
    else:
        raise ValueError("KL selection status is invalid")
    _checkpoint(receipt["parent_joint_checkpoint"], "parent_joint_checkpoint")
    for field in (
        "selection_rule_sha256",
        "paired_sample_contract_sha256",
        "image_dataset_contract_sha256",
        "text_replay_dataset_contract_sha256",
        "trial_comparison_contract_sha256",
        "parent_joint_stage_result_sha256",
        "admission_sha256",
        "dataset_ready_admission_sha256",
        "source_closure_sha256",
    ):
        _sha(receipt[field], field)
    return deepcopy(receipt)


__all__ = [
    "ANYRES_GRPO_KL_SELECTION_KIND",
    "ANYRES_GRPO_KL_SELECTION_RULE",
    "ANYRES_GRPO_PILOT_RESULT_KIND",
    "ANYRES_GRPO_PROGRESS_KIND",
    "ANYRES_GRPO_RUN_KIND",
    "build_anyres_grpo_progress",
    "build_anyres_grpo_run_contract",
    "build_text_schedule_cursor_state",
    "paired_stratified_validation_delta",
    "build_kl_pilot_result",
    "select_kl_pilot_results",
    "validate_anyres_grpo_progress",
    "validate_anyres_grpo_run_contract",
    "validate_kl_pilot_result",
    "validate_kl_selection_receipt",
]
