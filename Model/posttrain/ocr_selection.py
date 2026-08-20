# -*- coding: utf-8 -*-

"""Terminal selection receipt that keeps locked golden closed during training."""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from Model.ocr.tokenization import canonical_json_sha256
from Model.posttrain.checkpointing import (
    load_verified_checkpoint_metadata_envelope,
    verified_file_sha256,
)
from Model.posttrain.ocr_anyres_grpo_run import (
    validate_anyres_grpo_progress,
    validate_anyres_grpo_run_contract,
    validate_kl_selection_receipt,
)
from Model.posttrain.ocr_joint_eval import (
    DEPLOYMENT_BUCKET_WEIGHTS,
    joint_eval_eligibility,
)
from Model.posttrain.ocr_manifest_builder import file_sha256 as _file_sha256
from Model.training.checkpoint import load_checkpoint_metadata

OCR_SELECTION_RECEIPT_SCHEMA_VERSION = 2
OCR_ANYRES_GRPO_SELECTION_RECEIPT_SCHEMA_VERSION = 1
OCR_ANYRES_GRPO_SELECTION_RECEIPT_KIND = (
    "dol_ocr_anyres_grpo_selection_finalized_v1"
)
OCR_ANYRES_GRPO_SELECTION_RECEIPT_FILENAME = "SELECTION_FINALIZED.json"
OCR_ANYRES_GRPO_SUCCESSFUL_STOP_REASONS = frozenset(
    {
        "completed",
        "max_optimizer_steps",
        "max_rollout_attempts",
        "validation_plateau",
    }
)
_STEP_RE = re.compile(r"step_(\d+)")
_ANYRES_STEP_RE = re.compile(r"step_(\d{8})")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

_ANYRES_CHECKPOINT_IDENTITY_KEYS = {
    "path",
    "model_sha256",
    "metadata_sha256",
    "outer_optimizer_step",
}
_ANYRES_SELECTION_RECEIPT_KEYS = {
    "schema_version",
    "kind",
    "selected_checkpoint",
    "terminal_checkpoint",
    "formal_run_sha256",
    "terminal_progress_sha256",
    "stop_reason",
    "kl_selection_receipt_sha256",
    "selected_kl_coef",
    "parent_joint_checkpoint",
    "parent_joint_stage_result_sha256",
    "reference_model_sha256",
    "reference_metadata_sha256",
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
    "source_closure_sha256",
    "formal_monitor_best_validation_sha256",
    "locked_golden_anchor_sha256",
    "canonical_sha256",
}

_ANYRES_RUN_METADATA_KEY = "ocr_anyres_grpo_run_contract"
_ANYRES_RUN_SHA256_METADATA_KEY = "ocr_anyres_grpo_run_contract_sha256"
_ANYRES_PROGRESS_METADATA_KEY = "ocr_anyres_grpo_progress"
_ANYRES_PROGRESS_SHA256_METADATA_KEY = "ocr_anyres_grpo_progress_sha256"
_ANYRES_FORMAL_MONITOR_METADATA_KEY = "formal_monitor_best_validation"
_ANYRES_RUNTIME_BASELINE_METADATA_KEY = "formal_monitor_runtime_baseline"
_ANYRES_DATASET_ADMISSION_METADATA_KEY = "dataset_admission_report"
_ANYRES_RUNTIME_SOURCE_METADATA_KEY = "runtime_source_receipt"


def _step_from_checkpoint(path: Path) -> int:
    match = _STEP_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"checkpoint name is not step_XXXXXXXX: {path.name}")
    return int(match.group(1))


def _validate_terminal_metadata(
    checkpoint: Path,
    *,
    step: int,
    stop_reason: str,
    data_contract: dict[str, Any],
    selected_checkpoint: Path,
    selected_step: int,
) -> None:
    metadata = load_checkpoint_metadata(checkpoint)
    errors: list[str] = []
    if metadata.get("phase") != "grpo" or metadata.get("task") != "ocr":
        errors.append("terminal checkpoint is not OCR GRPO")
    if metadata.get("final") is not True:
        errors.append("terminal checkpoint final flag is not true")
    if metadata.get("stop_reason") != stop_reason:
        errors.append("terminal checkpoint stop_reason differs")
    health = metadata.get("health_state")
    if not isinstance(health, dict):
        errors.append("terminal checkpoint has no health_state")
    else:
        if health.get("stop_reason") != stop_reason:
            errors.append("terminal health_state stop_reason differs")
        if health.get("best_val_eligible") is not True:
            errors.append("terminal health_state selected best is not eligible")
        best_val_step = health.get("best_val_step")
        if (
            type(best_val_step) is not int
            or best_val_step != selected_step
        ):
            errors.append("terminal health_state best_val_step differs")
        best_checkpoint = health.get("best_checkpoint")
        if not isinstance(best_checkpoint, str) or not best_checkpoint:
            errors.append("terminal health_state best_checkpoint is missing")
        else:
            recorded_best = Path(best_checkpoint)
            try:
                recorded_step = _step_from_checkpoint(recorded_best)
            except ValueError as exc:
                errors.append(str(exc))
            else:
                if (
                    recorded_best.parent.name != "best"
                    or recorded_best.name != selected_checkpoint.name
                    or recorded_step != selected_step
                ):
                    errors.append(
                        "terminal health_state best_checkpoint differs"
                    )
        expected_selected = checkpoint.parent / "best" / selected_checkpoint.name
        try:
            current_selected = selected_checkpoint.resolve(strict=True)
            canonical_selected = expected_selected.resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            errors.append(
                "selected checkpoint cannot be resolved in canonical best/: "
                f"{exc}"
            )
        else:
            if current_selected != canonical_selected:
                errors.append(
                    "selected checkpoint is not terminal sibling best/<step>"
                )
    if metadata.get("data_contract") != data_contract:
        errors.append("terminal checkpoint data contract differs")
    if _step_from_checkpoint(checkpoint) != step:
        errors.append("terminal checkpoint step differs")
    if errors:
        raise ValueError(
            "unsafe terminal selection metadata:\n  - "
            + "\n  - ".join(errors)
        )


def build_selection_receipt(
    *,
    selected_checkpoint: str | Path,
    selected_step: int,
    terminal_checkpoint: str | Path,
    terminal_step: int,
    stop_reason: str,
    data_contract: dict[str, Any],
) -> dict[str, Any]:
    selected = Path(selected_checkpoint)
    terminal = Path(terminal_checkpoint)
    if stop_reason in {"", "running"}:
        raise ValueError("selection cannot be finalized while training is running")
    if type(selected_step) is not int or selected_step <= 0:
        raise ValueError("selection requires a post-update best checkpoint")
    if type(terminal_step) is not int or terminal_step < selected_step:
        raise ValueError(
            "terminal_step must be an integer at or after selected_step"
        )
    for name, checkpoint in (
        ("selected", selected),
        ("terminal", terminal),
    ):
        if not (checkpoint / "COMPLETE").is_file():
            raise ValueError(f"{name} checkpoint has no COMPLETE marker")
        if not (checkpoint / "model.pt").is_file():
            raise ValueError(f"{name} checkpoint has no model.pt")
    if _step_from_checkpoint(selected) != selected_step:
        raise ValueError("selected checkpoint name differs from selected_step")
    _validate_terminal_metadata(
        terminal,
        step=terminal_step,
        stop_reason=stop_reason,
        data_contract=data_contract,
        selected_checkpoint=selected,
        selected_step=selected_step,
    )
    return {
        "schema_version": OCR_SELECTION_RECEIPT_SCHEMA_VERSION,
        "kind": "ocr_grpo_selection_finalized",
        "selected_checkpoint_name": selected.name,
        "selected_step": int(selected_step),
        "selected_model_sha256": _file_sha256(selected / "model.pt"),
        "terminal_checkpoint_name": terminal.name,
        "terminal_step": int(terminal_step),
        "terminal_model_sha256": _file_sha256(terminal / "model.pt"),
        "stop_reason": stop_reason,
        "data_contract_canonical_sha256": canonical_json_sha256(data_contract),
    }


def write_selection_receipt(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    if destination.exists():
        existing = destination.read_text(encoding="utf-8")
        if existing == rendered:
            return
        raise FileExistsError(
            f"refusing to replace a different selection receipt: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_and_validate_selection_receipt(
    path: str | Path,
    *,
    selected_checkpoint: str | Path,
    data_contract: dict[str, Any],
) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(
            "locked golden remains closed: terminal selection receipt is missing"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid selection receipt JSON: {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("selection receipt must be a JSON object")
    selected = Path(selected_checkpoint)
    errors: list[str] = []
    canonical_source = selected.parent / "SELECTION_FINALIZED.json"
    if source.is_symlink() or source.absolute() != canonical_source.absolute():
        errors.append(
            "selection receipt must be canonical beside the selected checkpoint"
        )
    if payload.get("schema_version") != OCR_SELECTION_RECEIPT_SCHEMA_VERSION:
        errors.append("schema_version differs from runtime")
    if payload.get("kind") != "ocr_grpo_selection_finalized":
        errors.append("kind is not an OCR selection finalization")
    if payload.get("selected_checkpoint_name") != selected.name:
        errors.append("selected checkpoint name differs")
    selected_step = payload.get("selected_step")
    selected_checkpoint_step = _step_from_checkpoint(selected)
    if (
        type(selected_step) is not int
        or selected_step != selected_checkpoint_step
    ):
        errors.append("selected step differs")
    if payload.get("selected_model_sha256") != _file_sha256(
        selected / "model.pt"
    ):
        errors.append("selected model SHA-256 differs")
    if payload.get("stop_reason") in {None, "", "running"}:
        errors.append("training stop_reason is not terminal")
    if payload.get(
        "data_contract_canonical_sha256"
    ) != canonical_json_sha256(data_contract):
        errors.append("training data contract differs")
    terminal_name = payload.get("terminal_checkpoint_name")
    terminal_step = payload.get("terminal_step")
    if not isinstance(terminal_name, str) or not _STEP_RE.fullmatch(
        terminal_name
    ):
        errors.append("terminal checkpoint name is invalid")
    elif type(terminal_step) is not int:
        errors.append("terminal step is invalid")
    else:
        terminal = source.parent.parent / terminal_name
        if not (terminal / "COMPLETE").is_file():
            errors.append("terminal checkpoint has no COMPLETE marker")
        elif not (terminal / "model.pt").is_file():
            errors.append("terminal checkpoint has no model.pt")
        else:
            if payload.get("terminal_model_sha256") != _file_sha256(
                terminal / "model.pt"
            ):
                errors.append("terminal model SHA-256 differs")
            try:
                _validate_terminal_metadata(
                    terminal,
                    step=terminal_step,
                    stop_reason=str(payload.get("stop_reason", "")),
                    data_contract=data_contract,
                    selected_checkpoint=selected,
                    selected_step=selected_checkpoint_step,
                )
            except (FileNotFoundError, TypeError, ValueError) as exc:
                errors.append(str(exc))
    if errors:
        raise ValueError(
            "unsafe OCR selection receipt:\n  - " + "\n  - ".join(errors)
        )
    return {**payload, "receipt_file_sha256": _file_sha256(source)}


def _require_anyres_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_exact_mapping(
    value: object,
    keys: set[str],
    field: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{field} fields differ from contract")
    return dict(value)


def _read_complete_marker(path: Path, *, expected_step: int) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    elif path.is_symlink():  # pragma: no cover - modern macOS/Linux
        raise ValueError(f"checkpoint COMPLETE must not be a symlink: {path}")
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"checkpoint COMPLETE is unreadable: {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise ValueError(
                f"checkpoint COMPLETE must be a non-empty regular file: {path}"
            )
        marker = os.read(descriptor, 128)
        if os.read(descriptor, 1):
            raise ValueError(f"checkpoint COMPLETE is unexpectedly large: {path}")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise ValueError(f"checkpoint COMPLETE changed while being read: {path}")
    if marker != f"step={expected_step}\n".encode("ascii"):
        raise ValueError(
            "checkpoint COMPLETE differs from metadata outer optimizer step: "
            f"{path}"
        )


def _anyres_checkpoint_identity(
    checkpoint: str | Path,
    *,
    role: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(checkpoint)
    try:
        canonical = source.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ValueError(f"{role} checkpoint cannot be resolved: {exc}") from exc
    if not canonical.is_dir():
        raise ValueError(f"{role} checkpoint must be a directory")
    if source.is_symlink() or not source.is_absolute() or source != canonical:
        raise ValueError(f"{role} checkpoint path must be canonical absolute")
    match = _ANYRES_STEP_RE.fullmatch(canonical.name)
    if match is None:
        raise ValueError(f"{role} checkpoint name must be step_XXXXXXXX")
    model_path = canonical / "model.pt"
    metadata_path = canonical / "meta.pt"
    complete_path = canonical / "COMPLETE"
    try:
        model_sha256 = verified_file_sha256(model_path)
        envelope, metadata_sha256 = load_verified_checkpoint_metadata_envelope(
            metadata_path
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"{role} checkpoint identity is invalid: {exc}") from exc
    outer_step = int(envelope["step"])
    if int(match.group(1)) != outer_step:
        raise ValueError(
            f"{role} checkpoint name differs from metadata outer optimizer step"
        )
    _read_complete_marker(complete_path, expected_step=outer_step)
    return (
        {
            "path": str(canonical),
            "model_sha256": model_sha256,
            "metadata_sha256": metadata_sha256,
            "outer_optimizer_step": outer_step,
        },
        envelope,
    )


def _require_anyres_resumable_checkpoint(
    identity: Mapping[str, Any],
    *,
    run: Mapping[str, Any],
    role: str,
) -> None:
    checkpoint = Path(str(identity["path"]))
    required = ["optimizer.pt", "scheduler.pt", "rng.pt"]
    if run["precision"] == "fp16":
        required.append("scaler.pt")
    for name in required:
        member = checkpoint / name
        try:
            verified_file_sha256(member)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            raise ValueError(
                f"{role} checkpoint is not fully resumable: {name}: {exc}"
            ) from exc


def _validate_anyres_checkpoint_metadata(
    envelope: Mapping[str, Any],
    *,
    run: Mapping[str, Any],
    role: str,
    locked_golden_anchor_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = envelope.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{role} checkpoint metadata must be an object")
    if metadata.get("phase") != "grpo" or metadata.get("task") != "ocr":
        raise ValueError(f"{role} checkpoint is not OCR GRPO")
    if metadata.get("training_stage") != "grpo_formal":
        raise ValueError(f"{role} checkpoint is not from formal GRPO")
    embedded_run = validate_anyres_grpo_run_contract(
        metadata.get(_ANYRES_RUN_METADATA_KEY)
    )
    if embedded_run != dict(run) or metadata.get(
        _ANYRES_RUN_SHA256_METADATA_KEY
    ) != run["canonical_sha256"]:
        raise ValueError(f"{role} checkpoint formal run contract differs")
    embedded_progress = validate_anyres_grpo_progress(
        metadata.get(_ANYRES_PROGRESS_METADATA_KEY),
        run_contract=run,
    )
    if metadata.get(_ANYRES_PROGRESS_SHA256_METADATA_KEY) != embedded_progress[
        "canonical_sha256"
    ]:
        raise ValueError(f"{role} checkpoint progress SHA-256 differs")
    if envelope.get("step") != embedded_progress["optimizer_steps"]:
        raise ValueError(
            f"{role} outer optimizer step differs from embedded progress"
        )
    if metadata.get("final") is not embedded_progress["terminal"]:
        raise ValueError(f"{role} final flag differs from embedded progress")
    if metadata.get("stop_reason") != embedded_progress["stop_reason"]:
        raise ValueError(f"{role} stop_reason differs from embedded progress")
    if metadata.get("locked_golden_opened") is not False:
        raise ValueError(f"{role} checkpoint does not keep locked golden closed")
    if metadata.get("locked_golden_anchor_sha256") != (
        locked_golden_anchor_sha256
    ):
        raise ValueError(f"{role} locked golden anchor SHA-256 differs")

    dataset_admission = metadata.get(_ANYRES_DATASET_ADMISSION_METADATA_KEY)
    if not isinstance(dataset_admission, Mapping) or canonical_json_sha256(
        dataset_admission
    ) != run["dataset_ready_admission_sha256"]:
        raise ValueError(f"{role} dataset admission report differs from formal run")
    runtime_source = metadata.get(_ANYRES_RUNTIME_SOURCE_METADATA_KEY)
    if not isinstance(runtime_source, Mapping):
        raise ValueError(f"{role} runtime source receipt is missing")
    source_base = dict(runtime_source)
    source_sha = source_base.pop("canonical_sha256", None)
    if (
        source_sha != run["source_closure_sha256"]
        or canonical_json_sha256(source_base) != source_sha
    ):
        raise ValueError(f"{role} runtime source receipt differs from formal run")

    runtime_baseline = metadata.get(_ANYRES_RUNTIME_BASELINE_METADATA_KEY)
    if not isinstance(runtime_baseline, Mapping) or canonical_json_sha256(
        runtime_baseline
    ) != run["baseline_validation_sha256"]:
        raise ValueError(f"{role} formal-monitor runtime baseline differs")
    best_validation = metadata.get(_ANYRES_FORMAL_MONITOR_METADATA_KEY)
    if not isinstance(best_validation, Mapping) or canonical_json_sha256(
        best_validation
    ) != embedded_progress["best_validation_sha256"]:
        raise ValueError(f"{role} formal-monitor best validation differs")
    _validate_formal_monitor_role(
        runtime_baseline,
        run=run,
        dataset_admission=dataset_admission,
        role=f"{role} formal-monitor runtime baseline",
    )
    _validate_formal_monitor_role(
        best_validation,
        run=run,
        dataset_admission=dataset_admission,
        role=f"{role} formal-monitor best validation",
    )
    try:
        baseline_bucket_cer = {
            bucket: float(
                runtime_baseline["real"]["buckets"][bucket][
                    "raw_grapheme_cer"
                ]
            )
            for bucket in DEPLOYMENT_BUCKET_WEIGHTS
        }
        recomputed_baseline_eligibility = joint_eval_eligibility(
            runtime_baseline,
            baseline_bucket_cer=baseline_bucket_cer,
            baseline_text_token_nll=float(
                runtime_baseline["text_replay"]["token_nll"]
            ),
            min_relative_cer_improvement=0.0,
        )
        recomputed_eligibility = joint_eval_eligibility(
            best_validation,
            baseline_bucket_cer=baseline_bucket_cer,
            baseline_text_token_nll=float(
                runtime_baseline["text_replay"]["token_nll"]
            ),
            min_relative_cer_improvement=0.0,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{role} formal-monitor validation cannot be revalidated: {exc}"
        ) from exc
    if runtime_baseline.get("eligibility") != (
        recomputed_baseline_eligibility
    ):
        raise ValueError(
            f"{role} formal-monitor baseline eligibility differs from "
            "recomputation"
        )
    if recomputed_baseline_eligibility["eligible"] is not True:
        raise ValueError(f"{role} formal-monitor runtime baseline is ineligible")
    if best_validation.get("eligibility") != recomputed_eligibility:
        raise ValueError(
            f"{role} formal-monitor eligibility differs from recomputation"
        )
    if recomputed_eligibility["eligible"] is not True:
        raise ValueError(f"{role} formal-monitor selected best is ineligible")
    return dict(metadata), embedded_progress


def _validate_formal_monitor_role(
    report: Mapping[str, Any],
    *,
    run: Mapping[str, Any],
    dataset_admission: Mapping[str, Any],
    role: str,
) -> None:
    if report.get("image_dataset_contract_sha256") != run[
        "formal_monitor_dataset_contract_sha256"
    ]:
        raise ValueError(f"{role} image dataset is not formal_monitor")
    text_report = report.get("text_replay")
    if not isinstance(text_report, Mapping) or text_report.get(
        "dataset_contract_sha256"
    ) != run["text_replay_formal_monitor_contract_sha256"]:
        raise ValueError(f"{role} text dataset is not formal_monitor")
    try:
        image_registration = dataset_admission["datasets"]["formal_monitor"]
        image_contract = image_registration["contract"]
        text_registration = dataset_admission["text_replay"]["splits"][
            "formal_monitor"
        ]
        text_contract = text_registration["contract"]
        registered_image_ids = list(image_contract["sample_ids"])
        registered_text_tokens = int(
            text_contract["token_stats"]["total_sequence_tokens"]
        ) - int(text_contract["token_stats"]["samples"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{role} dataset admission lacks formal_monitor coverage: {exc}"
        ) from exc
    if (
        image_registration.get("contract_sha256")
        != run["formal_monitor_dataset_contract_sha256"]
        or image_contract.get("contract_sha256")
        != run["formal_monitor_dataset_contract_sha256"]
        or text_registration.get("contract_sha256")
        != run["text_replay_formal_monitor_contract_sha256"]
        or text_contract.get("contract_sha256")
        != run["text_replay_formal_monitor_contract_sha256"]
    ):
        raise ValueError(f"{role} dataset admission contract differs")
    records = report.get("selection_records")
    if not isinstance(records, list):
        raise ValueError(f"{role} has no per-sample coverage records")
    try:
        evaluated_image_ids = [str(record["sample_id"]) for record in records]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{role} sample coverage records are invalid") from exc
    if (
        evaluated_image_ids != sorted(set(evaluated_image_ids))
        or set(evaluated_image_ids) != set(registered_image_ids)
    ):
        raise ValueError(f"{role} does not cover the full formal_monitor image set")
    if text_report.get("target_tokens") != registered_text_tokens:
        raise ValueError(f"{role} does not cover all formal_monitor text targets")


def _validate_progress_selected_best(
    progress: Mapping[str, Any],
    *,
    selected_identity: Mapping[str, Any],
    formal_monitor_best_validation_sha256: str,
    role: str,
) -> None:
    if progress.get("best_validation_eligible") is not True:
        raise ValueError(f"{role} progress selected best is not eligible")
    if progress.get("best_validation_step") != selected_identity[
        "outer_optimizer_step"
    ]:
        raise ValueError(f"{role} progress best step differs from selected")
    if progress.get("best_checkpoint") != selected_identity["path"]:
        raise ValueError(f"{role} progress best checkpoint differs from selected")
    if progress.get("best_validation_sha256") != (
        formal_monitor_best_validation_sha256
    ):
        raise ValueError(
            f"{role} progress best validation differs from formal monitor best"
        )


def _validate_anyres_checkpoint_layout(
    selected_identity: Mapping[str, Any],
    terminal_identity: Mapping[str, Any],
) -> None:
    selected = Path(str(selected_identity["path"]))
    terminal = Path(str(terminal_identity["path"]))
    output = terminal.parent
    expected_selected = output / "best" / selected.name
    if selected != expected_selected:
        raise ValueError("selected checkpoint must be <output>/best/step_XXXXXXXX")
    if terminal.parent != output or terminal.parent.name == "best":
        raise ValueError("terminal checkpoint must be <output>/step_XXXXXXXX")
    if selected.parent.parent != output:
        raise ValueError("selected and terminal checkpoints use different outputs")


def _anyres_lineage_from_run(run: Mapping[str, Any]) -> dict[str, Any]:
    admission = run["admission"]
    if not isinstance(admission, Mapping):  # already rejected by run validator
        raise ValueError("formal run admission must be an object")
    reference_model = _require_anyres_sha256(
        admission.get("reference_checkpoint_sha256"),
        "reference model SHA-256",
    )
    reference_metadata = _require_anyres_sha256(
        admission.get("reference_metadata_sha256"),
        "reference metadata SHA-256",
    )
    if admission.get("dataset_admission_report_sha256") != run[
        "dataset_ready_admission_sha256"
    ]:
        raise ValueError("formal run admission dataset report SHA-256 differs")
    return {
        "formal_run_sha256": run["canonical_sha256"],
        "kl_selection_receipt_sha256": run[
            "formal_selection_receipt_sha256"
        ],
        "selected_kl_coef": float(run["kl_trial"]["selected_kl_coef"]),
        "parent_joint_checkpoint": deepcopy(run["parent_joint_checkpoint"]),
        "parent_joint_stage_result_sha256": run[
            "parent_joint_stage_result_sha256"
        ],
        "reference_model_sha256": reference_model,
        "reference_metadata_sha256": reference_metadata,
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
        "source_closure_sha256": run["source_closure_sha256"],
    }


def build_anyres_grpo_selection_receipt(
    *,
    selected_checkpoint: str | Path,
    terminal_checkpoint: str | Path,
    formal_run_contract: Mapping[str, Any],
    terminal_progress: Mapping[str, Any],
    formal_monitor_best_validation_sha256: str,
    locked_golden_anchor_sha256: str,
) -> dict[str, Any]:
    """Finalize a formal anyres checkpoint without opening locked golden.

    ``locked_golden_anchor_sha256`` is treated only as an opaque, label-free
    anchor.  This function never resolves or reads a locked-golden artifact.
    """

    run = validate_anyres_grpo_run_contract(formal_run_contract)
    if run["mode"] != "formal":
        raise ValueError("anyres final selection requires a formal GRPO run")
    progress = validate_anyres_grpo_progress(
        terminal_progress,
        run_contract=run,
    )
    formal_monitor_sha = _require_anyres_sha256(
        formal_monitor_best_validation_sha256,
        "formal monitor best validation SHA-256",
    )
    golden_anchor_sha = _require_anyres_sha256(
        locked_golden_anchor_sha256,
        "locked golden anchor SHA-256",
    )
    selected_identity, _ = _anyres_checkpoint_identity(
        selected_checkpoint,
        role="selected",
    )
    terminal_identity, _ = _anyres_checkpoint_identity(
        terminal_checkpoint,
        role="terminal",
    )
    payload = {
        "schema_version": OCR_ANYRES_GRPO_SELECTION_RECEIPT_SCHEMA_VERSION,
        "kind": OCR_ANYRES_GRPO_SELECTION_RECEIPT_KIND,
        "selected_checkpoint": selected_identity,
        "terminal_checkpoint": terminal_identity,
        **_anyres_lineage_from_run(run),
        "terminal_progress_sha256": progress["canonical_sha256"],
        "stop_reason": progress["stop_reason"],
        "formal_monitor_best_validation_sha256": formal_monitor_sha,
        "locked_golden_anchor_sha256": golden_anchor_sha,
    }
    payload["canonical_sha256"] = canonical_json_sha256(payload)
    return validate_anyres_grpo_selection_receipt(
        payload,
        formal_run_contract=run,
        terminal_progress=progress,
    )


def validate_anyres_grpo_selection_receipt(
    value: object,
    *,
    formal_run_contract: Mapping[str, Any],
    terminal_progress: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate an anyres formal-selection receipt against live bytes."""

    receipt = _require_exact_mapping(
        value,
        _ANYRES_SELECTION_RECEIPT_KEYS,
        "anyres GRPO selection receipt",
    )
    if receipt["schema_version"] != (
        OCR_ANYRES_GRPO_SELECTION_RECEIPT_SCHEMA_VERSION
    ) or receipt["kind"] != OCR_ANYRES_GRPO_SELECTION_RECEIPT_KIND:
        raise ValueError("unsupported anyres GRPO selection receipt")
    canonical_sha = _require_anyres_sha256(
        receipt["canonical_sha256"],
        "anyres selection canonical SHA-256",
    )
    unhashed = {
        key: deepcopy(item)
        for key, item in receipt.items()
        if key != "canonical_sha256"
    }
    if canonical_json_sha256(unhashed) != canonical_sha:
        raise ValueError("anyres GRPO selection receipt canonical SHA-256 mismatch")

    run = validate_anyres_grpo_run_contract(formal_run_contract)
    if run["mode"] != "formal":
        raise ValueError("anyres final selection requires a formal GRPO run")
    progress = validate_anyres_grpo_progress(
        terminal_progress,
        run_contract=run,
    )
    if progress["terminal"] is not True or progress["stop_reason"] == "running":
        raise ValueError("formal GRPO progress is not terminal")
    if progress["stop_reason"] not in OCR_ANYRES_GRPO_SUCCESSFUL_STOP_REASONS:
        raise ValueError(
            "formal GRPO stop_reason is not a successful terminal reason"
        )
    if receipt["terminal_progress_sha256"] != progress["canonical_sha256"]:
        raise ValueError("terminal progress SHA-256 differs from receipt")
    if receipt["stop_reason"] != progress["stop_reason"]:
        raise ValueError("terminal stop_reason differs from receipt")

    kl_selection = validate_kl_selection_receipt(
        run["formal_selection_receipt"]
    )
    if (
        kl_selection["status"] != "selected"
        or kl_selection["formal_run_allowed"] is not True
        or kl_selection["canonical_sha256"]
        != run["formal_selection_receipt_sha256"]
    ):
        raise ValueError("formal run KL selection receipt does not authorize training")
    if float(kl_selection["selected_kl_coef"]) != float(
        run["kl_trial"]["selected_kl_coef"]
    ):
        raise ValueError("formal run selected KL differs from KL selection receipt")

    expected_lineage = _anyres_lineage_from_run(run)
    for field, expected in expected_lineage.items():
        if receipt[field] != expected:
            raise ValueError(f"anyres selection receipt lineage differs for {field}")
    for field in (
        "terminal_progress_sha256",
        "formal_monitor_best_validation_sha256",
        "locked_golden_anchor_sha256",
    ):
        _require_anyres_sha256(receipt[field], field)

    selected_recorded = _require_exact_mapping(
        receipt["selected_checkpoint"],
        _ANYRES_CHECKPOINT_IDENTITY_KEYS,
        "selected checkpoint identity",
    )
    terminal_recorded = _require_exact_mapping(
        receipt["terminal_checkpoint"],
        _ANYRES_CHECKPOINT_IDENTITY_KEYS,
        "terminal checkpoint identity",
    )
    selected_actual, selected_envelope = _anyres_checkpoint_identity(
        selected_recorded["path"],
        role="selected",
    )
    terminal_actual, terminal_envelope = _anyres_checkpoint_identity(
        terminal_recorded["path"],
        role="terminal",
    )
    if selected_recorded != selected_actual:
        raise ValueError("selected checkpoint identity changed after finalization")
    if terminal_recorded != terminal_actual:
        raise ValueError("terminal checkpoint identity changed after finalization")
    _validate_anyres_checkpoint_layout(selected_actual, terminal_actual)
    _require_anyres_resumable_checkpoint(
        terminal_actual,
        run=run,
        role="terminal",
    )
    if (
        selected_actual["outer_optimizer_step"]
        == terminal_actual["outer_optimizer_step"]
        and selected_actual["model_sha256"] != terminal_actual["model_sha256"]
    ):
        raise ValueError(
            "same-step selected and terminal model bytes differ"
        )

    locked_anchor_sha = receipt["locked_golden_anchor_sha256"]
    _, selected_progress = _validate_anyres_checkpoint_metadata(
        selected_envelope,
        run=run,
        role="selected",
        locked_golden_anchor_sha256=locked_anchor_sha,
    )
    _, terminal_embedded_progress = _validate_anyres_checkpoint_metadata(
        terminal_envelope,
        run=run,
        role="terminal",
        locked_golden_anchor_sha256=locked_anchor_sha,
    )
    if terminal_embedded_progress != progress:
        raise ValueError("terminal checkpoint embeds another progress state")
    if terminal_actual["outer_optimizer_step"] != progress[
        "optimizer_steps"
    ]:
        raise ValueError("terminal outer optimizer step differs from progress")
    formal_monitor_sha = receipt["formal_monitor_best_validation_sha256"]
    _validate_progress_selected_best(
        selected_progress,
        selected_identity=selected_actual,
        formal_monitor_best_validation_sha256=formal_monitor_sha,
        role="selected",
    )
    _validate_progress_selected_best(
        progress,
        selected_identity=selected_actual,
        formal_monitor_best_validation_sha256=formal_monitor_sha,
        role="terminal",
    )
    return deepcopy(receipt)


def _fsync_parent_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        # Individual receipt bytes are still fsynced before the atomic rename.
        pass


def write_anyres_grpo_selection_receipt(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    formal_run_contract: Mapping[str, Any],
    terminal_progress: Mapping[str, Any],
) -> None:
    """Validate and atomically publish the canonical formal receipt."""

    receipt = validate_anyres_grpo_selection_receipt(
        payload,
        formal_run_contract=formal_run_contract,
        terminal_progress=terminal_progress,
    )
    destination = Path(path)
    expected = (
        Path(receipt["selected_checkpoint"]["path"]).parent
        / OCR_ANYRES_GRPO_SELECTION_RECEIPT_FILENAME
    )
    if destination.absolute() != expected:
        raise ValueError(
            "anyres selection receipt must be canonical beside best checkpoints"
        )
    rendered = json.dumps(
        receipt,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    if destination.is_symlink():
        raise ValueError("anyres selection receipt destination must not be a symlink")
    if destination.exists():
        existing = destination.read_text(encoding="utf-8")
        if existing == rendered:
            return
        raise FileExistsError(
            f"refusing to replace a different anyres selection receipt: "
            f"{destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.tmp-",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_parent_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = [
    "OCR_ANYRES_GRPO_SELECTION_RECEIPT_FILENAME",
    "OCR_ANYRES_GRPO_SELECTION_RECEIPT_KIND",
    "OCR_ANYRES_GRPO_SELECTION_RECEIPT_SCHEMA_VERSION",
    "OCR_ANYRES_GRPO_SUCCESSFUL_STOP_REASONS",
    "OCR_SELECTION_RECEIPT_SCHEMA_VERSION",
    "build_anyres_grpo_selection_receipt",
    "build_selection_receipt",
    "load_and_validate_selection_receipt",
    "validate_anyres_grpo_selection_receipt",
    "write_anyres_grpo_selection_receipt",
    "write_selection_receipt",
]
