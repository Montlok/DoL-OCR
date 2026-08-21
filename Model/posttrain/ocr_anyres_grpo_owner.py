# -*- coding: utf-8 -*-

"""Transactional persistence primitives for the single-process GRPO owner."""

from __future__ import annotations

import os
import random
import stat
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from Model.ocr.tokenization import canonical_json_sha256
from Model.posttrain.checkpointing import (
    load_verified_checkpoint_metadata_envelope,
    load_verified_torch_payload,
    verified_file_sha256,
)
from Model.posttrain.ocr_anyres_grpo_protocol import (
    FIXED_KL_PILOT_ROLLOUT_ATTEMPTS,
    fixed_kl_pilot_attempt,
    validate_fixed_kl_pilot_protocol,
    validate_sampler_resume_state,
)
from Model.posttrain.ocr_anyres_grpo_formal_protocol import (
    formal_grpo_attempt,
    validate_formal_grpo_protocol,
    validate_formal_sampler_boundary,
)
from Model.posttrain.ocr_anyres_grpo_run import (
    build_anyres_grpo_progress,
    build_text_schedule_cursor_state,
    validate_anyres_grpo_progress,
    validate_anyres_grpo_run_contract,
)
from Model.posttrain.ocr_quota_sampler import OCRQuotaSampler
from Model.training.checkpoint import (
    NO_UPDATE_PROGRESS_FILENAME,
    clear_no_update_progress as clear_shared_no_update_progress,
    load_no_update_progress as load_shared_no_update_progress,
    resolve_checkpoint_dir,
    restore_rng_state,
    save_no_update_progress as save_shared_no_update_progress,
)


PILOT_PROTOCOL_METADATA_KEY = "ocr_anyres_grpo_pilot_protocol"
PILOT_PROTOCOL_SHA256_METADATA_KEY = (
    "ocr_anyres_grpo_pilot_protocol_sha256"
)
OWNER_NO_UPDATE_FILENAME = NO_UPDATE_PROGRESS_FILENAME
OWNER_NO_UPDATE_KIND = "dol_ocr_anyres_grpo_no_update_progress_v1"

_JOURNAL_KEYS = {
    "schema_version",
    "kind",
    "anchor_checkpoint",
    "run_contract_sha256",
    "protocol_sha256",
    "anchor_progress_sha256",
    "progress",
    "rng_state",
}


def validate_protocol_run_bindings(
    run_contract: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind every schedule/dataset/seed field shared by run and protocol."""

    run = validate_anyres_grpo_run_contract(run_contract)
    if run["mode"] == "pilot":
        payload = validate_fixed_kl_pilot_protocol(protocol)
        protocol_sha = protocol.get("canonical_sha256")
        if run["pilot_protocol"] != dict(protocol) or run[
            "pilot_protocol_sha256"
        ] != protocol_sha:
            raise ValueError("run does not bind the complete pilot protocol")
    else:
        payload = validate_formal_grpo_protocol(protocol)
    schedule = run["schedule"]
    expected = {
        "global_batch_size": payload["global_batch_size"],
        "text_batch_size": payload["text_batch_size"],
        "world_size": payload["world_size"],
        "ocr_prompt_schedule_sha256": payload[
            "ocr_prompt_schedule_sha256"
        ],
        "text_batch_schedule_sha256": payload[
            "text_batch_schedule_sha256"
        ],
        "rollout_seed_schedule_sha256": payload[
            "rollout_seed_schedule_sha256"
        ],
    }
    for field, value in expected.items():
        if schedule[field] != value:
            raise ValueError(f"run schedule differs from protocol {field}")
    if schedule["dist_mode"] != "single" or payload["world_size"] != 1:
        raise ValueError("pilot protocol requires single/world_size=1")
    if run["seed"] != payload["base_seed"]:
        raise ValueError("run seed differs from protocol base seed")
    if run["train_dataset_contract_sha256"] != payload[
        "ocr_dataset_contract_sha256"
    ]:
        raise ValueError("run train image contract differs from protocol")
    if run["text_replay_train_contract_sha256"] != payload[
        "text_dataset_contract_sha256"
    ]:
        raise ValueError("run text train contract differs from protocol")
    if run["mode"] == "pilot" and schedule["max_rollout_attempts"] != FIXED_KL_PILOT_ROLLOUT_ATTEMPTS:
        raise ValueError("run does not contain the fixed 200-attempt budget")
    if run["mode"] == "formal":
        for field in (
            "max_rollout_attempts",
            "max_optimizer_steps",
            "eval_every",
            "save_every",
            "early_stop_patience",
        ):
            if schedule[field] != payload[field]:
                raise ValueError(f"formal run differs from protocol {field}")
    return run, payload


def _protocol_attempt(run: Mapping[str, Any], protocol: Mapping[str, Any], index: int):
    return (
        fixed_kl_pilot_attempt(protocol, index)
        if run["mode"] == "pilot"
        else formal_grpo_attempt(protocol, index)
    )


def _validate_sampler_boundary(run, protocol, sampler, index) -> None:
    if run["mode"] == "pilot":
        validate_sampler_resume_state(protocol, sampler=sampler, attempt_index=index)
    else:
        validate_formal_sampler_boundary(protocol, sampler, index)


def initial_pilot_progress(
    run_contract: Mapping[str, Any],
    *,
    sampler: OCRQuotaSampler,
) -> dict[str, Any]:
    run = validate_anyres_grpo_run_contract(run_contract)
    if sampler.pending_global_batch is not None or sampler.draw_counter != 0:
        raise ValueError("fresh GRPO sampler must be at its zero boundary")
    return build_anyres_grpo_progress(
        run,
        optimizer_steps=0,
        rollout_attempts=0,
        no_update_count=0,
        consecutive_no_update=0,
        optimizer_skip_count=0,
        consecutive_optimizer_skips=0,
        sampler_state=sampler.state_dict(),
        text_cursor_state=build_text_schedule_cursor_state(
            dataset_contract_sha256=run[
                "text_replay_train_contract_sha256"
            ],
            batch_size=int(run["schedule"]["text_batch_size"]),
            schedule_cursor=0,
            ce_steps=0,
        ),
        text_ce_steps=0,
        eval_count=0,
        bad_eval_count=0,
        best_validation_eligible=False,
        best_validation_step=0,
        best_checkpoint=None,
        best_validation_sha256=None,
        last_validation_sha256=None,
        stop_reason="running",
        terminal=False,
    )


def begin_live_attempt(
    protocol: Mapping[str, Any],
    *,
    sampler: OCRQuotaSampler,
    progress: Mapping[str, Any],
    run_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Prepare, but never commit, the exact registered live sampler batch."""

    run, _ = validate_protocol_run_bindings(run_contract, protocol)
    state = validate_anyres_grpo_progress(progress, run_contract=run)
    attempt_index = int(state["rollout_attempts"])
    if sampler.state_dict() != state["sampler_state"]:
        raise ValueError("live sampler state differs from checkpoint progress")
    _validate_sampler_boundary(run, protocol, sampler, attempt_index)
    attempt = _protocol_attempt(run, protocol, attempt_index)
    pending_ids = list(sampler.prepare_global_batch())
    if pending_ids != attempt["ocr_prompt_ids"]:
        raise ValueError("live sampler pending IDs differ from pilot protocol")
    return attempt


def preview_attempt_commit(
    run_contract: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    sampler: OCRQuotaSampler,
    progress: Mapping[str, Any],
    metrics: Mapping[str, Any],
    eval_count: int | None = None,
    bad_eval_count: int | None = None,
    best_validation_eligible: bool | None = None,
    best_validation_step: int | None = None,
    best_checkpoint: str | None = None,
    best_validation_sha256: str | None = None,
    last_validation_sha256: str | None = None,
    stop_reason: str = "running",
    terminal: bool = False,
) -> dict[str, Any]:
    """Validate the complete next progress value without mutating live state."""

    run, _ = validate_protocol_run_bindings(run_contract, protocol)
    before = validate_anyres_grpo_progress(progress, run_contract=run)
    attempt_index = int(before["rollout_attempts"])
    attempt = _protocol_attempt(run, protocol, attempt_index)
    if list(sampler.pending_global_batch or ()) != attempt["ocr_prompt_ids"]:
        raise ValueError("attempt commit requires the registered pending OCR IDs")
    stepped = metrics.get("stepped")
    if type(stepped) is not bool:
        raise ValueError("trainer metrics must contain a boolean stepped flag")
    if stepped:
        if metrics.get("optimizer_steps") != 1 or metrics.get(
            "scheduler_steps"
        ) != 1 or metrics.get("used_text_replay") is not True:
            raise ValueError("updated attempt must step optimizer/scheduler/text CE once")
    elif (
        metrics.get("optimizer_steps") != 0
        or metrics.get("scheduler_steps") != 0
        or metrics.get("used_text_replay") is not False
        or metrics.get("skip_reason") != "no_active_reward_groups"
    ):
        raise ValueError("no-active attempt metrics differ from transaction contract")

    staged = deepcopy(sampler)
    staged.commit_global_batch()
    if canonical_json_sha256(staged.state_dict()) != attempt[
        "sampler_state_after_sha256"
    ]:
        raise ValueError("staged sampler after-state differs from pilot protocol")
    optimizer_steps = int(before["optimizer_steps"]) + int(stepped)
    rollout_attempts = attempt_index + 1
    no_updates = int(before["no_update_count"]) + int(not stepped)
    consecutive_no_updates = (
        0 if stepped else int(before["consecutive_no_update"]) + 1
    )
    if eval_count is None:
        eval_count = int(before["eval_count"])
    if bad_eval_count is None:
        bad_eval_count = int(before["bad_eval_count"])
    if best_validation_eligible is None:
        best_validation_eligible = bool(before["best_validation_eligible"])
    if best_validation_step is None:
        best_validation_step = int(before["best_validation_step"])
    if best_checkpoint is None and best_validation_eligible:
        best_checkpoint = before["best_checkpoint"]
    if best_validation_sha256 is None and best_validation_eligible:
        best_validation_sha256 = before["best_validation_sha256"]
    if last_validation_sha256 is None:
        last_validation_sha256 = before["last_validation_sha256"]
    result = build_anyres_grpo_progress(
        run,
        optimizer_steps=optimizer_steps,
        rollout_attempts=rollout_attempts,
        no_update_count=no_updates,
        consecutive_no_update=consecutive_no_updates,
        optimizer_skip_count=int(before["optimizer_skip_count"]),
        consecutive_optimizer_skips=int(
            before["consecutive_optimizer_skips"]
        ),
        sampler_state=staged.state_dict(),
        text_cursor_state=build_text_schedule_cursor_state(
            dataset_contract_sha256=run[
                "text_replay_train_contract_sha256"
            ],
            batch_size=int(run["schedule"]["text_batch_size"]),
            schedule_cursor=rollout_attempts,
            ce_steps=optimizer_steps,
        ),
        text_ce_steps=optimizer_steps,
        eval_count=eval_count,
        bad_eval_count=bad_eval_count,
        best_validation_eligible=best_validation_eligible,
        best_validation_step=best_validation_step,
        best_checkpoint=best_checkpoint,
        best_validation_sha256=best_validation_sha256,
        last_validation_sha256=last_validation_sha256,
        stop_reason=stop_reason,
        terminal=terminal,
    )
    return result


def commit_live_attempt(
    sampler: OCRQuotaSampler,
    *,
    protocol: Mapping[str, Any],
    previewed_progress: Mapping[str, Any],
) -> None:
    """Commit only after compute and complete progress validation succeeded."""

    sampler.commit_global_batch()
    if sampler.state_dict() != previewed_progress.get("sampler_state"):
        raise RuntimeError("live sampler commit differs from previewed progress")
    expected_index = int(previewed_progress["rollout_attempts"]) - 1
    if expected_index < 0:
        raise RuntimeError("committed progress has no completed attempt")
    # The progress run SHA was validated before this commit preview, so the
    # protocol kind can be selected by its own reviewed schema.
    try:
        validate_fixed_kl_pilot_protocol(protocol)
    except ValueError:
        attempt = formal_grpo_attempt(protocol, expected_index)
    else:
        attempt = fixed_kl_pilot_attempt(protocol, expected_index)
    if canonical_json_sha256(sampler.state_dict()) != attempt[
        "sampler_state_after_sha256"
    ]:
        raise RuntimeError("live sampler after-state differs from pilot protocol")


def capture_safe_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "cpu": torch.get_rng_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def checkpoint_identity(path: str | Path) -> dict[str, str]:
    checkpoint = Path(path).resolve(strict=True)
    return {
        "path": str(checkpoint),
        "model_sha256": verified_file_sha256(checkpoint / "model.pt"),
        "metadata_sha256": verified_file_sha256(checkpoint / "meta.pt"),
    }


def validate_no_update_delta(
    run_contract: Mapping[str, Any],
    *,
    anchor_progress: Mapping[str, Any],
    journal_progress: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    run = validate_anyres_grpo_run_contract(run_contract)
    anchor = validate_anyres_grpo_progress(anchor_progress, run_contract=run)
    journal = validate_anyres_grpo_progress(journal_progress, run_contract=run)
    delta = int(journal["rollout_attempts"]) - int(anchor["rollout_attempts"])
    if delta <= 0:
        raise ValueError("no-update journal must advance at least one attempt")
    invariants = (
        "optimizer_steps",
        "optimizer_skip_count",
        "consecutive_optimizer_skips",
        "text_ce_steps",
        "eval_count",
        "bad_eval_count",
        "best_validation_eligible",
        "best_validation_step",
        "best_checkpoint",
        "best_validation_sha256",
        "last_validation_sha256",
    )
    for field in invariants:
        if journal[field] != anchor[field]:
            raise ValueError(f"no-update journal illegally changes {field}")
    if int(journal["no_update_count"]) - int(anchor["no_update_count"]) != delta:
        raise ValueError("no-update journal accounting differs from attempt delta")
    if int(journal["consecutive_no_update"]) != int(
        anchor["consecutive_no_update"]
    ) + delta:
        raise ValueError("no-update journal consecutive count differs")
    if journal["terminal"] or journal["stop_reason"] != "running":
        raise ValueError("terminal progress must be persisted as a full checkpoint")
    return anchor, journal


def save_no_update_journal(
    output_dir: str | Path,
    *,
    anchor_checkpoint: Mapping[str, str],
    run_contract: Mapping[str, Any],
    protocol: Mapping[str, Any],
    anchor_progress: Mapping[str, Any],
    progress: Mapping[str, Any],
) -> Path:
    if Path(anchor_checkpoint["path"]).resolve().parent != Path(output_dir).resolve():
        raise ValueError("no-update anchor is outside the pilot output root")
    run, _ = validate_protocol_run_bindings(run_contract, protocol)
    anchor, journal = validate_no_update_delta(
        run,
        anchor_progress=anchor_progress,
        journal_progress=progress,
    )
    state = {
        "schema_version": 1,
        "kind": OWNER_NO_UPDATE_KIND,
        "anchor_checkpoint": deepcopy(dict(anchor_checkpoint)),
        "run_contract_sha256": run["canonical_sha256"],
        "protocol_sha256": protocol["canonical_sha256"],
        "anchor_progress_sha256": anchor["canonical_sha256"],
        "progress": journal,
    }
    return save_shared_no_update_progress(
        anchor_checkpoint["path"],
        contract_sha256=run["canonical_sha256"],
        state=state,
    )


def load_no_update_journal(
    output_dir: str | Path,
    *,
    anchor_checkpoint: Mapping[str, str],
    run_contract: Mapping[str, Any],
    protocol: Mapping[str, Any],
    anchor_progress: Mapping[str, Any],
) -> dict[str, Any] | None:
    if Path(anchor_checkpoint["path"]).resolve().parent != Path(output_dir).resolve():
        raise ValueError("no-update anchor is outside the pilot output root")
    payload = load_shared_no_update_progress(
        anchor_checkpoint["path"],
        contract_sha256=run_contract["canonical_sha256"],
    )
    if payload is None:
        return None
    state = payload.get("state")
    expected_state_keys = _JOURNAL_KEYS - {"rng_state"}
    if not isinstance(state, Mapping) or set(state) != expected_state_keys:
        raise ValueError("no-update journal fields differ from contract")
    if state["schema_version"] != 1 or state["kind"] != OWNER_NO_UPDATE_KIND:
        raise ValueError("no-update journal kind differs")
    run, _ = validate_protocol_run_bindings(run_contract, protocol)
    anchor = validate_anyres_grpo_progress(anchor_progress, run_contract=run)
    if state["anchor_checkpoint"] != dict(anchor_checkpoint):
        raise ValueError("no-update journal anchors another checkpoint")
    if state["run_contract_sha256"] != run["canonical_sha256"]:
        raise ValueError("no-update journal run SHA differs")
    if state["protocol_sha256"] != protocol["canonical_sha256"]:
        raise ValueError("no-update journal protocol SHA differs")
    if state["anchor_progress_sha256"] != anchor["canonical_sha256"]:
        raise ValueError("no-update journal anchor progress differs")
    _, progress = validate_no_update_delta(
        run,
        anchor_progress=anchor,
        journal_progress=state["progress"],
    )
    if not isinstance(payload["rng_state"], Mapping):
        raise ValueError("no-update journal RNG state differs")
    restore_rng_state(dict(payload["rng_state"]))
    return progress


def clear_no_update_journal(output_dir: str | Path) -> None:
    output = Path(output_dir)
    destination = output / OWNER_NO_UPDATE_FILENAME
    if destination.exists() or destination.is_symlink():
        clear_shared_no_update_progress(output)


def restore_full_checkpoint(
    checkpoint: str | Path,
    *,
    model,
    optimizer,
    scheduler,
    scaler,
) -> tuple[Path, dict[str, Any], str]:
    """Restore a complete single-process checkpoint without unsafe unpickling."""

    root = resolve_checkpoint_dir(checkpoint).resolve(strict=True)
    required = [
        "COMPLETE",
        "model.pt",
        "optimizer.pt",
        "scheduler.pt",
        "rng.pt",
        "meta.pt",
    ]
    if scaler is not None:
        required.append("scaler.pt")
    for name in required:
        member = root / name
        if member.is_symlink():
            raise ValueError(f"resume checkpoint {name} must not be a symlink")
        try:
            info = member.stat()
        except FileNotFoundError as exc:
            raise ValueError(f"resume checkpoint is missing {name}") from exc
        if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
            raise ValueError(f"resume checkpoint {name} must be non-empty regular")
    envelope, meta_sha = load_verified_checkpoint_metadata_envelope(root / "meta.pt")
    marker_path = root / "COMPLETE"
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(marker_path, flags)
    try:
        before = os.fstat(descriptor)
        marker = os.read(descriptor, 128)
        if os.read(descriptor, 1):
            raise ValueError("resume COMPLETE marker is unexpectedly large")
        after = os.fstat(descriptor)
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
            raise ValueError("resume COMPLETE changed while being read")
    finally:
        os.close(descriptor)
    if marker != f"step={envelope['step']}\n".encode("ascii"):
        raise ValueError("resume COMPLETE differs from metadata outer step")
    model_state, _ = load_verified_torch_payload(root / "model.pt")
    optimizer_state, _ = load_verified_torch_payload(root / "optimizer.pt")
    scheduler_state, _ = load_verified_torch_payload(root / "scheduler.pt")
    rng_state, _ = load_verified_torch_payload(root / "rng.pt")
    if not all(
        isinstance(value, Mapping)
        for value in (model_state, optimizer_state, scheduler_state, rng_state)
    ):
        raise ValueError("resume checkpoint state payload is malformed")
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
    scheduler.load_state_dict(scheduler_state)
    if scaler is not None:
        scaler_state, _ = load_verified_torch_payload(root / "scaler.pt")
        if not isinstance(scaler_state, Mapping):
            raise ValueError("resume scaler state is malformed")
        scaler.load_state_dict(scaler_state)
    restore_rng_state(dict(rng_state))
    return root, envelope, meta_sha


__all__ = [
    "OWNER_NO_UPDATE_FILENAME",
    "PILOT_PROTOCOL_METADATA_KEY",
    "PILOT_PROTOCOL_SHA256_METADATA_KEY",
    "begin_live_attempt",
    "capture_safe_rng_state",
    "checkpoint_identity",
    "clear_no_update_journal",
    "commit_live_attempt",
    "initial_pilot_progress",
    "load_no_update_journal",
    "preview_attempt_commit",
    "restore_full_checkpoint",
    "save_no_update_journal",
    "validate_no_update_delta",
    "validate_protocol_run_bindings",
]
