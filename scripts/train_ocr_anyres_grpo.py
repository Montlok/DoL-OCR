#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Own anyres OCR GRPO admission, pilot selection, and formal runs.

The ``pilot`` owner runs/resumes one isolated 200-attempt KL trial; ``select-kl``
authenticates three completed trials and emits either a formal KL selection
receipt or an explicit NO-GO receipt.  Neither path opens locked golden data or
treats a pilot checkpoint as the formal initialization checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Model.posttrain.checkpointing import (  # noqa: E402
    load_verified_checkpoint_metadata_envelope,
    verified_file_sha256,
)
from Model.config import BOS_ID, EOS_ID, PAD_ID, TrainingConfig  # noqa: E402
from Model.ocr.anyres_preprocess_contract import (  # noqa: E402
    validate_anyres_preprocess_contract,
)
from Model.ocr.tokenization import (  # noqa: E402
    canonical_json_sha256,
    make_ocr_target_encoder,
    native_tokenization_contract,
)
from Model.posttrain.grpo import GRPOConfig  # noqa: E402
from Model.posttrain.ocr_anyres_collator import AnyresOCRSFTCollator  # noqa: E402
from Model.posttrain.ocr_anyres_data import AnyresOCRDataset  # noqa: E402
from Model.posttrain.ocr_anyres_grpo_owner import (  # noqa: E402
    PILOT_PROTOCOL_METADATA_KEY,
    PILOT_PROTOCOL_SHA256_METADATA_KEY,
    begin_live_attempt,
    checkpoint_identity,
    clear_no_update_journal,
    commit_live_attempt,
    initial_pilot_progress,
    load_no_update_journal,
    preview_attempt_commit,
    restore_full_checkpoint,
    save_no_update_journal,
    validate_protocol_run_bindings,
)
from Model.posttrain.ocr_anyres_grpo_protocol import (  # noqa: E402
    apply_attempt_seed,
    build_fixed_kl_pilot_protocol,
    validate_fixed_kl_pilot_protocol,
    validate_sampler_resume_state,
)
from Model.posttrain.ocr_anyres_grpo_formal_protocol import (  # noqa: E402
    FORMAL_GRPO_PROTOCOL_METADATA_KEY,
    FORMAL_GRPO_PROTOCOL_SHA256_METADATA_KEY,
    apply_formal_attempt_seed,
    build_formal_grpo_protocol,
    rebuild_and_match_formal_protocol,
    validate_formal_sampler_boundary,
)
from Model.posttrain.ocr_anyres_grpo_run import (  # noqa: E402
    build_anyres_grpo_run_contract,
    build_kl_pilot_result,
    select_kl_pilot_results,
    validate_anyres_grpo_progress,
    validate_anyres_grpo_run_contract,
    validate_kl_pilot_result,
    validate_kl_selection_receipt,
)
from Model.posttrain.ocr_anyres_grpo_trainer import (  # noqa: E402
    AnyresGRPOKLAblationTrial,
    train_anyres_grpo_cycle,
)
from Model.posttrain.ocr_anyres_reward import (  # noqa: E402
    build_anyres_ocr_reward_adapter,
)
from Model.posttrain.ocr_joint_contract import (  # noqa: E402
    admit_joint_policy_for_grpo,
)
from Model.posttrain.ocr_joint_eval import (  # noqa: E402
    DEPLOYMENT_BUCKET_WEIGHTS,
    evaluate_ocr_joint,
    joint_eval_eligibility,
)
from Model.posttrain.ocr_quota_sampler import OCRQuotaSampler  # noqa: E402
from Model.posttrain.ocr_selection import (  # noqa: E402
    OCR_ANYRES_GRPO_SELECTION_RECEIPT_FILENAME,
    build_anyres_grpo_selection_receipt,
    write_anyres_grpo_selection_receipt,
)
from Model.posttrain.text_replay import (  # noqa: E402
    TextReplayCollator,
    TextReplayPartition,
)
from Model.posttrain.verified_prefetch import VerifiedBatchPrefetcher  # noqa: E402
from Model.training.checkpoint import save_checkpoint  # noqa: E402
from Model.training.optim import build_ocr_joint_adamw, build_scheduler  # noqa: E402
from Tokenizer.multimodal import NativeImageProcessorV2, PILImageProcessor  # noqa: E402
from Tokenizer.unified.bundle import TokenizerBundle  # noqa: E402
from scripts import train_ocr_anyres_joint_sft as joint_cli  # noqa: E402
from scripts import train_ocr_anyres_sft as visual_cli  # noqa: E402
from scripts.validate_ocr_anyres_dataset import (  # noqa: E402
    _load_exclusions,
    _strict_json,
    build_report as build_dataset_report,
)


GRPO_RUN_METADATA_KEY = "ocr_anyres_grpo_run_contract"
GRPO_RUN_SHA256_METADATA_KEY = "ocr_anyres_grpo_run_contract_sha256"
GRPO_PROGRESS_METADATA_KEY = "ocr_anyres_grpo_progress"
GRPO_PROGRESS_SHA256_METADATA_KEY = "ocr_anyres_grpo_progress_sha256"
KL_SELECTION_FILENAME = "KL_SELECTION.json"
KL_NO_GO_FILENAME = "KL_SELECTION_NO_GO.json"
_PILOT_ATTEMPTS = 200
_GRPO_EXTRA_SCRIPTS = (
    "scripts/train_ocr_anyres_joint_sft.py",
    "scripts/train_ocr_anyres_grpo.py",
)
_GRPO_SOURCE_ALLOWLIST = {
    "Model/posttrain/ocr_anyres_grpo.py",
    "Model/posttrain/ocr_anyres_grpo_owner.py",
    "Model/posttrain/ocr_anyres_grpo_formal_protocol.py",
    "Model/posttrain/ocr_anyres_grpo_protocol.py",
    "Model/posttrain/ocr_anyres_grpo_run.py",
    "Model/posttrain/ocr_anyres_grpo_trainer.py",
    "Model/posttrain/ocr_anyres_reward.py",
    "scripts/train_ocr_anyres_grpo.py",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    select = subcommands.add_parser(
        "select-kl",
        help="authenticate and compare the three isolated KL pilots",
    )
    select.add_argument(
        "--pilot-result",
        action="append",
        required=True,
        help="PILOT_RESULT JSON; provide exactly three times",
    )
    select.add_argument(
        "--existing-receipt",
        default="",
        help="optional prior receipt that must exactly equal recomputation",
    )
    select.add_argument("--output-dir", required=True)
    pilot = subcommands.add_parser(
        "pilot",
        help="run or resume one isolated fixed 200-attempt KL pilot",
    )
    pilot.add_argument("--dist", choices=["single"], default="single")
    pilot.add_argument("--resume", default="")
    pilot.add_argument("--joint-stage-result", required=True)
    pilot.add_argument("--tokenizer", required=True)
    pilot.add_argument("--root", required=True)
    pilot.add_argument("--assets", required=True)
    pilot.add_argument("--views", required=True)
    pilot.add_argument("--train-samples", required=True)
    pilot.add_argument("--sft-validation-samples", required=True)
    pilot.add_argument("--kl-selection-samples", required=True)
    pilot.add_argument("--formal-monitor-samples", required=True)
    pilot.add_argument("--preprocess-contract", required=True)
    pilot.add_argument("--text-replay", required=True)
    pilot.add_argument("--reviewed-exclusions", required=True)
    pilot.add_argument("--output-dir", required=True)
    pilot.add_argument("--kl-coef", type=float, choices=[0.04, 0.01, 0.0], required=True)
    pilot.add_argument("--global-batch-size", type=int, default=20)
    pilot.add_argument("--text-batch-size", type=int, default=4)
    pilot.add_argument("--group-size", type=int, default=4)
    pilot.add_argument("--seed", type=int, default=42)
    pilot.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    pilot.add_argument("--device", default="cuda:0")
    pilot.add_argument("--tower-lr", type=float, required=True)
    pilot.add_argument("--bridge-lr", type=float, required=True)
    pilot.add_argument("--projector-lr", type=float, required=True)
    pilot.add_argument("--lm-lr", type=float, required=True)
    pilot.add_argument("--weight-decay", type=float, default=0.01)
    pilot.add_argument("--warmup-steps", type=int, default=0)
    pilot.add_argument("--grad-clip", type=float, default=1.0)
    pilot.add_argument("--text-weight", type=float, default=0.2)
    pilot.add_argument("--loss-chunk-size", type=int, default=4096)
    pilot.add_argument("--max-behavior-log-ratio", type=float, required=True)
    pilot.add_argument("--max-consecutive-no-update", type=int, default=20)
    pilot.add_argument("--max-consecutive-optimizer-skips", type=int, default=5)
    pilot.add_argument("--keep-last", type=int, default=2)
    formal = subcommands.add_parser(
        "formal",
        help="run or resume formal anyres GRPO from an authenticated KL receipt",
    )
    formal.add_argument("--dist", choices=["single"], default="single")
    formal.add_argument("--resume", default="")
    formal.add_argument("--pilot-result", action="append", required=True)
    formal.add_argument("--kl-selection-receipt", required=True)
    formal.add_argument("--joint-stage-result", required=True)
    formal.add_argument("--locked-golden-anchor-sha256", required=True)
    formal.add_argument("--tokenizer", required=True)
    formal.add_argument("--root", required=True)
    formal.add_argument("--assets", required=True)
    formal.add_argument("--views", required=True)
    formal.add_argument("--train-samples", required=True)
    formal.add_argument("--sft-validation-samples", required=True)
    formal.add_argument("--kl-selection-samples", required=True)
    formal.add_argument("--formal-monitor-samples", required=True)
    formal.add_argument("--preprocess-contract", required=True)
    formal.add_argument("--text-replay", required=True)
    formal.add_argument("--reviewed-exclusions", required=True)
    formal.add_argument("--output-dir", required=True)
    formal.add_argument("--max-rollout-attempts", type=int, required=True)
    formal.add_argument("--max-optimizer-steps", type=int, required=True)
    formal.add_argument("--eval-every", type=int, required=True)
    formal.add_argument("--save-every", type=int, required=True)
    formal.add_argument("--early-stop-patience", type=int, required=True)
    formal.add_argument("--global-batch-size", type=int, default=20)
    formal.add_argument("--text-batch-size", type=int, default=4)
    formal.add_argument("--group-size", type=int, default=4)
    formal.add_argument("--seed", type=int, required=True)
    formal.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    formal.add_argument("--device", default="cuda:0")
    formal.add_argument("--tower-lr", type=float, required=True)
    formal.add_argument("--bridge-lr", type=float, required=True)
    formal.add_argument("--projector-lr", type=float, required=True)
    formal.add_argument("--lm-lr", type=float, required=True)
    formal.add_argument("--weight-decay", type=float, default=0.01)
    formal.add_argument("--warmup-steps", type=int, default=0)
    formal.add_argument("--grad-clip", type=float, default=1.0)
    formal.add_argument("--text-weight", type=float, default=0.2)
    formal.add_argument("--loss-chunk-size", type=int, default=4096)
    formal.add_argument("--max-behavior-log-ratio", type=float, required=True)
    formal.add_argument("--max-consecutive-no-update", type=int, default=20)
    formal.add_argument("--max-consecutive-optimizer-skips", type=int, default=5)
    formal.add_argument("--keep-last", type=int, default=2)
    return parser.parse_args(argv)


def _strict_json_file(path: str | Path, *, where: str) -> dict[str, Any]:
    source = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    elif source.is_symlink():  # pragma: no cover
        raise ValueError(f"{where} must not be a symlink")
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise ValueError(f"cannot open {where}: {source}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{where} must be a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(after):
            raise ValueError(f"{where} changed while being read")
    finally:
        os.close(descriptor)

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{where} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            b"".join(chunks).decode("utf-8", errors="strict"),
            object_pairs_hook=object_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{where} is not strict UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{where} must contain one JSON object")
    return payload


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _checkpoint_dir(identity: Mapping[str, Any], *, where: str) -> Path:
    path_value = identity.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{where}.path must be a non-empty string")
    checkpoint = Path(path_value)
    try:
        resolved = checkpoint.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ValueError(f"{where}.path does not resolve: {checkpoint}") from exc
    if path_value != str(resolved) or checkpoint.is_symlink():
        raise ValueError(f"{where}.path must be canonical absolute")
    if not resolved.is_dir():
        raise ValueError(f"{where}.path must be a checkpoint directory")
    return resolved


def _require_regular_member(
    path: Path,
    *,
    where: str,
    nonempty: bool = True,
) -> None:
    if path.is_symlink():
        raise ValueError(f"{where} must not be a symlink")
    try:
        info = path.stat()
    except FileNotFoundError as exc:
        raise ValueError(f"{where} is missing") from exc
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{where} must be a regular file")
    if nonempty and info.st_size <= 0:
        raise ValueError(f"{where} must be non-empty")


def _stable_regular_bytes(path: Path, *, where: str) -> bytes:
    _require_regular_member(path, where=where)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 4096)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(after):
            raise ValueError(f"{where} changed while being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _metadata_contract(
    metadata: Mapping[str, Any],
    *,
    result: Mapping[str, Any],
    checkpoint_role: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if metadata.get("phase") != "grpo" or metadata.get("task") != "ocr":
        raise ValueError(f"{checkpoint_role} is not OCR GRPO metadata")
    if metadata.get("final") is not True:
        raise ValueError(f"{checkpoint_role} final flag must be true")
    if metadata.get("training_stage") != "grpo_kl_pilot":
        raise ValueError(f"{checkpoint_role} training stage differs")
    if metadata.get("locked_golden_opened") is not False:
        raise ValueError(f"{checkpoint_role} does not prove golden stayed closed")
    run = validate_anyres_grpo_run_contract(metadata.get(GRPO_RUN_METADATA_KEY))
    protocol = metadata.get(PILOT_PROTOCOL_METADATA_KEY)
    validate_fixed_kl_pilot_protocol(protocol)
    if metadata.get(PILOT_PROTOCOL_SHA256_METADATA_KEY) != protocol.get(
        "canonical_sha256"
    ):
        raise ValueError(f"{checkpoint_role} embedded protocol SHA differs")
    validate_protocol_run_bindings(run, protocol)
    progress = validate_anyres_grpo_progress(
        metadata.get(GRPO_PROGRESS_METADATA_KEY),
        run_contract=run,
    )
    if metadata.get(GRPO_RUN_SHA256_METADATA_KEY) != run["canonical_sha256"]:
        raise ValueError(f"{checkpoint_role} embedded run SHA differs")
    if metadata.get(GRPO_PROGRESS_SHA256_METADATA_KEY) != progress[
        "canonical_sha256"
    ]:
        raise ValueError(f"{checkpoint_role} embedded progress SHA differs")
    if run["mode"] != "pilot":
        raise ValueError(f"{checkpoint_role} did not come from a pilot run")
    if run["schedule"]["max_rollout_attempts"] != _PILOT_ATTEMPTS or run[
        "kl_trial"
    ]["trial_steps"] != _PILOT_ATTEMPTS:
        raise ValueError(f"{checkpoint_role} pilot budget is not 200 attempts")
    if (
        progress["rollout_attempts"] != _PILOT_ATTEMPTS
        or progress["terminal"] is not True
        or progress["stop_reason"] != "pilot_budget_complete"
    ):
        raise ValueError(f"{checkpoint_role} is not terminal at 200 attempts")
    if progress["eval_count"] != 4:
        raise ValueError(f"{checkpoint_role} does not bind four pilot evaluations")
    if canonical_json_sha256(progress["sampler_state"]) != protocol["payload"][
        "attempts"
    ][-1]["sampler_state_after_sha256"]:
        raise ValueError(f"{checkpoint_role} sampler is not at attempt-200 boundary")
    if progress["last_validation_sha256"] != result[
        "selection_validation_sha256"
    ]:
        raise ValueError(f"{checkpoint_role} last validation is not attempt 200")
    baseline = metadata.get("grpo_runtime_baseline")
    selection_validation = metadata.get("selection_validation")
    if not isinstance(baseline, Mapping) or canonical_json_sha256(baseline) != result[
        "baseline_validation_sha256"
    ] or dict(baseline) != result["baseline_validation"]:
        raise ValueError(f"{checkpoint_role} baseline report differs")
    if not isinstance(selection_validation, Mapping) or canonical_json_sha256(
        selection_validation
    ) != result["selection_validation_sha256"] or dict(
        selection_validation
    ) != result["selection_validation"]:
        raise ValueError(f"{checkpoint_role} selection report differs")
    dataset_report = metadata.get("dataset_admission_report")
    if not isinstance(dataset_report, Mapping) or canonical_json_sha256(
        dataset_report
    ) != run["dataset_ready_admission_sha256"]:
        raise ValueError(f"{checkpoint_role} dataset admission differs")
    try:
        expected_sample_ids = dataset_report["datasets"]["kl_selection"][
            "contract"
        ]["sample_ids"]
        text_stats = dataset_report["text_replay"]["splits"]["kl_selection"][
            "contract"
        ]["token_stats"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{checkpoint_role} KL selection coverage is absent") from exc
    if (
        not isinstance(expected_sample_ids, list)
        or len(expected_sample_ids) != len(set(expected_sample_ids))
    ):
        raise ValueError(f"{checkpoint_role} KL selection sample IDs are invalid")
    expected_ids = set(expected_sample_ids)
    for name, report in (("baseline", baseline), ("selection", selection_validation)):
        records = report.get("selection_records")
        if not isinstance(records, list) or {
            row.get("sample_id") for row in records if isinstance(row, Mapping)
        } != expected_ids or len(records) != len(expected_ids):
            raise ValueError(f"{checkpoint_role} {name} does not cover full KL split")
        text_report = report.get("text_replay")
        expected_targets = int(text_stats["total_sequence_tokens"]) - int(
            text_stats["samples"]
        )
        if not isinstance(text_report, Mapping) or text_report.get(
            "target_tokens"
        ) != expected_targets:
            raise ValueError(f"{checkpoint_role} {name} text coverage differs")
    if _attach_baseline_gate(baseline) != dict(baseline):
        raise ValueError(f"{checkpoint_role} baseline eligibility was not recomputed")
    if _attach_selection_gate(
        selection_validation,
        baseline=baseline,
        optimizer_steps=int(progress["optimizer_steps"]),
    ) != dict(selection_validation):
        raise ValueError(f"{checkpoint_role} selection eligibility was not recomputed")
    runtime_source = visual_cli._validate_runtime_source_receipt(
        metadata.get("runtime_source_receipt")
    )
    if runtime_source["canonical_sha256"] != run["source_closure_sha256"]:
        raise ValueError(f"{checkpoint_role} runtime source closure differs")
    runtime_environment = metadata.get("runtime_environment")
    if not isinstance(runtime_environment, Mapping) or canonical_json_sha256(
        runtime_environment
    ) != run["runtime_environment_sha256"]:
        raise ValueError(f"{checkpoint_role} runtime environment differs")
    optimizer_contract = metadata.get("grpo_optimizer_contract")
    if not isinstance(optimizer_contract, Mapping) or optimizer_contract.get(
        "canonical_sha256"
    ) != run["optimizer_contract_sha256"]:
        raise ValueError(f"{checkpoint_role} optimizer contract differs")
    expected_run_fields = {
        "canonical_sha256": "run_contract_sha256",
        "parent_joint_checkpoint": "parent_joint_checkpoint",
        "parent_joint_stage_result_sha256": "parent_joint_stage_result_sha256",
        "admission_sha256": "admission_sha256",
        "kl_trial": "kl_trial",
        "kl_trial_sha256": "kl_trial_sha256",
        "trial_comparison_contract_sha256": "trial_comparison_contract_sha256",
        "baseline_validation_sha256": "baseline_validation_sha256",
        "kl_selection_dataset_contract_sha256": (
            "image_dataset_contract_sha256"
        ),
        "text_replay_kl_selection_contract_sha256": (
            "text_replay_dataset_contract_sha256"
        ),
        "source_closure_sha256": "source_closure_sha256",
        "dataset_ready_admission_sha256": "dataset_ready_admission_sha256",
    }
    for run_field, result_field in expected_run_fields.items():
        if run[run_field] != result[result_field]:
            raise ValueError(
                f"{checkpoint_role} run differs from PILOT_RESULT {result_field}"
            )
    for field in (
        "optimizer_steps",
        "rollout_attempts",
        "no_update_count",
        "optimizer_skip_count",
    ):
        if progress[field] != result[field]:
            raise ValueError(
                f"{checkpoint_role} progress differs from PILOT_RESULT {field}"
            )
    if result["hard_eligible"]:
        selection = result["selection_checkpoint"]
        if not isinstance(selection, Mapping) or progress["best_checkpoint"] != selection[
            "path"
        ]:
            raise ValueError(
                f"{checkpoint_role} progress best differs from selection checkpoint"
            )
        if progress["best_validation_sha256"] != result[
            "selection_validation_sha256"
        ]:
            raise ValueError(
                f"{checkpoint_role} progress best validation differs"
            )
    elif progress["best_validation_eligible"] is not False:
        raise ValueError(f"{checkpoint_role} publishes an ineligible best")
    return run, progress


def _authenticate_parent_joint(identity: Mapping[str, Any], *, where: str) -> None:
    checkpoint = _checkpoint_dir(identity, where=where)
    for name, digest_field in (
        ("model.pt", "model_sha256"),
        ("meta.pt", "metadata_sha256"),
    ):
        member = checkpoint / name
        _require_regular_member(member, where=f"{where}/{name}")
        actual = verified_file_sha256(
            member,
            expected_sha256=identity.get(digest_field),
        )
        if actual != identity.get(digest_field):
            raise ValueError(f"{where}/{name} differs from PILOT_RESULT")
    envelope, _ = load_verified_checkpoint_metadata_envelope(
        checkpoint / "meta.pt",
        expected_sha256=identity.get("metadata_sha256"),
    )
    metadata = envelope["metadata"]
    if metadata.get("training_stage") != "joint" or metadata.get("final") is not False:
        raise ValueError(f"{where} is not a non-final joint best checkpoint")


def _authenticate_checkpoint(
    identity: Mapping[str, Any],
    *,
    result: Mapping[str, Any],
    role: str,
) -> dict[str, Any]:
    checkpoint = _checkpoint_dir(identity, where=role)
    complete = checkpoint / "COMPLETE"
    model_path = checkpoint / "model.pt"
    meta_path = checkpoint / "meta.pt"
    optimizer_path = checkpoint / "optimizer.pt"
    scheduler_path = checkpoint / "scheduler.pt"
    rng_path = checkpoint / "rng.pt"
    _require_regular_member(complete, where=f"{role}/COMPLETE")
    _require_regular_member(model_path, where=f"{role}/model.pt")
    _require_regular_member(meta_path, where=f"{role}/meta.pt")
    for name, member in (
        ("optimizer.pt", optimizer_path),
        ("scheduler.pt", scheduler_path),
        ("rng.pt", rng_path),
    ):
        _require_regular_member(member, where=f"{role}/{name}")
    model_sha = verified_file_sha256(
        model_path,
        expected_sha256=identity.get("model_sha256"),
    )
    envelope, metadata_sha = load_verified_checkpoint_metadata_envelope(
        meta_path,
        expected_sha256=identity.get("metadata_sha256"),
    )
    if model_sha != identity.get("model_sha256") or metadata_sha != identity.get(
        "metadata_sha256"
    ):
        raise ValueError(f"{role} bytes differ from PILOT_RESULT identity")
    run, progress = _metadata_contract(
        envelope["metadata"],
        result=result,
        checkpoint_role=role,
    )
    outer_step = int(envelope["step"])
    if outer_step != progress["optimizer_steps"]:
        raise ValueError(
            f"{role} outer step differs from progress.optimizer_steps"
        )
    marker = _stable_regular_bytes(complete, where=f"{role}/COMPLETE")
    try:
        marker_text = marker.decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{role}/COMPLETE must be strict ASCII") from exc
    if marker_text != f"step={outer_step}\n":
        raise ValueError(f"{role}/COMPLETE differs from outer checkpoint step")
    if run["precision"] == "fp16":
        _require_regular_member(
            checkpoint / "scaler.pt",
            where=f"{role}/scaler.pt",
        )
    return {
        "path": str(checkpoint),
        "model_sha256": model_sha,
        "metadata_sha256": metadata_sha,
        "step": outer_step,
        "run": run,
        "progress": progress,
    }


def _authenticate_pilot_result(payload: object, *, where: str) -> dict[str, Any]:
    result = validate_kl_pilot_result(payload)
    _authenticate_parent_joint(
        result["parent_joint_checkpoint"],
        where=f"{where} parent joint checkpoint",
    )
    terminal = _authenticate_checkpoint(
        result["terminal_checkpoint"],
        result=result,
        role=f"{where} terminal checkpoint",
    )
    parent_path = Path(result["parent_joint_checkpoint"]["path"])
    terminal_root = Path(result["terminal_checkpoint"]["path"]).parent
    if (
        parent_path == terminal_root
        or parent_path in terminal_root.parents
        or terminal_root in parent_path.parents
    ):
        raise ValueError(f"{where} pilot output is not disjoint from joint parent")
    if {
        key: terminal[key]
        for key in ("path", "model_sha256", "metadata_sha256")
    } != result["terminal_checkpoint"]:
        raise ValueError(f"{where} terminal checkpoint identity changed")
    selection_identity = result["selection_checkpoint"]
    if selection_identity is not None:
        if selection_identity == result["terminal_checkpoint"]:
            selection = terminal
        else:
            selection = _authenticate_checkpoint(
                selection_identity,
                result=result,
                role=f"{where} selection checkpoint",
            )
        if {
            key: selection[key]
            for key in ("path", "model_sha256", "metadata_sha256")
        } != selection_identity:
            raise ValueError(f"{where} selection checkpoint identity changed")
    return result


def select_authenticated_pilots(
    result_paths: Sequence[str | Path],
    *,
    existing_receipt: str | Path | None = None,
) -> dict[str, Any]:
    if len(result_paths) != 3:
        raise ValueError("select-kl requires exactly three --pilot-result files")
    source_before = _current_source_closure()
    results = [
        _authenticate_pilot_result(
            _strict_json_file(path, where=f"PILOT_RESULT[{index}]"),
            where=f"PILOT_RESULT[{index}]",
        )
        for index, path in enumerate(result_paths)
    ]
    output_roots = [Path(result["terminal_checkpoint"]["path"]).parent for result in results]
    if len(set(output_roots)) != 3 or any(
        left in right.parents or right in left.parents
        for index, left in enumerate(output_roots)
        for right in output_roots[index + 1 :]
    ):
        raise ValueError("KL pilots must use three disjoint output roots")
    if any(
        result["source_closure_sha256"] != source_before["canonical_sha256"]
        for result in results
    ):
        raise ValueError("pilot results were produced by another source closure")
    receipt = select_kl_pilot_results(results)
    if existing_receipt:
        previous = validate_kl_selection_receipt(
            _strict_json_file(existing_receipt, where="existing KL receipt")
        )
        if previous != receipt:
            raise ValueError("existing KL receipt differs from full recomputation")
    if _current_source_closure() != source_before:
        raise RuntimeError("selector production source changed during recomputation")
    return receipt


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.tmp-",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:  # pragma: no cover
            pass
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _run_select_kl(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    if output.exists() or output.is_symlink():
        raise ValueError("--output-dir must not already exist")
    receipt = select_authenticated_pilots(
        args.pilot_result,
        existing_receipt=(args.existing_receipt or None),
    )
    selected = receipt.get("status") == "selected" and receipt.get(
        "formal_run_allowed"
    ) is True
    no_go = receipt.get("status") == "no_candidate_admitted" and receipt.get(
        "formal_run_allowed"
    ) is False
    if not selected and not no_go:
        raise RuntimeError("KL selector returned an inconsistent authorization state")

    output.mkdir(parents=True, exist_ok=False)
    destination = output / (
        KL_SELECTION_FILENAME if selected else KL_NO_GO_FILENAME
    )
    _atomic_json(destination, receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0 if selected else 2


def _validate_pilot_args(args: argparse.Namespace) -> None:
    if args.dist != "single":
        raise ValueError("anyres GRPO pilot requires --dist=single")
    for name in (
        "global_batch_size",
        "text_batch_size",
        "group_size",
        "loss_chunk_size",
        "max_consecutive_no_update",
        "max_consecutive_optimizer_skips",
        "keep_last",
    ):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.global_batch_size % 20:
        raise ValueError("--global-batch-size must be divisible by 20")
    if args.group_size < 2:
        raise ValueError("--group-size must be at least two")
    if isinstance(args.seed, bool) or not isinstance(args.seed, int) or args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if (
        isinstance(args.warmup_steps, bool)
        or not isinstance(args.warmup_steps, int)
        or args.warmup_steps < 0
    ):
        raise ValueError("--warmup-steps must be non-negative")
    for name in (
        "tower_lr",
        "bridge_lr",
        "projector_lr",
        "lm_lr",
        "grad_clip",
        "max_behavior_log_ratio",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("weight_decay", "text_weight"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative")
    output = Path(args.output_dir).resolve()
    if args.resume:
        if not output.is_dir() or output.is_symlink():
            raise ValueError("pilot --resume requires the existing canonical output root")
    elif output.exists() or output.is_symlink():
        raise ValueError("fresh pilot --output-dir must not already exist")


def _validate_formal_args(args: argparse.Namespace) -> None:
    _validate_pilot_args(args)
    if len(args.pilot_result) != 3:
        raise ValueError("formal requires exactly three --pilot-result values")
    for name in (
        "max_rollout_attempts",
        "max_optimizer_steps",
        "eval_every",
        "save_every",
        "early_stop_patience",
    ):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_optimizer_steps > args.max_rollout_attempts:
        raise ValueError("max optimizer steps cannot exceed rollout attempts")
    anchor = args.locked_golden_anchor_sha256
    if not isinstance(anchor, str) or len(anchor) != 64 or any(
        character not in "0123456789abcdef" for character in anchor
    ):
        raise ValueError("--locked-golden-anchor-sha256 must be lowercase SHA-256")


def _pilot_validator_args(args: argparse.Namespace) -> argparse.Namespace:
    return SimpleNamespace(
        root=args.root,
        assets=args.assets,
        views=args.views,
        train_samples=args.train_samples,
        sft_validation_samples=args.sft_validation_samples,
        kl_selection_samples=args.kl_selection_samples,
        formal_monitor_samples=args.formal_monitor_samples,
        preprocess_contract=args.preprocess_contract,
        tokenizer=args.tokenizer,
        text_replay=args.text_replay,
        reviewed_exclusions=args.reviewed_exclusions,
        out="",
    )


def _joint_best_identity(result: Mapping[str, Any]) -> dict[str, str]:
    if result.get("grpo_promotion_allowed") is not True:
        raise ValueError("JOINT_STAGE_RESULT does not authorize GRPO")
    best = result.get("best_eligible_checkpoint")
    if not isinstance(best, Mapping) or set(best) != {
        "path",
        "model_sha256",
        "metadata_sha256",
    }:
        raise ValueError("JOINT_STAGE_RESULT has no exact eligible best identity")
    return {key: str(best[key]) for key in ("path", "model_sha256", "metadata_sha256")}


def _current_source_closure() -> dict[str, Any]:
    return visual_cli._runtime_source_receipt(extra_scripts=_GRPO_EXTRA_SCRIPTS)


def _assert_parent_source_extension(
    parent_metadata: Mapping[str, Any],
    current: Mapping[str, Any],
) -> None:
    saved = visual_cli._validate_runtime_source_receipt(
        parent_metadata.get("runtime_source_receipt")
    )
    current = visual_cli._validate_runtime_source_receipt(current)
    saved_files = {row["path"]: row["sha256"] for row in saved["files"]}
    current_files = {row["path"]: row["sha256"] for row in current["files"]}
    added = set(current_files) - set(saved_files)
    removed = set(saved_files) - set(current_files)
    if (
        not added
        or "scripts/train_ocr_anyres_grpo.py" not in added
        or not added.issubset(_GRPO_SOURCE_ALLOWLIST)
        or removed
    ):
        raise ValueError(
            "GRPO source closure differs outside the explicit GRPO allowlist: "
            f"added={sorted(added)}, removed={sorted(removed)}"
        )
    drift = {
        path for path, digest in saved_files.items()
        if current_files.get(path) != digest
    }
    if drift:
        raise ValueError(f"production source drift since joint stage: {sorted(drift)}")


def _require_source_unchanged(expected: Mapping[str, Any]) -> None:
    current = _current_source_closure()
    if current != expected:
        raise RuntimeError("GRPO production source changed during the pilot")


def _attach_baseline_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(report))
    buckets = {
        bucket: float(result["real"]["buckets"][bucket]["raw_grapheme_cer"])
        for bucket in DEPLOYMENT_BUCKET_WEIGHTS
    }
    text_nll = float(result["text_replay"]["token_nll"])
    result["eligibility"] = joint_eval_eligibility(
        result,
        baseline_bucket_cer=buckets,
        baseline_text_token_nll=text_nll,
        min_relative_cer_improvement=0.0,
    )
    return result


def _attach_selection_gate(
    report: Mapping[str, Any],
    *,
    baseline: Mapping[str, Any],
    optimizer_steps: int,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(report))
    baseline_buckets = {
        bucket: float(baseline["real"]["buckets"][bucket]["raw_grapheme_cer"])
        for bucket in DEPLOYMENT_BUCKET_WEIGHTS
    }
    gate = joint_eval_eligibility(
        result,
        baseline_bucket_cer=baseline_buckets,
        baseline_text_token_nll=float(baseline["text_replay"]["token_nll"]),
        min_relative_cer_improvement=0.0,
    )
    if optimizer_steps <= 0:
        gate = copy.deepcopy(gate)
        gate["eligible"] = False
        gate["reasons"] = [*gate["reasons"], "no_optimizer_updates"]
    result["eligibility"] = gate
    return result


def _optimizer_contract(optimizer) -> dict[str, Any]:
    return joint_cli._optimizer_contract(optimizer)


def _scheduler_contract(args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "kind": "dol_ocr_anyres_grpo_cosine_scheduler_v1",
        "max_optimizer_steps": int(getattr(args, "max_optimizer_steps", 200)),
        "warmup_steps": args.warmup_steps,
        "min_lr_ratio": 0.1,
        "step_unit": "successful_optimizer_update",
    }
    return {**payload, "canonical_sha256": canonical_json_sha256(payload)}


def _pilot_metadata(
    prepared,
    *,
    run: Mapping[str, Any],
    protocol: Mapping[str, Any],
    progress: Mapping[str, Any],
    baseline: Mapping[str, Any],
    selection_validation: Mapping[str, Any] | None,
    last_validation: Mapping[str, Any] | None,
    dataset_report: Mapping[str, Any],
    optimizer_contract: Mapping[str, Any],
    runtime_source: Mapping[str, Any],
    runtime_environment: Mapping[str, Any],
    final: bool,
) -> dict[str, Any]:
    return {
        **copy.deepcopy(prepared.metadata),
        "phase": "grpo",
        "task": "ocr",
        "training_stage": "grpo_kl_pilot",
        "final": final,
        "stop_reason": progress["stop_reason"],
        GRPO_RUN_METADATA_KEY: copy.deepcopy(dict(run)),
        GRPO_RUN_SHA256_METADATA_KEY: run["canonical_sha256"],
        GRPO_PROGRESS_METADATA_KEY: copy.deepcopy(dict(progress)),
        GRPO_PROGRESS_SHA256_METADATA_KEY: progress["canonical_sha256"],
        PILOT_PROTOCOL_METADATA_KEY: copy.deepcopy(dict(protocol)),
        PILOT_PROTOCOL_SHA256_METADATA_KEY: protocol["canonical_sha256"],
        "grpo_parent_joint_checkpoint": copy.deepcopy(run["parent_joint_checkpoint"]),
        "grpo_admission": copy.deepcopy(run["admission"]),
        "dataset_admission_report": copy.deepcopy(dict(dataset_report)),
        "grpo_optimizer_contract": copy.deepcopy(dict(optimizer_contract)),
        "grpo_runtime_baseline": copy.deepcopy(dict(baseline)),
        "selection_validation": (
            None if selection_validation is None
            else copy.deepcopy(dict(selection_validation))
        ),
        "last_validation": (
            None if last_validation is None else copy.deepcopy(dict(last_validation))
        ),
        "runtime_source_receipt": copy.deepcopy(dict(runtime_source)),
        "runtime_environment": copy.deepcopy(dict(runtime_environment)),
        "locked_golden_opened": False,
    }


def _text_index_by_id(dataset) -> dict[str, int]:
    index_by_id: dict[str, int] = {}
    for index in range(len(dataset)):
        sample_id = str(dataset[index]["metadata"]["id"])
        if sample_id in index_by_id:
            raise ValueError("text train dataset contains duplicate registered IDs")
        index_by_id[sample_id] = index
    if set(index_by_id) != set(dataset.dataset_contract["selected_row_ids"]):
        raise ValueError("text train runtime IDs differ from dataset contract")
    return index_by_id


def _load_text_by_ids(
    dataset,
    sample_ids: Sequence[str],
    collator,
    *,
    index_by_id: Mapping[str, int],
):
    try:
        rows = [dataset[index_by_id[sample_id]] for sample_id in sample_ids]
    except KeyError as exc:
        raise ValueError("pilot protocol references an unknown text train ID") from exc
    return collator(rows)


def _atomic_result(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_json(path, payload)


def _finish_pilot_result(
    output: Path,
    *,
    run: Mapping[str, Any],
    progress: Mapping[str, Any],
    terminal_checkpoint: Mapping[str, Any],
    baseline: Mapping[str, Any],
    selection_validation: Mapping[str, Any],
) -> dict[str, Any]:
    eligible = selection_validation.get("eligibility", {}).get("eligible") is True
    result = build_kl_pilot_result(
        run_contract=run,
        progress=progress,
        terminal_checkpoint=terminal_checkpoint,
        selection_checkpoint=terminal_checkpoint if eligible else None,
        baseline_validation=baseline,
        selection_validation=selection_validation,
    )
    result_path = output / "PILOT_RESULT.json"
    if result_path.exists() or result_path.is_symlink():
        existing = validate_kl_pilot_result(
            _strict_json_file(result_path, where="existing PILOT_RESULT")
        )
        if existing != result:
            raise ValueError("existing PILOT_RESULT differs from terminal recomputation")
    else:
        _atomic_result(result_path, result)
    return result


def _formal_metadata(
    prepared,
    *,
    run,
    protocol,
    progress,
    baseline,
    best_validation,
    last_validation,
    dataset_report,
    optimizer_contract,
    runtime_source,
    runtime_environment,
    locked_golden_anchor_sha256,
    final,
) -> dict[str, Any]:
    return {
        **copy.deepcopy(prepared.metadata),
        "phase": "grpo",
        "task": "ocr",
        "training_stage": "grpo_formal",
        "final": final,
        "stop_reason": progress["stop_reason"],
        GRPO_RUN_METADATA_KEY: copy.deepcopy(run),
        GRPO_RUN_SHA256_METADATA_KEY: run["canonical_sha256"],
        GRPO_PROGRESS_METADATA_KEY: copy.deepcopy(progress),
        GRPO_PROGRESS_SHA256_METADATA_KEY: progress["canonical_sha256"],
        FORMAL_GRPO_PROTOCOL_METADATA_KEY: copy.deepcopy(protocol),
        FORMAL_GRPO_PROTOCOL_SHA256_METADATA_KEY: protocol["canonical_sha256"],
        "formal_monitor_runtime_baseline": copy.deepcopy(baseline),
        "formal_monitor_best_validation": copy.deepcopy(best_validation),
        "last_validation": copy.deepcopy(last_validation),
        "dataset_admission_report": copy.deepcopy(dataset_report),
        "grpo_optimizer_contract": copy.deepcopy(optimizer_contract),
        "runtime_source_receipt": copy.deepcopy(runtime_source),
        "runtime_environment": copy.deepcopy(runtime_environment),
        "locked_golden_opened": False,
        "locked_golden_anchor_sha256": locked_golden_anchor_sha256,
    }


def _finish_formal(
    output: Path,
    *,
    run,
    progress,
    terminal_checkpoint: Path,
    best_validation,
    locked_golden_anchor_sha256: str,
) -> bool:
    if not progress["best_validation_eligible"]:
        no_go = {
            "schema_version": 1,
            "kind": "dol_ocr_anyres_grpo_formal_no_go_v1",
            "formal_run_sha256": run["canonical_sha256"],
            "terminal_progress_sha256": progress["canonical_sha256"],
            "stop_reason": progress["stop_reason"],
            "reason": "no_eligible_formal_monitor_checkpoint",
            "selection_receipt_written": False,
            "locked_golden_anchor_sha256": locked_golden_anchor_sha256,
        }
        no_go["canonical_sha256"] = canonical_json_sha256(no_go)
        _atomic_json(output / "FORMAL_NO_GO.json", no_go)
        return False
    if not isinstance(best_validation, Mapping):
        raise ValueError("formal terminal progress has no best validation report")
    selected = Path(progress["best_checkpoint"])
    receipt = build_anyres_grpo_selection_receipt(
        selected_checkpoint=selected,
        terminal_checkpoint=terminal_checkpoint,
        formal_run_contract=run,
        terminal_progress=progress,
        formal_monitor_best_validation_sha256=progress[
            "best_validation_sha256"
        ],
        locked_golden_anchor_sha256=locked_golden_anchor_sha256,
    )
    write_anyres_grpo_selection_receipt(
        selected.parent / OCR_ANYRES_GRPO_SELECTION_RECEIPT_FILENAME,
        receipt,
        formal_run_contract=run,
        terminal_progress=progress,
    )
    return True


def _run_pilot(args: argparse.Namespace) -> int:
    _validate_pilot_args(args)
    output = Path(args.output_dir).resolve()
    dataset_report = build_dataset_report(_pilot_validator_args(args))
    visual_cli._require_image_manifests_unchanged(args, dataset_report)
    preprocess_payload, _ = _strict_json(
        args.preprocess_contract,
        where="anyres preprocess contract",
    )
    preprocess = validate_anyres_preprocess_contract(preprocess_payload)
    preprocess_sha = preprocess["contract_canonical_sha256"]

    bundle = TokenizerBundle.from_dir(args.tokenizer)
    issues = bundle.validate()
    if issues:
        raise ValueError("invalid tokenizer bundle: " + "; ".join(issues))
    native_encoder = make_ocr_target_encoder(bundle.tokenizer, mode="native")
    tokenizer_contract = native_tokenization_contract(
        bundle.tokenizer,
        args.tokenizer,
    )
    reward_adapter = build_anyres_ocr_reward_adapter(bundle)
    joint_result = _strict_json_file(
        args.joint_stage_result,
        where="JOINT_STAGE_RESULT",
    )
    identity = _joint_best_identity(joint_result)
    parent_path = Path(identity["path"]).resolve()
    if (
        output == parent_path
        or output in parent_path.parents
        or parent_path in output.parents
    ):
        raise ValueError("pilot output and immutable joint parent must be disjoint")
    prepared = admit_joint_policy_for_grpo(
        parent_path,
        identity["model_sha256"],
        identity["metadata_sha256"],
        joint_result,
        reward_contract_sha256=reward_adapter.contract["canonical_sha256"],
        kl_coef=args.kl_coef,
    )
    current_source = _current_source_closure()
    _assert_parent_source_extension(prepared.metadata, current_source)

    max_decode_pixels = int(
        preprocess["budgets"]["decode"]["max_pixels_per_asset"]
    )
    max_views = int(
        preprocess["budgets"]["window"]["max_windows_per_asset"]
    )
    train_dataset = AnyresOCRDataset(
        root=args.root,
        assets_manifest=args.assets,
        views_manifest=args.views,
        samples_manifest=args.train_samples,
        split="train",
        expected_preprocess_contract_sha256=preprocess_sha,
        max_decode_pixels=max_decode_pixels,
        max_views_per_sample=max_views,
        image_delivery="bytes",
    )
    kl_dataset = AnyresOCRDataset(
        root=args.root,
        assets_manifest=args.assets,
        views_manifest=args.views,
        samples_manifest=args.kl_selection_samples,
        split="kl_selection",
        expected_preprocess_contract_sha256=preprocess_sha,
        max_decode_pixels=max_decode_pixels,
        max_views_per_sample=max_views,
        image_delivery="bytes",
    )
    if train_dataset.dataset_contract != visual_cli._image_split_contract(
        dataset_report, "train"
    ):
        raise ValueError("pilot train image dataset differs from admission report")
    if kl_dataset.dataset_contract != visual_cli._image_split_contract(
        dataset_report, "kl_selection"
    ):
        raise ValueError("pilot KL image dataset differs from admission report")
    excluded_documents, excluded_ngrams, _ = _load_exclusions(
        args.reviewed_exclusions
    )
    text_partition = TextReplayPartition(
        args.text_replay,
        native_encoder=native_encoder,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        max_seq_len=int(
            preprocess["budgets"]["context"]["max_sequence_tokens"]
        ),
        exclusion_document_ids=excluded_documents,
        exclusion_ngram_sha256=excluded_ngrams,
    )
    text_train = text_partition.dataset("train")
    text_kl = text_partition.dataset("kl_selection")
    for split, dataset in text_partition.datasets.items():
        if dataset.dataset_contract != visual_cli._text_split_contract(
            dataset_report, split
        ):
            raise ValueError(
                f"pilot text replay {split} differs from admission report"
            )

    expected_metadata = {
        "dataset_admission_report": dataset_report,
        "train_dataset_contract": train_dataset.dataset_contract,
        "kl_selection_dataset_contract": kl_dataset.dataset_contract,
        "text_replay_train_contract": text_train.dataset_contract,
        "text_replay_kl_selection_contract": text_kl.dataset_contract,
        "text_replay_partition_contract": text_partition.partition_contract,
        "anyres_preprocess_contract": preprocess,
        "anyres_preprocess_contract_sha256": preprocess_sha,
        "tokenizer_contract": tokenizer_contract,
    }
    for field, value in expected_metadata.items():
        if prepared.metadata.get(field) != value:
            raise ValueError(f"joint parent differs from current pilot {field}")
    if canonical_json_sha256(dataset_report) != (
        prepared.admission.dataset_admission_report_sha256
    ):
        raise ValueError("dataset admission report differs from GRPO admission")

    omvt_cfg = prepared.metadata["omvt_config"]
    from Model.config import OMVTConfig

    omvt_cfg = OMVTConfig(**omvt_cfg)
    collator = AnyresOCRSFTCollator(
        encode_reference=native_encoder,
        omvt_cfg=omvt_cfg,
        global_processor=PILImageProcessor(
            image_size=omvt_cfg.image_size,
            in_channels=omvt_cfg.in_channels,
        ),
        native_processor=NativeImageProcessorV2(
            in_channels=omvt_cfg.in_channels,
            max_decode_pixels=max_decode_pixels,
        ),
        max_raw_patch_tokens_per_view=int(
            preprocess["budgets"]["patch"]["max_raw_tokens_per_view"]
        ),
        max_seq_len=int(prepared.policy.cfg.max_seq_len),
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_id=PAD_ID,
    )
    validation_batches = visual_cli._validation_batches(kl_dataset, collator)
    text_validation_batches = visual_cli._text_batches(text_kl, pad_id=PAD_ID)
    text_train_collator = TextReplayCollator(pad_id=PAD_ID)
    text_train_ids = tuple(text_train.dataset_contract["selected_row_ids"])
    text_train_index = _text_index_by_id(text_train)
    sampler = OCRQuotaSampler(
        dict(zip(train_dataset.sample_ids, train_dataset.quota_buckets, strict=True)),
        global_batch_size=args.global_batch_size,
        seed=args.seed,
        world_size=1,
    )
    protocol = build_fixed_kl_pilot_protocol(
        sampler=sampler,
        ocr_dataset_contract_sha256=train_dataset.contract_sha256,
        text_sample_ids=text_train_ids,
        text_dataset_contract_sha256=text_train.contract_sha256,
        text_batch_size=args.text_batch_size,
        base_seed=args.seed,
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    policy = prepared.policy.to(device)
    policy.reverse_loss_enabled = False
    reference = prepared.reference
    if reference is not None:
        reference = reference.to(device)
        reference.reverse_loss_enabled = False
        reference.requires_grad_(False)
        reference.eval()
    train_cfg = TrainingConfig(
        learning_rate=args.lm_lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        max_steps=200,
        warmup_steps=args.warmup_steps,
        lr_decay_steps=200,
        precision=args.precision,
        output_dir=args.output_dir,
        optimizer="adamw",
        adam_use_atan2=False,
    )
    optimizer = build_ocr_joint_adamw(
        policy,
        train_cfg,
        lm_lr=args.lm_lr,
        tower_lr=args.tower_lr,
        projector_lr=args.projector_lr,
        bridge_lr=args.bridge_lr,
    )
    scheduler = build_scheduler(optimizer, train_cfg)
    scaler = None
    if args.precision == "fp16":
        if device.type != "cuda":
            raise ValueError("fp16 pilot requires CUDA")
        scaler = torch.amp.GradScaler("cuda")
    optimizer_contract = _optimizer_contract(optimizer)
    scheduler_contract = _scheduler_contract(args)
    trial = AnyresGRPOKLAblationTrial(
        selected_index=(0.04, 0.01, 0.0).index(float(args.kl_coef))
    )
    grpo_config = GRPOConfig(
        clip_eps=None,
        kl_coef=float(args.kl_coef),
        recurrent_steps=None,
        group_size=args.group_size,
        max_new_tokens=prepared.admission.recommended_max_new_tokens,
        temperature=1.0,
        top_p=None,
        log_ratio_clip=20.0,
        advantage_mode="centered",
        min_reward_spread=0.005,
        max_behavior_log_ratio=args.max_behavior_log_ratio,
    )
    runtime_environment = visual_cli._runtime_environment(device, args.precision)
    decode = reward_adapter.decode_completion
    _require_source_unchanged(current_source)
    baseline = evaluate_ocr_joint(
        policy,
        validation_batches,
        decode,
        max_new_tokens=prepared.admission.recommended_max_new_tokens,
        device=device,
        expected_image_dataset_contract_sha256=kl_dataset.contract_sha256,
        text_replay_batches=text_validation_batches,
        expected_text_replay_contract_sha256=text_kl.contract_sha256,
    )
    baseline = _attach_baseline_gate(baseline)
    if baseline["eligibility"]["eligible"] is not True:
        raise ValueError(
            "joint parent fails the current KL-selection baseline gate: "
            f"{baseline['eligibility']['reasons']}"
        )
    schedule = {
        "max_optimizer_steps": 200,
        "max_rollout_attempts": 200,
        "global_batch_size": args.global_batch_size,
        "text_batch_size": args.text_batch_size,
        "eval_every": 50,
        "save_every": 50,
        "early_stop_patience": 5,
        "text_weight": args.text_weight,
        "grad_clip": args.grad_clip,
        "loss_chunk_size": args.loss_chunk_size,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "scheduler_contract_sha256": scheduler_contract["canonical_sha256"],
        "world_size": 1,
        "dist_mode": "single",
        "max_consecutive_no_update": args.max_consecutive_no_update,
        "max_consecutive_optimizer_skips": args.max_consecutive_optimizer_skips,
        "ocr_prompt_schedule_sha256": protocol["payload"]["ocr_prompt_schedule_sha256"],
        "text_batch_schedule_sha256": protocol["payload"]["text_batch_schedule_sha256"],
        "rollout_seed_schedule_sha256": protocol["payload"]["rollout_seed_schedule_sha256"],
    }
    run = build_anyres_grpo_run_contract(
        mode="pilot",
        parent_joint_checkpoint={
            "path": str(parent_path),
            "model_sha256": identity["model_sha256"],
            "metadata_sha256": identity["metadata_sha256"],
        },
        parent_joint_stage_result_sha256=joint_result["canonical_sha256"],
        admission=prepared.admission,
        dataset_ready_admission_sha256=canonical_json_sha256(dataset_report),
        reward_contract_sha256=reward_adapter.contract["canonical_sha256"],
        source_closure_sha256=current_source["canonical_sha256"],
        runtime_environment_sha256=canonical_json_sha256(runtime_environment),
        baseline_validation_sha256=canonical_json_sha256(baseline),
        pilot_protocol=protocol,
        kl_trial=trial,
        formal_selection_receipt=None,
        grpo_config=asdict(grpo_config),
        optimizer_contract_sha256=optimizer_contract["canonical_sha256"],
        schedule=schedule,
        seed=args.seed,
        precision=args.precision,
    )
    validate_protocol_run_bindings(run, protocol)

    last_validation: Mapping[str, Any] | None = None
    selection_validation: Mapping[str, Any] | None = None
    if args.resume:
        checkpoint, envelope, _ = restore_full_checkpoint(
            args.resume,
            model=policy,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        if checkpoint.parent != output:
            raise ValueError("--resume checkpoint is not directly under --output-dir")
        metadata = envelope["metadata"]
        if (
            metadata.get("phase") != "grpo"
            or metadata.get("task") != "ocr"
            or metadata.get("training_stage") != "grpo_kl_pilot"
        ):
            raise ValueError("resume checkpoint is not an anyres OCR GRPO pilot")
        saved_run = validate_anyres_grpo_run_contract(
            metadata.get(GRPO_RUN_METADATA_KEY)
        )
        saved_protocol = metadata.get(PILOT_PROTOCOL_METADATA_KEY)
        validate_fixed_kl_pilot_protocol(saved_protocol)
        if saved_run != run or saved_protocol != protocol:
            raise ValueError("resume run/protocol differs from fresh reconstruction")
        if metadata.get(GRPO_RUN_SHA256_METADATA_KEY) != run["canonical_sha256"]:
            raise ValueError("resume run SHA metadata differs")
        if metadata.get(PILOT_PROTOCOL_SHA256_METADATA_KEY) != protocol[
            "canonical_sha256"
        ]:
            raise ValueError("resume protocol SHA metadata differs")
        resume_objects = {
            "grpo_runtime_baseline": baseline,
            "dataset_admission_report": dataset_report,
            "grpo_optimizer_contract": optimizer_contract,
            "runtime_source_receipt": current_source,
            "runtime_environment": runtime_environment,
        }
        for field, expected_value in resume_objects.items():
            if metadata.get(field) != expected_value:
                raise ValueError(f"resume checkpoint differs from current {field}")
        if metadata.get("locked_golden_opened") is not False:
            raise ValueError("resume checkpoint does not prove golden stayed closed")
        progress = validate_anyres_grpo_progress(
            metadata.get(GRPO_PROGRESS_METADATA_KEY),
            run_contract=run,
        )
        if metadata.get(GRPO_PROGRESS_SHA256_METADATA_KEY) != progress[
            "canonical_sha256"
        ] or envelope["step"] != progress["optimizer_steps"]:
            raise ValueError("resume progress/outer step differs")
        sampler.load_state_dict(progress["sampler_state"])
        validate_sampler_resume_state(
            protocol,
            sampler=sampler,
            attempt_index=progress["rollout_attempts"],
        )
        anchor_progress = progress
        anchor_checkpoint = checkpoint_identity(checkpoint)
        journal_progress = load_no_update_journal(
            output,
            anchor_checkpoint=anchor_checkpoint,
            run_contract=run,
            protocol=protocol,
            anchor_progress=anchor_progress,
        )
        if journal_progress is not None:
            progress = journal_progress
            sampler.load_state_dict(progress["sampler_state"])
            validate_sampler_resume_state(
                protocol,
                sampler=sampler,
                attempt_index=progress["rollout_attempts"],
            )
        last_validation = metadata.get("last_validation")
        selection_validation = metadata.get("selection_validation")
    else:
        progress = initial_pilot_progress(run, sampler=sampler)
        output.mkdir(parents=True, exist_ok=False)
        _atomic_json(output / "DATA_ADMISSION.json", dataset_report)
        _atomic_json(output / "PILOT_PROTOCOL.json", protocol)
        _atomic_json(output / "BASELINE_VALIDATION.json", baseline)
        _atomic_json(output / "RUN_CONTRACT.json", run)
        _require_source_unchanged(current_source)
        step_zero = save_checkpoint(
            output,
            0,
            policy,
            optimizer,
            scheduler,
            metadata=_pilot_metadata(
                prepared,
                run=run,
                protocol=protocol,
                progress=progress,
                baseline=baseline,
                selection_validation=None,
                last_validation=None,
                dataset_report=dataset_report,
                optimizer_contract=optimizer_contract,
                runtime_source=current_source,
                runtime_environment=runtime_environment,
                final=False,
            ),
            keep_last_n=args.keep_last,
            scaler=scaler,
        )
        if step_zero is None:
            raise RuntimeError("single-process step-0 checkpoint was not saved")
        anchor_progress = progress
        anchor_checkpoint = checkpoint_identity(step_zero)

    if progress["terminal"]:
        if selection_validation is None:
            raise ValueError("terminal resume checkpoint has no selection validation")
        _require_source_unchanged(current_source)
        selection_path = output / "SELECTION_VALIDATION.json"
        if selection_path.exists() or selection_path.is_symlink():
            if _strict_json_file(
                selection_path,
                where="existing SELECTION_VALIDATION",
            ) != selection_validation:
                raise ValueError(
                    "existing SELECTION_VALIDATION differs from terminal metadata"
                )
        else:
            _atomic_json(selection_path, selection_validation)
        terminal = checkpoint_identity(
            Path(anchor_checkpoint["path"])
        )
        _finish_pilot_result(
            output,
            run=run,
            progress=progress,
            terminal_checkpoint=terminal,
            baseline=baseline,
            selection_validation=selection_validation,
        )
        return 0

    while progress["rollout_attempts"] < 200:
        attempt_index = int(progress["rollout_attempts"])
        attempt = begin_live_attempt(
            protocol,
            sampler=sampler,
            progress=progress,
            run_contract=run,
        )
        apply_attempt_seed(protocol, attempt_index)
        scheduled_ids = tuple(attempt["ocr_prompt_ids"])

        def load_ocr(ids):
            return collator(
                [train_dataset.get_by_sample_id(sample_id) for sample_id in ids]
            )

        with VerifiedBatchPrefetcher([scheduled_ids], load_ocr, max_prefetch=1) as prefetch:
            batch_ids, ocr_batch = next(prefetch)
        if tuple(batch_ids) != scheduled_ids:
            raise RuntimeError("verified prefetch returned another OCR attempt")
        text_batch = _load_text_by_ids(
            text_train,
            attempt["text_sample_ids"],
            text_train_collator,
            index_by_id=text_train_index,
        )
        metrics = train_anyres_grpo_cycle(
            policy,
            reference,
            ocr_batch,
            reward_adapter,
            prepared.admission,
            grpo_config,
            optimizer,
            kl_trial=trial,
            text_batch=text_batch,
            text_weight=args.text_weight,
            device=device,
            scheduler=scheduler,
            scaler=scaler,
            precision=args.precision,
            grad_clip=args.grad_clip,
            loss_chunk_size=args.loss_chunk_size,
        )
        attempt_after = attempt_index + 1
        should_eval = attempt_after in {50, 100, 150, 200}
        evaluated = None
        eval_count = int(progress["eval_count"])
        bad_eval_count = int(progress["bad_eval_count"])
        best_eligible = False
        best_step = 0
        best_path = None
        best_sha = None
        last_sha = progress["last_validation_sha256"]
        optimizer_after = int(progress["optimizer_steps"]) + int(metrics["stepped"])
        if should_eval:
            _require_source_unchanged(current_source)
            evaluated = evaluate_ocr_joint(
                policy,
                validation_batches,
                decode,
                max_new_tokens=prepared.admission.recommended_max_new_tokens,
                device=device,
                expected_image_dataset_contract_sha256=kl_dataset.contract_sha256,
                text_replay_batches=text_validation_batches,
                expected_text_replay_contract_sha256=text_kl.contract_sha256,
            )
            evaluated = _attach_selection_gate(
                evaluated,
                baseline=baseline,
                optimizer_steps=optimizer_after,
            )
            eval_count += 1
            bad_eval_count += int(
                evaluated["eligibility"]["eligible"] is not True
            )
            last_sha = canonical_json_sha256(evaluated)
            if attempt_after == 200:
                selection_validation = evaluated
                best_eligible = evaluated["eligibility"]["eligible"] is True
                if best_eligible:
                    best_step = optimizer_after
                    best_path = str(output / f"step_{optimizer_after:08d}")
                    best_sha = last_sha
        terminal = attempt_after == 200
        preview = preview_attempt_commit(
            run,
            protocol,
            sampler=sampler,
            progress=progress,
            metrics=metrics,
            eval_count=eval_count,
            bad_eval_count=bad_eval_count,
            best_validation_eligible=best_eligible,
            best_validation_step=best_step,
            best_checkpoint=best_path,
            best_validation_sha256=best_sha,
            last_validation_sha256=last_sha,
            stop_reason="pilot_budget_complete" if terminal else "running",
            terminal=terminal,
        )
        if preview["consecutive_no_update"] > args.max_consecutive_no_update:
            raise RuntimeError("pilot exceeded max_consecutive_no_update")
        commit_live_attempt(
            sampler,
            protocol=protocol,
            previewed_progress=preview,
        )
        progress = preview
        if evaluated is not None:
            last_validation = evaluated
        must_full_save = bool(should_eval or terminal)
        if must_full_save:
            _require_source_unchanged(current_source)
            if progress["optimizer_steps"] == anchor_progress["optimizer_steps"]:
                # A same-step metadata refresh would otherwise leave a journal
                # whose step matches but whose anchor meta hash is stale after
                # a crash between the atomic checkpoint swap and journal clear.
                # Removing it first is safe: the old full checkpoint remains a
                # deterministic replay boundary until the new swap succeeds.
                clear_no_update_journal(output)
            checkpoint = save_checkpoint(
                output,
                int(progress["optimizer_steps"]),
                policy,
                optimizer,
                scheduler,
                metadata=_pilot_metadata(
                    prepared,
                    run=run,
                    protocol=protocol,
                    progress=progress,
                    baseline=baseline,
                    selection_validation=selection_validation,
                    last_validation=last_validation,
                    dataset_report=dataset_report,
                    optimizer_contract=optimizer_contract,
                    runtime_source=current_source,
                    runtime_environment=runtime_environment,
                    final=terminal,
                ),
                keep_last_n=args.keep_last,
                scaler=scaler,
            )
            if checkpoint is None:
                raise RuntimeError("single-process GRPO checkpoint was not saved")
            clear_no_update_journal(output)
            anchor_progress = progress
            anchor_checkpoint = checkpoint_identity(checkpoint)
        elif (
            metrics["stepped"] is False
            and anchor_progress["optimizer_steps"] == progress["optimizer_steps"]
        ):
            _require_source_unchanged(current_source)
            save_no_update_journal(
                output,
                anchor_checkpoint=anchor_checkpoint,
                run_contract=run,
                protocol=protocol,
                anchor_progress=anchor_progress,
                progress=progress,
            )
        if evaluated is not None:
            name = (
                "SELECTION_VALIDATION.json"
                if terminal
                else f"MONITOR_ATTEMPT_{attempt_after:03d}.json"
            )
            _atomic_json(output / name, evaluated)
        print(
            json.dumps(
                {
                    "rollout_attempt": attempt_after,
                    "optimizer_steps": progress["optimizer_steps"],
                    **metrics,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if selection_validation is None:
        raise RuntimeError("fixed pilot ended without attempt-200 selection validation")
    _require_source_unchanged(current_source)
    terminal_identity = checkpoint_identity(anchor_checkpoint["path"])
    _finish_pilot_result(
        output,
        run=run,
        progress=progress,
        terminal_checkpoint=terminal_identity,
        baseline=baseline,
        selection_validation=selection_validation,
    )
    return 0


def _formal_stop_reason(
    *,
    optimizer_steps: int,
    rollout_attempts: int,
    max_optimizer_steps: int,
    max_rollout_attempts: int,
    plateau: bool,
) -> str | None:
    if plateau:
        return "validation_plateau"
    optimizer_done = optimizer_steps >= max_optimizer_steps
    rollout_done = rollout_attempts >= max_rollout_attempts
    if optimizer_done and rollout_done:
        return "completed"
    if optimizer_done:
        return "max_optimizer_steps"
    if rollout_done:
        return "max_rollout_attempts"
    return None


def _formal_validation_decision(
    report: Mapping[str, Any],
    *,
    best_validation: Mapping[str, Any] | None,
    optimizer_steps: int,
    bad_eval_count: int,
) -> dict[str, Any]:
    eligible = report.get("eligibility", {}).get("eligible") is True
    current_cer = float(report["deployment_weighted_cer"])
    previous_cer = (
        float("inf")
        if best_validation is None
        else float(best_validation["deployment_weighted_cer"])
    )
    improved = bool(
        eligible
        and optimizer_steps > 0
        and visual_cli._strict_relative_improvement(current_cer, previous_cer)
    )
    if improved:
        return {"new_best": True, "bad_eval_count": 0}
    return {
        "new_best": False,
        "bad_eval_count": (
            bad_eval_count + 1 if best_validation is not None else bad_eval_count
        ),
    }


def _run_formal(args: argparse.Namespace) -> int:
    _validate_formal_args(args)
    kl_receipt = select_authenticated_pilots(
        args.pilot_result,
        existing_receipt=args.kl_selection_receipt,
    )
    if kl_receipt["status"] != "selected" or kl_receipt[
        "formal_run_allowed"
    ] is not True:
        raise ValueError("KL selection receipt does not authorize formal training")
    selected_kl = float(kl_receipt["selected_kl_coef"])
    pilot_seeds: set[int] = set()
    authenticated_result_shas = set(kl_receipt["candidate_result_sha256"])
    for index, path in enumerate(args.pilot_result):
        result = validate_kl_pilot_result(
            _strict_json_file(path, where=f"formal PILOT_RESULT[{index}]")
        )
        if result["canonical_sha256"] not in authenticated_result_shas:
            raise ValueError("PILOT_RESULT changed after authenticated selection")
        envelope, _ = load_verified_checkpoint_metadata_envelope(
            Path(result["terminal_checkpoint"]["path"]) / "meta.pt",
            expected_sha256=result["terminal_checkpoint"]["metadata_sha256"],
        )
        pilot_run = validate_anyres_grpo_run_contract(
            envelope["metadata"].get(GRPO_RUN_METADATA_KEY)
        )
        pilot_seeds.add(int(pilot_run["seed"]))
    if args.seed in pilot_seeds:
        raise ValueError("formal seed must differ from every pilot seed")

    output = Path(args.output_dir).resolve()
    joint_result = _strict_json_file(
        args.joint_stage_result,
        where="formal JOINT_STAGE_RESULT",
    )
    identity = _joint_best_identity(joint_result)
    parent_path = Path(identity["path"]).resolve()
    if output == parent_path or output in parent_path.parents or parent_path in output.parents:
        raise ValueError("formal output must be disjoint from parent joint checkpoint")
    if {
        "path": str(Path(identity["path"]).resolve()),
        "model_sha256": identity["model_sha256"],
        "metadata_sha256": identity["metadata_sha256"],
    } != kl_receipt["formal_init_checkpoint"]:
        raise ValueError("formal init joint checkpoint differs from KL receipt")
    if joint_result.get("canonical_sha256") != kl_receipt[
        "parent_joint_stage_result_sha256"
    ]:
        raise ValueError("formal JOINT_STAGE_RESULT differs from KL receipt")
    for result_path in args.pilot_result:
        pilot_root = Path(
            _strict_json_file(result_path, where="formal pilot path")[
                "terminal_checkpoint"
            ]["path"]
        ).parent
        if output == pilot_root or output in pilot_root.parents or pilot_root in output.parents:
            raise ValueError("formal output must be disjoint from every pilot output")

    dataset_report = build_dataset_report(_pilot_validator_args(args))
    visual_cli._require_image_manifests_unchanged(args, dataset_report)
    preprocess_payload, _ = _strict_json(
        args.preprocess_contract,
        where="formal anyres preprocess contract",
    )
    preprocess = validate_anyres_preprocess_contract(preprocess_payload)
    preprocess_sha = preprocess["contract_canonical_sha256"]
    bundle = TokenizerBundle.from_dir(args.tokenizer)
    issues = bundle.validate()
    if issues:
        raise ValueError("invalid tokenizer bundle: " + "; ".join(issues))
    native_encoder = make_ocr_target_encoder(bundle.tokenizer, mode="native")
    tokenizer_contract = native_tokenization_contract(bundle.tokenizer, args.tokenizer)
    reward_adapter = build_anyres_ocr_reward_adapter(bundle)
    prepared = admit_joint_policy_for_grpo(
        parent_path,
        identity["model_sha256"],
        identity["metadata_sha256"],
        joint_result,
        reward_contract_sha256=reward_adapter.contract["canonical_sha256"],
        kl_coef=selected_kl,
    )
    if prepared.admission.canonical_sha256 != kl_receipt["admission_sha256"]:
        raise ValueError("fresh formal admission differs from KL selection receipt")
    current_source = _current_source_closure()
    if current_source["canonical_sha256"] != kl_receipt["source_closure_sha256"]:
        raise ValueError("formal source closure differs from authenticated pilots")
    _assert_parent_source_extension(prepared.metadata, current_source)

    max_decode_pixels = int(preprocess["budgets"]["decode"]["max_pixels_per_asset"])
    max_views = int(preprocess["budgets"]["window"]["max_windows_per_asset"])
    train_dataset = AnyresOCRDataset(
        root=args.root,
        assets_manifest=args.assets,
        views_manifest=args.views,
        samples_manifest=args.train_samples,
        split="train",
        expected_preprocess_contract_sha256=preprocess_sha,
        max_decode_pixels=max_decode_pixels,
        max_views_per_sample=max_views,
        image_delivery="bytes",
    )
    monitor_dataset = AnyresOCRDataset(
        root=args.root,
        assets_manifest=args.assets,
        views_manifest=args.views,
        samples_manifest=args.formal_monitor_samples,
        split="formal_monitor",
        expected_preprocess_contract_sha256=preprocess_sha,
        max_decode_pixels=max_decode_pixels,
        max_views_per_sample=max_views,
        image_delivery="bytes",
    )
    if train_dataset.dataset_contract != visual_cli._image_split_contract(dataset_report, "train"):
        raise ValueError("formal train image dataset differs from admission")
    if monitor_dataset.dataset_contract != visual_cli._image_split_contract(dataset_report, "formal_monitor"):
        raise ValueError("formal monitor image dataset differs from admission")
    excluded_documents, excluded_ngrams, _ = _load_exclusions(args.reviewed_exclusions)
    text_partition = TextReplayPartition(
        args.text_replay,
        native_encoder=native_encoder,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        max_seq_len=int(preprocess["budgets"]["context"]["max_sequence_tokens"]),
        exclusion_document_ids=excluded_documents,
        exclusion_ngram_sha256=excluded_ngrams,
    )
    for split, dataset in text_partition.datasets.items():
        if dataset.dataset_contract != visual_cli._text_split_contract(dataset_report, split):
            raise ValueError(f"formal text replay {split} differs from admission")
    text_train = text_partition.dataset("train")
    text_monitor = text_partition.dataset("formal_monitor")
    expected_metadata = {
        "dataset_admission_report": dataset_report,
        "train_dataset_contract": train_dataset.dataset_contract,
        "formal_monitor_dataset_contract": monitor_dataset.dataset_contract,
        "text_replay_train_contract": text_train.dataset_contract,
        "text_replay_formal_monitor_contract": text_monitor.dataset_contract,
        "text_replay_partition_contract": text_partition.partition_contract,
        "anyres_preprocess_contract": preprocess,
        "anyres_preprocess_contract_sha256": preprocess_sha,
        "tokenizer_contract": tokenizer_contract,
    }
    for field, expected in expected_metadata.items():
        if prepared.metadata.get(field) != expected:
            raise ValueError(f"joint parent differs from current formal {field}")

    from Model.config import OMVTConfig

    omvt_cfg = OMVTConfig(**prepared.metadata["omvt_config"])
    collator = AnyresOCRSFTCollator(
        encode_reference=native_encoder,
        omvt_cfg=omvt_cfg,
        global_processor=PILImageProcessor(
            image_size=omvt_cfg.image_size,
            in_channels=omvt_cfg.in_channels,
        ),
        native_processor=NativeImageProcessorV2(
            in_channels=omvt_cfg.in_channels,
            max_decode_pixels=max_decode_pixels,
        ),
        max_raw_patch_tokens_per_view=int(
            preprocess["budgets"]["patch"]["max_raw_tokens_per_view"]
        ),
        max_seq_len=int(prepared.policy.cfg.max_seq_len),
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_id=PAD_ID,
    )
    monitor_batches = visual_cli._validation_batches(monitor_dataset, collator)
    text_monitor_batches = visual_cli._text_batches(text_monitor, pad_id=PAD_ID)
    text_train_collator = TextReplayCollator(pad_id=PAD_ID)
    text_train_ids = tuple(text_train.dataset_contract["selected_row_ids"])
    text_train_index = _text_index_by_id(text_train)
    sampler = OCRQuotaSampler(
        dict(zip(train_dataset.sample_ids, train_dataset.quota_buckets, strict=True)),
        global_batch_size=args.global_batch_size,
        seed=args.seed,
        world_size=1,
    )
    formal_protocol = build_formal_grpo_protocol(
        sampler=sampler,
        ocr_dataset_contract_sha256=train_dataset.contract_sha256,
        text_sample_ids=text_train_ids,
        text_dataset_contract_sha256=text_train.contract_sha256,
        text_batch_size=args.text_batch_size,
        base_seed=args.seed,
        max_rollout_attempts=args.max_rollout_attempts,
        max_optimizer_steps=args.max_optimizer_steps,
        eval_every=args.eval_every,
        save_every=args.save_every,
        early_stop_patience=args.early_stop_patience,
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    policy = prepared.policy.to(device)
    policy.reverse_loss_enabled = False
    reference = prepared.reference
    if reference is not None:
        reference = reference.to(device)
        reference.reverse_loss_enabled = False
        reference.requires_grad_(False)
        reference.eval()
    train_cfg = TrainingConfig(
        learning_rate=args.lm_lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        max_steps=args.max_optimizer_steps,
        warmup_steps=args.warmup_steps,
        lr_decay_steps=args.max_optimizer_steps,
        precision=args.precision,
        output_dir=args.output_dir,
        optimizer="adamw",
        adam_use_atan2=False,
    )
    optimizer = build_ocr_joint_adamw(
        policy,
        train_cfg,
        lm_lr=args.lm_lr,
        tower_lr=args.tower_lr,
        projector_lr=args.projector_lr,
        bridge_lr=args.bridge_lr,
    )
    scheduler = build_scheduler(optimizer, train_cfg)
    scaler = None
    if args.precision == "fp16":
        if device.type != "cuda":
            raise ValueError("fp16 formal training requires CUDA")
        scaler = torch.amp.GradScaler("cuda")
    optimizer_contract = _optimizer_contract(optimizer)
    scheduler_contract = _scheduler_contract(args)
    selected_index = (0.04, 0.01, 0.0).index(selected_kl)
    trial = AnyresGRPOKLAblationTrial(
        selected_index=selected_index,
        trial_steps=args.max_rollout_attempts,
        experiment_id="ocr_anyres_formal_v1",
    )
    grpo_config = GRPOConfig(
        clip_eps=None,
        kl_coef=selected_kl,
        recurrent_steps=None,
        group_size=args.group_size,
        max_new_tokens=prepared.admission.recommended_max_new_tokens,
        temperature=1.0,
        top_p=None,
        advantage_mode="centered",
        min_reward_spread=0.005,
        max_behavior_log_ratio=args.max_behavior_log_ratio,
    )
    runtime_environment = visual_cli._runtime_environment(device, args.precision)
    decode = reward_adapter.decode_completion
    _require_source_unchanged(current_source)
    baseline = evaluate_ocr_joint(
        policy,
        monitor_batches,
        decode,
        max_new_tokens=prepared.admission.recommended_max_new_tokens,
        device=device,
        expected_image_dataset_contract_sha256=monitor_dataset.contract_sha256,
        text_replay_batches=text_monitor_batches,
        expected_text_replay_contract_sha256=text_monitor.contract_sha256,
    )
    baseline = _attach_baseline_gate(baseline)
    if baseline["eligibility"]["eligible"] is not True:
        raise ValueError("joint parent fails formal-monitor baseline gate")
    schedule = {
        "max_optimizer_steps": args.max_optimizer_steps,
        "max_rollout_attempts": args.max_rollout_attempts,
        "global_batch_size": args.global_batch_size,
        "text_batch_size": args.text_batch_size,
        "eval_every": args.eval_every,
        "save_every": args.save_every,
        "early_stop_patience": args.early_stop_patience,
        "text_weight": args.text_weight,
        "grad_clip": args.grad_clip,
        "loss_chunk_size": args.loss_chunk_size,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "scheduler_contract_sha256": scheduler_contract["canonical_sha256"],
        "world_size": 1,
        "dist_mode": "single",
        "max_consecutive_no_update": args.max_consecutive_no_update,
        "max_consecutive_optimizer_skips": args.max_consecutive_optimizer_skips,
        "ocr_prompt_schedule_sha256": formal_protocol["payload"]["ocr_prompt_schedule_sha256"],
        "text_batch_schedule_sha256": formal_protocol["payload"]["text_batch_schedule_sha256"],
        "rollout_seed_schedule_sha256": formal_protocol["payload"]["rollout_seed_schedule_sha256"],
    }
    run = build_anyres_grpo_run_contract(
        mode="formal",
        parent_joint_checkpoint=kl_receipt["formal_init_checkpoint"],
        parent_joint_stage_result_sha256=joint_result["canonical_sha256"],
        admission=prepared.admission,
        dataset_ready_admission_sha256=canonical_json_sha256(dataset_report),
        reward_contract_sha256=reward_adapter.contract["canonical_sha256"],
        source_closure_sha256=current_source["canonical_sha256"],
        runtime_environment_sha256=canonical_json_sha256(runtime_environment),
        baseline_validation_sha256=canonical_json_sha256(baseline),
        pilot_protocol=None,
        formal_protocol=formal_protocol,
        kl_trial=trial,
        formal_selection_receipt=kl_receipt,
        grpo_config=asdict(grpo_config),
        optimizer_contract_sha256=optimizer_contract["canonical_sha256"],
        schedule=schedule,
        seed=args.seed,
        precision=args.precision,
    )
    validate_protocol_run_bindings(run, formal_protocol)

    best_validation = None
    last_validation = None
    if args.resume:
        checkpoint, envelope, _ = restore_full_checkpoint(
            args.resume,
            model=policy,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        if checkpoint.parent != output:
            raise ValueError("formal resume checkpoint is outside output root")
        metadata = envelope["metadata"]
        if (
            metadata.get("phase") != "grpo"
            or metadata.get("task") != "ocr"
            or metadata.get("training_stage") != "grpo_formal"
            or metadata.get("locked_golden_opened") is not False
            or metadata.get("locked_golden_anchor_sha256")
            != args.locked_golden_anchor_sha256
        ):
            raise ValueError("formal resume metadata differs")
        saved_run = validate_anyres_grpo_run_contract(metadata.get(GRPO_RUN_METADATA_KEY))
        saved_protocol = metadata.get(FORMAL_GRPO_PROTOCOL_METADATA_KEY)
        rebuild_and_match_formal_protocol(
            saved_protocol,
            sampler=sampler,
            ocr_dataset_contract_sha256=train_dataset.contract_sha256,
            text_sample_ids=text_train_ids,
            text_dataset_contract_sha256=text_train.contract_sha256,
        )
        if saved_run != run or saved_protocol != formal_protocol:
            raise ValueError("formal resume run/protocol differs from fresh reconstruction")
        if (
            metadata.get(GRPO_RUN_SHA256_METADATA_KEY) != run["canonical_sha256"]
            or metadata.get(FORMAL_GRPO_PROTOCOL_SHA256_METADATA_KEY)
            != formal_protocol["canonical_sha256"]
        ):
            raise ValueError("formal resume run/protocol SHA differs")
        for field, expected_value in {
            "formal_monitor_runtime_baseline": baseline,
            "dataset_admission_report": dataset_report,
            "grpo_optimizer_contract": optimizer_contract,
            "runtime_source_receipt": current_source,
            "runtime_environment": runtime_environment,
        }.items():
            if metadata.get(field) != expected_value:
                raise ValueError(f"formal resume differs from current {field}")
        progress = validate_anyres_grpo_progress(
            metadata.get(GRPO_PROGRESS_METADATA_KEY),
            run_contract=run,
        )
        if envelope["step"] != progress["optimizer_steps"]:
            raise ValueError("formal resume outer step differs from progress")
        if (
            metadata.get(GRPO_PROGRESS_SHA256_METADATA_KEY)
            != progress["canonical_sha256"]
            or metadata.get("final") is not progress["terminal"]
            or metadata.get("stop_reason") != progress["stop_reason"]
        ):
            raise ValueError("formal resume progress metadata differs")
        sampler.load_state_dict(progress["sampler_state"])
        validate_formal_sampler_boundary(formal_protocol, sampler, progress["rollout_attempts"])
        anchor_progress = progress
        anchor_checkpoint = checkpoint_identity(checkpoint)
        journal = load_no_update_journal(
            output,
            anchor_checkpoint=anchor_checkpoint,
            run_contract=run,
            protocol=formal_protocol,
            anchor_progress=anchor_progress,
        )
        if journal is not None:
            progress = journal
            sampler.load_state_dict(progress["sampler_state"])
            validate_formal_sampler_boundary(formal_protocol, sampler, progress["rollout_attempts"])
        best_validation = metadata.get("formal_monitor_best_validation")
        last_validation = metadata.get("last_validation")
    else:
        progress = initial_pilot_progress(run, sampler=sampler)
        output.mkdir(parents=True, exist_ok=False)
        for filename, payload in (
            ("DATA_ADMISSION.json", dataset_report),
            ("FORMAL_PROTOCOL.json", formal_protocol),
            ("FORMAL_MONITOR_BASELINE.json", baseline),
            ("RUN_CONTRACT.json", run),
        ):
            _atomic_json(output / filename, payload)
        _require_source_unchanged(current_source)
        step_zero = save_checkpoint(
            output,
            0,
            policy,
            optimizer,
            scheduler,
            metadata=_formal_metadata(
                prepared,
                run=run,
                protocol=formal_protocol,
                progress=progress,
                baseline=baseline,
                best_validation=None,
                last_validation=None,
                dataset_report=dataset_report,
                optimizer_contract=optimizer_contract,
                runtime_source=current_source,
                runtime_environment=runtime_environment,
                locked_golden_anchor_sha256=args.locked_golden_anchor_sha256,
                final=False,
            ),
            keep_last_n=args.keep_last,
            scaler=scaler,
        )
        if step_zero is None:
            raise RuntimeError("formal step-0 checkpoint was not saved")
        anchor_progress = progress
        anchor_checkpoint = checkpoint_identity(step_zero)

    if progress["terminal"]:
        _require_source_unchanged(current_source)
        selected = _finish_formal(
            output,
            run=run,
            progress=progress,
            terminal_checkpoint=Path(anchor_checkpoint["path"]),
            best_validation=best_validation,
            locked_golden_anchor_sha256=args.locked_golden_anchor_sha256,
        )
        return 0 if selected else 2

    while not progress["terminal"]:
        attempt_index = int(progress["rollout_attempts"])
        attempt = begin_live_attempt(
            formal_protocol,
            sampler=sampler,
            progress=progress,
            run_contract=run,
        )
        apply_formal_attempt_seed(formal_protocol, attempt_index)
        scheduled_ids = tuple(attempt["ocr_prompt_ids"])

        def load_ocr(ids):
            return collator([train_dataset.get_by_sample_id(value) for value in ids])

        with VerifiedBatchPrefetcher([scheduled_ids], load_ocr, max_prefetch=1) as prefetch:
            batch_ids, ocr_batch = next(prefetch)
        if tuple(batch_ids) != scheduled_ids:
            raise RuntimeError("formal prefetch returned another attempt")
        text_batch = _load_text_by_ids(
            text_train,
            attempt["text_sample_ids"],
            text_train_collator,
            index_by_id=text_train_index,
        )
        metrics = train_anyres_grpo_cycle(
            policy,
            reference,
            ocr_batch,
            reward_adapter,
            prepared.admission,
            grpo_config,
            optimizer,
            kl_trial=trial,
            text_batch=text_batch,
            text_weight=args.text_weight,
            device=device,
            scheduler=scheduler,
            scaler=scaler,
            precision=args.precision,
            grad_clip=args.grad_clip,
            loss_chunk_size=args.loss_chunk_size,
        )
        attempt_after = attempt_index + 1
        optimizer_after = int(progress["optimizer_steps"]) + int(metrics["stepped"])
        should_eval = (
            attempt_after % args.eval_every == 0
            or optimizer_after >= args.max_optimizer_steps
            or attempt_after >= args.max_rollout_attempts
        )
        eval_count = int(progress["eval_count"])
        bad_eval_count = int(progress["bad_eval_count"])
        best_eligible = bool(progress["best_validation_eligible"])
        best_step = int(progress["best_validation_step"])
        best_path = progress["best_checkpoint"]
        best_sha = progress["best_validation_sha256"]
        new_best = False
        evaluated = None
        if should_eval:
            _require_source_unchanged(current_source)
            evaluated = evaluate_ocr_joint(
                policy,
                monitor_batches,
                decode,
                max_new_tokens=prepared.admission.recommended_max_new_tokens,
                device=device,
                expected_image_dataset_contract_sha256=monitor_dataset.contract_sha256,
                text_replay_batches=text_monitor_batches,
                expected_text_replay_contract_sha256=text_monitor.contract_sha256,
            )
            evaluated = _attach_selection_gate(
                evaluated,
                baseline=baseline,
                optimizer_steps=optimizer_after,
            )
            eval_count += 1
            decision = _formal_validation_decision(
                evaluated,
                best_validation=best_validation,
                optimizer_steps=optimizer_after,
                bad_eval_count=bad_eval_count,
            )
            new_best = decision["new_best"]
            if new_best:
                best_validation = evaluated
                best_eligible = True
                best_step = optimizer_after
                best_path = str(output / "best" / f"step_{optimizer_after:08d}")
                best_sha = canonical_json_sha256(evaluated)
                bad_eval_count = 0
            else:
                bad_eval_count = decision["bad_eval_count"]
            last_validation = evaluated
        plateau = bool(
            best_eligible and bad_eval_count >= args.early_stop_patience
        )
        stop_reason = _formal_stop_reason(
            optimizer_steps=optimizer_after,
            rollout_attempts=attempt_after,
            max_optimizer_steps=args.max_optimizer_steps,
            max_rollout_attempts=args.max_rollout_attempts,
            plateau=plateau,
        )
        terminal = stop_reason is not None
        preview = preview_attempt_commit(
            run,
            formal_protocol,
            sampler=sampler,
            progress=progress,
            metrics=metrics,
            eval_count=eval_count,
            bad_eval_count=bad_eval_count,
            best_validation_eligible=best_eligible,
            best_validation_step=best_step,
            best_checkpoint=best_path,
            best_validation_sha256=best_sha,
            last_validation_sha256=(
                None if last_validation is None
                else canonical_json_sha256(last_validation)
            ),
            stop_reason=stop_reason or "running",
            terminal=terminal,
        )
        if preview["consecutive_no_update"] > args.max_consecutive_no_update:
            raise RuntimeError("formal run exceeded max_consecutive_no_update")
        commit_live_attempt(
            sampler,
            protocol=formal_protocol,
            previewed_progress=preview,
        )
        progress = preview
        if new_best:
            _require_source_unchanged(current_source)
            best_checkpoint = save_checkpoint(
                output / "best",
                optimizer_after,
                policy,
                optimizer,
                scheduler,
                metadata=_formal_metadata(
                    prepared,
                    run=run,
                    protocol=formal_protocol,
                    progress=progress,
                    baseline=baseline,
                    best_validation=best_validation,
                    last_validation=last_validation,
                    dataset_report=dataset_report,
                    optimizer_contract=optimizer_contract,
                    runtime_source=current_source,
                    runtime_environment=runtime_environment,
                    locked_golden_anchor_sha256=args.locked_golden_anchor_sha256,
                    final=terminal,
                ),
                keep_last_n=args.keep_last,
                scaler=scaler,
            )
            if best_checkpoint is None or str(best_checkpoint.resolve()) != best_path:
                raise RuntimeError("formal eligible best checkpoint was not saved")
        must_root_save = (
            terminal or should_eval or attempt_after % args.save_every == 0
        )
        if must_root_save:
            _require_source_unchanged(current_source)
            if progress["optimizer_steps"] == anchor_progress["optimizer_steps"]:
                clear_no_update_journal(output)
            root_checkpoint = save_checkpoint(
                output,
                int(progress["optimizer_steps"]),
                policy,
                optimizer,
                scheduler,
                metadata=_formal_metadata(
                    prepared,
                    run=run,
                    protocol=formal_protocol,
                    progress=progress,
                    baseline=baseline,
                    best_validation=best_validation,
                    last_validation=last_validation,
                    dataset_report=dataset_report,
                    optimizer_contract=optimizer_contract,
                    runtime_source=current_source,
                    runtime_environment=runtime_environment,
                    locked_golden_anchor_sha256=args.locked_golden_anchor_sha256,
                    final=terminal,
                ),
                keep_last_n=args.keep_last,
                scaler=scaler,
            )
            if root_checkpoint is None:
                raise RuntimeError("formal root checkpoint was not saved")
            clear_no_update_journal(output)
            anchor_progress = progress
            anchor_checkpoint = checkpoint_identity(root_checkpoint)
        elif (
            metrics["stepped"] is False
            and anchor_progress["optimizer_steps"] == progress["optimizer_steps"]
        ):
            _require_source_unchanged(current_source)
            save_no_update_journal(
                output,
                anchor_checkpoint=anchor_checkpoint,
                run_contract=run,
                protocol=formal_protocol,
                anchor_progress=anchor_progress,
                progress=progress,
            )
        if evaluated is not None:
            _atomic_json(output / f"FORMAL_MONITOR_ATTEMPT_{attempt_after:08d}.json", evaluated)
        if terminal:
            break

    _require_source_unchanged(current_source)
    selected = _finish_formal(
        output,
        run=run,
        progress=progress,
        terminal_checkpoint=Path(anchor_checkpoint["path"]),
        best_validation=best_validation,
        locked_golden_anchor_sha256=args.locked_golden_anchor_sha256,
    )
    return 0 if selected else 2


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "select-kl":
        return _run_select_kl(args)
    if args.command == "pilot":
        return _run_pilot(args)
    if args.command == "formal":
        return _run_formal(args)
    raise ValueError(f"unsupported command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
