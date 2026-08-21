# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
import copy
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import torch

from Model.ocr.tokenization import canonical_json_sha256
from Model.posttrain.ocr_anyres_grpo_run import (
    build_anyres_grpo_run_contract,
    build_anyres_grpo_progress,
    build_kl_pilot_result,
    build_text_schedule_cursor_state,
)
from Model.tests.test_ocr_anyres_grpo_run import (
    _admission as fixture_admission,
    _pilot_protocol as fixture_protocol,
    _schedule as fixture_schedule,
    _validation as fixture_validation,
)
from Model.posttrain.ocr_anyres_grpo_trainer import AnyresGRPOKLAblationTrial
from Model.posttrain.ocr_quota_sampler import OCRQuotaSampler
from scripts import train_ocr_anyres_grpo as cli


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def _terminal_sampler_state_cached() -> dict:
    buckets = {
        f"{bucket}-{index:03d}": bucket
        for bucket in (
            "print",
            "handwritten_good",
            "handwritten_medium",
            "handwritten_poor",
        )
        for index in range(20)
    }
    sampler = OCRQuotaSampler(
        buckets,
        global_batch_size=20,
        seed=811,
        world_size=1,
    )
    for _ in range(200):
        sampler.prepare_global_batch()
        sampler.commit_global_batch()
    return sampler.state_dict()


def _progress(
    run: dict,
    *,
    eligible: bool,
    selection_path: Path | None,
    selection_sha: str,
) -> dict:
    optimizer_steps = 195
    return build_anyres_grpo_progress(
        run,
        optimizer_steps=optimizer_steps,
        rollout_attempts=200,
        no_update_count=5,
        consecutive_no_update=0,
        optimizer_skip_count=0,
        consecutive_optimizer_skips=0,
        sampler_state=copy.deepcopy(_terminal_sampler_state_cached()),
        text_cursor_state=build_text_schedule_cursor_state(
            dataset_contract_sha256=run["text_replay_train_contract_sha256"],
            batch_size=4,
            schedule_cursor=200,
            ce_steps=optimizer_steps,
        ),
        text_ce_steps=optimizer_steps,
        eval_count=4,
        bad_eval_count=0,
        best_validation_eligible=eligible,
        best_validation_step=optimizer_steps if eligible else 0,
        best_checkpoint=str(selection_path) if eligible else None,
        best_validation_sha256=selection_sha if eligible else None,
        last_validation_sha256=selection_sha,
        stop_reason="pilot_budget_complete",
        terminal=True,
    )


def _checkpoint_bytes(
    run: dict,
    progress: dict,
    protocol: dict,
    *,
    baseline: dict,
    selection: dict,
    dataset_report: dict,
    runtime_source: dict,
    runtime_environment: dict,
    outer_step: int,
) -> bytes:
    metadata = {
        "phase": "grpo",
        "task": "ocr",
        "final": True,
        cli.GRPO_RUN_METADATA_KEY: run,
        cli.GRPO_RUN_SHA256_METADATA_KEY: run["canonical_sha256"],
        cli.GRPO_PROGRESS_METADATA_KEY: progress,
        cli.GRPO_PROGRESS_SHA256_METADATA_KEY: progress["canonical_sha256"],
        cli.PILOT_PROTOCOL_METADATA_KEY: protocol,
        cli.PILOT_PROTOCOL_SHA256_METADATA_KEY: protocol["canonical_sha256"],
        "training_stage": "grpo_kl_pilot",
        "locked_golden_opened": False,
        "grpo_runtime_baseline": baseline,
        "selection_validation": selection,
        "dataset_admission_report": dataset_report,
        "runtime_source_receipt": runtime_source,
        "runtime_environment": runtime_environment,
        "grpo_optimizer_contract": {
            "canonical_sha256": run["optimizer_contract_sha256"]
        },
    }
    buffer = io.BytesIO()
    torch.save({"step": outer_step, "metadata": metadata}, buffer)
    return buffer.getvalue()


def _write_checkpoint(
    path: Path,
    *,
    model: bytes,
    meta: bytes,
    marker_step: int,
    include_scaler: bool,
) -> dict[str, str]:
    path.mkdir(parents=True)
    (path / "COMPLETE").write_text(
        f"step={marker_step}\n",
        encoding="ascii",
    )
    (path / "model.pt").write_bytes(model)
    (path / "meta.pt").write_bytes(meta)
    for name in ("optimizer.pt", "scheduler.pt", "rng.pt"):
        (path / name).write_bytes(f"tiny-{name}".encode("ascii"))
    if include_scaler:
        (path / "scaler.pt").write_bytes(b"tiny-scaler")
    return {
        "path": str(path.resolve()),
        "model_sha256": _sha256(path / "model.pt"),
        "metadata_sha256": _sha256(path / "meta.pt"),
    }


def _parent_checkpoint(root: Path) -> dict[str, str]:
    path = (root / "joint-parent").resolve()
    if not path.exists():
        path.mkdir(parents=True)
        (path / "model.pt").write_bytes(b"tiny-joint-parent")
        buffer = io.BytesIO()
        torch.save(
            {
                "step": 7,
                "metadata": {"training_stage": "joint", "final": False},
            },
            buffer,
        )
        (path / "meta.pt").write_bytes(buffer.getvalue())
    return {
        "path": str(path),
        "model_sha256": _sha256(path / "model.pt"),
        "metadata_sha256": _sha256(path / "meta.pt"),
    }


def _run_with_parent(
    selected_index: int,
    *,
    baseline: dict,
    parent: dict[str, str],
    protocol: dict,
    precision: str,
    dataset_report: dict,
    runtime_source: dict,
    runtime_environment: dict,
) -> dict:
    trial = AnyresGRPOKLAblationTrial(selected_index=selected_index)
    admission = replace(
        fixture_admission(),
        policy_checkpoint_sha256=parent["model_sha256"],
        policy_metadata_sha256=parent["metadata_sha256"],
        reference_checkpoint_sha256=parent["model_sha256"],
        reference_metadata_sha256=parent["metadata_sha256"],
        dataset_admission_report_sha256=canonical_json_sha256(dataset_report),
    )
    return build_anyres_grpo_run_contract(
        mode="pilot",
        parent_joint_checkpoint=parent,
        parent_joint_stage_result_sha256="3" * 64,
        admission=admission,
        dataset_ready_admission_sha256=canonical_json_sha256(dataset_report),
        reward_contract_sha256="a" * 64,
        source_closure_sha256=runtime_source["canonical_sha256"],
        runtime_environment_sha256=canonical_json_sha256(runtime_environment),
        baseline_validation_sha256=canonical_json_sha256(baseline),
        pilot_protocol=protocol,
        kl_trial=trial,
        formal_selection_receipt=None,
        grpo_config={
            "kl_coef": trial.selected_kl_coef,
            "max_new_tokens": 1,
            "min_reward_spread": 0.005,
            "clip_eps": None,
            "advantage_mode": "centered",
        },
        optimizer_contract_sha256="8" * 64,
        schedule=fixture_schedule(protocol),
        seed=42,
        precision=precision,
    )


def _provenance() -> tuple[dict, dict, dict]:
    sample_ids = [
        f"{bucket}-{index}"
        for bucket in (
            "print",
            "handwritten_good",
            "handwritten_medium",
            "handwritten_poor",
        )
        for index in range(2)
    ]
    dataset_report = {
        "kind": "tiny-dataset-admission",
        "schema_version": 1,
        "datasets": {
            "kl_selection": {"contract": {"sample_ids": sample_ids}}
        },
        "text_replay": {
            "splits": {
                "kl_selection": {
                    "contract": {
                        "token_stats": {
                            "samples": 8,
                            "total_sequence_tokens": 24,
                        }
                    }
                }
            }
        },
    }
    runtime_source = cli._current_source_closure()
    runtime_environment = {"device": "cpu", "precision": "bf16"}
    return dataset_report, runtime_source, runtime_environment


def _complete_validation(value: dict) -> dict:
    result = json.loads(json.dumps(value))
    cer = float(result["deployment_weighted_cer"])
    result["real"]["overall"].update(
        {"eos_rate": 1.0, "invalid_count": 0, "hit_cap_rate": 0.0}
    )
    result["real"]["buckets"] = {
        bucket: {"raw_grapheme_cer": cer}
        for bucket in (
            "print",
            "handwritten_good",
            "handwritten_medium",
            "handwritten_poor",
        )
    }
    interval = {"mean": 1.0, "low": 0.5, "high": 1.5}
    result["grounding"] = {
        "blank_cer_gap": 1.0,
        "shuffled_cer_gap": 1.0,
        "blank_first_token_nll_gap": 1.0,
        "shuffled_first_token_nll_gap": 1.0,
        "paired_bootstrap_95ci": {
            "blank_cer_gap": interval,
            "shuffled_cer_gap": interval,
            "blank_first_token_nll_gap": interval,
            "shuffled_first_token_nll_gap": interval,
        },
    }
    result["text_replay"]["target_tokens"] = 16
    return result


def _materialize_pilot(
    root: Path,
    selected_index: int,
    *,
    baseline_edits: int = 20,
    candidate_edits: int = 10,
    eligible: bool = True,
    metadata_selected_index: int | None = None,
    precision: str = "bf16",
    include_scaler: bool | None = None,
    outer_step: int = 195,
    marker_step: int | None = None,
) -> tuple[Path, dict]:
    root.mkdir(parents=True)
    baseline = cli._attach_baseline_gate(
        _complete_validation(fixture_validation(baseline_edits))
    )
    protocol = fixture_protocol()
    parent = _parent_checkpoint(root.parent)
    dataset_report, runtime_source, runtime_environment = _provenance()
    run = _run_with_parent(
        selected_index,
        baseline=baseline,
        parent=parent,
        protocol=protocol,
        precision=precision,
        dataset_report=dataset_report,
        runtime_source=runtime_source,
        runtime_environment=runtime_environment,
    )
    selection = cli._attach_selection_gate(
        _complete_validation(fixture_validation(candidate_edits)),
        baseline=baseline,
        optimizer_steps=195,
    )
    if not eligible:
        selection["eligibility"]["eligible"] = False
        selection["eligibility"]["reasons"].append("synthetic_gate_failure")
    selection_sha = canonical_json_sha256(selection)
    terminal_path = (root / "terminal").resolve()
    selection_path = terminal_path if eligible else None
    progress = _progress(
        run,
        eligible=eligible,
        selection_path=selection_path,
        selection_sha=selection_sha,
    )

    metadata_run = (
        run
        if metadata_selected_index is None
        else _run_with_parent(
            metadata_selected_index,
            baseline=baseline,
            parent=parent,
            protocol=protocol,
            precision=precision,
            dataset_report=dataset_report,
            runtime_source=runtime_source,
            runtime_environment=runtime_environment,
        )
    )
    metadata_progress = _progress(
        metadata_run,
        eligible=eligible,
        selection_path=selection_path,
        selection_sha=selection_sha,
    )
    model_bytes = b"tiny-anyres-grpo-policy-v1"
    meta_bytes = _checkpoint_bytes(
        metadata_run,
        metadata_progress,
        protocol,
        baseline=baseline,
        selection=selection,
        dataset_report=dataset_report,
        runtime_source=runtime_source,
        runtime_environment=runtime_environment,
        outer_step=outer_step,
    )
    terminal_identity = _write_checkpoint(
        terminal_path,
        model=model_bytes,
        meta=meta_bytes,
        marker_step=outer_step if marker_step is None else marker_step,
        include_scaler=(
            precision == "fp16" if include_scaler is None else include_scaler
        ),
    )
    selection_identity = terminal_identity if selection_path is not None else None
    result = build_kl_pilot_result(
        run_contract=run,
        progress=progress,
        terminal_checkpoint=terminal_identity,
        selection_checkpoint=selection_identity,
        baseline_validation=baseline,
        selection_validation=selection,
    )
    result_path = root / "PILOT_RESULT.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return result_path, result


class TrainOCRAnyresGRPOSelectKLTest(unittest.TestCase):
    def test_pilot_defaults_to_explicit_cuda_zero(self) -> None:
        args = cli.parse_args(
            [
                "pilot",
                "--joint-stage-result", "joint.json",
                "--tokenizer", "tokenizer",
                "--root", "data",
                "--assets", "assets.jsonl",
                "--views", "views.jsonl",
                "--train-samples", "train.jsonl",
                "--sft-validation-samples", "sft.jsonl",
                "--kl-selection-samples", "kl.jsonl",
                "--formal-monitor-samples", "formal.jsonl",
                "--preprocess-contract", "preprocess.json",
                "--text-replay", "text.jsonl",
                "--reviewed-exclusions", "exclusions.json",
                "--output-dir", "output",
                "--kl-coef", "0.04",
                "--tower-lr", "1e-5",
                "--bridge-lr", "1e-5",
                "--projector-lr", "1e-5",
                "--lm-lr", "1e-6",
                "--max-behavior-log-ratio", "2.0",
            ]
        )
        self.assertEqual(args.device, "cuda:0")

    def test_checkpoint_byte_tamper_and_missing_complete_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result_path, result = _materialize_pilot(root / "tamper", 0)
            terminal = Path(result["terminal_checkpoint"]["path"])
            (terminal / "model.pt").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                cli.select_authenticated_pilots(
                    [result_path, result_path, result_path]
                )

            result_path, result = _materialize_pilot(root / "missing", 0)
            Path(result["terminal_checkpoint"]["path"]).joinpath(
                "COMPLETE"
            ).unlink()
            with self.assertRaisesRegex(ValueError, "COMPLETE.*missing"):
                cli.select_authenticated_pilots(
                    [result_path, result_path, result_path]
                )

    def test_embedded_metadata_run_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result_path, _ = _materialize_pilot(
                Path(temporary) / "mismatch",
                0,
                metadata_selected_index=1,
            )
            with self.assertRaisesRegex(ValueError, "run differs from PILOT_RESULT"):
                cli.select_authenticated_pilots(
                    [result_path, result_path, result_path]
                )

    def test_terminal_resume_members_marker_and_outer_step_are_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result_path, result = _materialize_pilot(
                root / "marker",
                0,
                marker_step=194,
            )
            with self.assertRaisesRegex(ValueError, "COMPLETE differs"):
                cli.select_authenticated_pilots(
                    [result_path, result_path, result_path]
                )

            result_path, result = _materialize_pilot(root / "optimizer", 0)
            Path(result["terminal_checkpoint"]["path"]).joinpath(
                "optimizer.pt"
            ).unlink()
            with self.assertRaisesRegex(ValueError, "optimizer.pt.*missing"):
                cli.select_authenticated_pilots(
                    [result_path, result_path, result_path]
                )

            result_path, _ = _materialize_pilot(
                root / "outer-step",
                0,
                outer_step=196,
                marker_step=196,
            )
            with self.assertRaisesRegex(
                ValueError,
                "outer step differs from progress.optimizer_steps",
            ):
                cli.select_authenticated_pilots(
                    [result_path, result_path, result_path]
                )

    def test_fp16_terminal_requires_scaler_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result_path, _ = _materialize_pilot(
                Path(temporary) / "fp16",
                0,
                precision="fp16",
                include_scaler=False,
            )
            with self.assertRaisesRegex(ValueError, "scaler.pt.*missing"):
                cli.select_authenticated_pilots(
                    [result_path, result_path, result_path]
                )

    def test_no_candidate_writes_only_explicit_no_go(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [
                _materialize_pilot(
                    root / f"pilot-{index}",
                    index,
                    baseline_edits=10,
                    candidate_edits=10,
                )[0]
                for index in range(3)
            ]
            output = root / "selection"
            status = cli.main(
                [
                    "select-kl",
                    *(value for path in paths for value in ("--pilot-result", str(path))),
                    "--output-dir",
                    str(output),
                ]
            )
            self.assertEqual(status, 2)
            self.assertFalse((output / cli.KL_SELECTION_FILENAME).exists())
            no_go = json.loads(
                (output / cli.KL_NO_GO_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(no_go["status"], "no_candidate_admitted")
            self.assertFalse(no_go["formal_run_allowed"])

    def test_selection_is_order_independent_and_formal_restarts_from_joint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths_and_results = [
                _materialize_pilot(
                    root / "pilot-004",
                    0,
                    candidate_edits=11,
                ),
                _materialize_pilot(
                    root / "pilot-001",
                    1,
                    candidate_edits=10,
                ),
                _materialize_pilot(
                    root / "pilot-000",
                    2,
                    candidate_edits=10,
                ),
            ]
            paths = [item[0] for item in paths_and_results]
            forward = cli.select_authenticated_pilots(paths)
            reverse = cli.select_authenticated_pilots(list(reversed(paths)))
            self.assertEqual(forward, reverse)
            self.assertEqual(forward["status"], "selected")
            self.assertTrue(forward["formal_run_allowed"])
            self.assertEqual(
                forward["formal_init_checkpoint"],
                paths_and_results[0][1]["parent_joint_checkpoint"],
            )
            pilot_paths = {
                result[field]["path"]
                for _, result in paths_and_results
                for field in ("terminal_checkpoint", "selection_checkpoint")
            }
            self.assertNotIn(
                forward["formal_init_checkpoint"]["path"], pilot_paths
            )

            existing = root / "EXISTING.json"
            existing.write_text(
                json.dumps(forward, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            self.assertEqual(
                cli.select_authenticated_pilots(
                    paths,
                    existing_receipt=existing,
                ),
                forward,
            )

    def test_strict_json_rejects_duplicate_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "duplicate.json"
            source.write_text('{"kind":"a","kind":"b"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                cli._strict_json_file(source, where="fixture")


if __name__ == "__main__":
    unittest.main()
