# -*- coding: utf-8 -*-

from __future__ import annotations

import copy
import unittest
from pathlib import Path
from functools import lru_cache

from Model.ocr.tokenization import canonical_json_sha256
from Model.posttrain.ocr_anyres_grpo import AnyresGRPOAdmission
from Model.posttrain.ocr_anyres_grpo_protocol import (
    build_fixed_kl_pilot_protocol,
)
from Model.posttrain.ocr_anyres_grpo_formal_protocol import (
    build_formal_grpo_protocol,
)
from Model.posttrain.ocr_anyres_grpo_run import (
    build_anyres_grpo_progress,
    build_anyres_grpo_run_contract,
    build_kl_pilot_result,
    build_text_schedule_cursor_state,
    select_kl_pilot_results,
    validate_anyres_grpo_progress,
    validate_anyres_grpo_run_contract,
    validate_kl_pilot_result,
    validate_kl_selection_receipt,
)
from Model.posttrain.ocr_anyres_grpo_trainer import AnyresGRPOKLAblationTrial
from Model.posttrain.ocr_quota_sampler import OCRQuotaSampler


_BUCKET_WEIGHTS = {
    "print": 0.6,
    "handwritten_good": 0.1,
    "handwritten_medium": 0.2,
    "handwritten_poor": 0.1,
}


def _checkpoint(path: str, seed: str = "1") -> dict[str, str]:
    return {
        "path": str(Path(path).resolve()),
        "model_sha256": seed * 64,
        "metadata_sha256": chr(ord(seed) + 1) * 64,
    }


def _admission() -> AnyresGRPOAdmission:
    return AnyresGRPOAdmission(
        policy_checkpoint_sha256="1" * 64,
        policy_metadata_sha256="2" * 64,
        reference_checkpoint_sha256="1" * 64,
        reference_metadata_sha256="2" * 64,
        joint_stage_result_sha256="3" * 64,
        tokenizer_contract_sha256="4" * 64,
        visual_contract_sha256="5" * 64,
        preprocess_contract_sha256="6" * 64,
        native_migration_receipt_sha256="7" * 64,
        trainability_contract_sha256="8" * 64,
        trainable_parameter_names_sha256="9" * 64,
        reward_contract_sha256="a" * 64,
        dataset_admission_report_sha256="b" * 64,
        train_dataset_contract_sha256="c" * 64,
        sft_validation_dataset_contract_sha256="d" * 64,
        kl_selection_dataset_contract_sha256="e" * 64,
        formal_monitor_dataset_contract_sha256="f" * 64,
        text_replay_train_contract_sha256="0" * 64,
        text_replay_sft_validation_contract_sha256="1" * 64,
        text_replay_kl_selection_contract_sha256="2" * 64,
        text_replay_formal_monitor_contract_sha256="3" * 64,
        runtime_source_receipt_sha256="0" * 64,
        recommended_max_new_tokens=1,
        minimum_reward_spread=0.005,
    )


@lru_cache(maxsize=1)
def _cached_pilot_protocol() -> dict[str, object]:
    sample_buckets = {
        f"{bucket}-{index:03d}": bucket
        for bucket in _BUCKET_WEIGHTS
        for index in range(20)
    }
    return build_fixed_kl_pilot_protocol(
        sampler=OCRQuotaSampler(
            sample_buckets,
            global_batch_size=20,
            seed=811,
            world_size=1,
        ),
        ocr_dataset_contract_sha256="c" * 64,
        text_sample_ids=tuple(f"text-{index:03d}" for index in range(11)),
        text_dataset_contract_sha256="0" * 64,
        text_batch_size=4,
        base_seed=42,
    )


def _pilot_protocol() -> dict[str, object]:
    return copy.deepcopy(_cached_pilot_protocol())


@lru_cache(maxsize=1)
def _cached_formal_protocol() -> dict[str, object]:
    sample_buckets = {
        f"{bucket}-{index:03d}": bucket
        for bucket in _BUCKET_WEIGHTS
        for index in range(20)
    }
    return build_formal_grpo_protocol(
        sampler=OCRQuotaSampler(
            sample_buckets,
            global_batch_size=20,
            seed=811,
            world_size=1,
        ),
        ocr_dataset_contract_sha256="c" * 64,
        text_sample_ids=tuple(f"text-{index:03d}" for index in range(11)),
        text_dataset_contract_sha256="0" * 64,
        text_batch_size=4,
        base_seed=42,
        max_rollout_attempts=200,
        max_optimizer_steps=200,
        eval_every=50,
        save_every=50,
        early_stop_patience=5,
    )


def _formal_protocol() -> dict[str, object]:
    return copy.deepcopy(_cached_formal_protocol())


def _schedule(
    pilot_protocol: dict[str, object] | None = None,
) -> dict[str, object]:
    protocol_payload = None if pilot_protocol is None else pilot_protocol["payload"]
    return {
        "max_optimizer_steps": 200,
        "max_rollout_attempts": 200,
        "global_batch_size": 20,
        "text_batch_size": 4,
        "eval_every": 50,
        "save_every": 50,
        "early_stop_patience": 5,
        "text_weight": 0.2,
        "grad_clip": 1.0,
        "loss_chunk_size": 4096,
        "weight_decay": 0.01,
        "warmup_steps": 0,
        "scheduler_contract_sha256": "1" * 64,
        "world_size": 1,
        "dist_mode": "single",
        "max_consecutive_no_update": 20,
        "max_consecutive_optimizer_skips": 5,
        "ocr_prompt_schedule_sha256": (
            "2" * 64
            if protocol_payload is None
            else protocol_payload["ocr_prompt_schedule_sha256"]
        ),
        "text_batch_schedule_sha256": (
            "3" * 64
            if protocol_payload is None
            else protocol_payload["text_batch_schedule_sha256"]
        ),
        "rollout_seed_schedule_sha256": (
            "4" * 64
            if protocol_payload is None
            else protocol_payload["rollout_seed_schedule_sha256"]
        ),
    }


def _run(
    selected_index: int,
    *,
    mode: str = "pilot",
    baseline_validation: dict | None = None,
    formal_selection_receipt: dict | None = None,
    dataset_ready_admission_sha256: str = "b" * 64,
) -> dict:
    trial = AnyresGRPOKLAblationTrial(selected_index=selected_index)
    baseline = _validation(20) if baseline_validation is None else baseline_validation
    protocol = _pilot_protocol() if mode == "pilot" else None
    formal_protocol = _formal_protocol() if mode == "formal" else None
    if mode == "formal" and formal_selection_receipt is None:
        formal_selection_receipt = _formal_receipt()
    return build_anyres_grpo_run_contract(
        mode=mode,
        parent_joint_checkpoint=_checkpoint("/tmp/joint-best"),
        parent_joint_stage_result_sha256="3" * 64,
        admission=_admission(),
        dataset_ready_admission_sha256=dataset_ready_admission_sha256,
        reward_contract_sha256="a" * 64,
        source_closure_sha256="0" * 64,
        runtime_environment_sha256="1" * 64,
        baseline_validation_sha256=canonical_json_sha256(baseline),
        pilot_protocol=protocol,
        formal_protocol=formal_protocol,
        kl_trial=trial,
        formal_selection_receipt=formal_selection_receipt,
        grpo_config={
            "kl_coef": trial.selected_kl_coef,
            "max_new_tokens": 1,
            "min_reward_spread": 0.005,
            "clip_eps": None,
            "advantage_mode": "centered",
        },
        optimizer_contract_sha256="8" * 64,
        schedule=_schedule(protocol if protocol is not None else formal_protocol),
        seed=42,
        precision="bf16",
    )


def _validation(
    edits: int,
    *,
    eligible: bool = True,
    text_nll: float = 1.0,
) -> dict:
    records = sorted(
        (
            {
                "sample_id": f"{bucket}-{index}",
                "bucket": bucket,
                "grapheme_edits": edits,
                "reference_graphemes": 100,
            }
            for bucket in _BUCKET_WEIGHTS
            for index in range(2)
        ),
        key=lambda row: row["sample_id"],
    )
    cer = edits / 100
    return {
        "image_dataset_contract_sha256": "e" * 64,
        "deployment_weighted_cer": cer,
        "deployment_weights": dict(_BUCKET_WEIGHTS),
        "worst_bucket": {"raw_grapheme_cer": cer},
        "text_replay": {
            "dataset_contract_sha256": "2" * 64,
            "token_nll": text_nll,
        },
        "real": {"overall": {"raw_line_exact": max(0.0, 1.0 - cer)}},
        "eligibility": {
            "eligible": eligible,
            "reasons": [] if eligible else ["synthetic_gate_failure"],
        },
        "selection_records": records,
    }


def _pilot_result(
    selected_index: int,
    root: str,
    *,
    baseline_edits: int,
    candidate_edits: int,
    hard_eligible: bool = True,
) -> dict:
    baseline = _validation(baseline_edits)
    run = _run(selected_index, baseline_validation=baseline)
    selection = _validation(candidate_edits, eligible=hard_eligible)
    selection_sha = canonical_json_sha256(selection)
    optimizer_steps = 195
    terminal_path = f"{root}/step_00000195"
    progress = build_anyres_grpo_progress(
        run,
        optimizer_steps=optimizer_steps,
        rollout_attempts=200,
        no_update_count=5,
        consecutive_no_update=0,
        optimizer_skip_count=0,
        consecutive_optimizer_skips=0,
        sampler_state={
            "draw_counter": 200 * 20,
            "pending_global_batch": None,
        },
        text_cursor_state=build_text_schedule_cursor_state(
            dataset_contract_sha256=run["text_replay_train_contract_sha256"],
            batch_size=4,
            schedule_cursor=200,
            ce_steps=optimizer_steps,
        ),
        text_ce_steps=optimizer_steps,
        eval_count=4,
        bad_eval_count=0,
        best_validation_eligible=hard_eligible,
        best_validation_step=optimizer_steps if hard_eligible else 0,
        best_checkpoint=terminal_path if hard_eligible else None,
        best_validation_sha256=selection_sha if hard_eligible else None,
        last_validation_sha256=selection_sha,
        stop_reason="pilot_budget_complete",
        terminal=True,
    )
    return build_kl_pilot_result(
        run_contract=run,
        progress=progress,
        terminal_checkpoint=_checkpoint(terminal_path, "4"),
        selection_checkpoint=(
            _checkpoint(terminal_path, "4")
            if hard_eligible
            else None
        ),
        baseline_validation=baseline,
        selection_validation=selection,
    )


@lru_cache(maxsize=1)
def _cached_formal_receipt() -> dict:
    return select_kl_pilot_results(
        [
            _pilot_result(
                0,
                "/tmp/formal-receipt-004",
                baseline_edits=20,
                candidate_edits=11,
            ),
            _pilot_result(
                1,
                "/tmp/formal-receipt-001",
                baseline_edits=20,
                candidate_edits=10,
            ),
            _pilot_result(
                2,
                "/tmp/formal-receipt-000",
                baseline_edits=20,
                candidate_edits=10,
            ),
        ]
    )


def _formal_receipt() -> dict:
    return copy.deepcopy(_cached_formal_receipt())


class AnyresGRPORunContractTest(unittest.TestCase):
    def test_run_and_progress_contracts_reject_cursor_or_accounting_drift(self) -> None:
        run = _run(0)
        self.assertEqual(validate_anyres_grpo_run_contract(run), run)
        progress = build_anyres_grpo_progress(
            run,
            optimizer_steps=1,
            rollout_attempts=2,
            no_update_count=1,
            consecutive_no_update=1,
            optimizer_skip_count=0,
            consecutive_optimizer_skips=0,
            sampler_state={"draw_counter": 40, "pending_global_batch": None},
            text_cursor_state=build_text_schedule_cursor_state(
                dataset_contract_sha256=run["text_replay_train_contract_sha256"],
                batch_size=4,
                schedule_cursor=2,
                ce_steps=1,
            ),
            text_ce_steps=1,
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
        self.assertEqual(
            validate_anyres_grpo_progress(progress, run_contract=run), progress
        )
        bad = copy.deepcopy(progress)
        bad["sampler_state"]["draw_counter"] = 20
        bad.pop("canonical_sha256")
        bad["canonical_sha256"] = canonical_json_sha256(bad)
        with self.assertRaisesRegex(ValueError, "draw_counter"):
            validate_anyres_grpo_progress(bad, run_contract=run)
        with self.assertRaisesRegex(ValueError, "dataset report differs"):
            _run(0, dataset_ready_admission_sha256="a" * 64)

    def test_formal_run_accepts_no_pilot_comparison_hash(self) -> None:
        run = _run(1, mode="formal")
        self.assertIsNone(run["trial_comparison_contract_sha256"])
        self.assertEqual(
            run["formal_selection_receipt_sha256"],
            run["formal_selection_receipt"]["canonical_sha256"],
        )
        self.assertEqual(validate_anyres_grpo_run_contract(run), run)

    def test_formal_run_rejects_no_go_or_another_selected_kl(self) -> None:
        receipt = _formal_receipt()
        with self.assertRaisesRegex(ValueError, "selected KL"):
            _run(
                0,
                mode="formal",
                formal_selection_receipt=receipt,
            )
        no_go = select_kl_pilot_results(
            [
                _pilot_result(
                    index,
                    f"/tmp/formal-no-go-{index}",
                    baseline_edits=10,
                    candidate_edits=10,
                )
                for index in range(3)
            ]
        )
        with self.assertRaisesRegex(ValueError, "does not authorize"):
            _run(
                1,
                mode="formal",
                formal_selection_receipt=no_go,
            )

    def test_paired_selector_restarts_from_joint_and_prefers_conservative_kl(self) -> None:
        results = [
            _pilot_result(
                0,
                "/tmp/trial-004",
                baseline_edits=20,
                candidate_edits=11,
            ),
            _pilot_result(
                1,
                "/tmp/trial-001",
                baseline_edits=20,
                candidate_edits=10,
            ),
            _pilot_result(
                2,
                "/tmp/trial-000",
                baseline_edits=20,
                candidate_edits=10,
            ),
        ]
        receipt = select_kl_pilot_results(results)
        self.assertEqual(receipt["status"], "selected")
        self.assertTrue(receipt["formal_run_allowed"])
        self.assertEqual(receipt["selected_kl_coef"], 0.01)
        self.assertEqual(receipt["statistically_equivalent_kl_coefs"], [0.01, 0.0])
        self.assertEqual(receipt["formal_init_checkpoint"], _checkpoint("/tmp/joint-best"))
        self.assertEqual(validate_kl_selection_receipt(receipt), receipt)
        self.assertEqual(receipt["selection_rule"]["golden_data"], "forbidden")

        tampered = copy.deepcopy(results)
        tampered[2]["parent_joint_checkpoint"] = _checkpoint("/tmp/other-joint")
        tampered[2]["canonical_sha256"] = canonical_json_sha256(
            {
                key: value
                for key, value in tampered[2].items()
                if key != "canonical_sha256"
            }
        )
        with self.assertRaisesRegex(ValueError, "parent_joint_checkpoint"):
            select_kl_pilot_results(tampered)

    def test_no_significant_candidate_is_explicit_no_go(self) -> None:
        results = [
            _pilot_result(
                index,
                f"/tmp/no-go-{index}",
                baseline_edits=10,
                candidate_edits=10,
            )
            for index in range(3)
        ]
        receipt = select_kl_pilot_results(results)
        self.assertEqual(receipt["status"], "no_candidate_admitted")
        self.assertFalse(receipt["formal_run_allowed"])
        self.assertIsNone(receipt["selected_kl_coef"])
        self.assertIsNone(receipt["formal_init_checkpoint"])
        self.assertEqual(validate_kl_selection_receipt(receipt), receipt)

    def test_selector_is_order_independent_and_hard_gate_has_priority(self) -> None:
        results = [
            _pilot_result(
                0,
                "/tmp/hard-gate-004",
                baseline_edits=20,
                candidate_edits=0,
                hard_eligible=False,
            ),
            _pilot_result(
                1,
                "/tmp/hard-gate-001",
                baseline_edits=20,
                candidate_edits=10,
            ),
            _pilot_result(
                2,
                "/tmp/hard-gate-000",
                baseline_edits=20,
                candidate_edits=10,
            ),
        ]
        receipt = select_kl_pilot_results(results)
        self.assertEqual(receipt, select_kl_pilot_results(list(reversed(results))))
        self.assertEqual(receipt["selected_kl_coef"], 0.01)
        high_kl = next(
            row for row in receipt["candidate_eligibility"] if row["kl_coef"] == 0.04
        )
        self.assertFalse(high_kl["admitted"])
        self.assertIn("hard_eligibility_failed", high_kl["reasons"])

    def test_standalone_pilot_result_rechecks_attempt_accounting(self) -> None:
        result = _pilot_result(
            0,
            "/tmp/accounting",
            baseline_edits=20,
            candidate_edits=10,
        )
        tampered = copy.deepcopy(result)
        tampered["optimizer_skip_count"] = 1
        tampered["canonical_sha256"] = canonical_json_sha256(
            {
                key: value
                for key, value in tampered.items()
                if key != "canonical_sha256"
            }
        )
        with self.assertRaisesRegex(ValueError, "rollout-attempt accounting"):
            validate_kl_pilot_result(tampered)

    def test_paired_validation_identity_mismatch_fails_closed(self) -> None:
        run = _run(0)
        baseline = _validation(20)
        selection = _validation(10)
        selection["selection_records"][0]["sample_id"] = "wrong-sample"
        selection["selection_records"] = sorted(
            selection["selection_records"], key=lambda row: row["sample_id"]
        )
        selection_sha = canonical_json_sha256(selection)
        progress = build_anyres_grpo_progress(
            run,
            optimizer_steps=200,
            rollout_attempts=200,
            no_update_count=0,
            consecutive_no_update=0,
            optimizer_skip_count=0,
            consecutive_optimizer_skips=0,
            sampler_state={"draw_counter": 4_000, "pending_global_batch": None},
            text_cursor_state=build_text_schedule_cursor_state(
                dataset_contract_sha256=run["text_replay_train_contract_sha256"],
                batch_size=4,
                schedule_cursor=200,
                ce_steps=200,
            ),
            text_ce_steps=200,
            eval_count=4,
            bad_eval_count=0,
            best_validation_eligible=True,
            best_validation_step=200,
            best_checkpoint="/tmp/mismatch/step_00000200",
            best_validation_sha256=selection_sha,
            last_validation_sha256=selection_sha,
            stop_reason="pilot_budget_complete",
            terminal=True,
        )
        with self.assertRaisesRegex(ValueError, "sample identity"):
            build_kl_pilot_result(
                run_contract=run,
                progress=progress,
                terminal_checkpoint=_checkpoint("/tmp/mismatch/step_00000200", "4"),
                selection_checkpoint=_checkpoint(
                    "/tmp/mismatch/step_00000200", "4"
                ),
                baseline_validation=baseline,
                selection_validation=selection,
            )


if __name__ == "__main__":
    unittest.main()
