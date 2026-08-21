# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts import train_ocr_anyres_grpo as cli


def _args(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        dist="single",
        resume="",
        pilot_result=["a.json", "b.json", "c.json"],
        kl_selection_receipt="selection.json",
        joint_stage_result="joint.json",
        locked_golden_anchor_sha256="d" * 64,
        tokenizer="tokenizer",
        root="data",
        assets="assets.jsonl",
        views="views.jsonl",
        train_samples="train.jsonl",
        sft_validation_samples="sft.jsonl",
        kl_selection_samples="kl.jsonl",
        formal_monitor_samples="formal.jsonl",
        preprocess_contract="preprocess.json",
        text_replay="text.jsonl",
        reviewed_exclusions="exclusions.json",
        output_dir=str(root / "formal-output"),
        max_rollout_attempts=20,
        max_optimizer_steps=15,
        eval_every=5,
        save_every=5,
        early_stop_patience=2,
        global_batch_size=20,
        text_batch_size=4,
        group_size=4,
        seed=20260821,
        precision="bf16",
        device="cuda:0",
        tower_lr=1e-5,
        bridge_lr=1e-5,
        projector_lr=1e-5,
        lm_lr=1e-6,
        weight_decay=0.01,
        warmup_steps=0,
        grad_clip=1.0,
        text_weight=0.2,
        loss_chunk_size=4096,
        max_behavior_log_ratio=2.0,
        max_consecutive_no_update=20,
        max_consecutive_optimizer_skips=5,
        keep_last=2,
    )


class FormalCLIControlTest(unittest.TestCase):
    def test_receipt_bypass_stops_before_output_or_parent_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = _args(Path(temporary))
            with mock.patch.object(
                cli,
                "select_authenticated_pilots",
                side_effect=ValueError("receipt recomputation mismatch"),
            ) as authenticate, mock.patch.object(
                cli,
                "admit_joint_policy_for_grpo",
            ) as admit:
                with self.assertRaisesRegex(ValueError, "recomputation mismatch"):
                    cli._run_formal(args)
            authenticate.assert_called_once_with(
                args.pilot_result,
                existing_receipt=args.kl_selection_receipt,
            )
            admit.assert_not_called()
            self.assertFalse(Path(args.output_dir).exists())

    def test_first_best_then_equal_or_ineligible_accumulates_patience(self):
        first = {
            "deployment_weighted_cer": 0.1,
            "eligibility": {"eligible": True},
        }
        decision = cli._formal_validation_decision(
            first,
            best_validation=None,
            optimizer_steps=1,
            bad_eval_count=0,
        )
        self.assertEqual(decision, {"new_best": True, "bad_eval_count": 0})
        equal = cli._formal_validation_decision(
            first,
            best_validation=first,
            optimizer_steps=2,
            bad_eval_count=0,
        )
        self.assertEqual(equal, {"new_best": False, "bad_eval_count": 1})
        zero = {
            "deployment_weighted_cer": 0.0,
            "eligibility": {"eligible": True},
        }
        equal_zero = cli._formal_validation_decision(
            zero,
            best_validation=zero,
            optimizer_steps=3,
            bad_eval_count=1,
        )
        self.assertEqual(equal_zero["bad_eval_count"], 2)
        ineligible = {
            "deployment_weighted_cer": 0.01,
            "eligibility": {"eligible": False},
        }
        rejected = cli._formal_validation_decision(
            ineligible,
            best_validation=first,
            optimizer_steps=4,
            bad_eval_count=1,
        )
        self.assertEqual(rejected, {"new_best": False, "bad_eval_count": 2})

    def test_no_best_is_explicit_no_go_and_anchor_is_never_opened(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            run = {"canonical_sha256": "a" * 64}
            progress = {
                "best_validation_eligible": False,
                "canonical_sha256": "b" * 64,
                "stop_reason": "max_rollout_attempts",
            }
            with mock.patch.object(Path, "open", side_effect=AssertionError("opened")):
                selected = cli._finish_formal(
                    output,
                    run=run,
                    progress=progress,
                    terminal_checkpoint=output / "step_00000000",
                    best_validation=None,
                    locked_golden_anchor_sha256="c" * 64,
                )
            self.assertFalse(selected)
            payload = json.loads(
                (output / "FORMAL_NO_GO.json").read_text(encoding="utf-8")
            )
            self.assertFalse(payload["selection_receipt_written"])
            self.assertFalse(
                (output / "best" / "SELECTION_FINALIZED.json").exists()
            )

    def test_stop_reason_priority_is_deterministic(self):
        self.assertEqual(
            cli._formal_stop_reason(
                optimizer_steps=10,
                rollout_attempts=10,
                max_optimizer_steps=10,
                max_rollout_attempts=10,
                plateau=False,
            ),
            "completed",
        )
        self.assertEqual(
            cli._formal_stop_reason(
                optimizer_steps=4,
                rollout_attempts=7,
                max_optimizer_steps=10,
                max_rollout_attempts=20,
                plateau=True,
            ),
            "validation_plateau",
        )


if __name__ == "__main__":
    unittest.main()
