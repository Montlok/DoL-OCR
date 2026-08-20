# -*- coding: utf-8 -*-

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from Model.posttrain.ocr_anyres_grpo_owner import (
    OWNER_NO_UPDATE_FILENAME,
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
from Model.posttrain.ocr_anyres_grpo_protocol import fixed_kl_pilot_attempt
from Model.posttrain.ocr_quota_sampler import OCRQuotaSampler
from Model.tests.test_ocr_anyres_grpo_run import _run
from Model.training.checkpoint import save_checkpoint


def _sampler() -> OCRQuotaSampler:
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
    return OCRQuotaSampler(
        buckets,
        global_batch_size=20,
        seed=811,
        world_size=1,
    )


def _stepped() -> dict:
    return {
        "stepped": True,
        "optimizer_steps": 1,
        "scheduler_steps": 1,
        "used_text_replay": True,
        "skip_reason": None,
    }


def _inactive() -> dict:
    return {
        "stepped": False,
        "optimizer_steps": 0,
        "scheduler_steps": 0,
        "used_text_replay": False,
        "skip_reason": "no_active_reward_groups",
    }


class AnyresGRPOOwnerTest(unittest.TestCase):
    def test_step_zero_and_crash_before_commit_leave_progress_unchanged(self):
        run = _run(0)
        protocol = run["pilot_protocol"]
        sampler = _sampler()
        progress = initial_pilot_progress(run, sampler=sampler)
        self.assertEqual(progress["optimizer_steps"], 0)
        self.assertEqual(progress["rollout_attempts"], 0)
        self.assertEqual(progress["text_cursor_state"]["schedule_cursor"], 0)

        attempt = begin_live_attempt(
            protocol,
            sampler=sampler,
            progress=progress,
            run_contract=run,
        )
        preview = preview_attempt_commit(
            run,
            protocol,
            sampler=sampler,
            progress=progress,
            metrics=_stepped(),
        )
        self.assertEqual(list(sampler.pending_global_batch or ()), attempt["ocr_prompt_ids"])
        self.assertEqual(progress["rollout_attempts"], 0)
        self.assertEqual(preview["rollout_attempts"], 1)
        self.assertEqual(sampler.draw_counter, 20)

    def test_no_active_commits_ocr_and_text_slot_without_ce_or_optimizer(self):
        run = _run(0)
        protocol = run["pilot_protocol"]
        sampler = _sampler()
        progress = initial_pilot_progress(run, sampler=sampler)
        begin_live_attempt(
            protocol,
            sampler=sampler,
            progress=progress,
            run_contract=run,
        )
        preview = preview_attempt_commit(
            run,
            protocol,
            sampler=sampler,
            progress=progress,
            metrics=_inactive(),
        )
        commit_live_attempt(
            sampler,
            protocol=protocol,
            previewed_progress=preview,
        )
        self.assertEqual(preview["rollout_attempts"], 1)
        self.assertEqual(preview["optimizer_steps"], 0)
        self.assertEqual(preview["no_update_count"], 1)
        self.assertEqual(preview["text_cursor_state"]["schedule_cursor"], 1)
        self.assertEqual(preview["text_cursor_state"]["ce_steps"], 0)

        restored = _sampler()
        restored.load_state_dict(preview["sampler_state"])
        next_attempt = begin_live_attempt(
            protocol,
            sampler=restored,
            progress=preview,
            run_contract=run,
        )
        self.assertEqual(next_attempt, fixed_kl_pilot_attempt(protocol, 1))

    def test_fixed_200_attempt_accounting_can_end_at_195_updates(self):
        run = _run(0)
        protocol = run["pilot_protocol"]
        validate_protocol_run_bindings(run, protocol)
        sampler = _sampler()
        progress = initial_pilot_progress(run, sampler=sampler)
        for attempt_index in range(200):
            begin_live_attempt(
                protocol,
                sampler=sampler,
                progress=progress,
                run_contract=run,
            )
            inactive = attempt_index >= 195
            progress = preview_attempt_commit(
                run,
                protocol,
                sampler=sampler,
                progress=progress,
                metrics=_inactive() if inactive else _stepped(),
                eval_count=4 if attempt_index == 199 else progress["eval_count"],
                stop_reason=(
                    "pilot_budget_complete" if attempt_index == 199 else "running"
                ),
                terminal=attempt_index == 199,
            )
            commit_live_attempt(
                sampler,
                protocol=protocol,
                previewed_progress=progress,
            )
        self.assertEqual(progress["rollout_attempts"], 200)
        self.assertEqual(progress["optimizer_steps"], 195)
        self.assertEqual(progress["no_update_count"], 5)
        self.assertEqual(progress["text_cursor_state"]["schedule_cursor"], 200)
        self.assertEqual(progress["text_cursor_state"]["ce_steps"], 195)
        self.assertTrue(progress["terminal"])

    def test_shared_no_update_journal_is_anchor_bound_and_resumable(self):
        run = _run(0)
        protocol = run["pilot_protocol"]
        sampler = _sampler()
        anchor_progress = initial_pilot_progress(run, sampler=sampler)
        begin_live_attempt(
            protocol,
            sampler=sampler,
            progress=anchor_progress,
            run_contract=run,
        )
        journal_progress = preview_attempt_commit(
            run,
            protocol,
            sampler=sampler,
            progress=anchor_progress,
            metrics=_inactive(),
        )
        commit_live_attempt(
            sampler,
            protocol=protocol,
            previewed_progress=journal_progress,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            model = nn.Linear(2, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            checkpoint = save_checkpoint(
                output,
                0,
                model,
                optimizer,
                scheduler,
                metadata={"progress": anchor_progress},
            )
            assert checkpoint is not None
            identity = checkpoint_identity(checkpoint)
            destination = save_no_update_journal(
                output,
                anchor_checkpoint=identity,
                run_contract=run,
                protocol=protocol,
                anchor_progress=anchor_progress,
                progress=journal_progress,
            )
            self.assertEqual(destination.name, OWNER_NO_UPDATE_FILENAME)
            restored = load_no_update_journal(
                output,
                anchor_checkpoint=identity,
                run_contract=run,
                protocol=protocol,
                anchor_progress=anchor_progress,
            )
            self.assertEqual(restored, journal_progress)
            clear_no_update_journal(output)
            self.assertFalse(destination.exists())

    def test_full_checkpoint_safe_restore_recovers_all_training_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            model = nn.Linear(2, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            model(torch.ones(2, 2)).sum().backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            before = {name: value.detach().clone() for name, value in model.state_dict().items()}
            checkpoint = save_checkpoint(
                output,
                0,
                model,
                optimizer,
                scheduler,
                metadata={"kind": "tiny"},
            )
            assert checkpoint is not None
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(10)
            restored_path, envelope, _ = restore_full_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
            )
            self.assertEqual(restored_path, checkpoint.resolve())
            self.assertEqual(envelope["step"], 0)
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, before[name]))


if __name__ == "__main__":
    unittest.main()
