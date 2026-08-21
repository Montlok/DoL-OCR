# -*- coding: utf-8 -*-

"""Unit tests for fp16 GradScaler checkpoint persistence/restore."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from Model.training.checkpoint import (
    load_checkpoint,
    load_checkpoint_metadata,
    resolve_checkpoint_dir,
    resume_state,
    save_checkpoint,
)
from Model.training.loop import TrainState


class _StubScaler:
    """Minimal scaler exposing the state_dict/load_state_dict contract."""

    def __init__(self, scale: float = 1024.0, growth_tracker: int = 7) -> None:
        self._state = {"scale": scale, "_growth_tracker": growth_tracker}

    def state_dict(self) -> dict:
        return dict(self._state)

    def load_state_dict(self, state: dict) -> None:
        self._state = dict(state)


class CheckpointScalerStateTest(unittest.TestCase):
    def _make_artifacts(self):
        model = nn.Linear(4, 4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return model, optimizer, scheduler

    def test_scaler_state_roundtrips_through_checkpoint(self) -> None:
        model, optimizer, scheduler = self._make_artifacts()
        scaler = _StubScaler(scale=2048.0, growth_tracker=3)
        with tempfile.TemporaryDirectory() as tmp:
            save_checkpoint(
                tmp, 5, model, optimizer, scheduler, scaler=scaler, keep_last_n=0
            )
            self.assertTrue((Path(tmp) / "latest" / "scaler.pt").exists())

            payload = load_checkpoint(Path(tmp) / "latest")
            self.assertEqual(payload.scaler_state, {"scale": 2048.0, "_growth_tracker": 3})

            # resume_state must stash the scaler state into TrainState.extra so the
            # lazily created scaler in train_one_step can restore its dynamic scale.
            state = TrainState()
            resume_state(Path(tmp) / "latest", model, optimizer, scheduler, state=state)
            self.assertEqual(
                state.extra["grad_scaler_state"],
                {"scale": 2048.0, "_growth_tracker": 3},
            )

            # A fresh scaler restoring the stashed state recovers the exact scale.
            fresh = _StubScaler()
            fresh.load_state_dict(state.extra["grad_scaler_state"])
            self.assertEqual(fresh.state_dict()["scale"], 2048.0)

    def test_missing_scaler_is_none_and_not_stashed(self) -> None:
        model, optimizer, scheduler = self._make_artifacts()
        with tempfile.TemporaryDirectory() as tmp:
            save_checkpoint(tmp, 1, model, optimizer, scheduler, keep_last_n=0)
            self.assertFalse((Path(tmp) / "latest" / "scaler.pt").exists())

            payload = load_checkpoint(Path(tmp) / "latest")
            self.assertIsNone(payload.scaler_state)

            state = TrainState()
            resume_state(Path(tmp) / "latest", model, optimizer, scheduler, state=state)
            self.assertNotIn("grad_scaler_state", state.extra)

    def test_missing_checkpoint_path_does_not_fallback_to_sibling_latest(self) -> None:
        model, optimizer, scheduler = self._make_artifacts()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_checkpoint(root, 1, model, optimizer, scheduler, keep_last_n=0)
            with self.assertRaises(FileNotFoundError):
                load_checkpoint(root / "typo")

    def test_run_directory_loads_latest(self) -> None:
        model, optimizer, scheduler = self._make_artifacts()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_checkpoint(root, 3, model, optimizer, scheduler, keep_last_n=0)
            payload = load_checkpoint(root)
            self.assertEqual(payload.step, 3)

    def test_metadata_and_step_dir_resolve_without_loading_optimizer(self) -> None:
        model, optimizer, scheduler = self._make_artifacts()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_checkpoint(
                root,
                7,
                model,
                optimizer,
                scheduler,
                metadata={"rdt_config": {"recurrent_steps": 4}},
            )
            self.assertEqual(resolve_checkpoint_dir(root).name, "step_00000007")
            self.assertEqual(
                load_checkpoint_metadata(root),
                {"rdt_config": {"recurrent_steps": 4}},
            )

    def test_atomic_save_is_complete_and_same_step_can_be_replaced(self) -> None:
        model, optimizer, scheduler = self._make_artifacts()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_checkpoint(root, 2, model, optimizer, scheduler, metadata={"v": 1})
            step = root / "step_00000002"
            self.assertTrue((step / "COMPLETE").is_file())
            save_checkpoint(root, 2, model, optimizer, scheduler, metadata={"v": 2})
            self.assertEqual(load_checkpoint_metadata(step), {"v": 2})
            self.assertFalse((root / ".step_00000002.backup").exists())
            self.assertFalse(any(root.glob(".step_00000002.tmp-*")))

    def test_broken_latest_recovers_complete_pre_swap_backup(self) -> None:
        model, optimizer, scheduler = self._make_artifacts()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_checkpoint(root, 4, model, optimizer, scheduler)
            step = root / "step_00000004"
            backup = root / ".step_00000004.backup"
            step.rename(backup)
            resolved = resolve_checkpoint_dir(root)
            self.assertEqual(resolved, backup)
            self.assertEqual(load_checkpoint(root).step, 4)

    def test_broken_latest_falls_back_to_newest_complete_step(self) -> None:
        model, optimizer, scheduler = self._make_artifacts()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_checkpoint(root, 3, model, optimizer, scheduler, keep_last_n=0)
            save_checkpoint(root, 4, model, optimizer, scheduler, keep_last_n=0)
            latest = root / "latest"
            latest.unlink()
            latest.symlink_to("step_99999999")
            self.assertEqual(
                resolve_checkpoint_dir(latest),
                root / "step_00000004",
            )


if __name__ == "__main__":
    unittest.main()
