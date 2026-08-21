# -*- coding: utf-8 -*-

"""Unit tests for checkpoint RNG capture/restore (Python + NumPy + torch)."""

from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from Model.training.checkpoint import (
    _restore_rng,
    _rng_state,
    clear_no_update_progress,
    load_no_update_progress,
    restore_rng_state,
    save_checkpoint,
    save_no_update_progress,
)


class CheckpointRngStateTest(unittest.TestCase):
    def test_python_and_torch_rng_roundtrip(self) -> None:
        random.seed(1234)
        torch.manual_seed(1234)
        # Capture state, then draw a reference sequence.
        state = _rng_state()
        ref_py = [random.random() for _ in range(5)]
        ref_torch = torch.randint(0, 1_000_000, (5,)).tolist()

        # Advance the generators so they diverge from the captured state.
        for _ in range(10):
            random.random()
            torch.randint(0, 1_000_000, (3,))

        # Restoring must reproduce the exact reference sequence.
        _restore_rng(state)
        self.assertEqual([random.random() for _ in range(5)], ref_py)
        self.assertEqual(torch.randint(0, 1_000_000, (5,)).tolist(), ref_torch)

    def test_numpy_rng_roundtrip(self) -> None:
        try:
            import numpy as np
        except ImportError:  # pragma: no cover
            self.skipTest("numpy not installed")

        np.random.seed(99)
        self.assertIn("numpy_safe_v1", _rng_state())
        state = _rng_state()
        ref = np.random.rand(5).tolist()
        np.random.rand(20)  # advance
        _restore_rng(state)
        self.assertEqual(np.random.rand(5).tolist(), ref)

    def test_no_update_journal_restores_cursor_and_rng(self) -> None:
        model = nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        contract = "a" * 64
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = save_checkpoint(
                root,
                3,
                model,
                optimizer,
                scheduler,
                metadata={"phase": "grpo"},
            )
            assert checkpoint is not None
            random.seed(321)
            torch.manual_seed(321)
            save_no_update_progress(
                checkpoint,
                contract_sha256=contract,
                state={"batches_consumed": 17, "degenerate_streak": 4},
            )
            expected_python = random.random()
            expected_torch = torch.randint(0, 1_000_000, (4,)).tolist()
            random.random()
            torch.randint(0, 1_000_000, (4,))

            payload = load_no_update_progress(
                checkpoint,
                contract_sha256=contract,
            )
            assert payload is not None
            self.assertEqual(payload["state"]["batches_consumed"], 17)
            restore_rng_state(payload["rng_state"])
            self.assertEqual(random.random(), expected_python)
            self.assertEqual(
                torch.randint(0, 1_000_000, (4,)).tolist(),
                expected_torch,
            )

            clear_no_update_progress(checkpoint)
            self.assertIsNone(
                load_no_update_progress(
                    checkpoint,
                    contract_sha256=contract,
                )
            )

    def test_no_update_journal_rejects_same_step_anchor_replacement(self) -> None:
        model = nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        contract = "b" * 64
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = save_checkpoint(
                root,
                5,
                model,
                optimizer,
                scheduler,
                metadata={"generation": 1},
            )
            assert checkpoint is not None
            save_no_update_progress(
                checkpoint,
                contract_sha256=contract,
                state={"batches_consumed": 1},
            )
            # Re-publishing the same numbered step creates a different atomic
            # checkpoint generation. Its old cursor must never be applied.
            save_checkpoint(
                root,
                5,
                model,
                optimizer,
                scheduler,
                metadata={"generation": 2},
            )
            with self.assertRaisesRegex(ValueError, "different checkpoint"):
                load_no_update_progress(
                    root / "step_00000005",
                    contract_sha256=contract,
                )


if __name__ == "__main__":
    unittest.main()
