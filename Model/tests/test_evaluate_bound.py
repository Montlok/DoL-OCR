# -*- coding: utf-8 -*-

"""Regression: ``evaluate`` must bound consumed batches by ``max_batches``
even when every batch has only ignore-index targets (M3 / Codex follow-up)."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from Model.config import IGNORE_INDEX, TrainingConfig
from Model.training.loop import evaluate


class _ConstLossModel(nn.Module):
    def forward(self, **kwargs):  # noqa: ANN003 - test stub
        return {"loss": torch.tensor(1.0)}


class _PartsModel(nn.Module):
    """Returns a combined loss that differs from its forward part, plus a cfg
    carrying a custom ignore_index, to exercise the eval normalization."""

    def __init__(self, ignore_index: int) -> None:
        super().__init__()
        self.cfg = type("Cfg", (), {"ignore_index": ignore_index})()

    def forward(self, **kwargs):  # noqa: ANN003 - test stub
        return {
            "loss": torch.tensor(9.0),  # combined (forward + reverse) — ignored
            "loss_parts": {"forward": 2.0, "reverse": 7.0},
        }


def _all_ignore_batches():
    """Infinite generator of batches whose shifted labels are all ignore."""
    while True:
        labels = torch.full((2, 5), IGNORE_INDEX, dtype=torch.long)
        yield {
            "input_ids": torch.zeros((2, 5), dtype=torch.long),
            "attention_mask": torch.ones((2, 5), dtype=torch.long),
            "labels": labels,
        }


class EvaluateBoundTest(unittest.TestCase):
    def test_all_ignore_batches_do_not_spin_forever(self) -> None:
        cfg = TrainingConfig(use_loss_chunking=False)
        # If the consumed-batch counter were only incremented on non-empty
        # batches, this call would never terminate on the infinite generator.
        out = evaluate(
            _ConstLossModel(),
            _all_ignore_batches(),
            cfg,
            device=torch.device("cpu"),
            max_batches=8,
        )
        self.assertEqual(out["eval_tokens"], 0.0)
        self.assertEqual(out["eval_loss"], 0.0)

    def test_uses_forward_part_not_combined_loss(self) -> None:
        cfg = TrainingConfig(use_loss_chunking=False)
        labels = torch.arange(2 * 5).reshape(2, 5)  # all valid (no ignore)
        batch = {
            "input_ids": torch.zeros((2, 5), dtype=torch.long),
            "attention_mask": torch.ones((2, 5), dtype=torch.long),
            "labels": labels,
        }
        out = evaluate(
            _PartsModel(ignore_index=IGNORE_INDEX),
            iter([batch]),
            cfg,
            device=torch.device("cpu"),
            max_batches=4,
        )
        # eval_loss must equal the forward part (2.0), NOT the combined 9.0.
        self.assertAlmostEqual(out["eval_loss"], 2.0)

    def test_honors_model_custom_ignore_index(self) -> None:
        cfg = TrainingConfig(use_loss_chunking=False)
        custom_ignore = 7
        # Forward targets are labels[:, 1:]; set them all to the custom ignore
        # so the batch is treated as having zero valid targets and is skipped.
        labels = torch.full((2, 5), custom_ignore, dtype=torch.long)
        batch = {
            "input_ids": torch.zeros((2, 5), dtype=torch.long),
            "attention_mask": torch.ones((2, 5), dtype=torch.long),
            "labels": labels,
        }
        out = evaluate(
            _PartsModel(ignore_index=custom_ignore),
            iter([batch]),
            cfg,
            device=torch.device("cpu"),
            max_batches=4,
        )
        self.assertEqual(out["eval_tokens"], 0.0)


if __name__ == "__main__":
    unittest.main()
