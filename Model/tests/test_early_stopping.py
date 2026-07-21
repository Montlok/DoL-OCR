# -*- coding: utf-8 -*-

"""Regression tests for resumable loss-plateau stopping."""

from __future__ import annotations

import unittest

from Model.training.early_stopping import (
    EarlyStoppingConfig,
    LossPlateauStopper,
)


class EarlyStoppingConfigTest(unittest.TestCase):
    def test_invalid_values_are_rejected(self) -> None:
        invalid = (
            {"patience": -1},
            {"min_steps": -1},
            {"min_delta": -0.1},
            {"min_delta": float("nan")},
            {"smoothing": "median"},
            {"ema_alpha": 0.0},
            {"ema_alpha": 1.1},
            {"window_size": 0},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                EarlyStoppingConfig(**kwargs)

    def test_zero_patience_disables_stopping(self) -> None:
        self.assertFalse(EarlyStoppingConfig(patience=0).enabled)
        self.assertTrue(EarlyStoppingConfig(patience=1).enabled)


class LossPlateauStopperTest(unittest.TestCase):
    def test_ema_stops_after_configured_patience(self) -> None:
        stopper = LossPlateauStopper(
            EarlyStoppingConfig(patience=2, ema_alpha=1.0)
        )
        self.assertFalse(stopper.observe(2.0, 1))
        self.assertFalse(stopper.observe(1.0, 2))
        self.assertFalse(stopper.observe(1.0, 3))
        self.assertTrue(stopper.observe(1.0, 4))
        self.assertEqual(stopper.best_loss, 1.0)
        self.assertEqual(stopper.bad_steps, 2)

    def test_minimum_step_burn_in_tracks_current_level(self) -> None:
        stopper = LossPlateauStopper(
            EarlyStoppingConfig(
                patience=2,
                min_steps=3,
                ema_alpha=1.0,
            )
        )
        self.assertFalse(stopper.observe(1.0, 1))
        self.assertFalse(stopper.observe(5.0, 2))
        self.assertEqual(stopper.best_loss, 5.0)
        self.assertFalse(stopper.observe(5.0, 3))
        self.assertTrue(stopper.observe(5.0, 4))

    def test_min_delta_requires_material_improvement(self) -> None:
        stopper = LossPlateauStopper(
            EarlyStoppingConfig(
                patience=3,
                min_delta=0.1,
                ema_alpha=1.0,
            )
        )
        stopper.observe(5.0, 1)
        stopper.observe(4.95, 2)
        self.assertEqual(stopper.bad_steps, 1)
        stopper.observe(4.8, 3)
        self.assertEqual(stopper.best_loss, 4.8)
        self.assertEqual(stopper.bad_steps, 0)

    def test_rolling_window_resumes_exactly(self) -> None:
        config = EarlyStoppingConfig(
            patience=2,
            smoothing="window",
            window_size=3,
        )
        original = LossPlateauStopper(config)
        original.observe(5.0, 1)
        original.observe(4.0, 2)
        restored = LossPlateauStopper.from_metadata(
            config,
            original.metadata_dict(),
            require_state=True,
        )
        self.assertEqual(restored.state_dict(), original.state_dict())

        for step, loss in enumerate((3.0, 3.0, 3.0), start=3):
            self.assertEqual(
                restored.observe(loss, step),
                original.observe(loss, step),
            )
            self.assertEqual(restored.state_dict(), original.state_dict())

    def test_resume_rejects_configuration_drift(self) -> None:
        original = LossPlateauStopper(EarlyStoppingConfig(patience=5))
        metadata = original.metadata_dict()
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            LossPlateauStopper.from_metadata(
                EarlyStoppingConfig(patience=6),
                metadata,
                require_state=True,
            )

    def test_enabled_resume_requires_saved_state(self) -> None:
        with self.assertRaisesRegex(ValueError, "no early-stop metadata"):
            LossPlateauStopper.from_metadata(
                EarlyStoppingConfig(patience=5),
                None,
                require_state=True,
            )
        restored = LossPlateauStopper.from_metadata(
            EarlyStoppingConfig(),
            None,
            require_state=True,
        )
        self.assertFalse(restored.enabled)

    def test_non_finite_loss_and_repeated_step_are_rejected(self) -> None:
        stopper = LossPlateauStopper(EarlyStoppingConfig(patience=2))
        with self.assertRaises(FloatingPointError):
            stopper.observe(float("nan"), 1)
        stopper.observe(1.0, 1)
        with self.assertRaisesRegex(ValueError, "steps must increase"):
            stopper.observe(1.0, 1)


if __name__ == "__main__":
    unittest.main()
