# -*- coding: utf-8 -*-

"""Tests for the shared OMVT checkpoint helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from Model.training.multimodal_cli import make_omvt_cfg
from Model.training.omvt_checkpoint import (
    load_omvt_payload,
    resolve_omvt_checkpoint_path,
    tower_state_from_payload,
)


class ResolveCheckpointPathTest(unittest.TestCase):
    def test_resolves_file_dir_and_latest_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            direct = root / "omvt_ssl.pt"
            torch.save({"step": 1}, direct)
            self.assertEqual(resolve_omvt_checkpoint_path(direct), direct)
            self.assertEqual(resolve_omvt_checkpoint_path(root), direct)

            run = root / "run"
            (run / "latest").mkdir(parents=True)
            latest = run / "latest" / "omvt_ssl.pt"
            torch.save({"step": 2}, latest)
            self.assertEqual(resolve_omvt_checkpoint_path(run), latest)

    def test_missing_checkpoint_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                resolve_omvt_checkpoint_path(Path(tmp) / "nope")


class TowerStateFromPayloadTest(unittest.TestCase):
    def test_ema_overlay_keeps_non_float_buffers(self) -> None:
        payload = {
            "tower": {
                "w": torch.zeros(2),
                "steps": torch.tensor(7, dtype=torch.long),
            },
            "tower_ema": {"w": torch.ones(2)},
        }
        plain = tower_state_from_payload(payload)
        self.assertTrue(torch.equal(plain["w"], torch.zeros(2)))

        ema = tower_state_from_payload(payload, use_ema=True)
        self.assertTrue(torch.equal(ema["w"], torch.ones(2)))
        self.assertEqual(int(ema["steps"]), 7)
        # The original payload must stay untouched (callers may reuse it).
        self.assertTrue(torch.equal(payload["tower"]["w"], torch.zeros(2)))

    def test_bare_state_dict_passes_through(self) -> None:
        bare = {"w": torch.zeros(1)}
        self.assertIs(tower_state_from_payload(bare), bare)

    def test_load_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "omvt_ssl.pt"
            torch.save({"step": 3, "tower": {"w": torch.zeros(1)}}, path)
            payload = load_omvt_payload(Path(tmp))
            self.assertEqual(payload["step"], 3)


class MakeOmvtCfgTest(unittest.TestCase):
    def test_derived_matches_legacy_grid(self) -> None:
        cfg = make_omvt_cfg(56, 64, 8)
        self.assertEqual(cfg.vertical_patch, (28, 14))
        self.assertEqual(cfg.horizontal_patch, (14, 28))
        self.assertEqual(cfg.square_patch, (14, 14))
        self.assertEqual(cfg.layout_patch, (56, 56))
        self.assertEqual(cfg.compress_to, 8)

    def test_prod_keeps_dataclass_geometry(self) -> None:
        cfg = make_omvt_cfg(448, 512, 256, preset="prod")
        self.assertEqual(cfg.vertical_patch, (32, 8))
        self.assertEqual(cfg.square_patch, (16, 16))
        self.assertEqual(cfg.d_vision, 512)
        self.assertEqual(cfg.compress_to, 256)

    def test_compress_to_defaults_to_patch_count(self) -> None:
        cfg = make_omvt_cfg(448, 64)
        self.assertEqual(cfg.compress_to, 256)

    def test_rejects_bad_inputs(self) -> None:
        with self.assertRaises(ValueError):
            make_omvt_cfg(0, 64)
        with self.assertRaises(ValueError):
            make_omvt_cfg(58, 64)
        with self.assertRaises(ValueError):
            make_omvt_cfg(56, 64, preset="bogus")


if __name__ == "__main__":
    unittest.main()
