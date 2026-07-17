# -*- coding: utf-8 -*-

"""Depth-contract tests for generative OCR evaluation."""

from __future__ import annotations

import contextlib
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from Model.config import OMVTConfig
from Model.training import save_checkpoint
from scripts.eval_vlm_ocr import _decode_batches, _restore_omvt_geometry


class _RecordingGenerator:
    def __init__(self, trained_depth: int) -> None:
        self.cfg = SimpleNamespace(recurrent_steps=trained_depth)
        self.seen_depths: list[int | None] = []

    def generate(self, ids: torch.Tensor, **kwargs) -> torch.Tensor:
        self.seen_depths.append(kwargs.get("recurrent_steps"))
        continuation = torch.full(
            (ids.shape[0], 1), 300, dtype=ids.dtype, device=ids.device
        )
        return torch.cat((ids, continuation), dim=1)


class EvalVlmDepthContractTest(unittest.TestCase):
    def _args(self, override=None):
        return SimpleNamespace(
            batch_size=2,
            max_new_tokens=1,
            repetition_penalty=1.0,
            recurrent_steps=override,
        )

    def test_checkpoint_depth_is_passed_explicitly(self) -> None:
        model = _RecordingGenerator(trained_depth=4)
        prompts = [[1, 9, 2], [1, 9, 2]]
        preds = _decode_batches(
            model,
            prompts,
            lambda _start, _end: {},
            self._args(),
            torch.device("cpu"),
            contextlib.nullcontext,
        )
        self.assertEqual(model.seen_depths, [4])
        self.assertEqual(preds, [[300], [300]])

    def test_explicit_depth_override_wins(self) -> None:
        model = _RecordingGenerator(trained_depth=4)
        _decode_batches(
            model,
            [[1, 9, 2]],
            lambda _start, _end: {},
            self._args(override=8),
            torch.device("cpu"),
            contextlib.nullcontext,
        )
        self.assertEqual(model.seen_depths, [8])

    def test_checkpoint_geometry_is_applied_before_deployment_preprocessing(self) -> None:
        omvt_cfg = OMVTConfig(
            image_size=16,
            vertical_patch=(8, 4),
            horizontal_patch=(4, 8),
            square_patch=(4, 4),
            layout_patch=(16, 16),
            d_vision=32,
            vision_n_heads=4,
            vision_ffn_hidden=64,
            compress_to=2,
            compressor_layers=1,
            compressor_heads=4,
            n_vertical_layers=1,
            n_horizontal_layers=1,
            n_local_attn_layers=1,
            n_layout_layers=1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_checkpoint(
                root,
                1,
                nn.Linear(1, 1),
                None,
                None,
                metadata={"omvt_config": asdict(omvt_cfg)},
            )
            args = SimpleNamespace(
                checkpoint=str(root), image_size=224, n_image_tokens=256
            )
            restored = _restore_omvt_geometry(args)
            self.assertIsNotNone(restored)
            self.assertEqual(args.image_size, 16)
            self.assertEqual(args.n_image_tokens, 2)


if __name__ == "__main__":
    unittest.main()
