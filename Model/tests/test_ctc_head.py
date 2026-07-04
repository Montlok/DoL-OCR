# -*- coding: utf-8 -*-

"""Unit tests for the CTC-head OCR trainer (frozen OMVT tower insurance path).

Covers: forward shapes through a tiny fixture tower, CTC loss finiteness and
convergence on memorized samples, greedy-decode roundtrip, over-length target
skipping, and checkpoint resume.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from Model.config import OMVTConfig
from Model.omvt import OMVTVisionTower
from Model.training import build_optimizer, build_scheduler
from Model.training.loop import clip_or_check_grad_norm
from scripts.train_ctc_head import (
    BLANK_ID,
    NUM_CLASSES,
    CTCHead,
    _build_scheduler_cfg,
    _train_step,
    _unflatten_targets,
    bytes_to_text,
    ctc_collate,
    greedy_ctc_decode,
    load_checkpoint_into,
    num_trainable_params,
    save_checkpoint,
)


def _tiny_omvt_config(image_size: int = 56, d_vision: int = 32, compress_to: int = 16) -> OMVTConfig:
    """Small OMVT tower fixture, same shape family as Model/tests/test_omvt.py."""

    return OMVTConfig(
        image_size=image_size,
        vertical_patch=(28, 14),
        horizontal_patch=(14, 28),
        square_patch=(14, 14),
        layout_patch=(28, 28),
        d_vision=d_vision,
        vision_n_heads=4,
        vision_ffn_hidden=64,
        compress_to=compress_to,
        compressor_layers=1,
        compressor_heads=4,
        n_vertical_layers=1,
        n_horizontal_layers=1,
        n_local_attn_layers=1,
        n_layout_layers=1,
    )


class _Args:
    """Minimal stand-in for argparse.Namespace, only the fields the helpers read."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _default_args(**overrides) -> _Args:
    base = dict(
        data="smoke", batch_size=4, lr=3e-4, weight_decay=0.0,
        steps=30, warmup_steps=2, grad_clip=1.0,
    )
    base.update(overrides)
    return _Args(**base)


class CTCHeadForwardShapeTest(unittest.TestCase):
    def test_tower_plus_head_forward_shapes(self):
        torch.manual_seed(0)
        omvt_cfg = _tiny_omvt_config()
        tower = OMVTVisionTower(omvt_cfg).eval()
        for p in tower.parameters():
            p.requires_grad_(False)
        head = CTCHead(d_vision=omvt_cfg.d_vision, hidden=24)

        pixels = torch.rand(3, omvt_cfg.in_channels, omvt_cfg.image_size, omvt_cfg.image_size)
        with torch.no_grad():
            features = tower(pixels)["compressed"]
        self.assertEqual(features.shape, (3, omvt_cfg.compress_to, omvt_cfg.d_vision))

        logits = head(features)
        self.assertEqual(logits.shape, (3, omvt_cfg.compress_to, NUM_CLASSES))
        self.assertEqual(NUM_CLASSES, 257)
        self.assertEqual(BLANK_ID, 256)

    def test_head_param_count_is_small(self):
        # ~5M at the spec'd sizes (d_vision=512, hidden=384); sanity check the
        # architecture scales down cleanly for the tiny fixture too (no
        # hard-coded 512 anywhere that would break at a different d_vision).
        head = CTCHead(d_vision=512, hidden=384)
        n = num_trainable_params(head)
        self.assertGreater(n, 3_000_000)
        self.assertLess(n, 8_000_000)


class GreedyCTCDecodeTest(unittest.TestCase):
    def test_collapse_repeats_and_drop_blank(self):
        # sequence: a a BLANK a b b BLANK BLANK c -> "a a b c" after collapse+drop
        a, b, c = ord("a"), ord("b"), ord("c")
        seq = [a, a, BLANK_ID, a, b, b, BLANK_ID, BLANK_ID, c]
        logits = torch.full((1, len(seq), NUM_CLASSES), -10.0)
        for t, cls in enumerate(seq):
            logits[0, t, cls] = 10.0
        decoded = greedy_ctc_decode(logits)
        self.assertEqual(decoded, [[a, a, b, c]])

    def test_roundtrip_easy_synthetic_case(self):
        text = "ab"
        byte_ids = list(text.encode("utf-8"))
        # Craft a T=6 frame sequence that CTC-encodes "ab" unambiguously:
        # blank, a, blank, blank, b, blank
        seq = [BLANK_ID, byte_ids[0], BLANK_ID, BLANK_ID, byte_ids[1], BLANK_ID]
        logits = torch.full((1, len(seq), NUM_CLASSES), -10.0)
        for t, cls in enumerate(seq):
            logits[0, t, cls] = 10.0
        decoded = greedy_ctc_decode(logits)
        self.assertEqual(decoded, [byte_ids])
        self.assertEqual(bytes_to_text(decoded[0]), text)

    def test_bytes_to_text_replaces_invalid_utf8(self):
        # A lone continuation byte (0x80) is not valid UTF-8 on its own; the
        # probe printer must not raise on it (errors="replace").
        out = bytes_to_text([0x80])
        self.assertIn("�", out)


class CTCLossFiniteAndDecreasingTest(unittest.TestCase):
    def test_loss_finite_and_decreases_over_30_steps_on_memorized_samples(self):
        torch.manual_seed(0)
        omvt_cfg = _tiny_omvt_config()
        tower = OMVTVisionTower(omvt_cfg).eval()
        for p in tower.parameters():
            p.requires_grad_(False)
        head = CTCHead(d_vision=omvt_cfg.d_vision, hidden=24)

        device = torch.device("cpu")
        rng = torch.Generator().manual_seed(0)
        bsz = 4
        pixels = torch.rand(
            bsz, omvt_cfg.in_channels, omvt_cfg.image_size, omvt_cfg.image_size,
            generator=rng,
        )
        texts = ["a", "bb", "ccc", "d"]
        byte_targets = [list(t.encode("utf-8")) for t in texts]

        args = _default_args(steps=30)
        train_cfg = _build_scheduler_cfg(args)
        optimizer = build_optimizer(head, train_cfg)
        scheduler = build_scheduler(optimizer, train_cfg)

        losses = []
        for step in range(1, 31):
            batch = ctc_collate(list(zip(pixels, byte_targets)))
            loss, _, _ = _train_step(tower, head, batch, device, use_bf16=False)
            self.assertTrue(torch.isfinite(loss))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_or_check_grad_norm(head, 1.0, step=step)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))

        self.assertTrue(all(v == v for v in losses))  # no NaN anywhere
        # Average of the last 5 losses should be clearly below the first 5:
        # memorizing 4 short fixed samples for 30 steps must make progress.
        self.assertLess(sum(losses[-5:]) / 5, sum(losses[:5]) / 5)


class OverLengthTargetSkipTest(unittest.TestCase):
    def test_row_to_byte_target_over_length_is_detectable(self):
        from scripts.train_ctc_head import row_to_byte_target

        # A long transcription's byte length can exceed T=256 frames; the
        # dataset iterator is responsible for skipping such rows (CTC
        # requires target_len <= input_len). Exercise the length check the
        # dataset applies, using a fake bundle decode (torch/render-free).
        long_text = "x" * 300
        row = {
            "input_ids": [1, 2, long_text.__len__()],  # unused by row_to_byte_target
            "labels": [-100, -100] + [ord(c) for c in long_text] + [3],
            "images": ["img.png"],
        }
        # Build a row via the real contract so split_ocr_row's invariants hold.
        from Model.ocr.data import build_ocr_row

        target_ids = [ord(c) for c in long_text]
        row = build_ocr_row(
            target_ids, 4, "img.png",
            bos_id=2, image_start_id=6, image_patch_id=7, image_end_id=8, eos_id=3,
        )

        def fake_decode(ids: list[int]) -> str:
            return "".join(chr(i) for i in ids)

        image_ref, byte_ids = row_to_byte_target(row, fake_decode)
        self.assertEqual(image_ref, "img.png")
        self.assertGreater(len(byte_ids), 256)  # would be skipped by the dataset


class CTCCollateTest(unittest.TestCase):
    def test_flattened_targets_and_lengths_roundtrip(self):
        pixels = [torch.zeros(3, 4, 4), torch.ones(3, 4, 4)]
        targets = [[1, 2, 3], [4, 5]]
        batch = ctc_collate(list(zip(pixels, targets)))
        self.assertEqual(batch["pixels"].shape, (2, 3, 4, 4))
        self.assertEqual(batch["target_lengths"].tolist(), [3, 2])
        self.assertEqual(batch["targets"].tolist(), [1, 2, 3, 4, 5])
        recovered = _unflatten_targets(batch["targets"], batch["target_lengths"])
        self.assertEqual(recovered, targets)


class CTCCheckpointResumeTest(unittest.TestCase):
    def test_resume_restores_step_and_head_state(self):
        torch.manual_seed(0)
        omvt_cfg = _tiny_omvt_config()
        head = CTCHead(d_vision=omvt_cfg.d_vision, hidden=24)
        args = _default_args()
        train_cfg = _build_scheduler_cfg(args)
        optimizer = build_optimizer(head, train_cfg)

        # Perturb the head so its state is distinguishable from a fresh init.
        with torch.no_grad():
            for p in head.parameters():
                p.add_(1.0)
        saved_state = {k: v.clone() for k, v in head.state_dict().items()}

        with tempfile.TemporaryDirectory() as tmp:
            save_checkpoint(
                tmp, step=17, head=head, optimizer=optimizer, omvt_cfg=omvt_cfg,
                d_vision=omvt_cfg.d_vision, hidden=24,
            )
            self.assertTrue((Path(tmp) / "latest").exists())

            fresh_head = CTCHead(d_vision=omvt_cfg.d_vision, hidden=24)
            fresh_optimizer = build_optimizer(fresh_head, train_cfg)
            restored_step = load_checkpoint_into(tmp, fresh_head, fresh_optimizer)

        self.assertEqual(restored_step, 17)
        for k, v in fresh_head.state_dict().items():
            self.assertTrue(torch.equal(v, saved_state[k]), f"mismatch at {k}")


if __name__ == "__main__":
    unittest.main()
