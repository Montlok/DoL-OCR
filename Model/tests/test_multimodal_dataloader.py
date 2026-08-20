# -*- coding: utf-8 -*-

"""End-to-end multimodal dataloader → train_one_step smoke test.

Asserts that a JSONL row with an ``images`` column flows through the
streaming dataloader, lands as a stacked ``pixel_values`` dict on the
model, and produces a finite loss after one optimizer step.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from dataclasses import replace

import torch

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore

from Model.config import (
    BOS_ID,
    EOS_ID,
    IMAGE_PATCH_ID,
    OMVTConfig,
    TrainingConfig,
    tiny_config,
)
from Model.model import RDTForCausalLM
from Model.omvt import OMVTInjector
from Model.training import (
    PretrainingCollator,
    TrainState,
    build_omvt_cfg,
    build_optimizer,
    build_scheduler,
    train_one_step,
)

from Tokenizer.multimodal import PILImageProcessor


@unittest.skipIf(Image is None, "Pillow not installed")
class MultimodalDataloaderTest(unittest.TestCase):
    def test_multimodal_cli_derives_image_token_default(self) -> None:
        args = type(
            "Args",
            (),
            {
                "multimodal": True,
                "image_size": 64,
                "n_image_tokens": None,
                "d_vision": 32,
            },
        )()
        cfg = build_omvt_cfg(args)
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.compress_to, 9)

    def test_pixel_aware_collator_and_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            img_path = os.path.join(tmp, "t.png")
            Image.new("RGB", (16, 16), color=(127, 64, 200)).save(img_path)

            rdt_cfg = replace(tiny_config(), max_seq_len=12)
            omvt_cfg = OMVTConfig(
                image_size=16,
                d_vision=32,
                vertical_patch=(8, 4),
                horizontal_patch=(4, 8),
                square_patch=(4, 4),
                layout_patch=(16, 16),
                compress_to=4,
            )

            model = RDTForCausalLM(rdt_cfg)
            model.vision._omvt_cfg = omvt_cfg
            model.vision.omvt = OMVTInjector(rdt_cfg, omvt_cfg)

            n_img = 4
            ids = [BOS_ID] + [IMAGE_PATCH_ID] * n_img + [301, 302, 303, 304, 305, 306] + [EOS_ID]
            row = {
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
                "labels": ids,
                "images": [img_path],
            }

            coll = PretrainingCollator(
                image_processor=PILImageProcessor(image_size=16),
                omvt_cfg=omvt_cfg,
            )
            batch = coll([row, row])

            self.assertIn("pixel_values", batch)
            self.assertIsInstance(batch["pixel_values"], dict)
            self.assertEqual(batch["pixel_values"]["images"].shape, (2, 3, 16, 16))
            self.assertIn("vertical_patches", batch["pixel_values"])

            def gen():
                while True:
                    yield batch

            cfg = TrainingConfig(
                train_data="",
                seq_len=len(ids),
                micro_batch_size=2,
                max_steps=1,
                warmup_steps=1,
                precision="fp32",
            )
            opt = build_optimizer(model, cfg)
            sch = build_scheduler(opt, cfg)
            state = TrainState()
            metrics = train_one_step(model, gen(), opt, sch, cfg, state, device=torch.device("cpu"))
            self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
            self.assertGreater(metrics["loss"], 0.0)

    def test_text_only_row_skips_pixel_values(self) -> None:
        coll = PretrainingCollator(
            image_processor=PILImageProcessor(image_size=16) if Image else None,
            omvt_cfg=None,
        )
        row = {
            "input_ids": [BOS_ID, 11, 12, EOS_ID],
            "attention_mask": [1, 1, 1, 1],
            "labels": [BOS_ID, 11, 12, EOS_ID],
        }
        batch = coll([row, row])
        self.assertNotIn("pixel_values", batch)

    def test_multi_image_per_row_is_rejected(self) -> None:
        # Current OMVT injector only supports 1 image per row; the
        # collator must refuse N!=1 explicitly so the failure mode is a
        # clear ValueError rather than a silent OMVT/B*N mismatch.
        from Model.config import OMVTConfig

        omvt_cfg = OMVTConfig(
            image_size=16,
            d_vision=32,
            vertical_patch=(8, 4),
            horizontal_patch=(4, 8),
            square_patch=(4, 4),
            layout_patch=(16, 16),
            compress_to=4,
        )
        coll = PretrainingCollator(
            image_processor=PILImageProcessor(image_size=16) if Image else None,
            omvt_cfg=omvt_cfg,
        )
        with tempfile.TemporaryDirectory() as tmp:
            img = os.path.join(tmp, "x.png")
            Image.new("RGB", (16, 16)).save(img)
            row = {
                "input_ids": [BOS_ID, IMAGE_PATCH_ID, EOS_ID],
                "attention_mask": [1, 1, 1],
                "labels": [BOS_ID, IMAGE_PATCH_ID, EOS_ID],
                "images": [img, img],  # 2 images on this row → rejected
            }
            with self.assertRaisesRegex(ValueError, "one image per row"):
                coll([row, row])

    def test_mixed_image_counts_in_batch_is_rejected(self) -> None:
        from Model.config import OMVTConfig

        omvt_cfg = OMVTConfig(
            image_size=16,
            d_vision=32,
            vertical_patch=(8, 4),
            horizontal_patch=(4, 8),
            square_patch=(4, 4),
            layout_patch=(16, 16),
            compress_to=4,
        )
        coll = PretrainingCollator(
            image_processor=PILImageProcessor(image_size=16) if Image else None,
            omvt_cfg=omvt_cfg,
        )
        with tempfile.TemporaryDirectory() as tmp:
            img = os.path.join(tmp, "x.png")
            Image.new("RGB", (16, 16)).save(img)
            with_img = {
                "input_ids": [BOS_ID, IMAGE_PATCH_ID, EOS_ID],
                "attention_mask": [1, 1, 1],
                "labels": [BOS_ID, IMAGE_PATCH_ID, EOS_ID],
                "images": [img],
            }
            without = {
                "input_ids": [BOS_ID, 5, EOS_ID],
                "attention_mask": [1, 1, 1],
                "labels": [BOS_ID, 5, EOS_ID],
            }
            with self.assertRaisesRegex(ValueError, "one image per row"):
                coll([with_img, without])

    def test_strict_ocr_collator_verifies_image_bytes_lazily(self) -> None:
        omvt_cfg = OMVTConfig(
            image_size=16,
            d_vision=32,
            vertical_patch=(8, 4),
            horizontal_patch=(4, 8),
            square_patch=(4, 4),
            layout_patch=(16, 16),
            compress_to=4,
        )
        coll = PretrainingCollator(
            image_processor=PILImageProcessor(image_size=16),
            omvt_cfg=omvt_cfg,
            require_verified_images=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            image_path = os.path.join(tmp, "bound.png")
            Image.new("RGB", (16, 16), color="white").save(image_path)
            image_bytes = open(image_path, "rb").read()
            row = {
                "input_ids": [BOS_ID, IMAGE_PATCH_ID, EOS_ID],
                "attention_mask": [1, 1, 1],
                "labels": [BOS_ID, IMAGE_PATCH_ID, EOS_ID],
                "images": [image_path],
                "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "image_size_bytes": len(image_bytes),
            }
            self.assertIn("pixel_values", coll([row]))

            mutated = bytearray(image_bytes)
            mutated[-1] ^= 1
            with open(image_path, "wb") as handle:
                handle.write(mutated)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                coll([row])

    def test_strict_ocr_collator_rejects_missing_image_binding(self) -> None:
        coll = PretrainingCollator(require_verified_images=True)
        row = {
            "input_ids": [BOS_ID, IMAGE_PATCH_ID, EOS_ID],
            "attention_mask": [1, 1, 1],
            "labels": [BOS_ID, IMAGE_PATCH_ID, EOS_ID],
            "images": ["/tmp/unbound.png"],
        }
        with self.assertRaisesRegex(ValueError, "image_sha256"):
            coll([row])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
