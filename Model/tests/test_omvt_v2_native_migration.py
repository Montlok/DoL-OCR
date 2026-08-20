# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import replace
import hashlib
import unittest

import torch

from Model.config import OMVTConfig, RDTConfig
from Model.omvt import pack_native_omvt_batch
from Model.vision import VisionInjector
from Tokenizer.multimodal.native_image_io import NativeImageTensor


def _omvt_cfg() -> OMVTConfig:
    return OMVTConfig(
        image_size=16,
        vertical_patch=(8, 4),
        horizontal_patch=(4, 8),
        square_patch=(4, 4),
        layout_patch=(16, 16),
        d_vision=8,
        n_vertical_layers=1,
        n_horizontal_layers=1,
        n_local_attn_layers=1,
        n_layout_layers=1,
        vision_n_heads=2,
        vision_ffn_hidden=16,
        vision_dropout=0.0,
        compress_to=3,
        compressor_layers=1,
        compressor_heads=2,
    )


def _rdt_cfg() -> RDTConfig:
    return RDTConfig(
        d_model=8,
        n_heads=1,
        head_dim=8,
        kv_lora_rank=4,
        rope_head_dim=4,
        nope_head_dim=4,
        ffn_hidden=16,
        ffn_multiple=8,
        n_prelude=1,
        n_coda=1,
        mamba_per_block=1,
        attn_per_block=1,
        recurrent_steps=1,
        mamba_d_state=4,
        mamba_expand=2,
        mamba_headdim=8,
        use_official_mamba=False,
        max_seq_len=32,
    )


def _sample(height: int, width: int) -> NativeImageTensor:
    pixels = torch.linspace(-1.0, 1.0, 3 * height * width).reshape(
        3, height, width
    )
    return NativeImageTensor(
        pixels=pixels,
        pixel_valid_mask=torch.ones((height, width), dtype=torch.bool),
        original_hw=(height, width),
    )


class NativeTowerMigrationTest(unittest.TestCase):
    def test_explicit_install_maps_every_tensor_and_preserves_v1_state(self):
        torch.manual_seed(41)
        omvt_cfg = _omvt_cfg()
        vision = VisionInjector(_rdt_cfg(), omvt_cfg=omvt_cfg)
        self.assertIsNone(vision.native_detail_tower)
        vision._ensure_omvt()

        expected_v1_keys = {
            f"encoder.{name}" for name in vision.encoder.state_dict()
        } | {f"omvt.{name}" for name in vision.omvt.state_dict()}
        self.assertEqual(set(vision.state_dict()), expected_v1_keys)
        self.assertFalse(
            any(name.startswith("native_detail_tower.") for name in vision.state_dict())
        )
        legacy_before = {
            name: tensor.detach().clone()
            for name, tensor in vision.omvt.tower.state_dict().items()
        }

        receipt = vision.install_native_detail_tower(
            omvt_cfg,
            max_detail_tokens=omvt_cfg.compress_to,
            ratio=7,
            initialize_from_legacy=True,
        )
        self.assertIs(receipt, vision.native_detail_migration_receipt)
        self.assertTrue(receipt.initialized_from_legacy)
        self.assertEqual(
            receipt.canonical_sha256,
            hashlib.sha256(receipt.canonical_json.encode("utf-8")).hexdigest(),
        )

        legacy_state = vision.omvt.tower.state_dict()
        native_state = vision.native_detail_tower.state_dict()
        copied_targets = set()
        classified_targets = set()
        for mapping in receipt.mappings:
            classified_targets.add(mapping.target)
            target = native_state[mapping.target]
            self.assertEqual(tuple(target.shape), mapping.shape)
            self.assertEqual(str(target.dtype), mapping.dtype)
            if mapping.action == "copy_exact":
                self.assertIsNotNone(mapping.source)
                self.assertTrue(torch.equal(legacy_state[mapping.source], target))
                copied_targets.add(mapping.target)
            elif mapping.action.startswith("zero_"):
                self.assertTrue(torch.equal(target, torch.zeros_like(target)))
            elif mapping.action == "identity_new_layer_norm_affine":
                if mapping.target.endswith(".weight"):
                    self.assertTrue(torch.equal(target, torch.ones_like(target)))
                else:
                    self.assertTrue(torch.equal(target, torch.zeros_like(target)))
            else:
                self.fail(f"unexpected legacy migration action: {mapping.action}")
        self.assertEqual(classified_targets, set(native_state))
        self.assertIn("encoders.vertical.patch_embed.weight", copied_targets)
        self.assertIn("encoders.square.layers.0.qkv.weight", copied_targets)
        self.assertIn("encoders.layout.layers.0.mixer.3.weight", copied_targets)
        self.assertIn("compressor.latents", copied_targets)
        self.assertIn(
            "compressor.blocks.0.cross.key_value_proj.weight",
            copied_targets,
        )
        for name, before in legacy_before.items():
            self.assertTrue(torch.equal(before, legacy_state[name]))

        with self.assertRaisesRegex(RuntimeError, "already installed"):
            vision.install_native_detail_tower(
                omvt_cfg,
                max_detail_tokens=omvt_cfg.compress_to,
                ratio=7,
            )

    def test_installed_native_tower_forwards_and_backpropagates(self):
        torch.manual_seed(53)
        omvt_cfg = _omvt_cfg()
        vision = VisionInjector(_rdt_cfg(), omvt_cfg=omvt_cfg)
        vision._ensure_omvt()
        vision.install_native_detail_tower(
            omvt_cfg,
            max_detail_tokens=omvt_cfg.compress_to,
            ratio=5,
        )
        packed = pack_native_omvt_batch(
            [_sample(13, 9), _sample(7, 15)],
            omvt_cfg,
            normalized_white=(1.0, 1.0, 1.0),
        )
        output = vision.native_detail_tower(packed)
        self.assertEqual(output["detail_cu_seqlens"].numel(), 3)
        output["detail_memory"].square().mean().backward()
        self.assertIsNotNone(
            vision.native_detail_tower.encoders["vertical"].patch_embed.weight.grad
        )
        self.assertGreater(
            float(
                vision.native_detail_tower.encoders[
                    "vertical"
                ].patch_embed.weight.grad.abs().sum()
            ),
            0.0,
        )

    def test_migration_fails_closed_on_missing_legacy_or_shape_and_layer_drift(self):
        omvt_cfg = _omvt_cfg()
        missing = VisionInjector(_rdt_cfg(), omvt_cfg=omvt_cfg)
        before = tuple(missing.state_dict())
        with self.assertRaisesRegex(RuntimeError, "must already be installed"):
            missing.install_native_detail_tower(
                omvt_cfg,
                max_detail_tokens=omvt_cfg.compress_to,
                ratio=4,
            )
        self.assertIsNone(missing.native_detail_tower)
        self.assertEqual(tuple(missing.state_dict()), before)

        extended = VisionInjector(_rdt_cfg(), omvt_cfg=omvt_cfg)
        extended._ensure_omvt()
        source_latents = extended.omvt.tower.compressor.latents.detach().clone()
        receipt = extended.install_native_detail_tower(
            omvt_cfg,
            max_detail_tokens=omvt_cfg.compress_to + 2,
            ratio=4,
        )
        target_latents = extended.native_detail_tower.compressor.latents
        self.assertTrue(
            torch.equal(target_latents[: omvt_cfg.compress_to], source_latents)
        )
        self.assertTrue(torch.equal(target_latents[-2:], source_latents[:2]))
        self.assertIn(
            "copy_prefix_repeat_extension",
            {mapping.action for mapping in receipt.mappings},
        )

        too_small = VisionInjector(_rdt_cfg(), omvt_cfg=omvt_cfg)
        too_small._ensure_omvt()
        with self.assertRaisesRegex(ValueError, "at least legacy compress_to"):
            too_small.install_native_detail_tower(
                omvt_cfg,
                max_detail_tokens=omvt_cfg.compress_to - 1,
                ratio=4,
            )
        self.assertIsNone(too_small.native_detail_tower)

        wrong_layers = VisionInjector(_rdt_cfg(), omvt_cfg=omvt_cfg)
        wrong_layers._ensure_omvt()
        with self.assertRaisesRegex(ValueError, "n_vertical_layers"):
            wrong_layers.install_native_detail_tower(
                replace(omvt_cfg, n_vertical_layers=2),
                max_detail_tokens=omvt_cfg.compress_to,
                ratio=4,
            )
        self.assertIsNone(wrong_layers.native_detail_tower)


if __name__ == "__main__":
    unittest.main()
