# -*- coding: utf-8 -*-

from __future__ import annotations

import io
import unittest

import torch
from PIL import Image

from Model.config import OMVTConfig, RDTConfig
from Model.model import RDTForCausalLM
from Model.omvt import OMVTInjector
from Model.posttrain.ocr_anyres_collator import AnyresOCRSFTCollator
from Model.posttrain.ocr_joint_forward import (
    forward_anyres_ocr_batch,
    move_native_batch,
)
from Tokenizer.multimodal import NativeImageProcessorV2, PILImageProcessor


class _NativeEncoder:
    mode = "native"

    def __call__(self, text: str) -> list[int]:
        return [300 + index for index, _ in enumerate(text)]


def _png(width: int, height: int, value: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (value, value, value)).save(
        buffer, format="PNG"
    )
    return buffer.getvalue()


def _item() -> dict:
    text = "abc"
    sample = {
        "visual_contract": "dol_ocr_anyres_v2",
        "qa_state": "accepted",
        "reference_model": {"text": text},
        "reference_token_count": len(text),
        "reading_order": ["view"],
        "style": "print",
        "difficulty": None,
    }
    return {
        "dataset_contract_sha256": "d" * 64,
        "sample": sample,
        "canonical_image": {
            "delivery": "bytes",
            "bytes": _png(7, 13, 20),
            "metadata": {},
        },
        "derived_images": [
            {
                "delivery": "bytes",
                "bytes": _png(5, 11, 80),
                "metadata": {"view_id": "view"},
            }
        ],
        "quota_bucket": "print",
    }


def _configs() -> tuple[RDTConfig, OMVTConfig]:
    rdt = RDTConfig(
        d_model=32,
        n_heads=4,
        head_dim=8,
        kv_lora_rank=8,
        rope_head_dim=4,
        nope_head_dim=4,
        ffn_hidden=64,
        ffn_multiple=32,
        n_prelude=1,
        n_coda=1,
        mamba_per_block=1,
        attn_per_block=1,
        recurrent_steps=1,
        mamba_d_state=8,
        mamba_expand=2,
        mamba_headdim=16,
        use_official_mamba=False,
        max_seq_len=512,
        bidirectional=False,
    )
    omvt = OMVTConfig(
        image_size=16,
        vertical_patch=(8, 4),
        horizontal_patch=(4, 8),
        square_patch=(4, 4),
        layout_patch=(16, 16),
        d_vision=32,
        vision_n_heads=4,
        vision_ffn_hidden=64,
        compress_to=256,
        compressor_layers=1,
        compressor_heads=4,
    )
    return rdt, omvt


class OCRJointForwardTest(unittest.TestCase):
    def test_native_float_payloads_follow_tower_precision(self) -> None:
        _rdt, omvt = _configs()
        collator = AnyresOCRSFTCollator(
            encode_reference=_NativeEncoder(),
            omvt_cfg=omvt,
            global_processor=PILImageProcessor(
                image_size=16, in_channels=3, mean=None, std=None
            ),
            native_processor=NativeImageProcessorV2(
                in_channels=3, mean=None, std=None, max_decode_pixels=1000
            ),
            max_raw_patch_tokens_per_view=100,
            max_seq_len=512,
        )
        packed = collator([_item()])["native_packed"]
        moved = move_native_batch(
            packed,
            torch.device("cpu"),
            dtype=torch.bfloat16,
        )
        for stream in moved.streams.values():
            self.assertEqual(stream.patches.dtype, torch.bfloat16)
            self.assertEqual(stream.bbox_norm_yxxy.dtype, torch.bfloat16)
            self.assertEqual(stream.valid_fraction.dtype, torch.bfloat16)
            self.assertEqual(stream.sample_ids.dtype, torch.long)
            self.assertEqual(stream.cu_seqlens.device.type, "cpu")

    def test_two_stage_zero_bridge_opens_gradient_into_native_tower(self) -> None:
        torch.manual_seed(37)
        rdt, omvt = _configs()
        collator = AnyresOCRSFTCollator(
            encode_reference=_NativeEncoder(),
            omvt_cfg=omvt,
            global_processor=PILImageProcessor(
                image_size=16, in_channels=3, mean=None, std=None
            ),
            native_processor=NativeImageProcessorV2(
                in_channels=3, mean=None, std=None, max_decode_pixels=1000
            ),
            max_raw_patch_tokens_per_view=100,
            max_seq_len=512,
        )
        batch = collator([_item()])
        model = RDTForCausalLM(rdt)
        model.vision._omvt_cfg = omvt
        model.vision.omvt = OMVTInjector(rdt, omvt)
        model.vision.install_native_detail_tower(
            omvt,
            max_detail_tokens=256,
            ratio=4,
            initialize_from_legacy=True,
        )
        bridge = model.install_vision_cross_attention(memory_dim=omvt.d_vision)

        first = forward_anyres_ocr_batch(
            model,
            batch,
            device=torch.device("cpu"),
        )
        first["loss"].backward()
        self.assertGreater(float(bridge.output_projection.weight.grad.abs().sum()), 0)
        tower_weight = model.vision.native_detail_tower.encoders[
            "vertical"
        ].patch_embed.weight
        self.assertTrue(tower_weight.grad is None or float(tower_weight.grad.abs().sum()) == 0)

        with torch.no_grad():
            bridge.output_projection.weight.add_(
                1e-3 * bridge.output_projection.weight.grad
            )
        model.zero_grad(set_to_none=True)
        second = forward_anyres_ocr_batch(
            model,
            batch,
            device=torch.device("cpu"),
        )
        second["loss"].backward()
        self.assertIsNotNone(tower_weight.grad)
        self.assertGreater(float(tower_weight.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
