# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest
from unittest import mock

import torch

from Model.config import OMVTConfig
from Model.omvt import NativeDetailCompressor, NativeOMVTDetailTower, pack_native_omvt_batch
from Model.omvt.native_mixers import (
    NativeSquareWindowAttention,
    NativeVerticalSSM,
    stable_lexicographic_order,
)
from Model.omvt.native_patcher import PackedNativeOMVTBatch, PackedPatchStream
from Tokenizer.multimodal.native_image_io import NativeImageTensor


def _cfg() -> OMVTConfig:
    return OMVTConfig(
        image_size=16,
        vertical_patch=(4, 2),
        horizontal_patch=(2, 4),
        square_patch=(2, 2),
        layout_patch=(4, 4),
        d_vision=8,
        n_vertical_layers=1,
        n_horizontal_layers=1,
        n_local_attn_layers=1,
        n_layout_layers=1,
        vision_n_heads=2,
        vision_ffn_hidden=16,
        vision_dropout=0.0,
        compress_to=5,
        compressor_layers=1,
        compressor_heads=2,
    )


def _sample(height: int, width: int, offset: float) -> NativeImageTensor:
    pixels = torch.arange(3 * height * width, dtype=torch.float32).reshape(
        3, height, width
    )
    pixels = torch.sin(pixels * 0.07 + offset)
    return NativeImageTensor(
        pixels=pixels,
        pixel_valid_mask=torch.ones((height, width), dtype=torch.bool),
        original_hw=(height, width),
    )


def _slice_stream(stream: PackedPatchStream, sample_index: int) -> PackedPatchStream:
    start = int(stream.cu_seqlens[sample_index].item())
    end = int(stream.cu_seqlens[sample_index + 1].item())
    count = end - start
    return PackedPatchStream(
        patches=stream.patches[start:end],
        bbox_px_yxxy=stream.bbox_px_yxxy[start:end],
        bbox_norm_yxxy=stream.bbox_norm_yxxy[start:end],
        valid_fraction=stream.valid_fraction[start:end],
        sample_ids=torch.zeros_like(stream.sample_ids[start:end]),
        cu_seqlens=torch.tensor([0, count], dtype=torch.int32),
        grid_yx=stream.grid_yx[start:end],
    )


def _slice_batch(
    batch: PackedNativeOMVTBatch,
    sample_index: int,
) -> PackedNativeOMVTBatch:
    return PackedNativeOMVTBatch(
        streams={
            kind: _slice_stream(stream, sample_index)
            for kind, stream in batch.streams.items()
        },
        original_hw=batch.original_hw[sample_index : sample_index + 1],
        raw_patch_tokens=batch.raw_patch_tokens[sample_index : sample_index + 1],
    )


class NativeOMVTV2MathContractTest(unittest.TestCase):
    def test_directional_scan_is_stable_and_resets_at_sample_boundaries(self):
        grid = torch.tensor([[1, 1], [0, 1], [0, 0], [1, 0], [0, 1]])
        self.assertEqual(
            stable_lexicographic_order(grid, direction="vertical").tolist(),
            [2, 1, 4, 3, 0],
        )
        self.assertEqual(
            stable_lexicographic_order(grid, direction="horizontal").tolist(),
            [2, 3, 1, 4, 0],
        )

        packed = pack_native_omvt_batch(
            [_sample(11, 7, 0.0), _sample(5, 13, 1.0)],
            _cfg(),
            normalized_white=(1.0, 1.0, 1.0),
        )
        stream = packed.streams["vertical"]
        layer = NativeVerticalSSM(8, 16).eval()
        features = torch.randn(stream.patches.shape[0], 8)
        batch_output = layer(features, stream)
        for sample_index in range(2):
            start = int(stream.cu_seqlens[sample_index].item())
            end = int(stream.cu_seqlens[sample_index + 1].item())
            single_output = layer(
                features[start:end],
                _slice_stream(stream, sample_index),
            )
            self.assertTrue(
                torch.allclose(batch_output[start:end], single_output, atol=1e-6)
            )

    def test_square_attention_uses_only_real_8x8_windows(self):
        packed = pack_native_omvt_batch(
            [_sample(34, 34, 0.0)],
            _cfg(),
            normalized_white=(1.0, 1.0, 1.0),
        )
        stream = packed.streams["square"]
        features = torch.randn(stream.patches.shape[0], 8)
        layer = NativeSquareWindowAttention(8, 2, 16).eval()
        original = torch.nn.functional.scaled_dot_product_attention
        with mock.patch(
            "Model.omvt.native_mixers.F.scaled_dot_product_attention",
            wraps=original,
        ) as attention:
            output = layer(features, stream)
        self.assertEqual(output.shape, features.shape)
        self.assertEqual(attention.call_count, 9)
        for call in attention.call_args_list:
            query = call.args[0]
            self.assertLessEqual(query.shape[-2], 64)
            self.assertNotIn("attn_mask", call.kwargs)

    def test_detail_budget_is_monotonic_and_hard_capped(self):
        compressor = NativeDetailCompressor(
            _cfg(),
            max_detail_tokens_per_sample=5,
            source_tokens_per_detail_token=7,
        )
        budgets = [compressor.detail_token_budget(count) for count in range(101)]
        self.assertEqual(budgets, sorted(budgets))
        self.assertEqual(budgets[0], 0)
        self.assertEqual(max(budgets), 5)
        self.assertEqual(compressor.detail_token_budget(1), 1)
        self.assertEqual(compressor.detail_token_budget(35), 5)
        self.assertEqual(compressor.detail_token_budget(10_000), 5)

    def test_variable_size_batch_matches_single_and_has_no_cross_sample_path(self):
        torch.manual_seed(17)
        cfg = _cfg()
        samples = [_sample(11, 7, 0.0), _sample(5, 13, 1.0)]
        packed = pack_native_omvt_batch(
            samples,
            cfg,
            normalized_white=(1.0, 1.0, 1.0),
        )
        tower = NativeOMVTDetailTower(
            cfg,
            max_detail_tokens_per_sample=5,
            source_tokens_per_detail_token=9,
        ).eval()
        batch_output = tower(packed)
        self.assertLessEqual(int(batch_output["detail_token_counts"].max()), 5)
        self.assertEqual(
            batch_output["detail_cu_seqlens"].tolist(),
            [
                0,
                int(batch_output["detail_token_counts"][0]),
                int(batch_output["detail_token_counts"].sum()),
            ],
        )

        for sample_index in range(2):
            single_output = tower(_slice_batch(packed, sample_index))
            start = int(batch_output["detail_cu_seqlens"][sample_index].item())
            end = int(batch_output["detail_cu_seqlens"][sample_index + 1].item())
            self.assertTrue(
                torch.allclose(
                    batch_output["detail_memory"][start:end],
                    single_output["detail_memory"],
                    atol=1e-6,
                    rtol=1e-6,
                )
            )

        changed = pack_native_omvt_batch(
            [samples[0], _sample(5, 13, 8.0)],
            cfg,
            normalized_white=(1.0, 1.0, 1.0),
        )
        changed_output = tower(changed)
        first_end = int(batch_output["detail_cu_seqlens"][1].item())
        self.assertTrue(
            torch.allclose(
                batch_output["detail_memory"][:first_end],
                changed_output["detail_memory"][:first_end],
                atol=1e-6,
                rtol=1e-6,
            )
        )
        second_start = first_end
        self.assertFalse(
            torch.allclose(
                batch_output["detail_memory"][second_start:],
                changed_output["detail_memory"][second_start:],
            )
        )

        tower.train()
        train_output = tower(packed)
        train_output["detail_memory"].square().mean().backward()
        gradients = [
            parameter.grad
            for parameter in tower.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(any(bool((gradient.abs() > 0).any()) for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
