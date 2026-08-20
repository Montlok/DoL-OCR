# -*- coding: utf-8 -*-

from __future__ import annotations

import io
import unittest

import torch

from Model.config import OMVTConfig
from Model.omvt import pack_native_omvt_batch
from Tokenizer.multimodal import NativeImageProcessorV2

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore


def _cfg() -> OMVTConfig:
    return OMVTConfig(
        image_size=16,
        vertical_patch=(8, 4),
        horizontal_patch=(4, 8),
        square_patch=(4, 4),
        layout_patch=(16, 16),
        d_vision=16,
        vision_n_heads=4,
        vision_ffn_hidden=32,
        compress_to=2,
    )


@unittest.skipIf(Image is None, "Pillow not installed")
class NativeOMVTGeometryTest(unittest.TestCase):
    @staticmethod
    def _png(width: int, height: int, color: tuple[int, int, int]) -> bytes:
        buffer = io.BytesIO()
        Image.new("RGB", (width, height), color=color).save(buffer, format="PNG")
        return buffer.getvalue()

    def test_processor_preserves_aspect_ratio_and_packer_is_sample_local(self):
        processor = NativeImageProcessorV2(max_decode_pixels=10_000)
        samples = processor(
            [
                self._png(13, 29, (255, 255, 255)),
                self._png(31, 7, (0, 0, 0)),
            ]
        )
        self.assertEqual([sample.original_hw for sample in samples], [(29, 13), (7, 31)])
        self.assertEqual(tuple(samples[0].pixels.shape), (3, 29, 13))
        self.assertEqual(tuple(samples[1].pixels.shape), (3, 7, 31))

        packed = pack_native_omvt_batch(
            samples,
            _cfg(),
            normalized_white=processor.normalized_white,
        )
        square = packed.streams["square"]
        self.assertEqual(square.cu_seqlens.tolist(), [0, 32, 48])
        self.assertEqual(packed.raw_patch_tokens.tolist(), [66, 34])
        self.assertTrue(bool((square.bbox_norm_yxxy >= 0).all()))
        self.assertTrue(bool((square.bbox_norm_yxxy <= 1).all()))
        self.assertTrue(bool((square.valid_fraction > 0).all()))
        self.assertTrue(bool((square.valid_fraction <= 1).all()))

        for sample_index, sample in enumerate(samples):
            single = pack_native_omvt_batch(
                [sample],
                _cfg(),
                normalized_white=processor.normalized_white,
            ).streams["square"]
            start = square.cu_seqlens[sample_index].item()
            end = square.cu_seqlens[sample_index + 1].item()
            self.assertTrue(torch.equal(square.patches[start:end], single.patches))
            self.assertTrue(
                torch.equal(square.bbox_norm_yxxy[start:end], single.bbox_norm_yxxy)
            )
            self.assertTrue(
                torch.equal(square.valid_fraction[start:end], single.valid_fraction)
            )

    def test_partial_patch_uses_normalized_white_and_records_fraction(self):
        processor = NativeImageProcessorV2(max_decode_pixels=100)
        sample = processor([self._png(3, 3, (0, 0, 0))])[0]
        packed = pack_native_omvt_batch(
            [sample],
            _cfg(),
            normalized_white=processor.normalized_white,
        )
        square = packed.streams["square"]
        self.assertEqual(square.patches.shape[0], 1)
        self.assertAlmostEqual(float(square.valid_fraction[0]), 9.0 / 16.0)
        patch = square.patches[0].reshape(3, 4, 4)
        expected_white = torch.tensor(processor.normalized_white)
        self.assertTrue(torch.allclose(patch[:, 3, 3], expected_white))

    def test_budget_fails_closed_without_streaming_planner(self):
        processor = NativeImageProcessorV2(max_decode_pixels=10_000)
        sample = processor([self._png(32, 32, (255, 255, 255))])[0]
        with self.assertRaisesRegex(ValueError, "streamed macro-window"):
            pack_native_omvt_batch(
                [sample],
                _cfg(),
                normalized_white=processor.normalized_white,
                max_raw_patch_tokens_per_sample=1,
            )

    def test_decode_budget_fails_before_materializing_native_tensor(self):
        processor = NativeImageProcessorV2(max_decode_pixels=8)
        with self.assertRaisesRegex(ValueError, "max_pixels"):
            processor([self._png(3, 3, (255, 255, 255))])


if __name__ == "__main__":
    unittest.main()
