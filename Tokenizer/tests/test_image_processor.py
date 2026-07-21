# -*- coding: utf-8 -*-

"""Tests for :class:`Tokenizer.multimodal.PILImageProcessor`."""

from __future__ import annotations

import os
import tempfile
import unittest

import torch

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore

from Tokenizer.multimodal import PILImageProcessor


@unittest.skipIf(Image is None, "Pillow not installed")
class PILImageProcessorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "a.png")
        Image.new("RGB", (32, 24), color=(200, 100, 50)).save(self.path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_path_input_returns_normalized_chw(self) -> None:
        proc = PILImageProcessor(image_size=16)
        out = proc([self.path])
        self.assertEqual(out.shape, (1, 3, 16, 16))
        self.assertEqual(out.dtype, torch.float32)
        # ImageNet normalize ⇒ values land in a sane range, not raw [0,1].
        self.assertGreater(float(out.max()), 0.0)
        self.assertLess(float(out.min()), 0.0)

    def test_bytes_input(self) -> None:
        with open(self.path, "rb") as f:
            data = f.read()
        proc = PILImageProcessor(image_size=8)
        out = proc([data])
        self.assertEqual(out.shape, (1, 3, 8, 8))

    def test_pil_image_input(self) -> None:
        img = Image.open(self.path).convert("RGB")
        proc = PILImageProcessor(image_size=8)
        out = proc([img])
        self.assertEqual(out.shape, (1, 3, 8, 8))

    def test_dict_with_path(self) -> None:
        proc = PILImageProcessor(image_size=8)
        out = proc([{"path": self.path}])
        self.assertEqual(out.shape, (1, 3, 8, 8))

    def test_batch_of_mixed_specs(self) -> None:
        proc = PILImageProcessor(image_size=8)
        with open(self.path, "rb") as f:
            data = f.read()
        out = proc([self.path, data, Image.open(self.path)])
        self.assertEqual(out.shape, (3, 3, 8, 8))

    def test_custom_mean_std(self) -> None:
        proc = PILImageProcessor(image_size=4, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
        out = proc([self.path])
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)

    def test_phone_exif_orientation_is_applied(self) -> None:
        from Tokenizer.multimodal.image_io import _open_to_rgb

        path = os.path.join(self.tmp.name, "oriented.jpg")
        image = Image.new("RGB", (2, 3), color=(200, 100, 50))
        exif = image.getexif()
        exif[274] = 6  # rotate 90 degrees clockwise for display
        image.save(path, exif=exif)
        self.assertEqual(_open_to_rgb(path).size, (3, 2))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
