# -*- coding: utf-8 -*-

"""Contracts for the retained line-image preprocessing used by CTC/OMVT."""

from __future__ import annotations

import io
import unittest

import numpy as np
from PIL import Image

from Model.ocr.image_preprocess import letterbox_grayscale_to_square


class OCRImagePreprocessTests(unittest.TestCase):
    def test_bytes_and_pil_inputs_are_pixel_identical(self) -> None:
        image = Image.new("L", (11, 173), 255)
        for y in range(7, 168):
            image.putpixel((5, y), (y * 13) % 180)
        encoded = io.BytesIO()
        image.save(encoded, format="PNG")

        from_bytes = letterbox_grayscale_to_square(encoded.getvalue(), 224)
        from_pil = letterbox_grayscale_to_square(image, 224)
        self.assertTrue(np.array_equal(np.asarray(from_bytes), np.asarray(from_pil)))

    def test_extreme_aspect_ratio_is_not_stretched(self) -> None:
        image = Image.new("L", (2, 400), 0)
        output = letterbox_grayscale_to_square(image, 224)
        dark = np.argwhere(np.asarray(output) < 128)
        height = int(dark[:, 0].max() - dark[:, 0].min() + 1)
        width = int(dark[:, 1].max() - dark[:, 1].min() + 1)
        self.assertGreaterEqual(height, 220)
        self.assertLessEqual(width, 4)

    def test_exif_orientation_is_applied_before_letterbox(self) -> None:
        image = Image.new("L", (80, 20), 0)
        exif = Image.Exif()
        exif[274] = 6
        encoded = io.BytesIO()
        image.save(encoded, format="JPEG", quality=100, exif=exif)

        output = letterbox_grayscale_to_square(encoded.getvalue(), 224)
        dark = np.argwhere(np.asarray(output) < 128)
        height = int(dark[:, 0].max() - dark[:, 0].min() + 1)
        width = int(dark[:, 1].max() - dark[:, 1].min() + 1)
        self.assertGreater(height, width * 3)


if __name__ == "__main__":
    unittest.main()
