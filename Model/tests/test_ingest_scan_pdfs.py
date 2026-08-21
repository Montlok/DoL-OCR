# -*- coding: utf-8 -*-

"""Unit tests for the pure helpers of ``scripts.ingest_scan_pdfs``."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore

from scripts.ingest_scan_pdfs import _ink_fraction, _mn_count, _pad_square  # noqa: E402


class MnCountTest(unittest.TestCase):
    def test_counts_only_mongolian_block(self):
        self.assertEqual(_mn_count("ᠮᠣᠩᠭᠣᠯ abc 123"), 6)
        self.assertEqual(_mn_count("plain latin"), 0)


@unittest.skipIf(Image is None, "Pillow not installed")
class PadSquareTest(unittest.TestCase):
    def test_landscape_page_pads_to_centered_square(self):
        img = Image.new("L", (40, 20), 0)
        out = _pad_square(img, fill=255)
        self.assertEqual(out.size, (40, 40))
        self.assertEqual(out.getpixel((0, 0)), 255)  # padding band
        self.assertEqual(out.getpixel((20, 20)), 0)  # original content

    def test_square_page_returned_unchanged(self):
        img = Image.new("L", (32, 32), 7)
        self.assertIs(_pad_square(img, fill=255), img)


@unittest.skipIf(Image is None, "Pillow not installed")
class InkFractionTest(unittest.TestCase):
    def test_half_dark_page_reads_half(self):
        img = Image.new("L", (64, 64), 255)
        img.paste(0, (0, 0, 32, 64))
        self.assertAlmostEqual(_ink_fraction(img), 0.5, places=3)

    def test_blank_page_reads_zero(self):
        self.assertEqual(_ink_fraction(Image.new("L", (64, 64), 255)), 0.0)


if __name__ == "__main__":
    unittest.main()
