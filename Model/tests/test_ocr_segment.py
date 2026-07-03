# -*- coding: utf-8 -*-

"""Unit tests for Model.ocr.segment (synthetic ink bars, no model, no fonts).

Covers the deployment segmentation contract: column detection via smoothed
ink-profile valleys (including the degenerate single-column, blank-page, and
exact min-width/min-gap boundary cases — not just the wide-margin happy
path), valley-preferring line cutting with hard-cut fallback, and the
PIL-facing ``detect_line_columns`` wrapper the annotation-pack builder
consumes (whole-page fallback + absolute-density semantics preserved).
"""

from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from Model.ocr.segment import (
    chunk_column_by_height,
    detect_columns,
    detect_line_columns,
    lines_from_column,
)


def blank_page(h: int, w: int) -> np.ndarray:
    return np.full((h, w), 255, dtype=np.uint8)


def add_bar(page: np.ndarray, x0: int, x1: int, y0: int, y1: int, value: int = 0):
    page[y0:y1, x0:x1] = value


class DetectColumnsTest(unittest.TestCase):
    def test_two_columns_found_in_left_to_right_order(self):
        page = blank_page(300, 260)
        add_bar(page, 40, 70, 20, 280)
        add_bar(page, 150, 180, 20, 280)
        boxes = detect_columns(page)
        self.assertEqual(len(boxes), 2)
        (ax0, ay0, ax1, ay1), (bx0, _, bx1, _) = boxes
        self.assertLess(ax0, bx0)  # left-to-right ordering
        # boxes are full page height
        self.assertEqual((ay0, ay1), (0, 300))
        # smoothing widens the run by at most the window; bar must be inside
        self.assertLessEqual(ax0, 40)
        self.assertGreaterEqual(ax1, 70)
        self.assertLessEqual(abs(ax0 - 40), 6)
        self.assertLessEqual(abs(ax1 - 70), 6)
        self.assertLessEqual(abs(bx0 - 150), 6)
        self.assertLessEqual(abs(bx1 - 180), 6)

    def test_single_column_page(self):
        page = blank_page(300, 200)
        add_bar(page, 80, 120, 10, 290)
        boxes = detect_columns(page)
        self.assertEqual(len(boxes), 1)
        x0, _, x1, _ = boxes[0]
        self.assertLessEqual(x0, 80)
        self.assertGreaterEqual(x1, 120)

    def test_blank_page_returns_empty(self):
        self.assertEqual(detect_columns(blank_page(100, 100)), [])

    def test_min_width_boundary(self):
        # A run exactly min_width wide survives; one pixel narrower does not.
        # smooth=1 keeps the profile run exactly as drawn.
        page = blank_page(100, 100)
        add_bar(page, 30, 42, 0, 100)  # width 12
        kept = detect_columns(page, smooth=1, min_width=12)
        self.assertEqual(len(kept), 1)
        dropped = detect_columns(page, smooth=1, min_width=13)
        self.assertEqual(dropped, [])

    def test_min_gap_boundary_merges_and_splits(self):
        # Two bars separated by a 5px valley: min_gap=6 merges (gap < 6),
        # min_gap=5 keeps them separate (gap == 5 is not < 5).
        page = blank_page(100, 120)
        add_bar(page, 20, 40, 0, 100)
        add_bar(page, 45, 65, 0, 100)
        merged = detect_columns(page, smooth=1, min_gap=6)
        self.assertEqual(len(merged), 1)
        self.assertEqual((merged[0][0], merged[0][2]), (20, 65))
        split = detect_columns(page, smooth=1, min_gap=5)
        self.assertEqual(len(split), 2)

    def test_narrow_noise_streak_filtered(self):
        page = blank_page(100, 100)
        add_bar(page, 20, 44, 0, 100)  # real column
        add_bar(page, 70, 73, 0, 100)  # 3px streak
        boxes = detect_columns(page, smooth=1, min_width=12)
        self.assertEqual(len(boxes), 1)
        self.assertEqual((boxes[0][0], boxes[0][2]), (20, 44))

    def test_rejects_non_2d_input(self):
        with self.assertRaises(ValueError):
            detect_columns(np.zeros((4, 4, 3), dtype=np.uint8))

    def test_parameter_validation(self):
        page = blank_page(10, 10)
        with self.assertRaises(ValueError):
            detect_columns(page, smooth=0)
        with self.assertRaises(ValueError):
            detect_columns(page, valley_frac=1.0)
        with self.assertRaises(ValueError):
            detect_columns(page, min_width=0)
        with self.assertRaises(ValueError):
            detect_columns(page, min_gap=-1)


class LinesFromColumnTest(unittest.TestCase):
    def _column_with_bars(self, bars, h=320, w=40):
        col = blank_page(h, w)
        for y0, y1 in bars:
            add_bar(col, 5, w - 5, y0, y1)
        return col

    def test_three_bars_three_strips_in_reading_order(self):
        bars = [(20, 80), (120, 180), (220, 280)]
        col = self._column_with_bars(bars)
        boxes = lines_from_column(col, target_px=100)
        self.assertEqual(len(boxes), 3)
        for (y0, y1), (b0, b1) in zip(boxes, bars):
            # each strip covers its bar, trimmed near the bar extents
            self.assertLessEqual(y0, b0)
            self.assertGreaterEqual(y1, b1)
            self.assertLessEqual(b0 - y0, 6)
            self.assertLessEqual(y1 - b1, 6)
        # reading order: top to bottom
        self.assertEqual(boxes, sorted(boxes))

    def test_blank_column_returns_empty(self):
        self.assertEqual(lines_from_column(blank_page(200, 40), 100), [])

    def test_solid_column_hard_cuts(self):
        # No valley anywhere: hard cuts at target_px, full coverage.
        col = self._column_with_bars([(0, 500)], h=500)
        boxes = lines_from_column(col, target_px=200, pad=0, smooth=1)
        self.assertEqual(len(boxes), 3)
        self.assertEqual(boxes[0][0], 0)
        self.assertEqual(boxes[-1][1], 500)
        for (_, prev_end), (next_start, _) in zip(boxes, boxes[1:]):
            self.assertEqual(prev_end, next_start)

    def test_dust_below_min_height_dropped(self):
        col = blank_page(100, 40)
        add_bar(col, 5, 35, 50, 52)  # 2px speck
        self.assertEqual(lines_from_column(col, 60, min_height=8), [])

    def test_single_short_bar_single_strip(self):
        col = self._column_with_bars([(40, 120)], h=200)
        boxes = lines_from_column(col, target_px=300)
        self.assertEqual(len(boxes), 1)

    def test_parameter_validation(self):
        col = blank_page(50, 20)
        with self.assertRaises(ValueError):
            lines_from_column(col, 0)
        with self.assertRaises(ValueError):
            lines_from_column(col, 10, min_height=20)  # window <= min_height
        with self.assertRaises(ValueError):
            lines_from_column(col, 100, overshoot=0.5)


def build_acceptance_page() -> np.ndarray:
    """The task's E2E fixture: 2 columns x 3 ink-bar lines each.

    Shared with ``Model.tests.test_ocr_deploy`` and the CPU smoke run as a
    plain function (importing the TestCase would make discovery run it
    twice).
    """
    page = blank_page(480, 360)
    for x0, x1 in ((60, 120), (200, 260)):
        for y0, y1 in ((30, 140), (180, 290), (330, 440)):
            add_bar(page, x0, x1, y0, y1)
    return page


class AcceptancePageTest(unittest.TestCase):
    def test_two_columns_three_lines_each(self):
        page = build_acceptance_page()
        cols = detect_columns(page)
        self.assertEqual(len(cols), 2)
        line_counts = []
        for x0, y0, x1, y1 in cols:
            strips = lines_from_column(page[y0:y1, x0:x1], target_px=120)
            line_counts.append(len(strips))
        self.assertEqual(line_counts, [3, 3])


class DetectLineColumnsWrapperTest(unittest.TestCase):
    """The annotation-pack consumer's contract must survive the refactor."""

    def test_requires_pil_image(self):
        with self.assertRaises(TypeError):
            detect_line_columns(blank_page(10, 10))

    def test_whole_page_fallback_on_blank(self):
        img = Image.new("L", (90, 60), 255)
        self.assertEqual(detect_line_columns(img), [(0, 0, 90, 60)])

    def test_absolute_density_threshold(self):
        # One dark pixel per column of height 100 = density 0.01: below the
        # 0.02 default -> fallback; a 3-dark-pixel column (0.03) is detected.
        arr = blank_page(100, 60)
        arr[0, 20:40] = 0  # density 0.01 over x in [20, 40)
        img = Image.fromarray(arr, mode="L")
        self.assertEqual(detect_line_columns(img), [(0, 0, 60, 100)])
        arr[1:3, 20:40] = 0  # now density 0.03
        img = Image.fromarray(arr, mode="L")
        boxes = detect_line_columns(img)
        self.assertEqual(len(boxes), 1)
        self.assertEqual((boxes[0][0], boxes[0][2]), (20, 40))

    def test_gap_merge_uses_inclusive_pixel_gap(self):
        # Historical rule: gaps <= min_gap_px merge. 3px gap with the
        # default min_gap_px=3 must merge into one column.
        arr = blank_page(50, 60)
        arr[:, 10:20] = 0
        arr[:, 23:33] = 0
        img = Image.fromarray(arr, mode="L")
        boxes = detect_line_columns(img)
        self.assertEqual(len(boxes), 1)
        self.assertEqual((boxes[0][0], boxes[0][2]), (10, 33))


class ChunkColumnByHeightTest(unittest.TestCase):
    def test_short_column_unchanged(self):
        self.assertEqual(
            chunk_column_by_height((5, 0, 25, 400), max_chunk_px=900),
            [(5, 0, 25, 400)],
        )

    def test_tall_column_chunked_with_short_tail(self):
        chunks = chunk_column_by_height((0, 0, 10, 2100), max_chunk_px=900)
        self.assertEqual(
            chunks, [(0, 0, 10, 900), (0, 900, 10, 1800), (0, 1800, 10, 2100)]
        )

    def test_rejects_bad_box(self):
        with self.assertRaises(ValueError):
            chunk_column_by_height((0, 10, 5, 10))


if __name__ == "__main__":
    unittest.main()
