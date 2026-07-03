# -*- coding: utf-8 -*-

"""Unit tests for Model.ocr.segment (synthetic ink bars, no model, no fonts).

Covers the deployment segmentation contract: column detection via smoothed
ink-profile valleys (including the degenerate single-column, blank-page, and
exact min-width/min-gap boundary cases — not just the wide-margin happy
path), valley-preferring line cutting with hard-cut fallback, the
PIL-facing ``detect_line_columns`` wrapper the annotation-pack builder
consumes (whole-page fallback + absolute-density semantics preserved), and
the real-scanned-book hardening added on top: scan-border trimming
(``trim_scan_borders``) and per-crop plausibility filtering
(``is_plausible_line``, wired into ``lines_from_column``).
"""

from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from Model.ocr.segment import (
    chunk_column_by_height,
    detect_columns,
    detect_line_columns,
    is_plausible_line,
    lines_from_column,
    trim_scan_borders,
)


def blank_page(h: int, w: int) -> np.ndarray:
    return np.full((h, w), 255, dtype=np.uint8)


def add_bar(page: np.ndarray, x0: int, x1: int, y0: int, y1: int, value: int = 0):
    page[y0:y1, x0:x1] = value


def add_textlike_bar(
    page: np.ndarray, x0: int, x1: int, y0: int, y1: int, value: int = 0, stripe: int = 3
):
    """A vertically-striped ink block standing in for real glyph strokes.

    A solid-filled ``add_bar`` block has ink_fraction ~1.0 within its own
    bounding box — indistinguishable, by the ink-density band
    :func:`Model.ocr.segment.is_plausible_line` checks, from the solid
    scan-border/gutter-shadow artifact that filter exists to reject. Real
    traditional-Mongolian glyph strokes are sparse (~1.5%-55% ink fraction
    is the plausible band); this helper paints 1px ink columns every
    ``stripe`` px instead of a solid fill, landing at ink_fraction ~1/stripe
    (default stripe=3 -> ~0.33), comfortably inside the plausible band,
    while every row in ``[y0, y1)`` still carries ink (nonzero row-ink
    count), so the row-profile valley/hard-cut logic
    :func:`~Model.ocr.segment.lines_from_column` depends on is unaffected —
    only the per-crop ink *fraction* the new quality filter reads changes.
    """
    for x in range(x0, x1, stripe):
        page[y0:y1, x : x + 1] = value


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
        # add_textlike_bar (sparse stripe), not add_bar (solid fill): the
        # resulting strips are read by is_plausible_line's ink-fraction
        # check now, and a solid-filled bar (ink_fraction ~1.0) is
        # indistinguishable from the solid-bar scan artifact that check
        # exists to reject. See add_textlike_bar's docstring.
        col = blank_page(h, w)
        for y0, y1 in bars:
            add_textlike_bar(col, 5, w - 5, y0, y1)
        return col

    def test_three_bars_three_strips_in_reading_order(self):
        bars = [(20, 80), (120, 180), (220, 280)]
        col = self._column_with_bars(bars)
        boxes, rejections = lines_from_column(col, target_px=100)
        self.assertEqual(len(boxes), 3)
        for (y0, y1), (b0, b1) in zip(boxes, bars):
            # each strip covers its bar, trimmed near the bar extents
            self.assertLessEqual(y0, b0)
            self.assertGreaterEqual(y1, b1)
            self.assertLessEqual(b0 - y0, 6)
            self.assertLessEqual(y1 - b1, 6)
        # reading order: top to bottom
        self.assertEqual(boxes, sorted(boxes))
        self.assertEqual(sum(rejections.values()), 0)

    def test_blank_column_returns_empty(self):
        boxes, rejections = lines_from_column(blank_page(200, 40), 100)
        self.assertEqual(boxes, [])
        self.assertEqual(sum(rejections.values()), 0)

    def test_solid_column_hard_cuts(self):
        # No valley anywhere: hard cuts at target_px, full coverage.
        col = self._column_with_bars([(0, 500)], h=500)
        boxes, rejections = lines_from_column(col, target_px=200, pad=0, smooth=1)
        self.assertEqual(len(boxes), 3)
        self.assertEqual(boxes[0][0], 0)
        self.assertEqual(boxes[-1][1], 500)
        for (_, prev_end), (next_start, _) in zip(boxes, boxes[1:]):
            self.assertEqual(prev_end, next_start)
        self.assertEqual(sum(rejections.values()), 0)

    def test_dust_below_min_height_dropped(self):
        col = blank_page(100, 40)
        add_bar(col, 5, 35, 50, 52)  # 2px speck
        boxes, rejections = lines_from_column(col, 60, min_height=8)
        self.assertEqual(boxes, [])
        # Dropped by the pre-existing min_height dust filter (upstream of
        # is_plausible_line entirely -- the loop `continue`s on a too-short
        # trimmed strip before the plausibility check ever runs), so this
        # must NOT show up as a too_short rejection in the new dict.
        self.assertEqual(sum(rejections.values()), 0)

    def test_single_short_bar_single_strip(self):
        col = self._column_with_bars([(40, 120)], h=200)
        boxes, rejections = lines_from_column(col, target_px=300)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(sum(rejections.values()), 0)

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
    twice). Uses ``add_textlike_bar`` (sparse stripe fill), not solid
    ``add_bar``: a solid-filled bar has ink_fraction ~1.0, which
    ``is_plausible_line`` (wired into ``lines_from_column``, and so into
    ``scripts.ocr_infer.segment_page`` which this fixture also drives via
    ``Model.tests.test_ocr_deploy``) now correctly rejects as
    indistinguishable from a solid scan-border/gutter-shadow artifact — the
    stripe fill still registers as one contiguous ink column to
    ``detect_columns`` (its default ``smooth=9`` bridges the sub-smoothing-
    window gaps between stripes; verified the column count/extent is
    unchanged from the solid-fill version) while reading as plausible
    sparse "text" to the new per-crop filter.
    """
    page = blank_page(480, 360)
    for x0, x1 in ((60, 120), (200, 260)):
        for y0, y1 in ((30, 140), (180, 290), (330, 440)):
            add_textlike_bar(page, x0, x1, y0, y1)
    return page


class AcceptancePageTest(unittest.TestCase):
    def test_two_columns_three_lines_each(self):
        page = build_acceptance_page()
        cols = detect_columns(page)
        self.assertEqual(len(cols), 2)
        line_counts = []
        for x0, y0, x1, y1 in cols:
            strips, rejections = lines_from_column(page[y0:y1, x0:x1], target_px=120)
            self.assertEqual(sum(rejections.values()), 0)
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


# ===========================================================================
# Real-scanned-book hardening: is_plausible_line, trim_scan_borders
# ===========================================================================


def _textlike_crop(w: int, h: int, ink_fraction: float = 0.3) -> np.ndarray:
    """A crop with a controlled, non-degenerate ink fraction (~text-like).

    Fills every ``round(1 / ink_fraction)``-th column solid, landing close
    to the requested fraction without the all-or-nothing extremes a single
    solid rectangle or an all-white array would give — used to probe
    ``is_plausible_line``'s ink-fraction band away from its edges.
    """
    crop = np.full((h, w), 255, dtype=np.uint8)
    period = max(1, round(1 / ink_fraction))
    for x in range(0, w, period):
        crop[:, x] = 0
    return crop


class IsPlausibleLineTest(unittest.TestCase):
    def test_plausible_textlike_crop(self):
        crop = _textlike_crop(64, 640, ink_fraction=0.3)
        plausible, reason = is_plausible_line(crop, page_w=400, page_h=2000)
        self.assertTrue(plausible)
        self.assertEqual(reason, "ok")

    def test_blank_crop_rejected(self):
        crop = blank_page(640, 64)
        plausible, reason = is_plausible_line(crop)
        self.assertFalse(plausible)
        self.assertEqual(reason, "blank")

    def test_ink_fraction_lower_boundary(self):
        # 64x1000 crop: exactly 1 ink pixel below/at/above the 0.015 floor.
        w, h = 64, 1000
        n_total = w * h
        # Just under the floor: reject as blank.
        crop = blank_page(h, w).copy()
        n_below = int(0.014 * n_total)
        flat = crop.reshape(-1)
        flat[:n_below] = 0
        plausible, reason = is_plausible_line(crop, min_ink_fraction=0.015)
        self.assertFalse(plausible)
        self.assertEqual(reason, "blank")
        # Comfortably over the floor: accepted (still under the ceiling).
        crop2 = blank_page(h, w).copy()
        n_over = int(0.05 * n_total)
        flat2 = crop2.reshape(-1)
        flat2[:n_over] = 0
        plausible2, reason2 = is_plausible_line(crop2, min_ink_fraction=0.015)
        self.assertTrue(plausible2)
        self.assertEqual(reason2, "ok")

    def test_solid_bar_rejected_above_max_ink_fraction(self):
        # A fully solid crop (ink_fraction 1.0) is the scan-border/gutter-
        # shadow/photo failure mode -- must reject regardless of geometry.
        crop = np.zeros((640, 64), dtype=np.uint8)
        plausible, reason = is_plausible_line(crop)
        self.assertFalse(plausible)
        self.assertEqual(reason, "solid_bar")

    def test_ink_fraction_upper_boundary(self):
        w, h = 100, 100
        # Just under the 0.55 default ceiling: accepted.
        crop_ok = np.full((h, w), 255, dtype=np.uint8)
        crop_ok[:54, :] = 0  # ink_fraction 0.54
        plausible, reason = is_plausible_line(crop_ok, min_line_width=1, min_line_height=1)
        self.assertTrue(plausible)
        self.assertEqual(reason, "ok")
        # Just over: rejected as solid_bar.
        crop_bad = np.full((h, w), 255, dtype=np.uint8)
        crop_bad[:56, :] = 0  # ink_fraction 0.56
        plausible2, reason2 = is_plausible_line(
            crop_bad, min_line_width=1, min_line_height=1
        )
        self.assertFalse(plausible2)
        self.assertEqual(reason2, "solid_bar")

    def test_too_narrow_rejected(self):
        crop = _textlike_crop(10, 200, ink_fraction=0.3)  # width 10 < default 18
        plausible, reason = is_plausible_line(crop)
        self.assertFalse(plausible)
        self.assertEqual(reason, "too_narrow")

    def test_width_boundary_exact(self):
        # width == min_line_width (18) passes; 17 does not.
        ok = _textlike_crop(18, 200, ink_fraction=0.3)
        plausible, reason = is_plausible_line(ok, min_line_height=1)
        self.assertTrue(plausible)
        self.assertEqual(reason, "ok")
        bad = _textlike_crop(17, 200, ink_fraction=0.3)
        plausible2, reason2 = is_plausible_line(bad, min_line_height=1)
        self.assertFalse(plausible2)
        self.assertEqual(reason2, "too_narrow")

    def test_too_wide_relative_to_page_rejected(self):
        # width 200 > 0.35 * page_w(400) = 140 -> reject as too_wide, even
        # though ink_fraction/height are otherwise plausible.
        crop = _textlike_crop(200, 300, ink_fraction=0.3)
        plausible, reason = is_plausible_line(crop, page_w=400, page_h=1000)
        self.assertFalse(plausible)
        self.assertEqual(reason, "too_wide")

    def test_too_wide_check_skipped_without_page_w(self):
        # Same crop as above, but page_w omitted: the width-vs-page-width
        # check must not fire (no page geometry to compare against).
        crop = _textlike_crop(200, 300, ink_fraction=0.3)
        plausible, reason = is_plausible_line(crop)
        self.assertTrue(plausible)
        self.assertEqual(reason, "ok")

    def test_too_short_rejected(self):
        crop = _textlike_crop(64, 40, ink_fraction=0.3)  # height 40 < default 60
        plausible, reason = is_plausible_line(crop)
        self.assertFalse(plausible)
        self.assertEqual(reason, "too_short")

    def test_height_boundary_exact(self):
        ok = _textlike_crop(64, 60, ink_fraction=0.3)
        plausible, reason = is_plausible_line(ok)
        self.assertTrue(plausible)
        self.assertEqual(reason, "ok")
        bad = _textlike_crop(64, 59, ink_fraction=0.3)
        plausible2, reason2 = is_plausible_line(bad)
        self.assertFalse(plausible2)
        self.assertEqual(reason2, "too_short")

    def test_rejects_non_2d_input(self):
        with self.assertRaises(ValueError):
            is_plausible_line(np.zeros((4, 4, 3), dtype=np.uint8))

    def test_parameter_validation(self):
        crop = _textlike_crop(64, 200)
        with self.assertRaises(ValueError):
            is_plausible_line(crop, min_ink_fraction=0.6, max_ink_fraction=0.5)
        with self.assertRaises(ValueError):
            is_plausible_line(crop, min_line_width=0)
        with self.assertRaises(ValueError):
            is_plausible_line(crop, max_width_frac=0.0)
        with self.assertRaises(ValueError):
            is_plausible_line(crop, min_line_height=0)


class TrimScanBordersTest(unittest.TestCase):
    def test_no_borders_is_noop(self):
        page = blank_page(200, 150)
        add_bar(page, 60, 90, 50, 150)  # ordinary interior content
        cropped, offset = trim_scan_borders(page)
        self.assertEqual(offset, (0, 0))
        np.testing.assert_array_equal(cropped, page)

    def test_solid_border_all_sides_trimmed(self):
        page = blank_page(300, 220)
        add_bar(page, 0, 220, 0, 10)  # top
        add_bar(page, 0, 220, 290, 300)  # bottom
        add_bar(page, 0, 10, 0, 300)  # left
        add_bar(page, 210, 220, 0, 300)  # right
        add_bar(page, 80, 140, 100, 200)  # interior content, must survive
        cropped, (y_off, x_off) = trim_scan_borders(page)
        self.assertGreaterEqual(y_off, 10)
        self.assertGreaterEqual(x_off, 10)
        # interior content box, translated into cropped-local coords, is
        # still present and still fully ink (nothing was clipped into it).
        cy0, cy1 = 100 - y_off, 200 - y_off
        cx0, cx1 = 80 - x_off, 140 - x_off
        self.assertTrue((cropped[cy0:cy1, cx0:cx1] == 0).all())

    def test_asymmetric_border_only_trims_qualifying_side(self):
        page = blank_page(150, 150)
        add_bar(page, 0, 150, 0, 20)  # top border only
        cropped, (y_off, x_off) = trim_scan_borders(page)
        self.assertGreaterEqual(y_off, 20)
        self.assertEqual(x_off, 0)  # untouched sides must not be trimmed

    def test_dense_page_capped_not_eaten(self):
        # A fully-solid-ink page (worst case: everything looks like border)
        # must still be capped at max_trim_frac per side, not trimmed away.
        page = np.zeros((100, 100), dtype=np.uint8)
        cropped, (y_off, x_off) = trim_scan_borders(page, max_trim_frac=0.15)
        self.assertLessEqual(y_off, int(100 * 0.15) + 3)  # + default safety_margin
        self.assertLessEqual(x_off, int(100 * 0.15) + 3)
        self.assertGreater(cropped.shape[0], 0)
        self.assertGreater(cropped.shape[1], 0)

    def test_parameter_validation(self):
        page = blank_page(50, 50)
        with self.assertRaises(ValueError):
            trim_scan_borders(page, border_ink_frac=0.0)
        with self.assertRaises(ValueError):
            trim_scan_borders(page, border_ink_frac=1.5)
        with self.assertRaises(ValueError):
            trim_scan_borders(page, safety_margin=-1)
        with self.assertRaises(ValueError):
            trim_scan_borders(page, max_trim_frac=0.5)


class RealScanAcceptanceTest(unittest.TestCase):
    """The task's E2E synthetic fixture: borders + blank margins + one solid
    artifact bar + real (sparse) text bars, run through the full
    trim -> detect -> lines_from_column pipeline."""

    def _build_page(self) -> np.ndarray:
        # 700 (w) x 900 (h): a page-scale canvas, wide enough to keep the
        # solid artifact bar's x-range clearly separated from both text
        # columns' x-ranges -- detect_columns's default smooth=9 bridges
        # anything within its window, so an artifact bar placed close enough
        # to touch a text column's smoothed profile would merge into one
        # giant "column" rather than exercising the reject-the-artifact path
        # this test is actually after (verified empirically before fixing
        # these coordinates).
        page = blank_page(900, 700)
        # Solid black scan borders, all four sides.
        add_bar(page, 0, 700, 0, 12)
        add_bar(page, 0, 700, 888, 900)
        add_bar(page, 0, 12, 0, 900)
        add_bar(page, 688, 700, 0, 900)
        # Two real text columns: sparse (dash-like) vertical ink blocks,
        # each with two "lines" separated by a valley, mimicking real
        # traditional-Mongolian glyph density (not solid fill).
        for x0, x1 in ((60, 124), (200, 264)):
            for y0, y1 in ((40, 180), (220, 360)):
                add_textlike_bar(page, x0, x1, y0, y1)
        # A wide, solid artifact bar (e.g. a gutter shadow or a photo the
        # detector should never treat as a text line) -- solid fill, x-range
        # well clear of both text columns above.
        add_bar(page, 450, 620, 100, 500)
        return page

    def test_borders_trimmed_bar_rejected_text_kept(self):
        page = self._build_page()
        trimmed, (y_off, x_off) = trim_scan_borders(page)
        self.assertGreater(y_off, 0)
        self.assertGreater(x_off, 0)
        # Border trim must not eat the page content: the leftmost text
        # column started at x=60 in original coordinates and must still be
        # present (with margin) in the trimmed page.
        self.assertLess(x_off, 60)

        cols = detect_columns(trimmed)
        # detect_columns only looks at x-ink-density, not per-crop
        # plausibility, so it reports 3 "columns" here: the two real sparse
        # text columns AND the wide solid artifact bar (their x-ranges are
        # deliberately kept clear of each other in _build_page so they don't
        # smear into one merged run). The artifact is then caught downstream
        # by lines_from_column's is_plausible_line filtering, which is
        # exactly what the rest of this test verifies.
        self.assertEqual(len(cols), 3)

        kept_total = 0
        rejected_total: dict[str, int] = {}
        trimmed_h, trimmed_w = trimmed.shape
        for x0, y0, x1, y1 in cols:
            column = trimmed[y0:y1, x0:x1]
            boxes, rejections = lines_from_column(
                column, target_px=200, page_w=trimmed_w, page_h=trimmed_h
            )
            kept_total += len(boxes)
            for reason, n in rejections.items():
                rejected_total[reason] = rejected_total.get(reason, 0) + n

        # The two real text columns contribute kept lines (2 "lines" each,
        # separated by the valley between the two add_textlike_bar blocks);
        # the solid artifact-bar column contributes zero kept lines -- its
        # full page-height column gets split into candidate chunks by
        # lines_from_column same as any other column, but every one of them
        # is solid ink (ink_fraction 1.0) and rejected as solid_bar.
        self.assertEqual(kept_total, 4)
        self.assertGreater(rejected_total.get("solid_bar", 0), 0)


if __name__ == "__main__":
    unittest.main()
