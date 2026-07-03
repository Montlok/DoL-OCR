# -*- coding: utf-8 -*-

"""Unit tests for the pure deployment helpers of ``scripts.ocr_infer`` and
``scripts.eval_l2_deploy`` (no checkpoint, no generation).

The generation path itself is proven by ``scripts.eval_vlm_ocr`` (reused, not
reimplemented) plus the CPU end-to-end smoke documented in the delivery
report; these tests pin the deployment-specific plumbing around it: prompt
layout, reading-order flattening, the numpy->PNG-bytes letterbox shim, column
synthesis, and row grouping.
"""

from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from Model.config import (
    BOS_ID,
    EOS_ID,
    IMAGE_END_ID,
    IMAGE_PATCH_ID,
    IMAGE_START_ID,
)
from Model.ocr.data import build_ocr_row
from Model.tests.test_ocr_segment import build_acceptance_page
from scripts.eval_l2_deploy import group_rows, per_sample_grapheme_cer, stack_lines
from scripts.ocr_infer import (
    build_inference_prompt,
    order_columns,
    segment_page,
    strip_to_letterboxed,
)


class BuildInferencePromptTest(unittest.TestCase):
    def test_layout_matches_row_contract(self):
        n = 7
        prompt = build_inference_prompt(n)
        self.assertEqual(len(prompt), n + 3)
        self.assertEqual(prompt[0], BOS_ID)
        self.assertEqual(prompt[1], IMAGE_START_ID)
        self.assertEqual(prompt[2 : 2 + n], [IMAGE_PATCH_ID] * n)
        self.assertEqual(prompt[-1], IMAGE_END_ID)
        # the throwaway target must not leak into the prompt
        self.assertNotIn(EOS_ID, prompt)

    def test_prompt_equals_training_row_prompt(self):
        # The masked prompt of a real training row (same n, no instruction)
        # must be byte-identical to the inference prompt.
        n = 5
        row = build_ocr_row(
            [400, 401],
            n,
            "img.png",
            bos_id=BOS_ID,
            image_start_id=IMAGE_START_ID,
            image_patch_id=IMAGE_PATCH_ID,
            image_end_id=IMAGE_END_ID,
            eos_id=EOS_ID,
        )
        prompt_len = sum(1 for x in row["labels"] if x == -100)
        self.assertEqual(build_inference_prompt(n), row["input_ids"][:prompt_len])


class OrderColumnsTest(unittest.TestCase):
    BOXES = [(120, 0, 160, 300), (20, 0, 60, 300), (220, 0, 260, 300)]

    def test_ltr_sorts_ascending_x(self):
        self.assertEqual(
            [b[0] for b in order_columns(self.BOXES, "ltr")], [20, 120, 220]
        )

    def test_rtl_reverses(self):
        self.assertEqual(
            [b[0] for b in order_columns(self.BOXES, "rtl")], [220, 120, 20]
        )


class StripToLetterboxedTest(unittest.TestCase):
    def test_square_geometry_and_centered_content(self):
        # 20x100 dark strip -> 64px letterboxed square: content centered,
        # corners white (the strip is pasted on a white max-side canvas).
        strip = np.zeros((100, 20), dtype=np.uint8)
        out = strip_to_letterboxed(strip, 64)
        self.assertEqual(out.mode, "L")
        self.assertEqual(out.size, (64, 64))
        arr = np.asarray(out)
        self.assertEqual(int(arr[0, 0]), 255)
        self.assertEqual(int(arr[0, -1]), 255)
        center_row = arr[32]
        self.assertLess(int(center_row[32]), 64)  # dark strip content
        self.assertEqual(int(center_row[0]), 255)  # white margin


class SegmentPageTest(unittest.TestCase):
    def test_reading_order_flattening_on_acceptance_page(self):
        page = build_acceptance_page()
        records = segment_page(page, target_line_px=120)
        self.assertEqual(len(records), 6)
        self.assertEqual(
            [(r["column"], r["line"]) for r in records],
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
        )
        # ltr default: column 0 is the left one
        self.assertLess(records[0]["box"][0], records[3]["box"][0])
        # boxes are page-absolute and contain their ink bars
        for r in records:
            x0, y0, x1, y1 = r["box"]
            self.assertTrue(0 <= x0 < x1 <= page.shape[1])
            self.assertTrue(0 <= y0 < y1 <= page.shape[0])
            self.assertEqual(r["strip"].shape, (y1 - y0, x1 - x0))
            self.assertTrue((r["strip"] < 200).any())

    def test_rtl_reverses_column_indices(self):
        page = build_acceptance_page()
        records = segment_page(page, target_line_px=120, column_order="rtl")
        # column 0 is now the right one
        self.assertGreater(records[0]["box"][0], records[3]["box"][0])

    def test_blank_page_yields_no_records(self):
        blank = np.full((200, 200), 255, dtype=np.uint8)
        self.assertEqual(segment_page(blank), [])


class GroupRowsTest(unittest.TestCase):
    def test_full_groups_and_dropped_tail(self):
        rows = list(range(11))
        groups, dropped = group_rows(rows, 3)
        self.assertEqual(groups, [[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        self.assertEqual(dropped, 2)

    def test_k_must_be_positive(self):
        with self.assertRaises(ValueError):
            group_rows([1, 2], 0)


class StackLinesTest(unittest.TestCase):
    def test_geometry_gap_and_centering(self):
        a = Image.fromarray(np.zeros((30, 40), dtype=np.uint8), mode="L")
        b = Image.fromarray(np.zeros((20, 20), dtype=np.uint8), mode="L")
        col = stack_lines([a, b], gap_px=5)
        self.assertEqual(col.shape, (55, 40))
        self.assertTrue((col[30:35, :] == 255).all())  # gap rows white
        self.assertTrue((col[35:, 10:30] == 0).all())  # b centered (40-20)//2
        self.assertTrue((col[35:, :10] == 255).all())

    def test_validation(self):
        with self.assertRaises(ValueError):
            stack_lines([], 4)
        img = Image.new("L", (5, 5), 0)
        with self.assertRaises(ValueError):
            stack_lines([img], -1)


class PerSampleGraphemeCerTest(unittest.TestCase):
    def test_identity_and_simple_error(self):
        self.assertEqual(per_sample_grapheme_cer("abc", "abc"), 0.0)
        self.assertAlmostEqual(per_sample_grapheme_cer("abd", "abc"), 1 / 3)
        # empty pred vs non-empty ref: full deletion penalty
        self.assertEqual(per_sample_grapheme_cer("", "ab"), 1.0)


if __name__ == "__main__":
    unittest.main()
