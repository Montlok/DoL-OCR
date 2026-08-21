# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import unittest

from Model.omvt.native_planner import (
    NativeMacroWindow,
    plan_native_macro_windows,
    prove_native_macro_window_coverage,
    raw_patch_token_count,
)


PATCH_SHAPES = {
    "vertical": (8, 4),
    "horizontal": (4, 8),
    "square": (4, 4),
    "layout": (16, 16),
}


class NativeMacroWindowPlannerTest(unittest.TestCase):
    def test_budget_fit_returns_canonical_identity_plan(self):
        height, width = 12, 10
        full_tokens = raw_patch_token_count(height, width, PATCH_SHAPES)
        plan = plan_native_macro_windows(
            height=height,
            width=width,
            patch_shapes=PATCH_SHAPES,
            max_raw_patch_tokens=full_tokens,
            max_windows=1,
            halo_px=7,
        )
        self.assertTrue(plan.identity)
        self.assertEqual(len(plan.windows), 1)
        self.assertEqual(plan.windows[0].asset_bbox_yxxy, (0, 0, height, width))
        self.assertEqual(plan.windows[0].ownership_bbox_yxxy, (0, 0, height, width))
        self.assertEqual(plan.windows[0].halo_tlbr, (0, 0, 0, 0))
        self.assertEqual(plan.windows[0].raw_patch_tokens, full_tokens)
        self.assertTrue(plan.coverage_proof.verified)
        self.assertEqual(plan.coverage_proof.ownership_gap_pixels, 0)
        self.assertEqual(plan.coverage_proof.ownership_overlap_pixels, 0)
        self.assertEqual(
            plan.canonical_sha256,
            hashlib.sha256(plan.canonical_json.encode("utf-8")).hexdigest(),
        )

    def test_over_budget_plan_is_reproducible_gapless_and_disjoint(self):
        arguments = dict(
            height=101,
            width=73,
            patch_shapes=PATCH_SHAPES,
            max_raw_patch_tokens=100,
            max_windows=256,
            halo_px=5,
        )
        first = plan_native_macro_windows(**arguments)
        second = plan_native_macro_windows(**arguments)
        self.assertEqual(first, second)
        self.assertEqual(first.canonical_sha256, second.canonical_sha256)
        self.assertFalse(first.identity)
        self.assertGreater(len(first.windows), 1)
        self.assertTrue(first.coverage_proof.verified)
        self.assertEqual(first.coverage_proof.ownership_gap_pixels, 0)
        self.assertEqual(first.coverage_proof.ownership_overlap_pixels, 0)
        self.assertEqual(first.coverage_proof.asset_window_gap_pixels, 0)
        self.assertLessEqual(
            first.coverage_proof.max_window_raw_patch_tokens,
            arguments["max_raw_patch_tokens"],
        )

        ownership = [[0] * arguments["width"] for _ in range(arguments["height"])]
        processing = [[0] * arguments["width"] for _ in range(arguments["height"])]
        for expected_index, window in enumerate(first.windows):
            self.assertEqual(window.index, expected_index)
            ay0, ax0, ay1, ax1 = window.asset_bbox_yxxy
            oy0, ox0, oy1, ox1 = window.ownership_bbox_yxxy
            self.assertTrue(ay0 <= oy0 < oy1 <= ay1)
            self.assertTrue(ax0 <= ox0 < ox1 <= ax1)
            self.assertEqual(
                window.halo_tlbr,
                (oy0 - ay0, ox0 - ax0, ay1 - oy1, ax1 - ox1),
            )
            self.assertTrue(all(0 <= halo <= arguments["halo_px"] for halo in window.halo_tlbr))
            self.assertEqual(
                window.raw_patch_tokens,
                raw_patch_token_count(ay1 - ay0, ax1 - ax0, PATCH_SHAPES),
            )
            self.assertLessEqual(
                window.raw_patch_tokens,
                arguments["max_raw_patch_tokens"],
            )
            for y in range(oy0, oy1):
                for x in range(ox0, ox1):
                    ownership[y][x] += 1
            for y in range(ay0, ay1):
                for x in range(ax0, ax1):
                    processing[y][x] += 1

        self.assertTrue(all(count == 1 for row in ownership for count in row))
        self.assertTrue(all(count >= 1 for row in processing for count in row))
        self.assertTrue(any(count > 1 for row in processing for count in row))

        changed_halo = plan_native_macro_windows(**{**arguments, "halo_px": 4})
        self.assertNotEqual(first.canonical_sha256, changed_halo.canonical_sha256)

    def test_minimum_halo_window_fails_and_proof_rejects_overlap(self):
        minimum_tokens = raw_patch_token_count(21, 21, PATCH_SHAPES)
        with self.assertRaisesRegex(ValueError, "minimum one-pixel"):
            plan_native_macro_windows(
                height=100,
                width=100,
                patch_shapes=PATCH_SHAPES,
                max_raw_patch_tokens=minimum_tokens - 1,
                max_windows=256,
                halo_px=10,
            )

        with self.assertRaisesRegex(ValueError, "max_windows"):
            plan_native_macro_windows(
                height=101,
                width=73,
                patch_shapes=PATCH_SHAPES,
                max_raw_patch_tokens=100,
                max_windows=1,
                halo_px=5,
            )

        full_tokens = raw_patch_token_count(10, 10, PATCH_SHAPES)
        duplicate_windows = (
            NativeMacroWindow(
                index=0,
                asset_bbox_yxxy=(0, 0, 10, 10),
                ownership_bbox_yxxy=(0, 0, 10, 10),
                halo_tlbr=(0, 0, 0, 0),
                raw_patch_tokens=full_tokens,
            ),
            NativeMacroWindow(
                index=1,
                asset_bbox_yxxy=(0, 0, 10, 10),
                ownership_bbox_yxxy=(0, 0, 10, 10),
                halo_tlbr=(0, 0, 0, 0),
                raw_patch_tokens=full_tokens,
            ),
        )
        proof = prove_native_macro_window_coverage(
            asset_hw=(10, 10),
            windows=duplicate_windows,
            patch_shapes=PATCH_SHAPES,
            max_raw_patch_tokens=full_tokens,
        )
        self.assertFalse(proof.verified)
        self.assertEqual(proof.ownership_gap_pixels, 0)
        self.assertEqual(proof.ownership_overlap_pixels, 100)


if __name__ == "__main__":
    unittest.main()
