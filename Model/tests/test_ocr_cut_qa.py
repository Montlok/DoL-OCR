# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest
import json

import numpy as np

from Model.ocr.cut_qa import analyze_cut_qa


class OCRCutQATest(unittest.TestCase):
    def test_single_hard_cut_quarantines_crossing_component(self) -> None:
        image = np.full((20, 12), 255, dtype=np.uint8)
        image[7:13, 4:8] = 0
        evidence = analyze_cut_qa(
            image,
            {"top": [0, 0, 12, 10]},
            canonical_pixel_sha256="a" * 64,
            plan_sha256="b" * 64,
            edge_band_px=1,
        )
        view = evidence.views[0]
        self.assertEqual(view.cc_crossing_count, 1)
        self.assertEqual(view.cc_uncovered_count, 1)
        self.assertTrue(view.cut_suspect)
        self.assertEqual(view.review_status, "quarantine")
        self.assertEqual(view.coverage, 0.0)
        self.assertEqual(view.manifest_qa()["cc_uncovered_count"], 1)
        self.assertEqual(
            json.loads(json.dumps(evidence.proof_payload)),
            evidence.proof_payload,
        )

    def test_overlapping_views_resolve_boundary_crossing(self) -> None:
        image = np.full((20, 12), 255, dtype=np.uint8)
        image[7:13, 4:8] = 0
        evidence = analyze_cut_qa(
            image,
            {
                "top": [0, 0, 12, 10],
                "overlap": [0, 5, 12, 16],
                "bottom": [0, 10, 12, 20],
            },
            canonical_pixel_sha256="a" * 64,
            plan_sha256="b" * 64,
            edge_band_px=1,
        )
        by_id = {view.view_id: view for view in evidence.views}
        self.assertEqual(evidence.uncovered_component_ids, ())
        self.assertEqual(by_id["top"].coverage, 1.0)
        self.assertEqual(by_id["top"].cc_crossing_count, 1)
        self.assertEqual(by_id["top"].cc_uncovered_count, 0)
        self.assertFalse(by_id["top"].cut_suspect)
        self.assertEqual(by_id["overlap"].cc_crossing_count, 0)

    def test_edge_ink_is_quarantined_even_when_component_is_contained(self) -> None:
        image = np.full((12, 12), 255, dtype=np.uint8)
        image[0:3, 3:9] = 0
        evidence = analyze_cut_qa(
            image,
            {"page": [0, 0, 12, 12]},
            canonical_pixel_sha256="a" * 64,
            plan_sha256="b" * 64,
            edge_band_px=1,
            max_edge_ink_fraction=0.01,
        )
        view = evidence.views[0]
        self.assertGreater(view.edge_ink_fraction["top"], 0.01)
        self.assertEqual(view.cc_uncovered_count, 0)
        self.assertTrue(view.cut_suspect)

    def test_proof_binds_source_pixels_plan_and_parameters(self) -> None:
        image = np.full((8, 8), 255, dtype=np.uint8)
        base = analyze_cut_qa(
            image,
            {"page": [0, 0, 8, 8]},
            canonical_pixel_sha256="a" * 64,
            plan_sha256="b" * 64,
        )
        changed_source = analyze_cut_qa(
            image,
            {"page": [0, 0, 8, 8]},
            canonical_pixel_sha256="c" * 64,
            plan_sha256="b" * 64,
        )
        changed_parameter = analyze_cut_qa(
            image,
            {"page": [0, 0, 8, 8]},
            canonical_pixel_sha256="a" * 64,
            plan_sha256="b" * 64,
            ink_threshold=199,
        )
        self.assertNotEqual(
            base.coverage_proof_sha256,
            changed_source.coverage_proof_sha256,
        )
        self.assertNotEqual(
            base.coverage_proof_sha256,
            changed_parameter.coverage_proof_sha256,
        )


if __name__ == "__main__":
    unittest.main()
