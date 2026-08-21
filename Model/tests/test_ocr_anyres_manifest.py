# -*- coding: utf-8 -*-

from __future__ import annotations

import copy
import hashlib
import unittest

from Model.omvt.native_planner import plan_native_macro_windows
from Model.ocr.near_duplicate import (
    PHASH_CONTRACT,
    near_duplicate_cluster_id,
)
from Model.posttrain.ocr_anyres_manifest import (
    ANYRES_VISUAL_CONTRACT,
    canonical_json_sha256,
    quota_counts,
    validate_anyres_assets_views_samples,
    validate_no_cross_split_leakage,
    validate_quota_60_10_20_10,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _reference(text: str) -> dict:
    return {
        "text": text,
        "utf8_sha256": _digest(text),
        "codepoints": [ord(char) for char in text],
    }


def _native_plan(width: int = 1000, height: int = 1500) -> dict:
    plan = plan_native_macro_windows(
        height=height,
        width=width,
        patch_shapes={
            "vertical": [32, 8],
            "horizontal": [8, 32],
            "square": [16, 16],
            "layout": [56, 56],
        },
        max_raw_patch_tokens=20_000,
        max_windows=1,
        halo_px=0,
    )
    return {
        "preprocess_contract_sha256": "5" * 64,
        "plan_sha256": plan.canonical_sha256,
        "payload": plan.canonical_payload(),
    }


def _cut_qa(
    *,
    view_id: str = "view-1",
    box: list[int] | None = None,
    pixel_sha256: str = "3" * 64,
) -> dict:
    native_plan = _native_plan()
    payload = {
        "contract": "connected_component_cut_qa_v2",
        "canonical_pixel_sha256": pixel_sha256,
        "plan_sha256": native_plan["plan_sha256"],
        "parameters": {
            "ink_threshold": 200,
            "min_component_pixels": 2,
            "edge_band_px": 2,
            "max_edge_ink_fraction": 0.01,
        },
        "image_hw": [1500, 1000],
        "components": [],
        "view_boxes": {view_id: box or [0, 0, 1000, 1500]},
        "complete_owners": {},
        "uncovered": [],
    }
    return {
        "coverage_proof_sha256": canonical_json_sha256(payload),
        "payload": payload,
    }


def _asset(
    asset_id: str = "asset-1",
    document_id: str = "document-1",
    capture_id: str = "capture-1",
) -> dict:
    return {
        "schema_version": 2,
        "asset_id": asset_id,
        "source_document_id": document_id,
        "source_capture_id": capture_id,
        "raw_relpath": f"raw/{asset_id}.png",
        "canonical_relpath": f"canonical/{asset_id}.png",
        "raw_sha256": "1" * 64,
        "canonical_sha256": "2" * 64,
        "pixel_sha256": "3" * 64,
        "raw_width": 1000,
        "raw_height": 1500,
        "canonical_width": 1000,
        "canonical_height": 1500,
        "native_plan": _native_plan(),
        "cut_qa": _cut_qa(),
        "near_duplicate": {
            "contract": PHASH_CONTRACT,
            "value": "0" * 16,
            "max_hamming_distance": 8,
            "cluster_id": near_duplicate_cluster_id((asset_id,)),
        },
    }


def _view(
    view_id: str = "view-1",
    asset_id: str = "asset-1",
    *,
    cut_suspect: bool = False,
    cc_crossing_count: int = 0,
    cc_uncovered_count: int = 0,
    coverage: float = 1.0,
) -> dict:
    native_plan = _native_plan()
    window = native_plan["payload"]["windows"][0]
    ay0, ax0, ay1, ax1 = window["asset_bbox_yxxy"]
    return {
        "schema_version": 2,
        "view_id": view_id,
        "asset_id": asset_id,
        "kind": "page",
        "derived_relpath": f"derived/{view_id}.png",
        "derived_width": ax1 - ax0,
        "derived_height": ay1 - ay0,
        "derived_sha256": "6" * 64,
        "box_xyxy": [ax0, ay0, ax1, ay1],
        "transform": {
            "contract": "native_macro_window_crop_v1",
            "plan_sha256": native_plan["plan_sha256"],
            "window_index": 0,
            "asset_bbox_yxxy": window["asset_bbox_yxxy"],
            "ownership_bbox_yxxy": window["ownership_bbox_yxxy"],
            "halo_tlbr": window["halo_tlbr"],
            "raw_patch_tokens": window["raw_patch_tokens"],
        },
        "pixel_sha256": "4" * 64,
        "qa": {
            "cut_suspect": cut_suspect,
            "edge_ink_fraction": {
                "top": 0.0,
                "bottom": 0.0,
                "left": 0.0,
                "right": 0.0,
            },
            "cc_crossing_count": cc_crossing_count,
            "cc_uncovered_count": cc_uncovered_count,
            "coverage": coverage,
            "coverage_proof_sha256": _cut_qa()["coverage_proof_sha256"],
            "review_status": "accepted",
            "adjudication_sha256": None,
        },
    }


def _sample(
    sample_id: str = "sample-1",
    split: str = "train",
    *,
    asset_id: str = "asset-1",
    view_id: str = "view-1",
    document_id: str = "document-1",
    capture_id: str = "capture-1",
    leakage_cluster: str = "leakage-1",
    near_duplicate_cluster: str = near_duplicate_cluster_id(("asset-1",)),
    style: str = "print",
    difficulty: str | None = None,
    writer_id: str | None = None,
    qa_state: str = "accepted",
) -> dict:
    text = "ᠪᠣᠯᠤᠨ\u180eᠠ᠃"
    return {
        "schema_version": 2,
        "sample_id": sample_id,
        "split": split,
        "visual_contract": ANYRES_VISUAL_CONTRACT,
        "asset_id": asset_id,
        "view_ids": [view_id],
        "preprocess_contract_sha256": "5" * 64,
        "reference_token_count": 12,
        "writing_mode": "vertical-lr",
        "reading_order": [view_id],
        "leakage_cluster": leakage_cluster,
        "group_ids": {
            "document_id": document_id,
            "capture_id": capture_id,
            "writer_id": writer_id,
            "near_duplicate_cluster": near_duplicate_cluster,
        },
        "style": style,
        "font_id": "NotoSansMongolian" if style == "print" else None,
        "writer_id": writer_id,
        "difficulty": difficulty,
        "reference_raw": _reference(text),
        "reference_model": _reference(text),
        "qa_state": qa_state,
    }


class AnyresManifestTest(unittest.TestCase):
    def test_public_split_contract_is_exact(self):
        for split in (
            "train",
            "sft_validation",
            "kl_selection",
            "formal_monitor",
        ):
            with self.subTest(split=split):
                normalized, quarantine = validate_anyres_assets_views_samples(
                    [_asset()],
                    [_view()],
                    [_sample(split=split)],
                )
                self.assertEqual(normalized["samples"][0]["split"], split)
                self.assertEqual(quarantine, [])
        for legacy in ("val", "validation", "test", "golden"):
            with self.subTest(legacy=legacy), self.assertRaisesRegex(
                ValueError,
                "split",
            ):
                validate_anyres_assets_views_samples(
                    [_asset()],
                    [_view()],
                    [_sample(split=legacy)],
                )

    def test_valid_rows_are_normalized_without_unicode_normalization(self):
        asset = _asset()
        asset["raw_sha256"] = asset["raw_sha256"].upper()
        sample = _sample(split="sft_validation")

        normalized, quarantine = validate_anyres_assets_views_samples(
            [asset], [_view()], [sample]
        )

        self.assertEqual(normalized["schema_version"], 2)
        self.assertEqual(normalized["samples"][0]["split"], "sft_validation")
        self.assertIn("\u180e", normalized["samples"][0]["reference_raw"]["text"])
        self.assertEqual(normalized["assets"][0]["raw_sha256"], "1" * 64)
        self.assertEqual(quarantine, [])
        self.assertEqual(
            canonical_json_sha256({"z": "ᠠ", "a": 1}),
            canonical_json_sha256({"a": 1, "z": "ᠠ"}),
        )

    def test_cut_crossing_and_zero_coverage_quarantine_view_and_sample(self):
        normalized, quarantine = validate_anyres_assets_views_samples(
            [_asset()],
            [
                _view(
                    cut_suspect=True,
                    cc_crossing_count=1,
                    cc_uncovered_count=1,
                    coverage=0,
                )
            ],
            [_sample()],
        )

        self.assertEqual(normalized["views"], [])
        self.assertEqual(normalized["samples"], [])
        self.assertEqual(
            quarantine[0],
            {
                "entity_type": "view",
                "entity_id": "view-1",
                "reasons": [
                    "cc_uncovered",
                    "cut_suspect",
                    "incomplete_coverage",
                ],
            },
        )
        self.assertEqual(
            quarantine[1]["reasons"],
            [
                "parent_view:view-1:cc_uncovered",
                "parent_view:view-1:cut_suspect",
                "parent_view:view-1:incomplete_coverage",
            ],
        )

    def test_manual_adjudication_can_clear_suspect_but_not_uncovered_components(self):
        reviewed = _view(cut_suspect=True)
        reviewed["qa"]["review_status"] = "manual_accepted"
        reviewed["qa"]["adjudication_sha256"] = "a" * 64
        normalized, quarantine = validate_anyres_assets_views_samples(
            [_asset()], [reviewed], [_sample()]
        )
        self.assertEqual(len(normalized["samples"]), 1)
        self.assertEqual(quarantine, [])

        reviewed["qa"]["cc_crossing_count"] = 1
        reviewed["qa"]["cc_uncovered_count"] = 1
        normalized, quarantine = validate_anyres_assets_views_samples(
            [_asset()], [reviewed], [_sample()]
        )
        self.assertEqual(normalized["samples"], [])
        self.assertIn("cc_uncovered", quarantine[0]["reasons"])

    def test_reference_hash_codepoints_and_invalid_unicode_fail_closed(self):
        bad_hash = _sample()
        bad_hash["reference_raw"]["utf8_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "does not match text bytes"):
            validate_anyres_assets_views_samples(
                [_asset()], [_view()], [bad_hash]
            )

        bad_codepoints = _sample()
        bad_codepoints["reference_model"]["codepoints"] = [0x1820]
        with self.assertRaisesRegex(ValueError, "codepoints do not match text"):
            validate_anyres_assets_views_samples(
                [_asset()], [_view()], [bad_codepoints]
            )

        bad_unicode = _sample()
        bad_unicode["reference_raw"]["text"] = "\ud800"
        bad_unicode["reference_raw"]["codepoints"] = [0xD800]
        with self.assertRaisesRegex(ValueError, "invalid Unicode surrogates"):
            validate_anyres_assets_views_samples(
                [_asset()], [_view()], [bad_unicode]
            )

        divergent = _sample()
        divergent["reference_model"] = _reference("ᠠ")
        with self.assertRaisesRegex(ValueError, "native canonicalization"):
            validate_anyres_assets_views_samples(
                [_asset()], [_view()], [divergent]
            )

    def test_schema_hash_refs_and_near_duplicate_field_are_strict(self):
        bad_asset = _asset()
        bad_asset["schema_version"] = 1
        with self.assertRaisesRegex(ValueError, "schema_version must be 2"):
            validate_anyres_assets_views_samples(
                [bad_asset], [_view()], [_sample()]
            )

        bad_view = _view()
        bad_view["pixel_sha256"] = "not-a-sha"
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            validate_anyres_assets_views_samples(
                [_asset()], [bad_view], [_sample()]
            )

        missing_near_duplicate = _sample()
        del missing_near_duplicate["group_ids"]["near_duplicate_cluster"]
        with self.assertRaisesRegex(ValueError, "near_duplicate_cluster"):
            validate_anyres_assets_views_samples(
                [_asset()], [_view()], [missing_near_duplicate]
            )

        invented_phash = _sample()
        invented_phash["group_ids"]["phash"] = "unsupported"
        with self.assertRaisesRegex(ValueError, "extra=.*phash"):
            validate_anyres_assets_views_samples(
                [_asset()], [_view()], [invented_phash]
            )

    def test_cross_split_leakage_rejects_every_identity_class(self):
        base = _sample()
        for kind in (
            "asset",
            "document",
            "capture",
            "writer",
            "near_duplicate",
            "leakage_cluster",
        ):
            with self.subTest(kind=kind):
                first = copy.deepcopy(base)
                second = copy.deepcopy(base)
                first.update({"sample_id": "first", "split": "train"})
                second.update({"sample_id": "second", "split": "kl_selection"})
                first["asset_id"] = "asset-a"
                second["asset_id"] = "asset-b"
                first["leakage_cluster"] = "leak-a"
                second["leakage_cluster"] = "leak-b"
                first["group_ids"] = {
                    "document_id": "doc-a",
                    "capture_id": "capture-a",
                    "writer_id": "writer-a",
                    "near_duplicate_cluster": "near-a",
                }
                second["group_ids"] = {
                    "document_id": "doc-b",
                    "capture_id": "capture-b",
                    "writer_id": "writer-b",
                    "near_duplicate_cluster": "near-b",
                }
                if kind == "asset":
                    second["asset_id"] = first["asset_id"]
                elif kind == "document":
                    second["group_ids"]["document_id"] = "doc-a"
                elif kind == "capture":
                    second["group_ids"]["capture_id"] = "capture-a"
                elif kind == "writer":
                    second["group_ids"]["writer_id"] = "writer-a"
                elif kind == "near_duplicate":
                    second["group_ids"]["near_duplicate_cluster"] = "near-a"
                else:
                    second["leakage_cluster"] = "leak-a"
                with self.assertRaisesRegex(ValueError, f"for {kind}="):
                    validate_no_cross_split_leakage([first, second])

    def test_parent_group_and_style_contracts_fail_closed(self):
        group_mismatch = _sample()
        group_mismatch["group_ids"]["document_id"] = "wrong-document"
        with self.assertRaisesRegex(ValueError, "differs from parent asset"):
            validate_anyres_assets_views_samples(
                [_asset()], [_view()], [group_mismatch]
            )

        handwritten = _sample(
            style="handwritten", writer_id="writer-1", difficulty="poor"
        )
        normalized, _ = validate_anyres_assets_views_samples(
            [_asset()], [_view()], [handwritten]
        )
        self.assertEqual(normalized["samples"][0]["writer_id"], "writer-1")

        bad_handwritten = copy.deepcopy(handwritten)
        bad_handwritten["difficulty"] = None
        with self.assertRaisesRegex(ValueError, "difficulty must be"):
            validate_anyres_assets_views_samples(
                [_asset()], [_view()], [bad_handwritten]
            )

    def test_quota_is_exactly_60_10_20_10(self):
        samples = [{"style": "print", "difficulty": None} for _ in range(6)]
        samples += [{"style": "handwritten", "difficulty": "good"}]
        samples += [
            {"style": "handwritten", "difficulty": "medium"} for _ in range(2)
        ]
        samples += [{"style": "handwritten", "difficulty": "poor"}]

        expected = {
            "print": 6,
            "handwritten_good": 1,
            "handwritten_medium": 2,
            "handwritten_poor": 1,
        }
        self.assertEqual(quota_counts(samples), expected)
        self.assertEqual(validate_quota_60_10_20_10(samples), expected)

        samples[-1] = {"style": "print", "difficulty": None}
        with self.assertRaisesRegex(ValueError, "exactly 60/10/20/10"):
            validate_quota_60_10_20_10(samples)


if __name__ == "__main__":
    unittest.main()
