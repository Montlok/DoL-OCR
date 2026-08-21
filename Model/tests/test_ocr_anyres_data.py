# -*- coding: utf-8 -*-

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from Model.ocr.cut_qa import analyze_cut_qa
from Model.ocr.near_duplicate import (
    PHASH_CONTRACT,
    near_duplicate_cluster_id,
    perceptual_hash,
)
from Model.omvt.native_planner import plan_native_macro_windows
from Model.posttrain.ocr_anyres_data import (
    AnyresOCRDataset,
    load_anyres_jsonl,
    rgba_pixel_sha256,
)
from Model.posttrain.ocr_anyres_manifest import canonical_json_sha256


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reference(text: str) -> dict:
    return {
        "text": text,
        "utf8_sha256": _sha256(text.encode("utf-8")),
        "codepoints": [ord(char) for char in text],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


class AnyresDatasetFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        for directory in ("raw", "canonical", "derived"):
            (root / directory).mkdir(parents=True, exist_ok=True)
        self.raw = root / "raw" / "asset.png"
        self.canonical = root / "canonical" / "asset.png"
        self.view_a = root / "derived" / "view-a.png"
        self.view_b = root / "derived" / "view-b.png"
        source = Image.new("RGB", (4, 6), (240, 240, 240))
        source.paste((210, 220, 230), (0, 0, 4, 3))
        source.paste((230, 220, 210), (0, 3, 4, 6))
        source.save(self.raw)
        source.save(self.canonical)
        Image.new("RGB", (4, 3), (210, 220, 230)).save(self.view_a)
        Image.new("RGB", (4, 3), (230, 220, 210)).save(self.view_b)

        self.plan = plan_native_macro_windows(
            height=6,
            width=4,
            patch_shapes={
                "vertical": [1, 1],
                "horizontal": [1, 1],
                "square": [1, 1],
                "layout": [1, 1],
            },
            max_raw_patch_tokens=64,
            max_windows=2,
            halo_px=0,
        )

        raw_bytes = self.raw.read_bytes()
        canonical_bytes = self.canonical.read_bytes()
        canonical_pixel, _ = rgba_pixel_sha256(canonical_bytes)
        canonical_phash = perceptual_hash(canonical_bytes)
        view_boxes = {}
        for index, view_id in enumerate(("view-a", "view-b")):
            ay0, ax0, ay1, ax1 = self.plan.windows[index].asset_bbox_yxxy
            view_boxes[view_id] = [ax0, ay0, ax1, ay1]
        self.cut_evidence = analyze_cut_qa(
            np.asarray(source.convert("L"), dtype=np.uint8),
            view_boxes,
            canonical_pixel_sha256=canonical_pixel,
            plan_sha256=self.plan.canonical_sha256,
        )
        self.asset = {
            "schema_version": 2,
            "asset_id": "asset-1",
            "source_document_id": "document-1",
            "source_capture_id": "capture-1",
            "raw_relpath": "raw/asset.png",
            "canonical_relpath": "canonical/asset.png",
            "raw_sha256": _sha256(raw_bytes),
            "canonical_sha256": _sha256(canonical_bytes),
            "pixel_sha256": canonical_pixel,
            "raw_width": 4,
            "raw_height": 6,
            "canonical_width": 4,
            "canonical_height": 6,
            "native_plan": {
                "preprocess_contract_sha256": "5" * 64,
                "plan_sha256": self.plan.canonical_sha256,
                "payload": self.plan.canonical_payload(),
            },
            "cut_qa": {
                "coverage_proof_sha256": (
                    self.cut_evidence.coverage_proof_sha256
                ),
                "payload": self.cut_evidence.proof_payload,
            },
            "near_duplicate": {
                "contract": PHASH_CONTRACT,
                "value": canonical_phash,
                "max_hamming_distance": 8,
                "cluster_id": near_duplicate_cluster_id(("asset-1",)),
            },
        }
        self.views = [
            self._view("view-a", "derived/view-a.png", self.view_a, 0),
            self._view("view-b", "derived/view-b.png", self.view_b, 1),
        ]
        text = "ᠪᠣᠯᠤᠨ\u180eᠠ᠃"
        self.sample = {
            "schema_version": 2,
            "sample_id": "sample-1",
            "split": "train",
            "visual_contract": "dol_ocr_anyres_v2",
            "asset_id": "asset-1",
            "view_ids": ["view-a", "view-b"],
            "preprocess_contract_sha256": "5" * 64,
            "reference_token_count": 9,
            "writing_mode": "vertical-lr",
            "reading_order": ["view-b", "view-a"],
            "leakage_cluster": "leakage-1",
            "group_ids": {
                "document_id": "document-1",
                "capture_id": "capture-1",
                "writer_id": None,
                "near_duplicate_cluster": near_duplicate_cluster_id(
                    ("asset-1",)
                ),
            },
            "style": "print",
            "font_id": "NotoSansMongolian",
            "writer_id": None,
            "difficulty": None,
            "reference_raw": _reference(text),
            "reference_model": _reference(text),
            "qa_state": "accepted",
        }
        self.assets_manifest = root / "assets.jsonl"
        self.views_manifest = root / "views.jsonl"
        self.samples_manifest = root / "samples.jsonl"
        self.write()

    def _view(
        self, view_id: str, relpath: str, path: Path, window_index: int
    ) -> dict:
        data = path.read_bytes()
        pixel_sha256, size = rgba_pixel_sha256(data)
        window = self.plan.canonical_payload()["windows"][window_index]
        ay0, ax0, ay1, ax1 = window["asset_bbox_yxxy"]
        qa_by_id = {
            evidence.view_id: evidence.manifest_qa()
            for evidence in self.cut_evidence.views
        }
        return {
            "schema_version": 2,
            "view_id": view_id,
            "asset_id": "asset-1",
            "kind": "tile",
            "derived_relpath": relpath,
            "derived_width": size[0],
            "derived_height": size[1],
            "derived_sha256": _sha256(data),
            "box_xyxy": [ax0, ay0, ax1, ay1],
            "transform": {
                "contract": "native_macro_window_crop_v1",
                "plan_sha256": self.plan.canonical_sha256,
                "window_index": window_index,
                "asset_bbox_yxxy": window["asset_bbox_yxxy"],
                "ownership_bbox_yxxy": window["ownership_bbox_yxxy"],
                "halo_tlbr": window["halo_tlbr"],
                "raw_patch_tokens": window["raw_patch_tokens"],
            },
            "pixel_sha256": pixel_sha256,
            "qa": qa_by_id[view_id],
        }

    def write(self, samples: list[dict] | None = None) -> None:
        _write_jsonl(self.assets_manifest, [self.asset])
        _write_jsonl(self.views_manifest, self.views)
        _write_jsonl(self.samples_manifest, samples or [self.sample])

    def dataset(self, *, image_delivery: str = "bytes") -> AnyresOCRDataset:
        return AnyresOCRDataset(
            root=self.root,
            assets_manifest=self.assets_manifest,
            views_manifest=self.views_manifest,
            samples_manifest=self.samples_manifest,
            split="train",
            expected_preprocess_contract_sha256="5" * 64,
            max_decode_pixels=1_000,
            max_views_per_sample=8,
            image_delivery=image_delivery,
        )


class AnyresOCRDatasetTest(unittest.TestCase):
    def test_only_four_public_splits_are_selectable(self):
        for split in (
            "train",
            "sft_validation",
            "kl_selection",
            "formal_monitor",
        ):
            with self.subTest(split=split), tempfile.TemporaryDirectory() as temporary:
                fixture = AnyresDatasetFixture(Path(temporary))
                fixture.sample["split"] = split
                fixture.write()
                dataset = AnyresOCRDataset(
                    root=fixture.root,
                    assets_manifest=fixture.assets_manifest,
                    views_manifest=fixture.views_manifest,
                    samples_manifest=fixture.samples_manifest,
                    split=split,
                    expected_preprocess_contract_sha256="5" * 64,
                    max_decode_pixels=1_000,
                    max_views_per_sample=8,
                )
                self.assertEqual(dataset.dataset_contract["split"], split)

        with tempfile.TemporaryDirectory() as temporary:
            fixture = AnyresDatasetFixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "split must be"):
                AnyresOCRDataset(
                    root=fixture.root,
                    assets_manifest=fixture.assets_manifest,
                    views_manifest=fixture.views_manifest,
                    samples_manifest=fixture.samples_manifest,
                    split="validation",
                    expected_preprocess_contract_sha256="5" * 64,
                    max_decode_pixels=1_000,
                    max_views_per_sample=8,
                )

    def test_bytes_delivery_preserves_reading_order_and_marks_quota(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AnyresDatasetFixture(Path(temporary))
            quarantined = copy.deepcopy(fixture.sample)
            quarantined["sample_id"] = "sample-quarantined"
            quarantined["qa_state"] = "quarantine"
            fixture.write([fixture.sample, quarantined])

            dataset = fixture.dataset()
            item = dataset[0]

            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.sample_ids, ("sample-1",))
            self.assertEqual(dataset.index_for_sample_id("sample-1"), 0)
            self.assertEqual(
                dataset.get_by_sample_id("sample-1")["sample"]["sample_id"],
                "sample-1",
            )
            self.assertEqual(
                dataset.dataset_contract["contract_sha256"],
                dataset.contract_sha256,
            )
            self.assertEqual(
                dataset.dataset_contract["preprocess_contract_sha256"],
                "5" * 64,
            )
            self.assertEqual(dataset.quota_counts["print"], 1)
            self.assertEqual(dataset.quota_buckets, ("print",))
            self.assertEqual(item["quota_bucket"], "print")
            self.assertEqual(
                item["canonical_image"]["bytes"], fixture.canonical.read_bytes()
            )
            self.assertEqual(
                [entry["metadata"]["view_id"] for entry in item["derived_images"]],
                ["view-b", "view-a"],
            )
            self.assertEqual(item["sample"]["reference_token_count"], 9)
            self.assertEqual(dataset.quarantine[0]["entity_id"], "sample-quarantined")

    def test_path_delivery_returns_verified_paths_without_image_processing(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AnyresDatasetFixture(Path(temporary))
            item = fixture.dataset(image_delivery="path")[0]

            self.assertEqual(item["canonical_image"]["delivery"], "path")
            self.assertEqual(
                item["canonical_image"]["path"], str(fixture.canonical.resolve())
            )
            self.assertNotIn("bytes", item["canonical_image"])
            self.assertEqual(
                [entry["metadata"]["derived_width"] for entry in item["derived_images"]],
                [4, 4],
            )

    def test_file_sha_size_and_decoded_pixel_mismatches_fail_closed(self):
        mutators = {
            "SHA-256 mismatch": lambda fixture: fixture.asset.__setitem__(
                "canonical_sha256", "0" * 64
            ),
            "asset_hw differs": lambda fixture: fixture.asset.__setitem__(
                "canonical_width", 5
            ),
            "decoded pixel SHA-256 mismatch": lambda fixture: fixture.views[0].__setitem__(
                "pixel_sha256", "0" * 64
            ),
            "perceptual hash differs": lambda fixture: fixture.asset[
                "near_duplicate"
            ].__setitem__("value", "f" * 16),
        }
        for expected, mutate in mutators.items():
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as temporary:
                fixture = AnyresDatasetFixture(Path(temporary))
                mutate(fixture)
                fixture.write()
                with self.assertRaisesRegex(ValueError, expected):
                    fixture.dataset()

    def test_raw_canonical_and_derived_crop_relationships_are_recomputed(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AnyresDatasetFixture(Path(temporary))
            Image.new("RGB", (4, 6), (1, 2, 3)).save(fixture.canonical)
            canonical_bytes = fixture.canonical.read_bytes()
            canonical_pixel, _ = rgba_pixel_sha256(canonical_bytes)
            fixture.asset["canonical_sha256"] = _sha256(canonical_bytes)
            fixture.asset["pixel_sha256"] = canonical_pixel
            fixture.asset["near_duplicate"]["value"] = perceptual_hash(
                canonical_bytes
            )
            fixture.asset["cut_qa"]["payload"][
                "canonical_pixel_sha256"
            ] = canonical_pixel
            fixture.asset["cut_qa"]["coverage_proof_sha256"] = (
                canonical_json_sha256(fixture.asset["cut_qa"]["payload"])
            )
            for view in fixture.views:
                view["qa"]["coverage_proof_sha256"] = fixture.asset[
                    "cut_qa"
                ]["coverage_proof_sha256"]
            fixture.write()
            with self.assertRaisesRegex(ValueError, "raw EXIF canonical pixels"):
                fixture.dataset()

        with tempfile.TemporaryDirectory() as temporary:
            fixture = AnyresDatasetFixture(Path(temporary))
            Image.new("RGB", (4, 3), (90, 91, 92)).save(fixture.view_a)
            data = fixture.view_a.read_bytes()
            pixel_sha, _ = rgba_pixel_sha256(data)
            fixture.views[0]["derived_sha256"] = _sha256(data)
            fixture.views[0]["pixel_sha256"] = pixel_sha
            fixture.write()
            with self.assertRaisesRegex(ValueError, "no-resize canonical crop"):
                fixture.dataset()

        with tempfile.TemporaryDirectory() as temporary:
            fixture = AnyresDatasetFixture(Path(temporary))
            exif = Image.Exif()
            exif[274] = 6
            Image.new("RGB", (4, 3), (210, 220, 230)).save(
                fixture.view_a,
                exif=exif,
            )
            data = fixture.view_a.read_bytes()
            pixel_sha, _ = rgba_pixel_sha256(data)
            fixture.views[0]["derived_sha256"] = _sha256(data)
            fixture.views[0]["pixel_sha256"] = pixel_sha
            fixture.write()
            with self.assertRaisesRegex(ValueError, "EXIF orientation"):
                fixture.dataset()

    def test_preprocess_contract_mismatch_fails_before_delivery(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AnyresDatasetFixture(Path(temporary))
            fixture.sample["preprocess_contract_sha256"] = "6" * 64
            fixture.write()
            with self.assertRaisesRegex(
                ValueError,
                "preprocess_contract_sha256 differs",
            ):
                fixture.dataset()

    def test_decode_and_view_count_budgets_fail_before_delivery(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AnyresDatasetFixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "max_decode_pixels"):
                AnyresOCRDataset(
                    root=fixture.root,
                    assets_manifest=fixture.assets_manifest,
                    views_manifest=fixture.views_manifest,
                    samples_manifest=fixture.samples_manifest,
                    split="train",
                    expected_preprocess_contract_sha256="5" * 64,
                    max_decode_pixels=8,
                    max_views_per_sample=8,
                )
            with self.assertRaisesRegex(ValueError, "max_views_per_sample"):
                AnyresOCRDataset(
                    root=fixture.root,
                    assets_manifest=fixture.assets_manifest,
                    views_manifest=fixture.views_manifest,
                    samples_manifest=fixture.samples_manifest,
                    split="train",
                    expected_preprocess_contract_sha256="5" * 64,
                    max_decode_pixels=1_000,
                    max_views_per_sample=1,
                )

    def test_post_admission_mutation_is_detected_before_delivery(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AnyresDatasetFixture(Path(temporary))
            dataset = fixture.dataset()
            Image.new("RGB", (2, 4), (99, 99, 99)).save(fixture.view_a)

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                _ = dataset[0]

    def test_symlinked_data_file_and_duplicate_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AnyresDatasetFixture(Path(temporary))
            external = fixture.root / "external.png"
            Image.new("RGB", (2, 4), (1, 2, 3)).save(external)
            fixture.view_a.unlink()
            fixture.view_a.symlink_to(external)
            with self.assertRaisesRegex(ValueError, "must not traverse a symlink"):
                fixture.dataset()

            duplicate = fixture.root / "duplicate.jsonl"
            duplicate.write_text('{"schema_version":2,"schema_version":2}\n')
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                load_anyres_jsonl(duplicate)

    def test_legacy_single_view_and_bad_reading_order_are_not_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AnyresDatasetFixture(Path(temporary))
            legacy = copy.deepcopy(fixture.sample)
            legacy["view_id"] = legacy.pop("view_ids")[0]
            fixture.write([legacy])
            with self.assertRaisesRegex(ValueError, "view_ids"):
                fixture.dataset()

            bad_order = copy.deepcopy(fixture.sample)
            bad_order["reading_order"] = ["view-a"]
            fixture.write([bad_order])
            with self.assertRaisesRegex(ValueError, "exact permutation"):
                fixture.dataset()


if __name__ == "__main__":
    unittest.main()
