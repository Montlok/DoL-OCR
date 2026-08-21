# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw, ImageOps

from Model.ocr.anyres_preprocess_contract import build_anyres_preprocess_contract
from Model.ocr.near_duplicate import PHASH_CONTRACT, perceptual_hash
from Model.omvt.native_planner import plan_native_macro_windows
from Model.posttrain.ocr_anyres_builder import (
    SPLIT_LOCK_KIND,
    SPLIT_POLICY_KIND,
    _assign_components,
    finalize_anyres_dataset,
    prepare_anyres_review_pack,
    split_identity_key,
    validate_split_policy,
)
from Model.posttrain.ocr_anyres_data import validate_anyres_ready_dataset
from Model.posttrain.ocr_anyres_manifest import (
    canonical_json_sha256,
    validate_anyres_assets_views_samples,
)


class _NativeEncoder:
    mode = "native"

    def __init__(self) -> None:
        self.stats = {
            "native": 0,
            "byte_fallback": 0,
            "canonicalized": 0,
        }

    def __call__(self, text: str) -> list[int]:
        self.stats["native"] += 1
        return [1000 + ord(character) for character in text]

    @staticmethod
    def decode(ids) -> str:
        return "".join(chr(int(value) - 1000) for value in ids)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _preprocess() -> dict:
    cfg = SimpleNamespace(
        in_channels=3,
        compress_to=256,
        vertical_patch=(32, 8),
        horizontal_patch=(8, 32),
        square_patch=(16, 16),
        layout_patch=(56, 56),
    )
    return build_anyres_preprocess_contract(
        cfg,
        max_decode_pixels=10_000,
        max_raw_patch_tokens_per_view=200,
        max_windows=4,
        halo_px=1,
        max_detail_tokens=64,
        source_tokens_per_detail_token=4,
        max_seq_len=512,
        recommended_max_new_tokens=2,
    )


def _canonical(path: Path) -> Image.Image:
    with Image.open(path) as image:
        result = ImageOps.exif_transpose(image).convert("RGB")
        result.load()
        return result


def _canonical_pixel_sha256(path: Path) -> str:
    image = _canonical(path).convert("RGBA")
    header = f"dol-ocr-rgba-v1\0{image.width}\0{image.height}\0".encode("ascii")
    return hashlib.sha256(header + image.tobytes()).hexdigest()


def _plan_sha(path: Path, preprocess: dict) -> str:
    image = _canonical(path)
    plan = plan_native_macro_windows(
        height=image.height,
        width=image.width,
        patch_shapes=preprocess["patch_contract"]["shapes_hw"],
        max_raw_patch_tokens=preprocess["budgets"]["patch"][
            "max_raw_tokens_per_view"
        ],
        max_windows=preprocess["budgets"]["window"]["max_windows_per_asset"],
        halo_px=preprocess["budgets"]["window"]["halo_pixels_per_side"],
    )
    if len(plan.windows) != 1:
        raise AssertionError("tiny fixture must use one identity window")
    return plan.canonical_sha256


def _save_pattern(
    path: Path,
    *,
    seed: int,
    exif_orientation: bool = False,
    touches_edge: bool = False,
) -> str:
    rng = random.Random(seed)
    width, height = ((56, 40) if exif_orientation else (48, 48))
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for _ in range(9):
        x = rng.randint(7, width - 11)
        y = rng.randint(7, height - 11)
        draw.rectangle((x, y, x + rng.randint(2, 5), y + rng.randint(2, 5)), fill="black")
    if touches_edge:
        draw.rectangle((0, 12, 4, 24), fill="black")
    if exif_orientation:
        exif = image.getexif()
        exif[274] = 6
        image.save(path, format="JPEG", quality=96, exif=exif)
    else:
        image.save(path, format="PNG", compress_level=9)
    return perceptual_hash(_canonical(path))


class AnyresBuilderTest(unittest.TestCase):
    def test_unlocked_component_assignment_is_order_invariant_and_four_way(self):
        policy = validate_split_policy(
            {
                "schema_version": 1,
                "kind": SPLIT_POLICY_KIND,
                "seed": 42,
                "fractions": {
                    "train": 0.4,
                    "sft_validation": 0.2,
                    "kl_selection": 0.2,
                    "formal_monitor": 0.2,
                },
                "phash_contract": PHASH_CONTRACT,
                "phash_max_hamming_distance": 0,
            }
        )
        policy_sha = canonical_json_sha256(policy)
        candidates = []
        clusters = []
        for index in range(128):
            asset_id = f"asset-{index:064x}"
            near_id = f"near-{index:04d}"
            candidates.append(
                SimpleNamespace(
                    sample_id=f"sample-{index:04d}",
                    source={
                        "source_document_id": f"document-{index:04d}",
                        "source_capture_id": f"capture-{index:04d}",
                        "writer_id": None,
                    },
                    asset={
                        "asset_id": asset_id,
                        "pixel_sha256": f"{index:064x}",
                        "near_duplicate": {"cluster_id": near_id},
                    },
                    phash=f"{index:016x}",
                )
            )
            clusters.append(
                SimpleNamespace(
                    member_ids=(asset_id,),
                    cluster_id=near_id,
                )
            )
        empty_lock = {
            "schema_version": 1,
            "kind": SPLIT_LOCK_KIND,
            "policy_canonical_sha256": policy_sha,
            "assignments": {},
            "phash_index": {},
        }
        forward, _, _ = _assign_components(
            candidates,
            combined_phash_clusters=clusters,
            policy=policy,
            existing_lock=empty_lock,
        )
        reverse, _, _ = _assign_components(
            list(reversed(candidates)),
            combined_phash_clusters=list(reversed(clusters)),
            policy=policy,
            existing_lock=empty_lock,
        )
        self.assertEqual(forward, reverse)
        self.assertEqual(
            {split for split, _component in forward.values()},
            {
                "train",
                "sft_validation",
                "kl_selection",
                "formal_monitor",
            },
        )

    def test_split_policy_requires_all_four_public_fractions(self):
        policy = {
            "schema_version": 1,
            "kind": SPLIT_POLICY_KIND,
            "seed": 42,
            "fractions": {
                "train": 0.4,
                "sft_validation": 0.2,
                "kl_selection": 0.2,
                "formal_monitor": 0.2,
            },
            "phash_contract": PHASH_CONTRACT,
            "phash_max_hamming_distance": 0,
        }
        normalized = validate_split_policy(policy)
        self.assertEqual(
            tuple(normalized["fractions"]),
            (
                "train",
                "sft_validation",
                "kl_selection",
                "formal_monitor",
            ),
        )
        legacy = {**policy, "fractions": {"train": 0.5, "validation": 0.5}}
        with self.assertRaisesRegex(ValueError, "must define"):
            validate_split_policy(legacy)

    def _fixture(
        self,
        root: Path,
        *,
        extra_sft_validation_print: int = 0,
    ) -> dict[str, Path | dict | list]:
        preprocess = _preprocess()
        images = root / "source-images"
        images.mkdir()
        quota_buckets = (
            "print",
            "handwritten_good",
            "handwritten_medium",
            "handwritten_poor",
        )
        buckets = [
            (split, bucket)
            for split in (
                "train",
                "sft_validation",
                "kl_selection",
                "formal_monitor",
            )
            for bucket in quota_buckets
            for _ in range(1 if split == "train" else 2)
        ] + [
            ("sft_validation", "print")
        ] * extra_sft_validation_print
        sources: list[dict] = []
        reviews: list[dict] = []
        assignments: dict[str, str] = {}
        seen_phashes: set[str] = set()
        for index, (split, bucket) in enumerate(buckets):
            sample_id = f"sample-{index:02d}"
            suffix = ".jpg" if index == 0 else ".png"
            image_path = images / f"{sample_id}{suffix}"
            attempt = 0
            while True:
                phash = _save_pattern(
                    image_path,
                    seed=10_000 + index + attempt * 101,
                    exif_orientation=(index == 0),
                    touches_edge=(index == 1),
                )
                if phash not in seen_phashes:
                    seen_phashes.add(phash)
                    break
                attempt += 1
            style = "print" if bucket == "print" else "handwritten"
            difficulty = None if style == "print" else bucket.removeprefix("handwritten_")
            document_id = f"document-{index:02d}"
            sources.append(
                {
                    "schema_version": 1,
                    "sample_id": sample_id,
                    "raw_relpath": image_path.relative_to(images).as_posix(),
                    "source_document_id": document_id,
                    "source_capture_id": f"capture-{index:02d}",
                    "style": style,
                    "font_id": "NotoSansMongolian" if style == "print" else None,
                    "writer_id": None if style == "print" else f"writer-{index:02d}",
                    "difficulty": difficulty,
                }
            )
            cut_adjudications = (
                {
                    "0": {
                        "decision": "manual_accepted",
                        "reason": "reviewed complete source-edge glyph",
                    }
                }
                if index == 1
                else {}
            )
            reviews.append(
                {
                    "schema_version": 1,
                    "sample_id": sample_id,
                    "decision": "approved",
                    "review_id": f"review-{index:02d}",
                    "reviewer_id": "reviewer-1",
                    "reviewed_at": "2026-08-20T12:00:00+08:00",
                    "reference_raw": "ᠠ",
                    "writing_mode": "vertical-lr",
                    "raw_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    "canonical_pixel_sha256": _canonical_pixel_sha256(image_path),
                    "native_plan_sha256": _plan_sha(image_path, preprocess),
                    "reading_order_window_indices": [0],
                    "cut_adjudications": cut_adjudications,
                }
            )
            assignments[split_identity_key("document", document_id)] = split

        missing_id = "sample-missing-review"
        missing_path = images / f"{missing_id}.png"
        _save_pattern(missing_path, seed=90_001)
        sources.append(
            {
                "schema_version": 1,
                "sample_id": missing_id,
                "raw_relpath": missing_path.name,
                "source_document_id": "document-missing",
                "source_capture_id": "capture-missing",
                "style": "print",
                "font_id": "NotoSansMongolian",
                "writer_id": None,
                "difficulty": None,
            }
        )

        stale_id = "sample-stale-review"
        stale_path = images / f"{stale_id}.png"
        _save_pattern(stale_path, seed=90_002)
        sources.append(
            {
                "schema_version": 1,
                "sample_id": stale_id,
                "raw_relpath": stale_path.name,
                "source_document_id": "document-stale",
                "source_capture_id": "capture-stale",
                "style": "print",
                "font_id": "NotoSansMongolian",
                "writer_id": None,
                "difficulty": None,
            }
        )
        reviews.append(
            {
                "schema_version": 1,
                "sample_id": stale_id,
                "decision": "approved",
                "review_id": "review-stale",
                "reviewer_id": "reviewer-1",
                "reviewed_at": "2026-08-20T12:00:00+08:00",
                "reference_raw": "ᠠ",
                "writing_mode": "vertical-lr",
                "raw_sha256": "0" * 64,
                "canonical_pixel_sha256": _canonical_pixel_sha256(stale_path),
                "native_plan_sha256": _plan_sha(stale_path, preprocess),
                "reading_order_window_indices": [0],
                "cut_adjudications": {},
            }
        )

        oversized_id = "sample-overbudget"
        oversized_path = images / f"{oversized_id}.png"
        oversized = Image.new("RGB", (101, 100), "white")
        ImageDraw.Draw(oversized).rectangle((20, 20, 30, 40), fill="black")
        oversized.save(oversized_path, format="PNG")
        sources.append(
            {
                "schema_version": 1,
                "sample_id": oversized_id,
                "raw_relpath": oversized_path.name,
                "source_document_id": "document-overbudget",
                "source_capture_id": "capture-overbudget",
                "style": "print",
                "font_id": "NotoSansMongolian",
                "writer_id": None,
                "difficulty": None,
            }
        )
        reviews.append(
            {
                "schema_version": 1,
                "sample_id": oversized_id,
                "decision": "approved",
                "review_id": "review-overbudget",
                "reviewer_id": "reviewer-1",
                "reviewed_at": "2026-08-20T12:00:00+08:00",
                "reference_raw": "ᠠ",
                "writing_mode": "vertical-lr",
                "raw_sha256": hashlib.sha256(oversized_path.read_bytes()).hexdigest(),
                "canonical_pixel_sha256": _canonical_pixel_sha256(oversized_path),
                "native_plan_sha256": _plan_sha(oversized_path, preprocess),
                "reading_order_window_indices": [0],
                "cut_adjudications": {},
            }
        )

        policy = {
            "schema_version": 1,
            "kind": SPLIT_POLICY_KIND,
            "seed": 42,
            "fractions": {
                "train": 0.4,
                "sft_validation": 0.2,
                "kl_selection": 0.2,
                "formal_monitor": 0.2,
            },
            "phash_contract": PHASH_CONTRACT,
            "phash_max_hamming_distance": 0,
        }
        split_lock = {
            "schema_version": 1,
            "kind": SPLIT_LOCK_KIND,
            "policy_canonical_sha256": canonical_json_sha256(policy),
            "assignments": assignments,
            "phash_index": {},
        }
        paths = {
            "sources": root / "sources.jsonl",
            "reviews": root / "approved_reviews.jsonl",
            "preprocess": root / "preprocess.json",
            "policy": root / "split_policy.json",
            "lock": root / "split_lock.json",
            "images": images,
            "output": root / "dataset-v1",
        }
        _write_jsonl(paths["sources"], sources)
        _write_jsonl(paths["reviews"], reviews)
        _write_json(paths["preprocess"], preprocess)
        _write_json(paths["policy"], policy)
        _write_json(paths["lock"], split_lock)
        return {**paths, "preprocess_payload": preprocess}

    def _build(self, fixture: dict) -> dict:
        encoder = _NativeEncoder()
        return finalize_anyres_dataset(
            sources_path=fixture["sources"],
            approved_reviews_path=fixture["reviews"],
            preprocess_contract_path=fixture["preprocess"],
            split_policy_path=fixture["policy"],
            split_lock_path=fixture["lock"],
            source_root=fixture["images"],
            output_dir=fixture["output"],
            native_encoder=encoder,
            decode_reference=encoder.decode,
            tokenizer_contract={"kind": "tiny-strict-native-tokenizer", "version": 1},
        )

    def test_finalize_produces_loader_verified_ready_dataset_and_quarantine(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            report = self._build(fixture)
            output = Path(report["output"])

            self.assertTrue((output / "READY").is_file())
            ready = json.loads((output / "READY").read_text(encoding="utf-8"))
            self.assertEqual(
                set(ready["manifests_sha256"]),
                {
                    "assets.jsonl",
                    "views.jsonl",
                    "train.jsonl",
                    "sft_validation.jsonl",
                    "kl_selection.jsonl",
                    "formal_monitor.jsonl",
                    "quarantine.jsonl",
                },
            )
            self.assertEqual(report["counts"]["accepted"], 28)
            self.assertEqual(report["counts"]["train"], 4)
            self.assertEqual(report["counts"]["sft_validation"], 8)
            self.assertEqual(report["counts"]["kl_selection"], 8)
            self.assertEqual(report["counts"]["formal_monitor"], 8)
            self.assertEqual(report["counts"]["quarantined"], 3)
            quarantine = _load_jsonl(output / "quarantine.jsonl")
            quarantine_by_id = {row["sample_id"]: row for row in quarantine}
            self.assertEqual(
                quarantine_by_id["sample-missing-review"]["reasons"],
                ["human_approved_review_missing"],
            )
            self.assertEqual(
                quarantine_by_id["sample-stale-review"]["reasons"],
                ["human_review_raw_sha256_mismatch"],
            )
            self.assertEqual(
                quarantine_by_id["sample-overbudget"]["reasons"],
                ["raw_image_pixel_budget_exceeded"],
            )

            train = _load_jsonl(output / "train.jsonl")
            sft_validation = _load_jsonl(output / "sft_validation.jsonl")
            kl_selection = _load_jsonl(output / "kl_selection.jsonl")
            formal_monitor = _load_jsonl(output / "formal_monitor.jsonl")
            self.assertEqual(len(train), 4)
            self.assertEqual(len(sft_validation), 8)
            self.assertEqual(len(kl_selection), 8)
            self.assertEqual(len(formal_monitor), 8)
            assets = _load_jsonl(output / "assets.jsonl")
            self.assertTrue(all(asset["near_duplicate"]["contract"] for asset in assets))
            components = [
                component
                for asset in assets
                for component in asset["cut_qa"]["payload"]["components"]
            ]
            self.assertTrue(components)
            self.assertTrue(
                all(
                    isinstance(component["box_xyxy"], list)
                    for component in components
                )
            )

            sample_zero = next(row for row in train if row["sample_id"] == "sample-00")
            asset_zero = next(row for row in assets if row["asset_id"] == sample_zero["asset_id"])
            self.assertEqual(
                (asset_zero["raw_width"], asset_zero["raw_height"]),
                (56, 40),
            )
            self.assertEqual(
                (asset_zero["canonical_width"], asset_zero["canonical_height"]),
                (40, 56),
            )

            views = _load_jsonl(output / "views.jsonl")
            sample_one = next(row for row in train if row["sample_id"] == "sample-01")
            reviewed_view = next(
                row for row in views if row["asset_id"] == sample_one["asset_id"]
            )
            self.assertEqual(reviewed_view["qa"]["review_status"], "manual_accepted")
            adjudications = _load_jsonl(
                output / "receipts" / "cut_adjudications.jsonl"
            )
            self.assertEqual(len(adjudications), 1)
            record = dict(adjudications[0])
            digest = record.pop("canonical_sha256")
            self.assertEqual(digest, canonical_json_sha256(record))
            self.assertEqual(digest, reviewed_view["qa"]["adjudication_sha256"])
            self.assertEqual(record["asset_pixel_sha256"], next(
                asset["pixel_sha256"] for asset in assets if asset["asset_id"] == sample_one["asset_id"]
            ))

            normalized, manifest_quarantine = validate_anyres_assets_views_samples(
                assets,
                views,
                [*train, *sft_validation, *kl_selection, *formal_monitor],
            )
            self.assertEqual(manifest_quarantine, [])
            ready_admission = validate_anyres_ready_dataset(
                output,
                expected_preprocess_contract_sha256=fixture[
                    "preprocess_payload"
                ]["contract_canonical_sha256"],
                assets_manifest=output / "assets.jsonl",
                views_manifest=output / "views.jsonl",
                train_manifest=output / "train.jsonl",
                sft_validation_manifest=output / "sft_validation.jsonl",
                kl_selection_manifest=output / "kl_selection.jsonl",
                formal_monitor_manifest=output / "formal_monitor.jsonl",
                normalized=normalized,
            )
            self.assertEqual(ready_admission["manual_adjudications"], 1)
            formal_monitor_path = output / "formal_monitor.jsonl"
            formal_monitor_path.write_bytes(
                formal_monitor_path.read_bytes() + b"\n"
            )
            with self.assertRaisesRegex(ValueError, "manifest SHA256"):
                validate_anyres_ready_dataset(
                    output,
                    expected_preprocess_contract_sha256=fixture[
                        "preprocess_payload"
                    ]["contract_canonical_sha256"],
                    assets_manifest=output / "assets.jsonl",
                    views_manifest=output / "views.jsonl",
                    train_manifest=output / "train.jsonl",
                    sft_validation_manifest=output / "sft_validation.jsonl",
                    kl_selection_manifest=output / "kl_selection.jsonl",
                    formal_monitor_manifest=formal_monitor_path,
                    normalized=normalized,
                )

            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                self._build(fixture)

    def test_cumulative_phash_bridge_between_locked_splits_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            source_rows = _load_jsonl(fixture["sources"])
            review_rows = _load_jsonl(fixture["reviews"])
            _write_jsonl(fixture["sources"], [source_rows[0]])
            _write_jsonl(fixture["reviews"], [review_rows[0]])
            current_path = Path(fixture["images"]) / source_rows[0]["raw_relpath"]
            current = int(perceptual_hash(_canonical(current_path)), 16)
            policy = json.loads(Path(fixture["policy"]).read_text(encoding="utf-8"))
            policy["phash_max_hamming_distance"] = 1
            _write_json(fixture["policy"], policy)
            lock = {
                "schema_version": 1,
                "kind": SPLIT_LOCK_KIND,
                "policy_canonical_sha256": canonical_json_sha256(policy),
                "assignments": {},
                "phash_index": {
                    f"asset-{'a' * 64}": {
                        "value": f"{current ^ 1:016x}",
                        "split": "train",
                    },
                    f"asset-{'b' * 64}": {
                        "value": f"{current ^ 2:016x}",
                        "split": "sft_validation",
                    },
                },
            }
            _write_json(fixture["lock"], lock)

            with self.assertRaisesRegex(ValueError, "cross-connects locked splits"):
                self._build(fixture)
            self.assertFalse(Path(fixture["output"]).exists())

    def test_prepare_review_pack_uses_same_pixels_plan_windows_and_cut_qa(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            review_output = root / "review-pack-v1"
            report = prepare_anyres_review_pack(
                sources_path=fixture["sources"],
                preprocess_contract_path=fixture["preprocess"],
                source_root=fixture["images"],
                output_dir=review_output,
            )

            self.assertEqual(report["counts"]["sources"], 31)
            self.assertEqual(report["counts"]["review_items"], 30)
            self.assertEqual(report["counts"]["quarantined"], 1)
            self.assertTrue((review_output / "READY").is_file())
            items = _load_jsonl(review_output / "review_items.jsonl")
            templates = _load_jsonl(
                review_output / "approved_reviews.template.jsonl"
            )
            item_by_id = {row["sample_id"]: row for row in items}
            template_by_id = {row["sample_id"]: row for row in templates}
            item = item_by_id["sample-00"]
            template = template_by_id["sample-00"]
            self.assertEqual(template["decision"], "pending")
            self.assertEqual(template["raw_sha256"], item["raw"]["sha256"])
            self.assertEqual(
                template["canonical_pixel_sha256"],
                item["canonical"]["pixel_sha256"],
            )
            self.assertEqual(
                template["native_plan_sha256"],
                item["native_plan"]["plan_sha256"],
            )
            self.assertEqual(
                template["reading_order_window_indices"],
                item["reading_order_proposals"]["vertical-lr"],
            )
            components = item["cut_qa"]["payload"]["components"]
            self.assertTrue(components)
            self.assertTrue(
                all(isinstance(component["box_xyxy"], list) for component in components)
            )
            window = item["windows"][0]
            with Image.open(review_output / window["preview_relpath"]) as preview:
                self.assertEqual(preview.format, "PNG")
                self.assertEqual(preview.size, (window["width"], window["height"]))
                self.assertIn(preview.getexif().get(274, 1), (None, 1))
            manual_template = template_by_id["sample-01"]
            self.assertEqual(
                manual_template["cut_adjudications"]["0"]["decision"],
                "manual_accepted",
            )
            self.assertEqual(
                manual_template["cut_adjudications"]["0"]["reason"],
                "",
            )

    def test_sft_validation_bucket_counts_three_and_five_are_ready_eligible(self):
        for expected_print_count in (3, 5):
            with self.subTest(expected_print_count=expected_print_count):
                with tempfile.TemporaryDirectory() as temporary:
                    fixture = self._fixture(
                        Path(temporary),
                        extra_sft_validation_print=expected_print_count - 2,
                    )
                    report = self._build(fixture)
                    validation = _load_jsonl(
                        Path(report["output"]) / "sft_validation.jsonl"
                    )
                    observed = sum(
                        row["style"] == "print" for row in validation
                    )
                    self.assertEqual(observed, expected_print_count)


if __name__ == "__main__":
    unittest.main()
