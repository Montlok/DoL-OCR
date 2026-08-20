# -*- coding: utf-8 -*-

"""Contract tests for deterministic OCR-RL manifest construction."""

from __future__ import annotations

import csv
import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from Model.ocr.tokenization import canonicalize_native_ocr_text
from Model.posttrain.ocr_manifest_builder import (
    DEFAULT_SPLIT_FRACTIONS,
    assign_group_splits,
    build_manifest_rows,
    file_sha256,
    load_annotation_pack,
    materialize_manifest_images,
    partition_manifest_rows,
    read_annotation_tsv,
)
from Model.posttrain.ocr_manifests import (
    assert_labeled_golden_matches_identity,
    golden_identity_semantic_sha256,
    load_golden_identity_manifest,
    load_locked_golden_receipt_anchor,
    verify_locked_golden_manifest,
)


def _native_encode(text: str) -> list[int]:
    return [
        ord(character)
        for character in canonicalize_native_ocr_text(text)
    ]


def _native_decode(ids) -> str:
    return "".join(chr(int(token)) for token in ids)


class OCRManifestBuilderTest(unittest.TestCase):
    def _pack(self, root: Path, *, duplicate_bytes: bool = False) -> Path:
        lines = []
        for index, name in enumerate(("a", "b", "c")):
            raw = root / f"{name}_raw.png"
            preview = root / f"{name}_224.png"
            raw_color = "red" if duplicate_bytes else (index * 50, 20, 30)
            raw_size = (9, 12) if duplicate_bytes else (9 + index, 12)
            Image.new("RGB", raw_size, raw_color).save(raw)
            Image.new("RGB", (224, 224), "white").save(preview)
            lines.append(
                {
                    "id": name,
                    "pdf": f"doc-{name}.pdf",
                    "page": index,
                    "raw_path": raw.name,
                    "preview_path": preview.name,
                }
            )
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps({"lines": lines}),
            encoding="utf-8",
        )
        return manifest

    def test_builder_uses_raw_crop_and_keeps_golden_identity_label_free(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = self._pack(root)
            image_root, provenance = load_annotation_pack(
                pack,
                domain="scan_book",
            )
            annotations = {
                "a": "a\u00a0b",
                "b": "9/7",
                "c": "ᠮᠣᠩᠭᠣᠯ᠋",
            }
            group_domains = {
                str(item["group_id"]): str(item["domain"])
                for item in provenance.values()
            }
            lock = assign_group_splits(
                group_domains,
                seed=7,
                fractions=DEFAULT_SPLIT_FRACTIONS,
            )
            rows = build_manifest_rows(
                annotations,
                provenance,
                lock["assignments"],
                image_root=image_root,
                encode_reference=_native_encode,
                decode_reference=_native_decode,
            )
            for row in rows:
                self.assertTrue(row["image"].endswith("_raw.png"))
                self.assertFalse(row["image"].endswith("_224.png"))
                self.assertEqual(
                    row["sha256"],
                    file_sha256(image_root / row["image"]),
                )
            self.assertEqual(
                next(row["reference"] for row in rows if row["id"] == "a"),
                "a b",
            )

            train, validation, golden, identity = partition_manifest_rows(rows)
            self.assertTrue(train)
            self.assertTrue(validation)
            self.assertTrue(golden)
            self.assertEqual(len(identity), len(golden))
            self.assertTrue(
                all(
                    "reference" not in row and "image" not in row
                    for row in identity
                )
            )
            group_splits = {
                row["group_id"]: row["split"] for row in rows
            }
            self.assertEqual(len(group_splits), 3)

    def test_materialized_pixels_are_physically_split_and_content_addressed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = self._pack(root)
            image_root, provenance = load_annotation_pack(pack, domain="photo")
            lock = assign_group_splits(
                {
                    str(item["group_id"]): str(item["domain"])
                    for item in provenance.values()
                },
                seed=0,
            )
            rows = build_manifest_rows(
                {"a": "a", "b": "b", "c": "c"},
                provenance,
                lock["assignments"],
                image_root=image_root,
                encode_reference=_native_encode,
                decode_reference=_native_decode,
            )
            public = root / "published" / "public"
            locked = root / "published" / "locked"
            materialized = materialize_manifest_images(
                rows,
                source_root=image_root,
                public_root=public,
                locked_root=locked,
            )
            for row in materialized:
                self.assertTrue(row["image"].startswith("images/"))
                self.assertNotIn("_raw", row["image"])
                base = locked if row["split"] == "golden" else public
                copied = base / row["image"]
                self.assertEqual(file_sha256(copied), row["sha256"])
                expected_mode = 0o600 if row["split"] == "golden" else 0o644
                self.assertEqual(
                    stat.S_IMODE(copied.stat().st_mode),
                    expected_mode,
                )
                other = public if row["split"] == "golden" else locked
                self.assertFalse((other / row["image"]).exists())

    def test_split_assignment_is_order_independent_and_incrementally_locked(self):
        groups = {
            "document:a": "photo",
            "document:b": "photo",
            "document:c": "photo",
            "document:d": "photo",
        }
        first = assign_group_splits(groups, seed=11)
        reversed_input = dict(reversed(list(groups.items())))
        rerun = assign_group_splits(reversed_input, seed=11)
        self.assertEqual(first, rerun)

        extended = {**groups, "document:e": "photo"}
        updated = assign_group_splits(
            extended,
            seed=11,
            existing_lock=first,
        )
        for group_id in groups:
            self.assertEqual(
                first["assignments"][group_id],
                updated["assignments"][group_id],
            )

    def test_duplicate_image_content_is_rejected_before_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = self._pack(root, duplicate_bytes=True)
            image_root, provenance = load_annotation_pack(pack, domain="photo")
            lock = assign_group_splits(
                {
                    str(item["group_id"]): str(item["domain"])
                    for item in provenance.values()
                },
                seed=0,
            )
            with self.assertRaisesRegex(ValueError, "duplicate raw image SHA"):
                build_manifest_rows(
                    {"a": "a", "b": "b", "c": "c"},
                    provenance,
                    lock["assignments"],
                    image_root=image_root,
                    encode_reference=_native_encode,
                    decode_reference=_native_decode,
                )

    def test_annotation_tsv_rejects_duplicate_ids_and_ignores_image_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotations.tsv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t")
                writer.writerow(
                    ["id", "image_relpath", "transcription", "notes"]
                )
                writer.writerow(["a", "../../untrusted-preview.png", "ᠮ", ""])
            self.assertEqual(read_annotation_tsv(path), {"a": "ᠮ"})

            with path.open("a", encoding="utf-8", newline="") as handle:
                csv.writer(handle, delimiter="\t").writerow(
                    ["a", "other.png", "ᠨ", ""]
                )
            with self.assertRaisesRegex(ValueError, "duplicate id"):
                read_annotation_tsv(path)

            path.write_text(
                "id\ttranscription\nbad\t\ufffd\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "already lossy"):
                read_annotation_tsv(path)

    def test_native_roundtrip_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = self._pack(root)
            image_root, provenance = load_annotation_pack(pack, domain="photo")
            lock = assign_group_splits(
                {
                    str(item["group_id"]): str(item["domain"])
                    for item in provenance.values()
                },
                seed=0,
            )
            with self.assertRaisesRegex(ValueError, "round-trip differs"):
                build_manifest_rows(
                    {"a": "a", "b": "b", "c": "c"},
                    provenance,
                    lock["assignments"],
                    image_root=image_root,
                    encode_reference=lambda _text: [ord("x")],
                    decode_reference=_native_decode,
                )

    def test_golden_identity_rejects_transcripts_and_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "identity.jsonl"
            row = {
                "schema_version": 1,
                "id": "secret",
                "group_id": "capture-1",
                "split": "golden",
                "image_sha256": "a" * 64,
                "reference": "must-not-be-visible",
                "image": "must-not-be-visible.png",
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "must not contain golden payload",
            ):
                load_golden_identity_manifest(path)

    def test_locked_golden_must_match_identity_exactly(self):
        identity = [
            {
                "schema_version": 1,
                "id": "a",
                "group_id": "document:a",
                "split": "golden",
                "image_sha256": "a" * 64,
            }
        ]

        class _Dataset:
            rows = [
                {
                    "id": "a",
                    "group_id": "document:a",
                    "sha256": "b" * 64,
                }
            ]

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, index):
                return self.rows[index]

        with self.assertRaisesRegex(ValueError, "group/hash mismatch"):
            assert_labeled_golden_matches_identity(_Dataset(), identity)

    def test_locked_receipt_is_canonical_and_publicly_anchored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            golden = root / "golden.jsonl"
            identity = root / "golden_identity.jsonl"
            identity_row = {
                "schema_version": 1,
                "id": "a",
                "group_id": "capture:a",
                "split": "golden",
                "image_sha256": "a" * 64,
            }
            identity.write_text(
                json.dumps(identity_row) + "\n",
                encoding="utf-8",
            )
            golden.write_text(
                json.dumps({**identity_row, "reference": "ᠠ"}) + "\n",
                encoding="utf-8",
            )
            identity_rows = load_golden_identity_manifest(identity)

            receipt = root / "build_receipt.json"
            receipt.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "kind": "ocr_locked_golden_build_receipt",
                        "locked_golden_sha256": file_sha256(golden),
                        "golden_identity_sha256": file_sha256(identity),
                        "golden_identity_semantic_sha256": (
                            golden_identity_semantic_sha256(identity_rows)
                        ),
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            public = root / "dataset_contract.json"
            public.write_text(
                json.dumps(
                    {
                        "locked_golden_receipt_sha256": file_sha256(receipt),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            from Model.posttrain import ocr_manifests

            original_file_sha256 = ocr_manifests._file_sha256
            original_path_open = Path.open

            def hash_public_artifact_only(path):
                if Path(path) == golden:
                    raise AssertionError(
                        "pre-claim receipt validation hashed golden labels"
                    )
                return original_file_sha256(path)

            def open_public_artifact_only(path, *args, **kwargs):
                if path == golden:
                    raise AssertionError(
                        "pre-claim receipt validation opened golden labels"
                    )
                return original_path_open(path, *args, **kwargs)

            with (
                patch.object(
                    ocr_manifests,
                    "_file_sha256",
                    side_effect=hash_public_artifact_only,
                ),
                patch.object(
                    Path,
                    "open",
                    new=open_public_artifact_only,
                ),
            ):
                loaded = load_locked_golden_receipt_anchor(
                    receipt,
                    golden_manifest=golden,
                    golden_identity_manifest=identity,
                    identity_rows=identity_rows,
                    public_dataset_contract=public,
                )
            self.assertEqual(
                loaded["receipt_file_sha256"],
                file_sha256(receipt),
            )
            self.assertEqual(
                verify_locked_golden_manifest(
                    loaded,
                    golden_manifest=golden,
                ),
                file_sha256(golden),
            )

            copied = root / "copied_receipt.json"
            copied.write_bytes(receipt.read_bytes())
            with self.assertRaisesRegex(ValueError, "canonical"):
                load_locked_golden_receipt_anchor(
                    copied,
                    golden_manifest=golden,
                    golden_identity_manifest=identity,
                    identity_rows=identity_rows,
                    public_dataset_contract=public,
                )

            receipt.write_text(
                receipt.read_text(encoding="utf-8") + " ",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "training-anchored"):
                load_locked_golden_receipt_anchor(
                    receipt,
                    golden_manifest=golden,
                    golden_identity_manifest=identity,
                    identity_rows=identity_rows,
                    public_dataset_contract=public,
                )


if __name__ == "__main__":
    unittest.main()
