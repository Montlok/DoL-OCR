# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from Model.posttrain.ocr_anyres_manifest import ANYRES_PUBLIC_SPLITS
from Model.posttrain.ocr_locked_contract import (
    LOCKED_IDENTITY_COMMITMENT_SCHEME,
    build_locked_golden_anchor,
    claim_locked_benchmark_once,
    preclaim_locked_benchmark,
    validate_locked_build_receipt,
)
from Model.posttrain.ocr_locked_data import (
    build_locked_combined_manifest,
    load_locked_benchmark_after_claim,
    locked_combined_manifest_commitments,
)
from Model.posttrain.text_replay import TEXT_REPLAY_PUBLIC_SPLITS


def _render(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(_render(value).rstrip(b"\n")).hexdigest()


def _scope() -> dict:
    return {
        "covered_inputs": [
            "source_pretraining_corpus",
            "anyres_train",
            "anyres_sft_validation",
            "anyres_kl_selection",
            "anyres_formal_monitor",
            "text_replay_train",
            "text_replay_sft_validation",
            "text_replay_kl_selection",
            "text_replay_formal_monitor",
        ],
        "modalities": ["image", "text"],
        "identity_dimensions": [
            "sample_id",
            "source_document_id",
            "source_capture_id",
            "writer_id",
            "canonical_pixel_sha256",
            "near_duplicate_cluster",
            "reference_text_sha256",
        ],
        "relation": "locked_golden_disjoint_from_all_covered_inputs_v1",
    }


class LockedDataTest(unittest.TestCase):
    def _fixture(self, root: Path):
        sealed = root / "sealed"
        assets_dir = sealed / "assets"
        assets_dir.mkdir(parents=True)
        images = []
        asset_entries = []
        buckets = (
            "print",
            "handwritten_good",
            "handwritten_medium",
            "handwritten_poor",
        )
        for bucket in buckets:
            for index in range(2):
                payload = f"png-{bucket}-{index}".encode("ascii")
                relpath = f"assets/{bucket}-{index}.png"
                (sealed / relpath).write_bytes(payload)
                reference = f"ᠮ{bucket[-1]}{index}"
                images.append(
                    {
                        "id": f"image-{bucket}-{index}",
                        "bucket": bucket,
                        "asset_relpath": relpath,
                        "asset_sha256": _sha(payload),
                        "asset_size_bytes": len(payload),
                        "reference": reference,
                        "reference_utf8_sha256": _sha(reference.encode("utf-8")),
                        "reference_token_count": len(reference),
                    }
                )
                asset_entries.append(
                    {
                        "relpath": relpath,
                        "sha256": _sha(payload),
                        "size_bytes": len(payload),
                    }
                )
        texts = []
        for index in range(8):
            reference = f"ᠠᠪ{index}"
            texts.append(
                {
                    "id": f"text-{index}",
                    "reference": reference,
                    "reference_utf8_sha256": _sha(reference.encode("utf-8")),
                    "reference_token_count": len(reference),
                }
            )
        manifest = build_locked_combined_manifest(
            image_records=sorted(images, key=lambda row: row["id"]),
            text_records=texts,
        )
        manifest_bytes = _render(manifest)
        manifest_path = sealed / "locked_manifest.json"
        manifest_path.write_bytes(manifest_bytes)
        commitments = locked_combined_manifest_commitments(manifest)
        counts = {bucket: 2 for bucket in buckets}
        lengths = [row["reference_token_count"] for row in texts]
        receipt = {
            "schema_version": 1,
            "kind": "dol_ocr_anyres_locked_build_receipt_v1",
            "manifest_file": {
                "relpath": "locked_manifest.json",
                "sha256": _sha(manifest_bytes),
                "size_bytes": len(manifest_bytes),
            },
            "asset_files": sorted(asset_entries, key=lambda row: row["relpath"]),
            **commitments,
            "preprocess_contract_sha256": "1" * 64,
            "tokenizer_contract_sha256": "2" * 64,
            "identity_commitment_scheme": LOCKED_IDENTITY_COMMITMENT_SCHEME,
            "cross_exclusion_receipt_sha256": "3" * 64,
            "contamination_scope": _scope(),
            "bucket_counts": counts,
            "token_stats": {
                "sample_count": len(texts),
                "minimum_reference_tokens": min(lengths),
                "maximum_reference_tokens": max(lengths),
                "total_reference_tokens": sum(lengths),
                "recommended_max_new_tokens": max(lengths) + 1,
            },
            "builder_source_sha256": "4" * 64,
        }
        receipt["canonical_sha256"] = _canonical_sha(receipt)
        receipt = validate_locked_build_receipt(receipt)
        receipt_bytes = _render(receipt)
        receipt_path = sealed / "LOCKED_BUILD_RECEIPT.json"
        receipt_path.write_bytes(receipt_bytes)
        anchor = build_locked_golden_anchor(
            build_receipt_sha256=_sha(receipt_bytes),
            build_receipt=receipt,
        )
        anchor_path = root / "LOCKED_GOLDEN_ANCHOR.json"
        anchor_path.write_bytes(_render(anchor))
        preclaim = preclaim_locked_benchmark(anchor_path, receipt_path)
        claim = claim_locked_benchmark_once(
            root / "ledger",
            preclaim.build_receipt_sha256,
            {"locked_golden_anchor_sha256": preclaim.anchor_canonical_sha256},
        )
        return sealed, preclaim, claim, manifest

    @staticmethod
    def _encode(text: str):
        return [ord(character) for character in text]

    @staticmethod
    def _decode(values):
        return "".join(chr(int(value)) for value in values)

    def test_claim_gated_capture_is_immutable_and_public_splits_stay_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            sealed, preclaim, claim, manifest = self._fixture(Path(temporary))
            source = load_locked_benchmark_after_claim(
                sealed,
                preclaim,
                claim,
                encode_reference=self._encode,
                decode_ids=self._decode,
            )
            self.assertEqual(len(source.images), 8)
            self.assertEqual(len(source.texts), 8)
            self.assertEqual(
                source.manifest_canonical_sha256,
                manifest["canonical_sha256"],
            )
            self.assertIsInstance(source.images[0].asset_bytes, bytes)
            self.assertNotIn("locked_golden", ANYRES_PUBLIC_SPLITS)
            self.assertNotIn("locked_golden", TEXT_REPLAY_PUBLIC_SPLITS)

    def test_wrong_claim_or_asset_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sealed, preclaim, claim, manifest = self._fixture(root)
            with self.assertRaisesRegex(TypeError, "LockedGoldenClaim"):
                load_locked_benchmark_after_claim(
                    sealed,
                    preclaim,
                    object(),
                    encode_reference=self._encode,
                    decode_ids=self._decode,
                )
            asset = sealed / manifest["image_records"][0]["asset_relpath"]
            asset.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "size|SHA-256"):
                load_locked_benchmark_after_claim(
                    sealed,
                    preclaim,
                    claim,
                    encode_reference=self._encode,
                    decode_ids=self._decode,
                )

    def test_asset_symlink_and_native_roundtrip_drift_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sealed, preclaim, claim, manifest = self._fixture(root)
            asset = sealed / manifest["image_records"][0]["asset_relpath"]
            original = asset.read_bytes()
            target = sealed / "target.bin"
            target.write_bytes(original)
            asset.unlink()
            asset.symlink_to(target)
            with self.assertRaises((OSError, ValueError)):
                load_locked_benchmark_after_claim(
                    sealed,
                    preclaim,
                    claim,
                    encode_reference=self._encode,
                    decode_ids=self._decode,
                )
        with tempfile.TemporaryDirectory() as temporary:
            sealed, preclaim, claim, _ = self._fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "roundtrip"):
                load_locked_benchmark_after_claim(
                    sealed,
                    preclaim,
                    claim,
                    encode_reference=self._encode,
                    decode_ids=lambda _values: "wrong",
                )


if __name__ == "__main__":
    unittest.main()
