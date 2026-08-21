# -*- coding: utf-8 -*-

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import tempfile
import threading
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

from Model.posttrain import ocr_locked_contract as locked


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


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


def _scope() -> dict[str, object]:
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


def _receipt(*, manifest_relpath: str = "sealed/golden.jsonl") -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": locked.LOCKED_CONTRACT_SCHEMA_VERSION,
        "kind": locked.LOCKED_BUILD_RECEIPT_KIND,
        "manifest_file": {
            "relpath": manifest_relpath,
            "sha256": "1" * 64,
            "size_bytes": 901,
        },
        "asset_files": [
            {
                "relpath": "sealed/assets/a.png",
                "sha256": "2" * 64,
                "size_bytes": 101,
            },
            {
                "relpath": "sealed/assets/b.png",
                "sha256": "3" * 64,
                "size_bytes": 202,
            },
        ],
        "image_dataset_contract_sha256": "4" * 64,
        "text_dataset_contract_sha256": "5" * 64,
        "preprocess_contract_sha256": "6" * 64,
        "tokenizer_contract_sha256": "7" * 64,
        "identity_commitment_scheme": locked.LOCKED_IDENTITY_COMMITMENT_SCHEME,
        "image_identity_commitment_sha256": "8" * 64,
        "text_identity_commitment_sha256": "9" * 64,
        "membership_commitment_sha256": "a" * 64,
        "cross_exclusion_receipt_sha256": "b" * 64,
        "contamination_scope": _scope(),
        "bucket_counts": {
            "print": 6,
            "handwritten_good": 2,
            "handwritten_medium": 2,
            "handwritten_poor": 2,
        },
        "token_stats": {
            "sample_count": 12,
            "minimum_reference_tokens": 2,
            "maximum_reference_tokens": 30,
            "total_reference_tokens": 180,
            "recommended_max_new_tokens": 31,
        },
        "builder_source_sha256": "c" * 64,
    }
    payload["canonical_sha256"] = _canonical_sha(payload)
    return payload


def _rehash(value: dict[str, object]) -> dict[str, object]:
    value = copy.deepcopy(value)
    value.pop("canonical_sha256", None)
    value["canonical_sha256"] = _canonical_sha(value)
    return value


class LockedContractValidationTest(unittest.TestCase):
    def test_anchor_is_label_free_and_binds_sealed_receipt(self):
        receipt = _receipt()
        receipt_sha = hashlib.sha256(_render(receipt)).hexdigest()
        anchor = locked.build_locked_golden_anchor(
            build_receipt_sha256=receipt_sha,
            build_receipt=receipt,
        )

        self.assertEqual(
            locked.validate_locked_golden_anchor(anchor),
            anchor,
        )
        self.assertEqual(anchor["build_receipt_sha256"], receipt_sha)
        self.assertEqual(
            anchor["bucket_counts"],
            receipt["bucket_counts"],
        )
        def object_keys(value: object) -> set[str]:
            if isinstance(value, dict):
                return set(value) | {
                    key
                    for child in value.values()
                    for key in object_keys(child)
                }
            if isinstance(value, list):
                return {
                    key for child in value for key in object_keys(child)
                }
            return set()

        keys = object_keys(anchor)
        for forbidden in (
            "reference",
            "transcription",
            "manifest_file",
            "asset_files",
            "relpath",
        ):
            self.assertNotIn(forbidden, keys)

    def test_anchor_and_receipt_reject_unknown_keys(self):
        receipt = _receipt()
        receipt["surprise"] = True
        receipt = _rehash(receipt)
        with self.assertRaisesRegex(ValueError, "unknown"):
            locked.validate_locked_build_receipt(receipt)

        valid = _receipt()
        anchor = locked.build_locked_golden_anchor(
            build_receipt_sha256="d" * 64,
            build_receipt=valid,
        )
        anchor["reference"] = "secret"
        anchor = _rehash(anchor)
        with self.assertRaisesRegex(ValueError, "unknown"):
            locked.validate_locked_golden_anchor(anchor)

    def test_receipt_rejects_unsafe_or_non_normalized_paths(self):
        unsafe = (
            "/absolute/golden.jsonl",
            "../golden.jsonl",
            "sealed/../golden.jsonl",
            "sealed\\golden.jsonl",
            "sealed/\x00golden.jsonl",
            "sealed//golden.jsonl",
            "./sealed/golden.jsonl",
        )
        for relpath in unsafe:
            with self.subTest(relpath=repr(relpath)):
                with self.assertRaisesRegex(ValueError, "POSIX path"):
                    locked.validate_locked_build_receipt(
                        _receipt(manifest_relpath=relpath)
                    )

    def test_receipt_requires_four_nontrivial_buckets_and_consistent_tokens(self):
        receipt = _receipt()
        receipt["bucket_counts"]["handwritten_poor"] = 1  # type: ignore[index]
        receipt = _rehash(receipt)
        with self.assertRaisesRegex(ValueError, "at least two"):
            locked.validate_locked_build_receipt(receipt)

        receipt = _receipt()
        receipt["token_stats"]["sample_count"] = 13  # type: ignore[index]
        receipt = _rehash(receipt)
        with self.assertRaisesRegex(ValueError, "differs from bucket_counts"):
            locked.validate_locked_build_receipt(receipt)

    def test_preclaim_rejects_duplicate_json_keys_and_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicate = root / "duplicate.json"
            duplicate.write_text(
                '{"schema_version":1,"schema_version":1}\n',
                encoding="utf-8",
            )
            receipt_path = root / "receipt.json"
            receipt_path.write_bytes(_render(_receipt()))
            with self.assertRaisesRegex(ValueError, "strict UTF-8 JSON"):
                locked.preclaim_locked_benchmark(duplicate, receipt_path)

            anchor_target = root / "anchor-target.json"
            anchor_target.write_text("{}\n", encoding="utf-8")
            anchor_link = root / "anchor-link.json"
            anchor_link.symlink_to(anchor_target)
            with self.assertRaisesRegex(ValueError, "opened safely"):
                locked.preclaim_locked_benchmark(anchor_link, receipt_path)

    def test_preclaim_reads_only_anchor_and_receipt_and_detects_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt = _receipt(manifest_relpath="sealed/MUST_NOT_OPEN.jsonl")
            receipt_path = root / "receipt.json"
            receipt_path.write_bytes(_render(receipt))
            receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
            anchor = locked.build_locked_golden_anchor(
                build_receipt_sha256=receipt_sha,
                build_receipt=receipt,
            )
            anchor_path = root / "anchor.json"
            anchor_path.write_bytes(_render(anchor))

            calls: list[str] = []
            real_read = locked._stable_read_regular

            def audited_read(path, *, where, limit):
                calls.append(os.fspath(path))
                if "MUST_NOT_OPEN" in os.fspath(path):
                    raise AssertionError("preclaim followed a sealed labeled path")
                return real_read(path, where=where, limit=limit)

            with patch.object(
                locked,
                "_stable_read_regular",
                side_effect=audited_read,
            ):
                preclaim = locked.preclaim_locked_benchmark(
                    anchor_path,
                    receipt_path,
                )
            self.assertEqual(calls, [str(anchor_path), str(receipt_path)])
            self.assertEqual(preclaim.build_receipt_sha256, receipt_sha)
            self.assertEqual(
                preclaim.build_receipt["manifest_file"]["relpath"],
                "sealed/MUST_NOT_OPEN.jsonl",
            )

            mismatched = copy.deepcopy(anchor)
            mismatched["preprocess_contract_sha256"] = "e" * 64
            mismatched = _rehash(mismatched)
            anchor_path.write_bytes(_render(mismatched))
            with self.assertRaisesRegex(ValueError, "preprocess_contract_sha256"):
                locked.preclaim_locked_benchmark(anchor_path, receipt_path)


class LockedClaimTest(unittest.TestCase):
    def test_concurrent_claims_admit_exactly_one_caller(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = root / "ledger"
            barrier = threading.Barrier(2)
            outcomes: list[tuple[str, object]] = []
            outcome_lock = threading.Lock()

            def contender(model_sha: str) -> None:
                barrier.wait()
                try:
                    capability = locked.claim_locked_benchmark_once(
                        ledger,
                        "d" * 64,
                        {"selected_model_sha256": model_sha},
                    )
                    result: tuple[str, object] = ("ok", capability)
                except Exception as exc:  # noqa: BLE001 - concurrency assertion
                    result = ("error", exc)
                with outcome_lock:
                    outcomes.append(result)

            threads = [
                threading.Thread(target=contender, args=(char * 64,))
                for char in ("1", "2")
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
            self.assertFalse(any(thread.is_alive() for thread in threads))
            successes = [value for status, value in outcomes if status == "ok"]
            failures = [value for status, value in outcomes if status == "error"]
            self.assertEqual(len(successes), 1)
            self.assertEqual(len(failures), 1)
            self.assertIsInstance(failures[0], FileExistsError)
            marker = locked.validate_locked_golden_claim(successes[0])
            self.assertEqual(marker["build_receipt_sha256"], "d" * 64)

    def test_second_model_is_rejected_by_receipt_key(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "ledger"
            locked.claim_locked_benchmark_once(
                ledger,
                "a" * 64,
                {"selected_model_sha256": "1" * 64},
            )
            with self.assertRaisesRegex(FileExistsError, "already claimed"):
                locked.claim_locked_benchmark_once(
                    ledger,
                    "a" * 64,
                    {"selected_model_sha256": "2" * 64},
                )

    def test_ledger_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "real"
            real.mkdir()
            ledger = root / "ledger"
            ledger.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "not a symlink"):
                locked.claim_locked_benchmark_once(
                    ledger,
                    "a" * 64,
                    {"selected_model_sha256": "1" * 64},
                )
            self.assertEqual(list(real.iterdir()), [])

    def test_capability_is_private_frozen_and_bound_to_marker_inode(self):
        with tempfile.TemporaryDirectory() as temporary:
            claim = locked.claim_locked_benchmark_once(
                Path(temporary) / "ledger",
                "b" * 64,
                {"selected_model_sha256": "1" * 64},
            )
            with self.assertRaises(TypeError):
                locked.LockedGoldenClaim(
                    factory_token=object(),
                    build_receipt_sha256="b" * 64,
                    marker_sha256=claim.marker_sha256,
                    claim_payload_sha256=claim.claim_payload_sha256,
                    ledger_path=str(claim.marker_path.parent),
                    marker_name=claim.marker_path.name,
                    ledger_device=0,
                    ledger_inode=0,
                    marker_device=0,
                    marker_inode=0,
                    marker_size=0,
                )
            with self.assertRaises(FrozenInstanceError):
                claim.marker_sha256 = "f" * 64  # type: ignore[misc]

            raw = claim.marker_path.read_bytes()
            claim.marker_path.unlink()
            claim.marker_path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "identity changed"):
                locked.validate_locked_golden_claim(claim)

    def test_directory_fsync_failure_keeps_consumed_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "ledger"
            real_fsync = os.fsync
            calls = 0

            def fail_directory_fsync(descriptor: int) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected directory fsync failure")
                real_fsync(descriptor)

            with patch.object(locked.os, "fsync", side_effect=fail_directory_fsync):
                with self.assertRaisesRegex(OSError, "injected"):
                    locked.claim_locked_benchmark_once(
                        ledger,
                        "c" * 64,
                        {"selected_model_sha256": "1" * 64},
                    )
            marker = ledger / f"{'c' * 64}.json"
            self.assertTrue(marker.is_file())
            self.assertEqual(stat.S_IMODE(marker.stat().st_mode), 0o600)
            with self.assertRaises(FileExistsError):
                locked.claim_locked_benchmark_once(
                    ledger,
                    "c" * 64,
                    {"selected_model_sha256": "2" * 64},
                )

    def test_incomplete_stub_is_atomic_no_clobber_and_bound_to_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            claim = locked.claim_locked_benchmark_once(
                root / "ledger",
                "e" * 64,
                {"selected_model_sha256": "1" * 64},
            )
            destination = root / "locked_report.json"
            actual = locked.write_claimed_evaluation_incomplete_stub(
                destination,
                claim,
                {"selected_checkpoint_metadata_sha256": "2" * 64},
            )
            self.assertEqual(actual, destination)
            stub = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(stub["status"], "claimed_evaluation_incomplete")
            self.assertEqual(stub["claim_marker_sha256"], claim.marker_sha256)
            self.assertEqual(
                stat.S_IMODE(destination.stat().st_mode),
                0o600,
            )
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                locked.write_claimed_evaluation_incomplete_stub(
                    destination,
                    claim,
                    {"selected_checkpoint_metadata_sha256": "3" * 64},
                )
            self.assertEqual(
                json.loads(destination.read_text(encoding="utf-8")),
                stub,
            )


if __name__ == "__main__":
    unittest.main()
