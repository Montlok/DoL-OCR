# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from Model.posttrain.ocr_locked_contract import (
    claim_locked_benchmark_once,
    write_claimed_evaluation_incomplete_stub,
)
from Model.posttrain.ocr_locked_data import (
    LockedBenchmarkBatchSource,
    LockedImageExample,
    LockedTextExample,
)
from Model.posttrain.ocr_locked_eval import (
    LOCKED_EVALUATION_REPORT_KIND,
    build_locked_evaluation_batches,
    build_locked_evaluation_report,
    finalize_locked_evaluation_report,
)


_BUCKETS = (
    "print",
    "handwritten_good",
    "handwritten_medium",
    "handwritten_poor",
)


def _source() -> LockedBenchmarkBatchSource:
    images = tuple(
        LockedImageExample(
            id=f"sample-{bucket}-{index}",
            bucket=bucket,
            asset_bytes=f"image-{bucket}-{index}".encode("ascii"),
            asset_sha256="1" * 64,
            reference="ᠠ",
            reference_token_ids=(300,),
        )
        for bucket in _BUCKETS
        for index in range(2)
    )
    texts = tuple(
        LockedTextExample(
            id=f"text-{index}",
            reference="ᠡ",
            reference_token_ids=(301,),
        )
        for index in range(8)
    )
    return LockedBenchmarkBatchSource(
        build_receipt_sha256="2" * 64,
        manifest_canonical_sha256="3" * 64,
        image_dataset_contract_sha256="4" * 64,
        text_dataset_contract_sha256="5" * 64,
        images=images,
        texts=texts,
    )


def _report(edits: int) -> dict:
    records = sorted(
        [
            {
                "sample_id": f"sample-{bucket}-{index}",
                "bucket": bucket,
                "grapheme_edits": edits,
                "reference_graphemes": 100,
            }
            for bucket in _BUCKETS
            for index in range(2)
        ],
        key=lambda row: row["sample_id"],
    )
    cer = edits / 100
    interval = {"mean": 1.0, "low": 0.5, "high": 1.5}
    return {
        "schema_version": 1,
        "image_dataset_contract_sha256": "4" * 64,
        "decode_contract": {},
        "real": {
            "overall": {
                "raw_grapheme_cer": cer,
                "raw_line_exact": 1.0 - cer,
                "eos_rate": 1.0,
                "invalid_count": 0,
                "hit_cap_rate": 0.0,
            },
            "buckets": {
                bucket: {
                    "raw_grapheme_cer": cer,
                    "raw_line_exact": 1.0 - cer,
                }
                for bucket in _BUCKETS
            },
        },
        "controls": {},
        "grounding": {
            "blank_cer_gap": 1.0,
            "shuffled_cer_gap": 1.0,
            "blank_first_token_nll_gap": 1.0,
            "shuffled_first_token_nll_gap": 1.0,
            "paired_bootstrap_95ci": {
                "blank_cer_gap": interval,
                "shuffled_cer_gap": interval,
                "blank_first_token_nll_gap": interval,
                "shuffled_first_token_nll_gap": interval,
            },
        },
        "deployment_weighted_cer": cer,
        "deployment_weights": {
            "print": 0.6,
            "handwritten_good": 0.1,
            "handwritten_medium": 0.2,
            "handwritten_poor": 0.1,
        },
        "worst_bucket": {"name": "print", "raw_grapheme_cer": cer},
        "text_replay": {
            "dataset_contract_sha256": "5" * 64,
            "token_nll": 1.0,
            "target_tokens": 16,
            "batches": 2,
            "position_contract": "boundary_v1",
        },
        "selection_records": records,
    }


class LockedEvalTest(unittest.TestCase):
    def test_batching_preserves_bucket_pairs_and_text_contract(self):
        captured = []

        def collator(rows):
            captured.append(rows)
            return {
                "dataset_contract_sha256": rows[0]["dataset_contract_sha256"],
                "quota_buckets": [row["quota_bucket"] for row in rows],
                "sample_metadata": [row["sample"] for row in rows],
            }

        images, texts = build_locked_evaluation_batches(
            _source(),
            image_collator=collator,
            text_batch_size=4,
        )
        self.assertEqual(len(images), 4)
        self.assertTrue(all(len(batch["quota_buckets"]) == 2 for batch in images))
        self.assertEqual(len(texts), 2)
        self.assertTrue(
            all(batch["dataset_contract_sha256"] == "5" * 64 for batch in texts)
        )

    def test_comparison_is_paired_significant_and_redacts_ids(self):
        report = build_locked_evaluation_report(
            selected_report=_report(1),
            reference_report=_report(2),
            selected_checkpoint={
                "path": str(Path("/tmp/selected").resolve()),
                "model_sha256": "1" * 64,
                "metadata_sha256": "2" * 64,
            },
            reference_checkpoint={
                "path": str(Path("/tmp/reference").resolve()),
                "model_sha256": "3" * 64,
                "metadata_sha256": "4" * 64,
            },
            selection_receipt_sha256="5" * 64,
            kl_selection_receipt_sha256="6" * 64,
            joint_stage_result_sha256="7" * 64,
            source_closure_sha256="8" * 64,
            locked_anchor_sha256="9" * 64,
            locked_build_receipt_sha256="a" * 64,
            claim_marker_sha256="b" * 64,
        )
        self.assertTrue(report["comparison"]["production_eligible"])
        self.assertTrue(report["comparison"]["evidence_significant"])
        rendered = json.dumps(report, ensure_ascii=False)
        self.assertNotIn("sample-print", rendered)
        self.assertNotIn("selection_records", rendered)
        self.assertIn("selection_record_commitment", rendered)

    def test_final_report_replaces_only_a_matching_incomplete_stub(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            claim = claim_locked_benchmark_once(
                root / "ledger",
                "a" * 64,
                {"locked_golden_anchor_sha256": "c" * 64},
            )
            output = root / "report.json"
            write_claimed_evaluation_incomplete_stub(
                output,
                claim,
                {"selection_receipt_sha256": "d" * 64},
            )
            report = {
                "schema_version": 1,
                "kind": LOCKED_EVALUATION_REPORT_KIND,
                "locked_build_receipt_sha256": claim.build_receipt_sha256,
                "claim_marker_sha256": claim.marker_sha256,
            }
            finalize_locked_evaluation_report(output, claim, report)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8"))["kind"],
                LOCKED_EVALUATION_REPORT_KIND,
            )


if __name__ == "__main__":
    unittest.main()
