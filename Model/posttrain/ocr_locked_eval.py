# -*- coding: utf-8 -*-

"""Shared batching, redaction, gates, and finalization for locked AnyRes OCR."""

from __future__ import annotations

import copy
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from Model.config import BOS_ID, EOS_ID, PAD_ID
from Model.ocr.tokenization import canonical_json_sha256
from Model.posttrain.ocr_anyres_grpo_run import (
    paired_stratified_validation_delta,
)
from Model.posttrain.ocr_joint_eval import (
    DEPLOYMENT_BUCKET_WEIGHTS,
    joint_eval_eligibility,
)
from Model.posttrain.ocr_locked_contract import (
    LOCKED_INCOMPLETE_STUB_KIND,
    LockedGoldenClaim,
    validate_locked_golden_claim,
)
from Model.posttrain.ocr_locked_data import LockedBenchmarkBatchSource
from Model.posttrain.text_replay import TextReplayCollator


LOCKED_EVALUATION_REPORT_KIND = "dol_ocr_anyres_locked_evaluation_report_v1"


def _require_sha256(value: object, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{where} must be lowercase SHA-256")
    return value


def _checkpoint_identity(value: Mapping[str, Any], where: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "model_sha256",
        "metadata_sha256",
    }:
        raise ValueError(f"{where} checkpoint identity fields differ")
    path = value["path"]
    if not isinstance(path, str) or path != str(Path(path).resolve()):
        raise ValueError(f"{where} checkpoint path must be absolute")
    return {
        "path": path,
        "model_sha256": _require_sha256(value["model_sha256"], f"{where}.model"),
        "metadata_sha256": _require_sha256(
            value["metadata_sha256"], f"{where}.metadata"
        ),
    }


def build_locked_evaluation_batches(
    source: LockedBenchmarkBatchSource,
    *,
    image_collator,
    text_batch_size: int = 4,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(source, LockedBenchmarkBatchSource):
        raise TypeError("source must be a LockedBenchmarkBatchSource")
    if not callable(image_collator):
        raise TypeError("image_collator must be callable")
    if isinstance(text_batch_size, bool) or not isinstance(text_batch_size, int) or (
        text_batch_size <= 0
    ):
        raise ValueError("text_batch_size must be positive")
    by_bucket = {bucket: [] for bucket in DEPLOYMENT_BUCKET_WEIGHTS}
    for record in source.images:
        style = "print" if record.bucket == "print" else "handwritten"
        difficulty = (
            None if style == "print" else record.bucket.removeprefix("handwritten_")
        )
        view_id = f"locked-view:{record.id}"
        sample = {
            "sample_id": record.id,
            "visual_contract": "dol_ocr_anyres_v2",
            "qa_state": "accepted",
            "reference_model": {"text": record.reference},
            "reference_token_count": len(record.reference_token_ids),
            "reading_order": [view_id],
            "style": style,
            "difficulty": difficulty,
        }
        image_payload = {
            "delivery": "bytes",
            "bytes": record.asset_bytes,
            "metadata": {"view_id": view_id},
        }
        by_bucket[record.bucket].append(
            {
                "dataset_contract_sha256": source.image_dataset_contract_sha256,
                "sample": sample,
                "canonical_image": image_payload,
                "derived_images": [image_payload],
                "quota_bucket": record.bucket,
            }
        )
    image_batches: list[dict[str, Any]] = []
    for bucket in DEPLOYMENT_BUCKET_WEIGHTS:
        rows = by_bucket[bucket]
        if len(rows) < 2:
            raise ValueError(f"locked bucket {bucket!r} needs at least two images")
        start = 0
        while start < len(rows):
            remaining = len(rows) - start
            width = 3 if remaining == 3 else 2
            if remaining == 1:
                raise ValueError(f"locked bucket {bucket!r} would create a singleton")
            image_batches.append(image_collator(rows[start:start + width]))
            start += width

    text_rows = [
        {
            "input_ids": [BOS_ID, *record.reference_token_ids, EOS_ID],
            "attention_mask": [1] * (len(record.reference_token_ids) + 2),
            "labels": [BOS_ID, *record.reference_token_ids, EOS_ID],
            "metadata": {
                "id": record.id,
                "dataset_contract_sha256": source.text_dataset_contract_sha256,
            },
        }
        for record in source.texts
    ]
    text_collator = TextReplayCollator(pad_id=PAD_ID)
    text_batches = [
        text_collator(text_rows[start:start + text_batch_size])
        for start in range(0, len(text_rows), text_batch_size)
    ]
    return image_batches, text_batches


def redact_locked_evaluation_report(report: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(report, Mapping):
        raise TypeError("locked evaluation report must be a mapping")
    normalized = copy.deepcopy(dict(report))
    records = normalized.pop("selection_records", None)
    if not isinstance(records, list) or not records:
        raise ValueError("locked evaluation report has no paired selection records")
    buckets = Counter()
    for row in records:
        if not isinstance(row, Mapping) or row.get("bucket") not in (
            DEPLOYMENT_BUCKET_WEIGHTS
        ):
            raise ValueError("locked selection record is malformed")
        buckets[str(row["bucket"])] += 1
    normalized["selection_record_commitment"] = {
        "canonical_sha256": canonical_json_sha256(records),
        "count": len(records),
        "bucket_counts": {
            bucket: buckets[bucket] for bucket in DEPLOYMENT_BUCKET_WEIGHTS
        },
        "redaction": "sample_ids_and_per_sample_edits_omitted",
    }
    return normalized


def build_locked_evaluation_report(
    *,
    selected_report: Mapping[str, Any],
    reference_report: Mapping[str, Any],
    selected_checkpoint: Mapping[str, Any],
    reference_checkpoint: Mapping[str, Any],
    selection_receipt_sha256: str,
    kl_selection_receipt_sha256: str,
    joint_stage_result_sha256: str,
    source_closure_sha256: str,
    locked_anchor_sha256: str,
    locked_build_receipt_sha256: str,
    claim_marker_sha256: str,
) -> dict[str, Any]:
    for name, report in (
        ("selected", selected_report),
        ("reference", reference_report),
    ):
        if not isinstance(report, Mapping):
            raise TypeError(f"{name}_report must be a mapping")
    if selected_report.get("image_dataset_contract_sha256") != reference_report.get(
        "image_dataset_contract_sha256"
    ):
        raise ValueError("selected/reference locked image contracts differ")
    selected_text = selected_report.get("text_replay")
    reference_text = reference_report.get("text_replay")
    if not isinstance(selected_text, Mapping) or not isinstance(
        reference_text, Mapping
    ) or selected_text.get("dataset_contract_sha256") != reference_text.get(
        "dataset_contract_sha256"
    ):
        raise ValueError("selected/reference locked text contracts differ")
    baseline_buckets = {
        bucket: float(
            reference_report["real"]["buckets"][bucket]["raw_grapheme_cer"]
        )
        for bucket in DEPLOYMENT_BUCKET_WEIGHTS
    }
    eligibility = joint_eval_eligibility(
        selected_report,
        baseline_bucket_cer=baseline_buckets,
        baseline_text_token_nll=float(reference_text["token_nll"]),
        min_relative_cer_improvement=0.0,
    )
    improvement = paired_stratified_validation_delta(
        reference_report,
        selected_report,
    )
    evidence_significant = float(improvement["ci95_low"]) > 0.0
    digests = {
        "selection_receipt_sha256": selection_receipt_sha256,
        "kl_selection_receipt_sha256": kl_selection_receipt_sha256,
        "joint_stage_result_sha256": joint_stage_result_sha256,
        "source_closure_sha256": source_closure_sha256,
        "locked_anchor_sha256": locked_anchor_sha256,
        "locked_build_receipt_sha256": locked_build_receipt_sha256,
        "claim_marker_sha256": claim_marker_sha256,
    }
    for field, digest in digests.items():
        _require_sha256(digest, field)
    payload = {
        "schema_version": 1,
        "kind": LOCKED_EVALUATION_REPORT_KIND,
        "selected_checkpoint": _checkpoint_identity(
            selected_checkpoint, "selected"
        ),
        "reference_checkpoint": _checkpoint_identity(
            reference_checkpoint, "reference"
        ),
        **digests,
        "selected": redact_locked_evaluation_report(selected_report),
        "reference": redact_locked_evaluation_report(reference_report),
        "comparison": {
            "production_eligible": eligibility["eligible"],
            "production_gate": eligibility,
            "paired_reference_minus_selected_weighted_cer": improvement,
            "evidence_significant": evidence_significant,
        },
    }
    payload["canonical_sha256"] = canonical_json_sha256(payload)
    return payload


def finalize_locked_evaluation_report(
    destination: str | Path,
    claim: LockedGoldenClaim,
    report: Mapping[str, Any],
) -> Path:
    marker = validate_locked_golden_claim(claim)
    target = Path(destination).absolute()
    if target.is_symlink() or not target.is_file():
        raise ValueError("locked report must already be an incomplete regular file")
    try:
        current = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("locked incomplete report is unreadable") from exc
    if (
        current.get("kind") != LOCKED_INCOMPLETE_STUB_KIND
        or current.get("status") != "claimed_evaluation_incomplete"
        or current.get("build_receipt_sha256") != claim.build_receipt_sha256
        or current.get("claim_marker_sha256") != claim.marker_sha256
    ):
        raise ValueError("locked incomplete report does not match the claim")
    payload = copy.deepcopy(dict(report))
    if payload.get("kind") != LOCKED_EVALUATION_REPORT_KIND:
        raise ValueError("final locked report kind differs")
    if payload.get("locked_build_receipt_sha256") != claim.build_receipt_sha256 or (
        payload.get("claim_marker_sha256") != claim.marker_sha256
    ) or marker["claim_payload_sha256"] != current.get("claim_payload_sha256"):
        raise ValueError("final locked report claim binding differs")
    rendered = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.tmp-",
            dir=target.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
        temporary_name = None
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
    return target


__all__ = [
    "LOCKED_EVALUATION_REPORT_KIND",
    "build_locked_evaluation_batches",
    "build_locked_evaluation_report",
    "finalize_locked_evaluation_report",
    "redact_locked_evaluation_report",
]
