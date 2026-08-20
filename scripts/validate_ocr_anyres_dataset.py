#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Read-only admission validator for all four DoL OCR anyres splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Model.config import BOS_ID, EOS_ID  # noqa: E402
from Model.ocr.anyres_preprocess_contract import (  # noqa: E402
    validate_anyres_preprocess_contract,
)
from Model.ocr.tokenization import (  # noqa: E402
    OCR_NATIVE_TARGET_ENCODING,
    make_ocr_target_encoder,
    tokenizer_manifest_canonical_sha256,
    tokenizer_vocab_sha256,
)
from Model.posttrain.ocr_anyres_data import (  # noqa: E402
    AnyresOCRDataset,
    load_anyres_jsonl,
    validate_anyres_ready_dataset,
)
from Model.posttrain.ocr_anyres_manifest import (  # noqa: E402
    ANYRES_PUBLIC_SPLITS,
    ANYRES_VALIDATION_SPLITS,
    canonical_json_sha256,
    validate_anyres_assets_views_samples,
)
from Model.posttrain.ocr_quota_sampler import OCRQuotaSampler  # noqa: E402
from Model.posttrain.text_replay import (  # noqa: E402
    TEXT_REPLAY_PUBLIC_SPLITS,
    TextReplayPartition,
)
from Model.omvt.native_planner import raw_patch_token_count  # noqa: E402
from Tokenizer.unified.bundle import TokenizerBundle  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--views", required=True)
    parser.add_argument("--train-samples", required=True)
    parser.add_argument("--sft-validation-samples", required=True)
    parser.add_argument("--kl-selection-samples", required=True)
    parser.add_argument("--formal-monitor-samples", required=True)
    parser.add_argument("--preprocess-contract", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--text-replay", required=True)
    parser.add_argument("--reviewed-exclusions", default="")
    parser.add_argument("--out", default="")
    return parser.parse_args(argv)


def _strict_json(path: str | Path, *, where: str) -> tuple[dict[str, Any], str]:
    source = Path(path)
    if source.is_symlink():
        raise ValueError(f"{where} must not be a symlink")
    try:
        raw = source.read_bytes()
    except FileNotFoundError as exc:
        raise ValueError(f"{where} does not exist: {source}") from exc

    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{where} has duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(raw.decode("utf-8", errors="strict"), object_pairs_hook=object_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{where} is not strict UTF-8 JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{where} must contain one JSON object")
    return payload, hashlib.sha256(raw).hexdigest()


def _load_exclusions(path: str) -> tuple[set[str], set[str], dict[str, Any]]:
    if not path:
        payload = {
            "schema_version": 1,
            "document_ids": [],
            "ngram_sha256": [],
        }
        return set(), set(), {
            "provided": False,
            "canonical_sha256": canonical_json_sha256(payload),
            "document_ids": 0,
            "ngram_sha256": 0,
        }
    payload, file_sha256 = _strict_json(path, where="reviewed exclusions")
    if set(payload) != {"schema_version", "document_ids", "ngram_sha256"}:
        raise ValueError("reviewed exclusions fields must be exact")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("reviewed exclusions schema_version must be 1")
    document_ids = payload["document_ids"]
    ngram_hashes = payload["ngram_sha256"]
    if not isinstance(document_ids, list) or any(
        not isinstance(value, str) or not value or value != value.strip()
        for value in document_ids
    ):
        raise ValueError("reviewed exclusion document_ids must be stripped strings")
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("reviewed exclusion document_ids must be unique")
    if not isinstance(ngram_hashes, list) or any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in ngram_hashes
    ):
        raise ValueError("reviewed exclusion ngram_sha256 values must be lowercase SHA256")
    if len(ngram_hashes) != len(set(ngram_hashes)):
        raise ValueError("reviewed exclusion ngram_sha256 values must be unique")
    return set(document_ids), set(ngram_hashes), {
        "provided": True,
        "file_sha256": file_sha256,
        "canonical_sha256": canonical_json_sha256(payload),
        "document_ids": len(document_ids),
        "ngram_sha256": len(ngram_hashes),
    }


def _require_manifest_split(rows: list[dict], allowed: set[str], name: str) -> None:
    wrong = [str(row.get("sample_id", "<unknown>")) for row in rows if row.get("split") not in allowed]
    if wrong:
        raise ValueError(f"{name} contains rows for another split: {wrong[:8]}")


def _length_summary(lengths: list[int]) -> dict[str, Any]:
    if not lengths:
        raise ValueError("token-length summary cannot be empty")
    ordered = sorted(lengths)

    def percentile(value: float) -> int:
        return ordered[round((len(ordered) - 1) * value)]

    return {
        "samples": len(lengths),
        "min": ordered[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
        "total": sum(ordered),
    }


def _verify_references(samples: list[dict], encoder, split: str) -> dict[str, Any]:
    lengths: list[int] = []
    for index, sample in enumerate(samples):
        reference = sample["reference_model"]["text"]
        token_ids = encoder(reference)
        expected = int(sample["reference_token_count"])
        if len(token_ids) != expected:
            raise ValueError(
                f"{split} sample {sample['sample_id']!r} native token count "
                f"differs: manifest={expected}, tokenizer={len(token_ids)}"
            )
        if EOS_ID in token_ids:
            raise ValueError(f"{split} sample {index} native reference contains EOS")
        lengths.append(len(token_ids))
    if encoder.stats["byte_fallback"] != 0:
        raise ValueError("native reference validation used byte fallback")
    if encoder.stats["canonicalized"] != 0:
        raise ValueError("reference_model text is not already native-canonical")
    summary = _length_summary(lengths)
    summary["including_eos_max"] = summary["max"] + 1
    return summary


def _sampler_contract(dataset: AnyresOCRDataset) -> dict[str, Any]:
    mapping = {
        sample_id: bucket
        for sample_id, bucket in zip(dataset.sample_ids, dataset.quota_buckets)
    }
    sampler = OCRQuotaSampler(
        mapping,
        global_batch_size=20,
        seed=0,
        world_size=1,
    )
    state = sampler.state_dict()
    return {
        "constructible": True,
        "config_sha256": state["config_sha256"],
        "sample_list_sha256": state["sample_list_sha256"],
    }


def _qa_summary(normalized: dict[str, Any]) -> dict[str, Any]:
    views = normalized["views"]
    statuses = Counter(str(view["qa"]["review_status"]) for view in views)
    return {
        "view_qa_schema_verified": True,
        "safe_views": len(views),
        "review_status": dict(sorted(statuses.items())),
        "coverage_min": min(float(view["qa"]["coverage"]) for view in views) if views else None,
        "cc_crossing_total": sum(int(view["qa"]["cc_crossing_count"]) for view in views),
        "cc_uncovered_total": sum(int(view["qa"]["cc_uncovered_count"]) for view in views),
    }


def _resource_budget_summary(
    normalized: dict[str, Any],
    preprocess: dict[str, Any],
) -> dict[str, Any]:
    patch_shapes = preprocess["patch_contract"]["shapes_hw"]
    max_decode = int(
        preprocess["budgets"]["decode"]["max_pixels_per_asset"]
    )
    max_windows = int(
        preprocess["budgets"]["window"]["max_windows_per_asset"]
    )
    halo_px = int(
        preprocess["budgets"]["window"]["halo_pixels_per_side"]
    )
    max_raw_per_view = int(
        preprocess["budgets"]["patch"]["max_raw_tokens_per_view"]
    )
    max_raw_per_asset = int(
        preprocess["budgets"]["patch"]["max_raw_tokens_per_asset"]
    )
    assets = {str(row["asset_id"]): row for row in normalized["assets"]}
    views = {str(row["view_id"]): row for row in normalized["views"]}
    referenced_assets = {str(row["asset_id"]) for row in normalized["samples"]}
    referenced_views = {
        str(view_id)
        for row in normalized["samples"]
        for view_id in row["view_ids"]
    }
    max_observed_view_tokens = 0
    max_observed_asset_tokens = 0
    max_observed_views = 0
    for asset_id in sorted(referenced_assets):
        asset = assets[asset_id]
        if asset["native_plan"]["preprocess_contract_sha256"] != preprocess[
            "contract_canonical_sha256"
        ]:
            raise ValueError(
                f"asset {asset_id!r} native plan preprocess hash differs"
            )
        for role in ("raw", "canonical"):
            pixels = int(asset[f"{role}_width"]) * int(asset[f"{role}_height"])
            if pixels > max_decode:
                raise ValueError(
                    f"asset {asset_id!r} {role} pixels exceed preprocess budget"
                )
        plan = asset["native_plan"]["payload"]
        if plan["patch_shapes"] != patch_shapes:
            raise ValueError(
                f"asset {asset_id!r} native plan patch shapes differ from preprocess"
            )
        if int(plan["max_raw_patch_tokens"]) != max_raw_per_view:
            raise ValueError(
                f"asset {asset_id!r} native plan raw-token budget differs"
            )
        if int(plan["max_windows"]) != max_windows:
            raise ValueError(f"asset {asset_id!r} native plan max_windows differs")
        if int(plan["halo_px"]) != halo_px:
            raise ValueError(f"asset {asset_id!r} native plan halo differs")

    per_asset_tokens: dict[str, int] = {asset_id: 0 for asset_id in referenced_assets}
    per_asset_views: dict[str, int] = {asset_id: 0 for asset_id in referenced_assets}
    for view_id in sorted(referenced_views):
        view = views[view_id]
        asset_id = str(view["asset_id"])
        width = int(view["derived_width"])
        height = int(view["derived_height"])
        if width * height > max_decode:
            raise ValueError(f"view {view_id!r} pixels exceed preprocess budget")
        observed = raw_patch_token_count(height, width, patch_shapes)
        declared = int(view["transform"]["raw_patch_tokens"])
        if observed != declared:
            raise ValueError(
                f"view {view_id!r} raw patch tokens differ from geometry"
            )
        if observed > max_raw_per_view:
            raise ValueError(f"view {view_id!r} exceeds per-view raw-token budget")
        per_asset_tokens[asset_id] += observed
        per_asset_views[asset_id] += 1
        max_observed_view_tokens = max(max_observed_view_tokens, observed)
    for asset_id in sorted(referenced_assets):
        if per_asset_views[asset_id] > max_windows:
            raise ValueError(f"asset {asset_id!r} exceeds max_windows")
        if per_asset_tokens[asset_id] > max_raw_per_asset:
            raise ValueError(f"asset {asset_id!r} exceeds total raw-token budget")
        max_observed_asset_tokens = max(
            max_observed_asset_tokens,
            per_asset_tokens[asset_id],
        )
        max_observed_views = max(max_observed_views, per_asset_views[asset_id])
    return {
        "planner_contract_verified": True,
        "assets_checked": len(referenced_assets),
        "views_checked": len(referenced_views),
        "max_observed_views_per_asset": max_observed_views,
        "max_observed_raw_tokens_per_view": max_observed_view_tokens,
        "max_observed_raw_tokens_per_asset": max_observed_asset_tokens,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    preprocess_raw, preprocess_file_sha256 = _strict_json(
        args.preprocess_contract,
        where="preprocess contract",
    )
    preprocess = validate_anyres_preprocess_contract(preprocess_raw)
    preprocess_sha256 = str(preprocess["contract_canonical_sha256"])
    output_budget = int(
        preprocess["budgets"]["output"]["recommended_max_new_tokens"]
    )

    bundle = TokenizerBundle.from_dir(args.tokenizer)
    issues = bundle.validate()
    if issues:
        raise ValueError("invalid tokenizer bundle:\n  - " + "\n  - ".join(issues))
    encoder = make_ocr_target_encoder(
        bundle.tokenizer,
        mode=OCR_NATIVE_TARGET_ENCODING,
    )

    assets = load_anyres_jsonl(args.assets)
    views = load_anyres_jsonl(args.views)
    manifest_paths = {
        "train": args.train_samples,
        "sft_validation": args.sft_validation_samples,
        "kl_selection": args.kl_selection_samples,
        "formal_monitor": args.formal_monitor_samples,
    }
    rows_by_split = {
        split: load_anyres_jsonl(manifest_paths[split])
        for split in ANYRES_PUBLIC_SPLITS
    }
    for split in ANYRES_PUBLIC_SPLITS:
        _require_manifest_split(
            rows_by_split[split],
            {split},
            f"{split} sample manifest",
        )
    normalized, quarantine = validate_anyres_assets_views_samples(
        assets,
        views,
        [
            row
            for split in ANYRES_PUBLIC_SPLITS
            for row in rows_by_split[split]
        ],
    )
    ready_admission = validate_anyres_ready_dataset(
        args.root,
        expected_preprocess_contract_sha256=preprocess_sha256,
        assets_manifest=args.assets,
        views_manifest=args.views,
        train_manifest=args.train_samples,
        sft_validation_manifest=args.sft_validation_samples,
        kl_selection_manifest=args.kl_selection_samples,
        formal_monitor_manifest=args.formal_monitor_samples,
        normalized=normalized,
    )
    resource_budgets = _resource_budget_summary(normalized, preprocess)

    datasets = {
        split: AnyresOCRDataset(
            root=args.root,
            assets_manifest=args.assets,
            views_manifest=args.views,
            samples_manifest=manifest_paths[split],
            split=split,
            expected_preprocess_contract_sha256=preprocess_sha256,
            max_decode_pixels=int(
                preprocess["budgets"]["decode"]["max_pixels_per_asset"]
            ),
            max_views_per_sample=int(
                preprocess["budgets"]["window"]["max_windows_per_asset"]
            ),
            image_delivery="path",
        )
        for split in ANYRES_PUBLIC_SPLITS
    }
    split_contract_hashes = {
        split: datasets[split].contract_sha256
        for split in ANYRES_PUBLIC_SPLITS
    }
    if len(set(split_contract_hashes.values())) != len(ANYRES_PUBLIC_SPLITS):
        raise ValueError("all four image splits must have distinct dataset contracts")
    if any(count <= 0 for count in datasets["train"].quota_counts.values()):
        raise ValueError("train split must have all four non-empty quota buckets")
    for split in ANYRES_VALIDATION_SPLITS:
        if any(count < 2 for count in datasets[split].quota_counts.values()):
            raise ValueError(
                f"{split} split needs at least two samples in every bucket"
            )

    accepted_by_split = {
        split: [
            row for row in normalized["samples"] if row["split"] == split
        ]
        for split in ANYRES_PUBLIC_SPLITS
    }
    for split in ANYRES_PUBLIC_SPLITS:
        accepted_ids = {
            str(row["sample_id"]) for row in accepted_by_split[split]
        }
        if accepted_ids != set(datasets[split].sample_ids):
            raise ValueError(
                f"global and {split}-only validation select different samples"
            )

    reference_lengths = {
        split: _verify_references(accepted_by_split[split], encoder, split)
        for split in ANYRES_PUBLIC_SPLITS
    }
    recommended = max(
        int(reference_lengths[split]["including_eos_max"])
        for split in ANYRES_PUBLIC_SPLITS
    )
    if recommended != output_budget:
        raise ValueError(
            "recommended_max_new_tokens must equal max native reference tokens "
            f"plus EOS: preprocess={output_budget}, observed={recommended}"
        )

    excluded_documents, excluded_ngrams, exclusions_report = _load_exclusions(
        args.reviewed_exclusions
    )
    text_partition = TextReplayPartition(
        args.text_replay,
        native_encoder=encoder,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        max_seq_len=int(
            preprocess["budgets"]["context"]["max_sequence_tokens"]
        ),
        exclusion_document_ids=excluded_documents,
        exclusion_ngram_sha256=excluded_ngrams,
    )
    text_replay_report = {
        "manifest_sha256": text_partition.partition_contract[
            "manifest_file_sha256"
        ],
        "partition_contract": text_partition.partition_contract,
        "partition_contract_sha256": text_partition.contract_sha256,
        "global_cross_split_leakage_validated": True,
        "splits": {
            split: {
                "contract": text_partition.dataset(split).dataset_contract,
                "contract_sha256": text_partition.dataset(split).contract_sha256,
                "document_sha256": text_partition.dataset(split).document_sha256,
                "text_sha256": text_partition.dataset(split).text_sha256,
                "token_stats": text_partition.dataset(split).token_stats,
            }
            for split in TEXT_REPLAY_PUBLIC_SPLITS
        },
    }

    return {
        "schema_version": 1,
        "kind": "dol_ocr_anyres_dataset_validation_report",
        "read_only": True,
        "model_loaded": False,
        "gpu_used": False,
        "golden_opened": False,
        "preprocess_contract": {
            "contract_canonical_sha256": preprocess_sha256,
            "file_sha256": preprocess_file_sha256,
            "planner_contract": preprocess["planner_contract"],
            "budgets": preprocess["budgets"],
            "implementation_sources": preprocess["implementation_sources"],
        },
        "ready_dataset": ready_admission,
        "manifests": {
            "assets_sha256": _file_sha256(args.assets),
            "views_sha256": _file_sha256(args.views),
            **{
                f"{split}_samples_sha256": _file_sha256(
                    manifest_paths[split]
                )
                for split in ANYRES_PUBLIC_SPLITS
            },
            "global_cross_split_leakage_validated": True,
        },
        "datasets": {
            split: {
                "contract": datasets[split].dataset_contract,
                "contract_sha256": datasets[split].contract_sha256,
                "quota_counts": dict(datasets[split].quota_counts),
                "native_reference_tokens": reference_lengths[split],
                "quota_sampler": _sampler_contract(datasets[split]),
            }
            for split in ANYRES_PUBLIC_SPLITS
        },
        "quarantine": {
            "entries": quarantine,
            "count": len(quarantine),
            "canonical_sha256": canonical_json_sha256(quarantine),
        },
        "qa": {
            **_qa_summary(normalized),
            "resource_budgets": resource_budgets,
        },
        "tokenizer": {
            "mode": OCR_NATIVE_TARGET_ENCODING,
            "manifest_canonical_sha256": tokenizer_manifest_canonical_sha256(
                args.tokenizer
            ),
            "vocab_sha256": tokenizer_vocab_sha256(bundle.tokenizer),
            "fallback_count": int(encoder.stats["byte_fallback"]),
        },
        "reviewed_exclusions": exclusions_report,
        "text_replay": text_replay_report,
        "recommended_max_new_tokens": recommended,
        "preprocess_output_budget": output_budget,
    }


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    rendered = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"
    if args.out:
        _atomic_write(args.out, rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
