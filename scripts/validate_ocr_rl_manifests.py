# -*- coding: utf-8 -*-

"""Validate OCR RL train/validation/golden manifests before allocating a GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Model.config import (  # noqa: E402
    BOS_ID,
    IMAGE_END_ID,
    IMAGE_PATCH_ID,
    IMAGE_START_ID,
)
from Model.posttrain.checkpointing import reconstruct_policy_from_checkpoint  # noqa: E402
from Model.posttrain.preference_data import OCRPromptDataset  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="completed visual checkpoint")
    parser.add_argument("--tokenizer", required=True, help="TokenizerBundle directory")
    parser.add_argument("--train", required=True, help="rl_train JSONL")
    parser.add_argument("--validation", required=True, help="rl_val JSONL")
    parser.add_argument("--golden", required=True, help="locked golden JSONL")
    parser.add_argument("--image-root", default="")
    parser.add_argument("--train-split", default="rl_train")
    parser.add_argument("--validation-split", default="rl_val")
    parser.add_argument("--golden-split", default="golden")
    parser.add_argument("--out", default="")
    return parser.parse_args(argv)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _keys(dataset: OCRPromptDataset) -> tuple[set[str], set[str], set[str], set[str]]:
    ids: set[str] = set()
    images: set[str] = set()
    hashes: set[str] = set()
    groups: set[str] = set()
    for idx in range(len(dataset)):
        row = dataset[idx]
        ids.add(row["id"])
        images.add(row["image"])
        hashes.add(row["sha256"])
        groups.add(row["group_id"])
    return ids, images, hashes, groups


def _percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * quantile)
    return ordered[index]


def _summary(
    dataset: OCRPromptDataset,
    *,
    include_reference_stats: bool = True,
) -> dict:
    rows = [dataset[idx] for idx in range(len(dataset))]
    if not include_reference_stats:
        return {
            "samples": len(rows),
            "groups": len({row["group_id"] for row in rows}),
            "locked": True,
        }
    summary = {
        "samples": len(rows),
        "groups": len({row["group_id"] for row in rows}),
        "image_bytes": sum(Path(row["image"]).stat().st_size for row in rows),
        "prompt_tokens_max": max(len(row["prompt_ids"]) for row in rows),
    }
    token_counts = [int(row["reference_token_count"]) + 1 for row in rows]
    summary.update(
        {
            "domains": dict(sorted(Counter(row["domain"] for row in rows).items())),
            "completion_tokens_including_eos": {
                "p50": _percentile(token_counts, 0.50),
                "p95": _percentile(token_counts, 0.95),
                "p99": _percentile(token_counts, 0.99),
                "max": max(token_counts),
            },
        },
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    restored = reconstruct_policy_from_checkpoint(args.checkpoint, require_vision=True)
    omvt_cfg = restored.omvt_config
    assert omvt_cfg is not None

    from Tokenizer.unified.bundle import TokenizerBundle
    from scripts.build_ocr_data import make_ocr_target_encoder

    bundle = TokenizerBundle.from_dir(args.tokenizer)
    issues = bundle.validate()
    if issues:
        raise ValueError("invalid tokenizer bundle:\n  - " + "\n  - ".join(issues))
    encode_reference = make_ocr_target_encoder(bundle.tokenizer)
    common = dict(
        encode=lambda text: bundle.encode(text, add_bos=False, add_eos=False),
        encode_reference=encode_reference,
        n_image_tokens=omvt_cfg.compress_to,
        bos_id=BOS_ID,
        image_start_id=IMAGE_START_ID,
        image_patch_id=IMAGE_PATCH_ID,
        image_end_id=IMAGE_END_ID,
        image_root=args.image_root or None,
        max_prompt_len=restored.rdt_config.max_seq_len - 1,
        validate_images=True,
        verify_image_decode=True,
        require_sha256=True,
        require_group_id=True,
        require_domain=True,
        verify_sha256=True,
    )
    golden = OCRPromptDataset(
        args.golden,
        required_split=args.golden_split,
        inspect_reference_tokens=False,
        retain_reference=False,
        inspect_prompt_tokens=False,
        **common,
    )
    golden_ids, golden_images, golden_hashes, golden_groups = _keys(golden)
    validation = OCRPromptDataset(
        args.validation,
        required_split=args.validation_split,
        excluded_ids=golden_ids,
        excluded_images=golden_images,
        excluded_sha256=golden_hashes,
        excluded_groups=golden_groups,
        **common,
    )
    val_ids, val_images, val_hashes, val_groups = _keys(validation)
    train = OCRPromptDataset(
        args.train,
        required_split=args.train_split,
        excluded_ids=golden_ids | val_ids,
        excluded_images=golden_images | val_images,
        excluded_sha256=golden_hashes | val_hashes,
        excluded_groups=golden_groups | val_groups,
        **common,
    )
    summaries = {
        "train": _summary(train),
        "validation": _summary(validation),
        # Do not expose domain or transcript-length statistics from the locked
        # final set before model selection is complete.
        "golden": _summary(golden, include_reference_stats=False),
    }
    recommended = max(
        summary["completion_tokens_including_eos"]["max"]
        for name, summary in summaries.items()
        if name != "golden"
    )
    max_prompt = max(
        summary["prompt_tokens_max"]
        for name, summary in summaries.items()
        if name != "golden"
    )
    if max_prompt + recommended > restored.rdt_config.max_seq_len:
        raise ValueError(
            f"max prompt {max_prompt} + required completion {recommended} exceeds "
            f"model max_seq_len={restored.rdt_config.max_seq_len}"
        )
    report = {
        "checkpoint": str(restored.checkpoint_dir),
        "model_max_seq_len": restored.rdt_config.max_seq_len,
        "image_tokens": omvt_cfg.compress_to,
        "recommended_max_new_tokens": recommended,
        "manifests": {
            "train_sha256": _sha256(args.train),
            "validation_sha256": _sha256(args.validation),
            "golden_sha256": _sha256(args.golden),
        },
        "splits": summaries,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
