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
from Model.ocr.tokenization import (  # noqa: E402
    OCR_NATIVE_TARGET_ENCODING,
    canonicalize_native_ocr_text,
    make_ocr_target_encoder,
    native_tokenization_contract,
)
from Model.posttrain.checkpointing import (  # noqa: E402
    load_verified_policy_metadata,
    reconstruct_policy_from_checkpoint,
)
from Model.posttrain.release_contract import (  # noqa: E402
    VISUAL_SOURCE_CONTRACT_ALIGNMENT_V3,
    VISUAL_SOURCE_CONTRACT_STREAMING_V2_RELEASE,
    admit_visual_ocr_source,
)
from Model.posttrain.ocr_manifest_builder import (  # noqa: E402
    reference_symbol_support,
)
from Model.posttrain.ocr_manifests import (  # noqa: E402
    golden_identity_keys,
    golden_identity_semantic_sha256,
    load_golden_identity_manifest,
    load_ocr_dataset_contract,
)
from Model.posttrain.preference_data import OCRPromptDataset  # noqa: E402
from Model.training.checkpoint import resolve_checkpoint_dir  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="completed visual checkpoint")
    parser.add_argument(
        "--visual-source-contract",
        choices=(
            VISUAL_SOURCE_CONTRACT_ALIGNMENT_V3,
            VISUAL_SOURCE_CONTRACT_STREAMING_V2_RELEASE,
        ),
        default=VISUAL_SOURCE_CONTRACT_ALIGNMENT_V3,
        help=(
            "visual checkpoint admission contract; historical streaming-v2 "
            "release admission is explicit and never used as a fallback"
        ),
    )
    parser.add_argument("--tokenizer", required=True, help="TokenizerBundle directory")
    parser.add_argument("--train", required=True, help="rl_train JSONL")
    parser.add_argument("--validation", required=True, help="rl_val JSONL")
    parser.add_argument(
        "--golden-identity",
        required=True,
        help="public transcript-free golden identity JSONL",
    )
    parser.add_argument("--image-root", default="")
    parser.add_argument(
        "--dataset-contract",
        default="",
        help="defaults to dataset_contract.json beside --train",
    )
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


def _visual_source_reconstruction_inputs(
    lineage: dict[str, object],
) -> tuple[Path, str, str]:
    raw_checkpoint = lineage.get("source_checkpoint")
    if not isinstance(raw_checkpoint, str) or not raw_checkpoint:
        raise ValueError("visual source lineage has no source_checkpoint")

    def required_sha256(field: str) -> str:
        value = lineage.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise ValueError(
                f"visual source lineage {field} must be a lowercase SHA256 digest"
            )
        return value

    return (
        resolve_checkpoint_dir(raw_checkpoint),
        required_sha256("source_checkpoint_model_sha256"),
        required_sha256("source_checkpoint_metadata_sha256"),
    )


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
            "reference_symbol_support": reference_symbol_support(rows),
        },
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from Tokenizer.unified.bundle import TokenizerBundle

    bundle = TokenizerBundle.from_dir(args.tokenizer)
    issues = bundle.validate()
    if issues:
        raise ValueError("invalid tokenizer bundle:\n  - " + "\n  - ".join(issues))
    token_contract = native_tokenization_contract(bundle.tokenizer, args.tokenizer)
    tokenizer_vocab_extent = (
        max(int(index) for index in bundle.tokenizer.vocab.values()) + 1
    )
    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint)
    source_metadata: dict[str, object] | None = None
    metadata_sha256_before_admission = ""
    if args.visual_source_contract == VISUAL_SOURCE_CONTRACT_ALIGNMENT_V3:
        (
            source_metadata,
            metadata_sha256_before_admission,
        ) = load_verified_policy_metadata(checkpoint_dir)
    visual_source = admit_visual_ocr_source(
        args.visual_source_contract,
        checkpoint_dir=checkpoint_dir,
        metadata=source_metadata,
        runtime_native_tokenization_contract=token_contract,
        tokenizer_vocab_extent=tokenizer_vocab_extent,
    )
    lineage = dict(visual_source["lineage"])
    (
        admitted_checkpoint_dir,
        expected_model_sha256,
        expected_metadata_sha256,
    ) = _visual_source_reconstruction_inputs(lineage)
    if admitted_checkpoint_dir.resolve() != checkpoint_dir.resolve():
        raise ValueError("admitted visual source resolves to a different checkpoint")
    if (
        metadata_sha256_before_admission
        and metadata_sha256_before_admission != expected_metadata_sha256
    ):
        raise ValueError("visual checkpoint metadata changed during source admission")

    # Source identity and tokenizer compatibility are proven before the 1B
    # model is allocated.  In particular, streaming-v2 metadata is loaded only
    # by the byte-verified admission path above.
    restored = reconstruct_policy_from_checkpoint(
        admitted_checkpoint_dir,
        require_vision=True,
        metadata_override=visual_source["metadata"],
        expected_metadata_sha256=expected_metadata_sha256,
        expected_model_sha256=expected_model_sha256,
    )
    if restored.metadata != visual_source["metadata"]:
        raise ValueError("visual checkpoint metadata changed after source admission")
    if restored.model_sha256 != expected_model_sha256:
        raise ValueError("visual checkpoint model SHA256 changed after source admission")
    if restored.metadata_sha256 != expected_metadata_sha256:
        raise ValueError(
            "visual checkpoint metadata SHA256 changed after source admission"
        )
    omvt_cfg = restored.omvt_config
    assert omvt_cfg is not None

    dataset_contract_path = (
        Path(args.dataset_contract)
        if args.dataset_contract
        else Path(args.train).parent / "dataset_contract.json"
    )
    dataset_contract = load_ocr_dataset_contract(
        dataset_contract_path,
        train_manifest=args.train,
        validation_manifest=args.validation,
        golden_identity_manifest=args.golden_identity,
        tokenization_contract=token_contract,
    )
    contract_image_root = Path(
        str(dataset_contract["resolved_image_root"])
    ).resolve()
    if args.image_root and Path(args.image_root).resolve() != contract_image_root:
        raise ValueError(
            "--image-root differs from the materialized public dataset root"
        )
    args.image_root = str(contract_image_root)
    encode_reference = make_ocr_target_encoder(
        bundle.tokenizer,
        mode=OCR_NATIVE_TARGET_ENCODING,
    )
    common = dict(
        encode=lambda text: bundle.encode(text, add_bos=False, add_eos=False),
        encode_reference=encode_reference,
        n_image_tokens=omvt_cfg.compress_to,
        bos_id=BOS_ID,
        image_start_id=IMAGE_START_ID,
        image_patch_id=IMAGE_PATCH_ID,
        image_end_id=IMAGE_END_ID,
        canonicalize_reference=canonicalize_native_ocr_text,
        image_root=args.image_root or None,
        max_prompt_len=restored.rdt_config.max_seq_len - 1,
        validate_images=True,
        verify_image_decode=True,
        require_sha256=True,
        require_group_id=True,
        require_domain=True,
        verify_sha256=True,
    )
    golden_identity = load_golden_identity_manifest(
        args.golden_identity,
        required_split=args.golden_split,
    )
    golden_ids, golden_hashes, golden_groups = golden_identity_keys(
        golden_identity
    )
    golden_images: set[str] = set()
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
        "golden_identity": {
            "samples": len(golden_identity),
            "groups": len(golden_groups),
            "locked": True,
        },
    }
    recommended = max(
        summary["completion_tokens_including_eos"]["max"]
        for name, summary in summaries.items()
        if name != "golden_identity"
    )
    max_prompt = max(
        summary["prompt_tokens_max"]
        for name, summary in summaries.items()
        if name != "golden_identity"
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
            "golden_identity_sha256": _sha256(args.golden_identity),
            "golden_identity_semantic_sha256": (
                golden_identity_semantic_sha256(golden_identity)
            ),
        },
        "ocr_tokenization_contract": token_contract,
        "visual_source": visual_source["lineage"],
        "dataset_contract_sha256": dataset_contract[
            "contract_file_sha256"
        ],
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
