#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Visual-stage supervised alignment for the reviewed DoL OCR anyres path."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Model.config import BOS_ID, EOS_ID, PAD_ID, TrainingConfig  # noqa: E402
from Model.ocr.anyres_preprocess_contract import (  # noqa: E402
    validate_anyres_preprocess_contract,
)
from Model.ocr.tokenization import (  # noqa: E402
    make_ocr_target_encoder,
    native_tokenization_contract,
)
from Model.posttrain.ocr_anyres_collator import AnyresOCRSFTCollator  # noqa: E402
from Model.posttrain.ocr_anyres_data import AnyresOCRDataset  # noqa: E402
from Model.posttrain.ocr_anyres_manifest import ANYRES_PUBLIC_SPLITS  # noqa: E402
from Model.posttrain.ocr_joint_contract import (  # noqa: E402
    admit_and_prepare_anyres_policy,
)
from Model.posttrain.ocr_joint_eval import (  # noqa: E402
    DEPLOYMENT_BUCKET_WEIGHTS,
    evaluate_ocr_joint,
    joint_eval_eligibility,
)
from Model.posttrain.ocr_joint_trainer import (  # noqa: E402
    configure_visual_stage_trainable,
    train_visual_cycle,
)
from Model.posttrain.ocr_quota_sampler import OCRQuotaSampler  # noqa: E402
from Model.posttrain.release_contract import (  # noqa: E402
    VISUAL_SOURCE_CONTRACT_STREAMING_V2_RELEASE,
    admit_visual_ocr_source,
)
from Model.posttrain.text_replay import (  # noqa: E402
    TEXT_REPLAY_PUBLIC_SPLITS,
    TextReplayCollator,
    TextReplayPartition,
)
from Model.posttrain.verified_prefetch import VerifiedBatchPrefetcher  # noqa: E402
from Model.training.checkpoint import save_checkpoint  # noqa: E402
from Model.training.optim import (  # noqa: E402
    build_ocr_joint_adamw,
    build_scheduler,
)
from Tokenizer.multimodal import (  # noqa: E402
    NativeImageProcessorV2,
    PILImageProcessor,
)
from Tokenizer.unified.bundle import TokenizerBundle  # noqa: E402
from scripts.validate_ocr_anyres_dataset import (  # noqa: E402
    _load_exclusions,
    _strict_json,
    build_report as build_dataset_report,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["visual", "joint"], default="visual")
    parser.add_argument("--dist", choices=["single", "ddp", "fsdp"], default="single")
    parser.add_argument("--resume", default="")
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--views", required=True)
    parser.add_argument("--train-samples", required=True)
    parser.add_argument("--sft-validation-samples", required=True)
    parser.add_argument("--kl-selection-samples", required=True)
    parser.add_argument("--formal-monitor-samples", required=True)
    parser.add_argument("--preprocess-contract", required=True)
    parser.add_argument("--text-replay", required=True)
    parser.add_argument("--reviewed-exclusions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cycles", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tower-lr", type=float, required=True)
    parser.add_argument("--bridge-lr", type=float, required=True)
    parser.add_argument("--projector-lr", type=float, required=True)
    parser.add_argument("--lm-lr", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-cycles", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--keep-last", type=int, default=3)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.stage != "visual":
        raise ValueError("this entry point only admits stage=visual")
    if args.dist != "single":
        raise ValueError("this entry point is not yet validated for distributed training")
    if args.resume:
        raise ValueError("resume is disabled until optimizer/sampler replay E2E is admitted")
    for name in (
        "cycles",
        "global_batch_size",
        "save_every",
        "eval_every",
        "early_stop_patience",
        "keep_last",
    ):
        value = getattr(args, name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if type(args.warmup_cycles) is not int or args.warmup_cycles < 0:
        raise ValueError("--warmup-cycles must be non-negative")
    if type(args.seed) is not int or args.seed < 0:
        raise ValueError("--seed must be non-negative")
    for name in ("tower_lr", "bridge_lr", "projector_lr", "lm_lr", "grad_clip"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("--weight-decay must be finite and non-negative")
    output = Path(args.output).resolve()
    source = Path(args.source_checkpoint).resolve()
    if output.exists() or output.is_symlink():
        raise ValueError("--output must not already exist")
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("output and immutable source checkpoint must be disjoint")


def _validator_args(args: argparse.Namespace) -> argparse.Namespace:
    return SimpleNamespace(
        root=args.root,
        assets=args.assets,
        views=args.views,
        train_samples=args.train_samples,
        sft_validation_samples=args.sft_validation_samples,
        kl_selection_samples=args.kl_selection_samples,
        formal_monitor_samples=args.formal_monitor_samples,
        preprocess_contract=args.preprocess_contract,
        tokenizer=args.tokenizer,
        text_replay=args.text_replay,
        reviewed_exclusions=args.reviewed_exclusions,
        out="",
    )


def _image_split_contract(
    dataset_report: Mapping[str, object],
    split: str,
) -> dict:
    if split not in ANYRES_PUBLIC_SPLITS:
        raise ValueError(f"unknown public image split {split!r}")
    report_datasets = dataset_report.get("datasets")
    if not isinstance(report_datasets, Mapping) or set(report_datasets) != set(
        ANYRES_PUBLIC_SPLITS
    ):
        raise ValueError("dataset admission report must contain all four splits")
    entry = report_datasets.get(split)
    if not isinstance(entry, Mapping):
        raise ValueError(f"dataset admission report has no {split} entry")
    contract = entry.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError(f"dataset admission report has no {split} contract")
    if entry.get("contract_sha256") != contract.get("contract_sha256"):
        raise ValueError(f"dataset admission report {split} contract SHA differs")
    return copy.deepcopy(dict(contract))


def _text_split_contract(
    dataset_report: Mapping[str, object],
    split: str,
) -> dict:
    if split not in TEXT_REPLAY_PUBLIC_SPLITS:
        raise ValueError(f"unknown public text split {split!r}")
    text_replay = dataset_report.get("text_replay")
    if not isinstance(text_replay, Mapping):
        raise ValueError("dataset admission report has no text replay partition")
    report_splits = text_replay.get("splits")
    if not isinstance(report_splits, Mapping) or set(report_splits) != set(
        TEXT_REPLAY_PUBLIC_SPLITS
    ):
        raise ValueError("dataset admission report must contain all four text splits")
    entry = report_splits.get(split)
    if not isinstance(entry, Mapping):
        raise ValueError(f"dataset admission report has no text {split} entry")
    contract = entry.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError(f"dataset admission report has no text {split} contract")
    if entry.get("contract_sha256") != contract.get("contract_sha256"):
        raise ValueError(f"dataset admission report text {split} contract SHA differs")
    return copy.deepcopy(dict(contract))


def _text_partition_contract(dataset_report: Mapping[str, object]) -> dict:
    text_replay = dataset_report.get("text_replay")
    if not isinstance(text_replay, Mapping):
        raise ValueError("dataset admission report has no text replay partition")
    contract = text_replay.get("partition_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("dataset admission report has no text partition contract")
    if text_replay.get("partition_contract_sha256") != contract.get(
        "contract_sha256"
    ):
        raise ValueError("dataset admission report text partition SHA differs")
    return copy.deepcopy(dict(contract))


def _require_image_manifests_unchanged(
    args: argparse.Namespace,
    dataset_report: Mapping[str, object],
) -> None:
    manifests = dataset_report.get("manifests")
    if not isinstance(manifests, Mapping):
        raise ValueError("dataset admission report has no manifest hashes")
    paths = {
        "assets_sha256": args.assets,
        "views_sha256": args.views,
        "train_samples_sha256": args.train_samples,
        "sft_validation_samples_sha256": args.sft_validation_samples,
        "kl_selection_samples_sha256": args.kl_selection_samples,
        "formal_monitor_samples_sha256": args.formal_monitor_samples,
    }
    current = {
        field: _file_sha256(Path(path)) for field, path in paths.items()
    }
    drifted = {
        field: (manifests.get(field), digest)
        for field, digest in current.items()
        if manifests.get(field) != digest
    }
    if drifted:
        raise ValueError(f"image manifests changed after validation: {drifted}")
    text_replay = dataset_report.get("text_replay")
    if not isinstance(text_replay, Mapping):
        raise ValueError("dataset admission report has no text replay partition")
    admitted_text_sha256 = text_replay.get("manifest_sha256")
    current_text_sha256 = _file_sha256(Path(args.text_replay))
    if admitted_text_sha256 != current_text_sha256:
        raise ValueError("text replay manifest changed after validation")


def _validation_batches(dataset, collator) -> list[dict]:
    by_bucket: dict[str, list[str]] = {bucket: [] for bucket in DEPLOYMENT_BUCKET_WEIGHTS}
    for sample_id, bucket in zip(dataset.sample_ids, dataset.quota_buckets, strict=True):
        by_bucket[bucket].append(sample_id)
    groups: list[list[str]] = []
    for bucket in DEPLOYMENT_BUCKET_WEIGHTS:
        values = by_bucket[bucket]
        if len(values) < 2:
            raise ValueError("validation requires at least two samples per bucket")
        cursor = 0
        while len(values) - cursor > 3:
            groups.append(values[cursor:cursor + 2])
            cursor += 2
        groups.append(values[cursor:])
    return [
        collator([dataset.get_by_sample_id(sample_id) for sample_id in group])
        for group in groups
    ]


def _planned_training_batches(sampler: OCRQuotaSampler, cycles: int) -> list[tuple[str, ...]]:
    staged = copy.deepcopy(sampler)
    result: list[tuple[str, ...]] = []
    for _ in range(cycles):
        result.append(staged.prepare_global_batch())
        staged.commit_global_batch()
    return result


def _strict_relative_improvement(
    current: float,
    best: float,
    *,
    minimum_relative: float = 0.005,
    absolute_epsilon: float = 1e-12,
) -> bool:
    if not math.isfinite(current) or current < 0:
        raise ValueError("validation CER must be finite and non-negative")
    if not math.isfinite(best):
        return True
    if best < 0:
        raise ValueError("best CER must be non-negative")
    required = max(abs(best) * minimum_relative, absolute_epsilon)
    return current < best - required


def _text_batches(dataset, *, pad_id: int, batch_size: int = 4) -> list[dict]:
    collator = TextReplayCollator(pad_id=pad_id)
    return [
        collator([dataset[index] for index in range(start, min(len(dataset), start + batch_size))])
        for start in range(0, len(dataset), batch_size)
    ]


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: object) -> str:
    data = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _checkpoint_identity(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "model_sha256": _file_sha256(path / "model.pt"),
        "metadata_sha256": _file_sha256(path / "meta.pt"),
    }


def _parameter_sha256(named_parameters) -> str:
    digest = hashlib.sha256()
    for name, parameter in named_parameters:
        tensor = parameter.detach().contiguous().view(torch.uint8).cpu()
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _language_parameter_sha256(model) -> str:
    prefixes = (
        "embed.",
        "prelude.",
        "recurrent.",
        "coda.",
        "final_norm.",
        "lm_head.",
        "reverse_head.",
    )
    return _parameter_sha256(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith(prefixes)
    )


def _legacy_omvt_parameter_sha256(model) -> str:
    if model.vision.omvt is None:
        raise RuntimeError("legacy OMVT is not installed")
    return _parameter_sha256(model.vision.omvt.named_parameters())


def _runtime_source_receipt(
    *,
    extra_scripts: tuple[str, ...] = (),
) -> dict:
    root = Path(_REPO_ROOT)
    production_python = [
        path
        for base in (root / "Model", root / "Tokenizer")
        for path in base.rglob("*.py")
        if "tests" not in path.relative_to(root).parts
        and "__pycache__" not in path.relative_to(root).parts
    ]
    relative_paths = sorted(
        {path.relative_to(root).as_posix() for path in production_python}
        | {
            "scripts/train_ocr_anyres_sft.py",
            "scripts/validate_ocr_anyres_dataset.py",
            *extra_scripts,
        }
    )
    rows = [
        {
            "path": relative,
            "sha256": _file_sha256(Path(_REPO_ROOT) / relative),
        }
        for relative in relative_paths
    ]
    base = {
        "schema_version": 1,
        "kind": "dol_ocr_anyres_production_source_closure_v1",
        "files": rows,
    }
    return {**base, "canonical_sha256": _json_sha256(base)}


def _validate_runtime_source_receipt(value: object) -> dict:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "kind",
        "files",
        "canonical_sha256",
    }:
        raise ValueError("runtime source receipt fields differ from contract")
    if value["schema_version"] != 1 or value["kind"] != (
        "dol_ocr_anyres_production_source_closure_v1"
    ):
        raise ValueError("runtime source receipt contract is unsupported")
    rows = value["files"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("runtime source receipt files must be non-empty")
    paths: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256"}:
            raise ValueError(f"runtime source receipt files[{index}] is invalid")
        path = str(row["path"])
        digest = str(row["sha256"])
        if not path or path.startswith("/") or ".." in Path(path).parts:
            raise ValueError("runtime source receipt contains an unsafe path")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("runtime source receipt contains an invalid SHA256")
        paths.append(path)
    if paths != sorted(set(paths)):
        raise ValueError("runtime source receipt paths must be sorted and unique")
    base = {
        "schema_version": value["schema_version"],
        "kind": value["kind"],
        "files": rows,
    }
    if value["canonical_sha256"] != _json_sha256(base):
        raise ValueError("runtime source receipt canonical SHA256 mismatch")
    return dict(value)


def _require_runtime_source_unchanged(
    expected: Mapping[str, object],
    *,
    extra_scripts: tuple[str, ...] = (),
) -> None:
    current = _runtime_source_receipt(extra_scripts=extra_scripts)
    if current != expected:
        raise RuntimeError("production source changed during training")


def _runtime_environment(device: torch.device, precision: str) -> dict:
    result = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": str(device),
        "precision": precision,
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        result.update(
            {
                "device_name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
        )
    return result


def _metadata(
    prepared,
    *,
    cycle: int,
    args: argparse.Namespace,
    train_dataset,
    val_dataset,
    text_partition,
    sampler,
    dataset_report,
    baseline_report,
    last_eval,
    final: bool,
    language_parameter_sha256: str,
    legacy_omvt_parameter_sha256: str,
    runtime_source_receipt: dict,
    runtime_environment: dict,
) -> dict:
    reported_contracts = {
        split: _image_split_contract(dataset_report, split)
        for split in ANYRES_PUBLIC_SPLITS
    }
    if train_dataset.dataset_contract != reported_contracts["train"]:
        raise ValueError("train dataset differs from admission report")
    if val_dataset.dataset_contract != reported_contracts["sft_validation"]:
        raise ValueError("SFT validation dataset differs from admission report")
    reported_text_contracts = {
        split: _text_split_contract(dataset_report, split)
        for split in TEXT_REPLAY_PUBLIC_SPLITS
    }
    for split in TEXT_REPLAY_PUBLIC_SPLITS:
        if (
            text_partition.dataset(split).dataset_contract
            != reported_text_contracts[split]
        ):
            raise ValueError(
                f"text replay {split} dataset differs from admission report"
            )
    if text_partition.partition_contract != _text_partition_contract(
        dataset_report
    ):
        raise ValueError("text replay partition differs from admission report")
    sft_text_contract = text_partition.dataset(
        "sft_validation"
    ).dataset_contract
    return {
        **prepared.metadata_template,
        "training_stage": "visual",
        "optimizer_cycle": cycle,
        "training_config": {
            "precision": args.precision,
            "global_batch_size": args.global_batch_size,
            "tower_lr": args.tower_lr,
            "bridge_lr": args.bridge_lr,
            "projector_lr": args.projector_lr,
            "lm_lr": args.lm_lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "cycles": args.cycles,
        },
        "train_dataset_contract": train_dataset.dataset_contract,
        "validation_dataset_contract": val_dataset.dataset_contract,
        "sft_validation_dataset_contract": val_dataset.dataset_contract,
        "kl_selection_dataset_contract": reported_contracts["kl_selection"],
        "formal_monitor_dataset_contract": reported_contracts["formal_monitor"],
        "text_replay_train_contract": text_partition.dataset(
            "train"
        ).dataset_contract,
        "text_replay_sft_validation_contract": sft_text_contract,
        "text_replay_kl_selection_contract": text_partition.dataset(
            "kl_selection"
        ).dataset_contract,
        "text_replay_formal_monitor_contract": text_partition.dataset(
            "formal_monitor"
        ).dataset_contract,
        "text_replay_validation_contract": sft_text_contract,
        "text_replay_partition_contract": text_partition.partition_contract,
        "quota_sampler_state": sampler.state_dict(),
        "dataset_admission_report": dataset_report,
        "baseline_validation": baseline_report,
        "last_validation": last_eval,
        "final": final,
        "stop_reason": "max_cycles" if final else "running",
        "frozen_parameter_contract": {
            "language_sha256": language_parameter_sha256,
            "legacy_omvt_sha256": legacy_omvt_parameter_sha256,
        },
        "runtime_source_receipt": runtime_source_receipt,
        "runtime_environment": runtime_environment,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_args(args)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_report = build_dataset_report(_validator_args(args))
    preprocess_payload, _ = _strict_json(
        args.preprocess_contract,
        where="anyres preprocess contract",
    )
    preprocess = validate_anyres_preprocess_contract(preprocess_payload)
    preprocess_sha = preprocess["contract_canonical_sha256"]

    bundle = TokenizerBundle.from_dir(args.tokenizer)
    issues = bundle.validate()
    if issues:
        raise ValueError("invalid tokenizer bundle: " + "; ".join(issues))
    token_contract = native_tokenization_contract(bundle.tokenizer, args.tokenizer)
    native_encoder = make_ocr_target_encoder(bundle.tokenizer, mode="native")
    vocab_extent = max(int(value) for value in bundle.tokenizer.vocab.values()) + 1

    train_dataset = AnyresOCRDataset(
        root=args.root,
        assets_manifest=args.assets,
        views_manifest=args.views,
        samples_manifest=args.train_samples,
        split="train",
        expected_preprocess_contract_sha256=preprocess_sha,
        max_decode_pixels=int(
            preprocess["budgets"]["decode"]["max_pixels_per_asset"]
        ),
        max_views_per_sample=int(
            preprocess["budgets"]["window"]["max_windows_per_asset"]
        ),
        image_delivery="bytes",
    )
    val_dataset = AnyresOCRDataset(
        root=args.root,
        assets_manifest=args.assets,
        views_manifest=args.views,
        samples_manifest=args.sft_validation_samples,
        split="sft_validation",
        expected_preprocess_contract_sha256=preprocess_sha,
        max_decode_pixels=int(
            preprocess["budgets"]["decode"]["max_pixels_per_asset"]
        ),
        max_views_per_sample=int(
            preprocess["budgets"]["window"]["max_windows_per_asset"]
        ),
        image_delivery="bytes",
    )
    if train_dataset.dataset_contract != _image_split_contract(
        dataset_report,
        "train",
    ):
        raise ValueError("train dataset changed after global validation")
    if val_dataset.dataset_contract != _image_split_contract(
        dataset_report,
        "sft_validation",
    ):
        raise ValueError("SFT validation dataset changed after global validation")
    _require_image_manifests_unchanged(args, dataset_report)
    excluded_documents, excluded_ngrams, _ = _load_exclusions(
        args.reviewed_exclusions
    )
    text_partition = TextReplayPartition(
        args.text_replay,
        native_encoder=native_encoder,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        max_seq_len=int(preprocess["budgets"]["context"]["max_sequence_tokens"]),
        exclusion_document_ids=excluded_documents,
        exclusion_ngram_sha256=excluded_ngrams,
    )
    text_dataset = text_partition.dataset("sft_validation")

    source_admission = admit_visual_ocr_source(
        VISUAL_SOURCE_CONTRACT_STREAMING_V2_RELEASE,
        checkpoint_dir=args.source_checkpoint,
        metadata=None,
        runtime_native_tokenization_contract=token_contract,
        tokenizer_vocab_extent=vocab_extent,
    )
    source_omvt = source_admission["metadata"].get("omvt_config")
    if not isinstance(source_omvt, dict):
        raise ValueError("reviewed source has no OMVT config")
    memory_dim = int(source_omvt["d_vision"])

    prepared = admit_and_prepare_anyres_policy(
        source_checkpoint=args.source_checkpoint,
        source_contract_kind=VISUAL_SOURCE_CONTRACT_STREAMING_V2_RELEASE,
        tokenizer_contract=token_contract,
        tokenizer_vocab_extent=vocab_extent,
        preprocess_contract=preprocess,
        stage="visual",
        target_max_seq_len=int(preprocess["budgets"]["context"]["max_sequence_tokens"]),
        native_detail_config={
            "max_detail_tokens": int(
                preprocess["budgets"]["detail"]["max_tokens_per_view"]
            ),
            "source_tokens_per_detail_token": int(
                preprocess["budgets"]["detail"]["source_tokens_per_detail_token"]
            ),
        },
        vision_cross_attention_config={
            "memory_dim": memory_dim,
            "n_heads": int(source_admission["metadata"]["rdt_config"]["n_heads"]),
            "dropout": 0.0,
        },
    )

    omvt_cfg = prepared.omvt_config
    collator = AnyresOCRSFTCollator(
        encode_reference=native_encoder,
        omvt_cfg=omvt_cfg,
        global_processor=PILImageProcessor(
            image_size=omvt_cfg.image_size,
            in_channels=omvt_cfg.in_channels,
        ),
        native_processor=NativeImageProcessorV2(
            in_channels=omvt_cfg.in_channels,
            max_decode_pixels=int(
                preprocess["budgets"]["decode"]["max_pixels_per_asset"]
            ),
        ),
        max_raw_patch_tokens_per_view=int(
            preprocess["budgets"]["patch"]["max_raw_tokens_per_view"]
        ),
        max_seq_len=prepared.rdt_config.max_seq_len,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_id=PAD_ID,
    )
    val_batches = _validation_batches(val_dataset, collator)
    replay_batches = _text_batches(
        text_dataset,
        pad_id=PAD_ID,
    )
    sample_buckets = dict(
        zip(train_dataset.sample_ids, train_dataset.quota_buckets, strict=True)
    )
    sampler = OCRQuotaSampler(
        sample_buckets,
        global_batch_size=args.global_batch_size,
        seed=args.seed,
        world_size=1,
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    model = prepared.policy.to(device)
    model.reverse_loss_enabled = False
    configure_visual_stage_trainable(model)
    runtime_source_receipt = _runtime_source_receipt()
    runtime_environment = _runtime_environment(device, args.precision)
    initial_language_sha256 = _language_parameter_sha256(model)
    initial_legacy_omvt_sha256 = _legacy_omvt_parameter_sha256(model)

    train_cfg = TrainingConfig(
        learning_rate=args.lm_lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        max_steps=args.cycles,
        warmup_steps=args.warmup_cycles,
        lr_decay_steps=args.cycles,
        precision=args.precision,
        output_dir=args.output,
        optimizer="adamw",
        adam_use_atan2=False,
    )
    optimizer = build_ocr_joint_adamw(
        model,
        train_cfg,
        lm_lr=args.lm_lr,
        tower_lr=args.tower_lr,
        projector_lr=args.projector_lr,
        bridge_lr=args.bridge_lr,
    )
    scheduler = build_scheduler(optimizer, train_cfg)
    scaler = None
    if args.precision == "fp16":
        if device.type != "cuda":
            raise ValueError("fp16 training requires CUDA")
        scaler = torch.amp.GradScaler("cuda")

    decode = lambda ids: bundle.tokenizer.decode(  # noqa: E731
        [int(value) for value in ids.tolist()]
    )
    max_new_tokens = int(
        preprocess["budgets"]["output"]["recommended_max_new_tokens"]
    )
    baseline = evaluate_ocr_joint(
        model,
        val_batches,
        decode,
        max_new_tokens=max_new_tokens,
        device=device,
        expected_image_dataset_contract_sha256=val_dataset.contract_sha256,
        text_replay_batches=replay_batches,
        expected_text_replay_contract_sha256=text_dataset.contract_sha256,
    )
    baseline_bucket_cer = {
        bucket: float(baseline["real"]["buckets"][bucket]["raw_grapheme_cer"])
        for bucket in DEPLOYMENT_BUCKET_WEIGHTS
    }
    baseline_text_nll = float(baseline["text_replay"]["token_nll"])

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(output_dir / "DATA_ADMISSION.json", dataset_report)
    _atomic_json(output_dir / "BASELINE_VALIDATION.json", baseline)

    last_eval = baseline
    best_cer = float("inf")
    best_checkpoint: Path | None = None
    best_eval: dict | None = None
    bad_evals = 0
    stop_reason = "max_cycles"
    completed_cycle = 0
    planned_batches = _planned_training_batches(sampler, args.cycles)

    def load_training_batch(sample_ids: tuple[str, ...]):
        return collator(
            [train_dataset.get_by_sample_id(sample_id) for sample_id in sample_ids]
        )

    with VerifiedBatchPrefetcher(
        planned_batches,
        load_training_batch,
        max_prefetch=1,
    ) as prefetcher:
        for cycle, (planned_ids, train_batch) in enumerate(prefetcher, start=1):
            sample_ids = sampler.prepare_global_batch()
            if sample_ids != planned_ids:
                raise RuntimeError("prefetched OCR batch differs from sampler state")
            metrics = train_visual_cycle(
                model,
                [train_batch],
                optimizer,
                device=device,
                scheduler=scheduler,
                scaler=scaler,
                precision=args.precision,
                grad_clip=args.grad_clip,
                loss_chunk_size=4096,
            )
            if not metrics["stepped"]:
                raise RuntimeError("visual optimizer cycle was skipped")
            sampler.commit_global_batch()
            completed_cycle = cycle

            should_eval = cycle % args.eval_every == 0 or cycle == args.cycles
            if should_eval:
                last_eval = evaluate_ocr_joint(
                    model,
                    val_batches,
                    decode,
                    max_new_tokens=max_new_tokens,
                    device=device,
                    expected_image_dataset_contract_sha256=(
                        val_dataset.contract_sha256
                    ),
                    text_replay_batches=replay_batches,
                    expected_text_replay_contract_sha256=(
                        text_dataset.contract_sha256
                    ),
                )
                eligibility = joint_eval_eligibility(
                    last_eval,
                    baseline_bucket_cer=baseline_bucket_cer,
                    baseline_text_token_nll=baseline_text_nll,
                    min_relative_cer_improvement=0.5,
                )
                last_eval["eligibility"] = eligibility
                current_cer = float(last_eval["deployment_weighted_cer"])
                materially_improved = _strict_relative_improvement(
                    current_cer,
                    best_cer,
                )
                if eligibility["eligible"] and materially_improved:
                    best_cer = current_cer
                    best_eval = last_eval
                    bad_evals = 0
                    _require_runtime_source_unchanged(runtime_source_receipt)
                    metadata = _metadata(
                        prepared,
                        cycle=cycle,
                        args=args,
                        train_dataset=train_dataset,
                        val_dataset=val_dataset,
                        text_partition=text_partition,
                        sampler=sampler,
                        dataset_report=dataset_report,
                        baseline_report=baseline,
                        last_eval=last_eval,
                        final=False,
                        language_parameter_sha256=initial_language_sha256,
                        legacy_omvt_parameter_sha256=initial_legacy_omvt_sha256,
                        runtime_source_receipt=runtime_source_receipt,
                        runtime_environment=runtime_environment,
                    )
                    saved_best = save_checkpoint(
                        output_dir / "best",
                        cycle,
                        model,
                        optimizer,
                        scheduler,
                        metadata=metadata,
                        keep_last_n=3,
                        scaler=scaler,
                    )
                    if saved_best is None:
                        raise RuntimeError(
                            "single-process best checkpoint was not saved"
                        )
                    best_checkpoint = saved_best
                elif math.isfinite(best_cer):
                    bad_evals += 1
                    if bad_evals >= args.early_stop_patience:
                        stop_reason = "validation_plateau"

            if cycle % args.save_every == 0:
                _require_runtime_source_unchanged(runtime_source_receipt)
                metadata = _metadata(
                    prepared,
                    cycle=cycle,
                    args=args,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    text_partition=text_partition,
                    sampler=sampler,
                    dataset_report=dataset_report,
                    baseline_report=baseline,
                    last_eval=last_eval,
                    final=False,
                    language_parameter_sha256=initial_language_sha256,
                    legacy_omvt_parameter_sha256=initial_legacy_omvt_sha256,
                    runtime_source_receipt=runtime_source_receipt,
                    runtime_environment=runtime_environment,
                )
                save_checkpoint(
                    output_dir,
                    cycle,
                    model,
                    optimizer,
                    scheduler,
                    metadata=metadata,
                    keep_last_n=args.keep_last,
                    scaler=scaler,
                )
            print(json.dumps({"cycle": cycle, **metrics}, sort_keys=True), flush=True)
            if stop_reason == "validation_plateau":
                break

    final_language_sha256 = _language_parameter_sha256(model)
    final_legacy_omvt_sha256 = _legacy_omvt_parameter_sha256(model)
    if final_language_sha256 != initial_language_sha256:
        raise RuntimeError("visual stage changed frozen language parameters")
    if final_legacy_omvt_sha256 != initial_legacy_omvt_sha256:
        raise RuntimeError("visual stage changed the frozen legacy OMVT anchor")
    _require_runtime_source_unchanged(runtime_source_receipt)
    final_metadata = _metadata(
        prepared,
        cycle=completed_cycle,
        args=args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        text_partition=text_partition,
        sampler=sampler,
        dataset_report=dataset_report,
        baseline_report=baseline,
        last_eval=last_eval,
        final=True,
        language_parameter_sha256=final_language_sha256,
        legacy_omvt_parameter_sha256=final_legacy_omvt_sha256,
        runtime_source_receipt=runtime_source_receipt,
        runtime_environment=runtime_environment,
    )
    final_metadata["stop_reason"] = stop_reason
    final_checkpoint = save_checkpoint(
        output_dir,
        completed_cycle,
        model,
        optimizer,
        scheduler,
        metadata=final_metadata,
        keep_last_n=args.keep_last,
        scaler=scaler,
    )
    if final_checkpoint is None:
        raise RuntimeError("single-process final checkpoint was not saved")
    _atomic_json(output_dir / "FINAL_VALIDATION.json", last_eval)
    last_eligibility = last_eval.get("eligibility")
    promotion_allowed = best_checkpoint is not None and best_eval is not None
    result = {
        "schema_version": 1,
        "kind": "dol_ocr_anyres_visual_stage_result_v1",
        "run_contract_sha256": prepared.run_contract[
            "contract_canonical_sha256"
        ],
        "final_checkpoint": _checkpoint_identity(final_checkpoint),
        "best_eligible_checkpoint": (
            None
            if best_checkpoint is None
            else _checkpoint_identity(best_checkpoint)
        ),
        "baseline_validation_sha256": _json_sha256(baseline),
        "final_validation_sha256": _json_sha256(last_eval),
        "best_validation_sha256": (
            None if best_eval is None else _json_sha256(best_eval)
        ),
        "last_eligibility": last_eligibility,
        "promotion_allowed": promotion_allowed,
        "completed_cycles": completed_cycle,
        "stop_reason": stop_reason,
        "frozen_parameter_contract": {
            "language_sha256": final_language_sha256,
            "legacy_omvt_sha256": final_legacy_omvt_sha256,
        },
        "runtime_source_receipt": runtime_source_receipt,
        "runtime_environment": runtime_environment,
    }
    result["canonical_sha256"] = _json_sha256(result)
    _atomic_json(output_dir / "VISUAL_STAGE_RESULT.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
