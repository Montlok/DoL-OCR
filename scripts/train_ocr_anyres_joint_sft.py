#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Joint LM+VM SFT, admitted only from the best eligible anyres visual run."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
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
    canonical_json_sha256,
    make_ocr_target_encoder,
    native_tokenization_contract,
)
from Model.posttrain.checkpointing import load_verified_policy_metadata  # noqa: E402
from Model.posttrain.ocr_anyres_collator import AnyresOCRSFTCollator  # noqa: E402
from Model.posttrain.ocr_anyres_data import AnyresOCRDataset  # noqa: E402
from Model.posttrain.ocr_joint_contract import (  # noqa: E402
    JOINT_STAGE_RESULT_KIND,
    promote_visual_policy_to_joint,
)
from Model.posttrain.ocr_joint_eval import (  # noqa: E402
    DEPLOYMENT_BUCKET_WEIGHTS,
    dual_baseline_joint_eligibility,
    evaluate_ocr_joint,
    joint_eval_eligibility,
)
from Model.posttrain.ocr_joint_trainer import (  # noqa: E402
    configure_joint_stage_trainable,
    train_joint_cycle,
)
from Model.posttrain.ocr_quota_sampler import OCRQuotaSampler  # noqa: E402
from Model.posttrain.text_replay import (  # noqa: E402
    TextReplayCollator,
    TextReplayPartition,
)
from Model.training.checkpoint import save_checkpoint  # noqa: E402
from Model.training.optim import build_ocr_joint_adamw, build_scheduler  # noqa: E402
from Tokenizer.multimodal import NativeImageProcessorV2, PILImageProcessor  # noqa: E402
from Tokenizer.unified.bundle import TokenizerBundle  # noqa: E402
from scripts import train_ocr_anyres_sft as visual_cli  # noqa: E402
from scripts.validate_ocr_anyres_dataset import (  # noqa: E402
    _load_exclusions,
    _strict_json,
    build_report as build_dataset_report,
)


JOINT_OCR_BATCHES_PER_CYCLE = 4


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", choices=["single", "ddp", "fsdp"], default="single")
    parser.add_argument("--resume", default="")
    parser.add_argument("--visual-stage-result", required=True)
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
    parser.add_argument("--text-batch-size", type=int, default=4)
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
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.dist != "single":
        raise ValueError("joint anyres SFT currently supports --dist=single only")
    if args.resume:
        raise ValueError("joint anyres SFT resume is not admitted")
    for name in (
        "cycles",
        "global_batch_size",
        "text_batch_size",
        "save_every",
        "eval_every",
        "early_stop_patience",
    ):
        value = getattr(args, name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.global_batch_size % 20 != 0:
        raise ValueError("--global-batch-size must be divisible by 20")
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
    if output.exists() or output.is_symlink():
        raise ValueError("--output must not already exist")


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


def _best_identity(result: object) -> dict[str, str]:
    if not isinstance(result, Mapping):
        raise ValueError("VISUAL_STAGE_RESULT must be an object")
    if result.get("promotion_allowed") is not True:
        raise ValueError("VISUAL_STAGE_RESULT is not promotion eligible")
    best = result.get("best_eligible_checkpoint")
    if not isinstance(best, Mapping) or set(best) != {
        "path",
        "model_sha256",
        "metadata_sha256",
    }:
        raise ValueError("VISUAL_STAGE_RESULT has no exact best checkpoint identity")
    return {key: str(best[key]) for key in ("path", "model_sha256", "metadata_sha256")}


def _assert_saved_data_contracts(
    metadata: Mapping[str, object],
    *,
    dataset_report: Mapping[str, object],
    train_dataset,
    val_dataset,
    text_partition,
    preprocess: Mapping[str, object],
    tokenizer_contract: Mapping[str, object],
    tokenizer_vocab_extent: int,
    runtime_source_receipt: Mapping[str, object],
) -> None:
    expected = {
        "train_dataset_contract": train_dataset.dataset_contract,
        "validation_dataset_contract": val_dataset.dataset_contract,
        "sft_validation_dataset_contract": val_dataset.dataset_contract,
        "kl_selection_dataset_contract": visual_cli._image_split_contract(
            dataset_report,
            "kl_selection",
        ),
        "formal_monitor_dataset_contract": visual_cli._image_split_contract(
            dataset_report,
            "formal_monitor",
        ),
        "text_replay_train_contract": visual_cli._text_split_contract(
            dataset_report,
            "train",
        ),
        "text_replay_sft_validation_contract": visual_cli._text_split_contract(
            dataset_report,
            "sft_validation",
        ),
        "text_replay_kl_selection_contract": visual_cli._text_split_contract(
            dataset_report,
            "kl_selection",
        ),
        "text_replay_formal_monitor_contract": visual_cli._text_split_contract(
            dataset_report,
            "formal_monitor",
        ),
        "text_replay_validation_contract": visual_cli._text_split_contract(
            dataset_report,
            "sft_validation",
        ),
        "text_replay_partition_contract": text_partition.partition_contract,
        "dataset_admission_report": dict(dataset_report),
        "anyres_preprocess_contract": dict(preprocess),
        "anyres_preprocess_contract_sha256": preprocess["contract_canonical_sha256"],
        "tokenizer_contract": dict(tokenizer_contract),
        "tokenizer_vocab_extent": tokenizer_vocab_extent,
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(f"visual best metadata differs from current {field}")
    saved_runtime = metadata.get("runtime_source_receipt")
    saved_runtime = visual_cli._validate_runtime_source_receipt(saved_runtime)
    current_runtime = visual_cli._validate_runtime_source_receipt(
        runtime_source_receipt
    )
    saved_rows = saved_runtime["files"]
    current_rows = current_runtime["files"]
    saved_files = {
        str(row["path"]): str(row["sha256"])
        for row in saved_rows
        if isinstance(row, Mapping) and "path" in row and "sha256" in row
    }
    current_files = {
        str(row["path"]): str(row["sha256"])
        for row in current_rows
        if isinstance(row, Mapping) and "path" in row and "sha256" in row
    }
    unexpected_paths = set(current_files) - set(saved_files) - {
        "scripts/train_ocr_anyres_joint_sft.py"
    }
    if unexpected_paths:
        raise ValueError(
            "unexpected production source files appeared since visual stage: "
            f"{sorted(unexpected_paths)}"
        )
    drifted = {
        path: (digest, current_files.get(path))
        for path, digest in saved_files.items()
        if current_files.get(path) != digest
    }
    if drifted:
        raise ValueError(f"runtime source drift since visual stage: {drifted}")


def _prepare_ocr_microbatches(sampler, dataset, collator):
    staged = copy.deepcopy(sampler)
    batches = []
    sample_ids = []
    for _ in range(JOINT_OCR_BATCHES_PER_CYCLE):
        ids = tuple(staged.prepare_global_batch())
        sample_ids.append(ids)
        batches.append(collator([dataset.get_by_sample_id(value) for value in ids]))
        staged.commit_global_batch()
    return batches, staged.state_dict(), sample_ids


def _text_batch_at_cursor(dataset, collator, *, cursor: int, batch_size: int):
    if type(cursor) is not int or cursor < 0:
        raise ValueError("text cursor must be non-negative")
    if len(dataset) <= 0:
        raise ValueError("text replay train dataset is empty")
    start = (cursor * batch_size) % len(dataset)
    indices = [(start + offset) % len(dataset) for offset in range(batch_size)]
    return collator([dataset[index] for index in indices]), cursor + 1, indices


def _text_cursor_state(dataset, *, cursor: int, batch_size: int) -> dict[str, object]:
    payload = {
        "schema_version": 1,
        "kind": "dol_text_replay_sequential_cursor_v1",
        "cursor": cursor,
        "batch_size": batch_size,
        "dataset_contract_sha256": dataset.contract_sha256,
        "wrap": "modulo_dataset_length_v1",
    }
    return {**payload, "canonical_sha256": canonical_json_sha256(payload)}


def _trainability_snapshot(model) -> dict[str, object]:
    names = sorted(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "stage": "joint",
        "policy": "joint_language_and_active_visual_v1",
        "trainable_parameter_names_sha256": canonical_json_sha256(names),
        "trainable_parameter_count": trainable,
        "frozen_parameter_count": total - trainable,
        "total_parameter_count": total,
    }


def _optimizer_contract(optimizer) -> dict[str, object]:
    groups = [
        {
            "role": group.get("ocr_joint_role"),
            "decay": group.get("ocr_joint_decay"),
            "base_lr": group.get("ocr_joint_base_lr"),
            "weight_decay": group.get("weight_decay"),
            "parameter_count": sum(parameter.numel() for parameter in group["params"]),
        }
        for group in optimizer.param_groups
    ]
    return {"groups": groups, "canonical_sha256": canonical_json_sha256(groups)}


def _runtime_source_receipt() -> dict[str, object]:
    return visual_cli._runtime_source_receipt(
        extra_scripts=("scripts/train_ocr_anyres_joint_sft.py",),
    )


def _metadata(
    prepared,
    *,
    cycle: int,
    args,
    train_dataset,
    val_dataset,
    text_train_dataset,
    text_validation_dataset,
    text_partition,
    sampler,
    text_cursor: int,
    dataset_report,
    historical_baseline,
    baseline,
    last_validation,
    optimizer_contract,
    runtime_source_receipt,
    runtime_environment,
    final: bool,
) -> dict[str, object]:
    reported_contracts = {
        split: visual_cli._image_split_contract(dataset_report, split)
        for split in (
            "train",
            "sft_validation",
            "kl_selection",
            "formal_monitor",
        )
    }
    if train_dataset.dataset_contract != reported_contracts["train"]:
        raise ValueError("train dataset differs from admission report")
    if val_dataset.dataset_contract != reported_contracts["sft_validation"]:
        raise ValueError("SFT validation dataset differs from admission report")
    for split, text_dataset in text_partition.datasets.items():
        if text_dataset.dataset_contract != visual_cli._text_split_contract(
            dataset_report,
            split,
        ):
            raise ValueError(
                f"text replay {split} dataset differs from admission report"
            )
    if text_partition.partition_contract != visual_cli._text_partition_contract(
        dataset_report
    ):
        raise ValueError("text replay partition differs from admission report")
    return {
        **prepared.metadata_template,
        "training_stage": "joint",
        "optimizer_cycle": cycle,
        "training_config": {
            "precision": args.precision,
            "global_batch_size": args.global_batch_size,
            "text_batch_size": args.text_batch_size,
            "tower_lr": args.tower_lr,
            "bridge_lr": args.bridge_lr,
            "projector_lr": args.projector_lr,
            "lm_lr": args.lm_lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "cycles": args.cycles,
            "ocr_batches_per_cycle": JOINT_OCR_BATCHES_PER_CYCLE,
            "text_batches_per_cycle": 1,
            "cadence": "4_ocr_microbatches_then_1_text_microbatch",
            "ocr_loss_aggregation": "mean_of_4_microbatch_losses",
            "text_weight": 0.2,
            "objective": "mean_ocr_plus_0.2_text_ce",
            "objective_ocr_to_text_coefficient_ratio": "1:0.2",
            "microbatch_losses": "mean_over_supervised_causal_tokens",
        },
        "train_dataset_contract": train_dataset.dataset_contract,
        "validation_dataset_contract": val_dataset.dataset_contract,
        "sft_validation_dataset_contract": val_dataset.dataset_contract,
        "kl_selection_dataset_contract": reported_contracts["kl_selection"],
        "formal_monitor_dataset_contract": reported_contracts["formal_monitor"],
        "text_replay_train_contract": text_train_dataset.dataset_contract,
        "text_replay_sft_validation_contract": (
            text_validation_dataset.dataset_contract
        ),
        "text_replay_kl_selection_contract": text_partition.dataset(
            "kl_selection"
        ).dataset_contract,
        "text_replay_formal_monitor_contract": text_partition.dataset(
            "formal_monitor"
        ).dataset_contract,
        "text_replay_validation_contract": (
            text_validation_dataset.dataset_contract
        ),
        "text_replay_partition_contract": text_partition.partition_contract,
        "quota_sampler_state": sampler.state_dict(),
        "text_replay_cursor": _text_cursor_state(
            text_train_dataset,
            cursor=text_cursor,
            batch_size=args.text_batch_size,
        ),
        "dataset_admission_report": dataset_report,
        "historical_visual_best_validation": historical_baseline,
        "joint_runtime_baseline": baseline,
        "last_validation": last_validation,
        "optimizer_contract": optimizer_contract,
        "runtime_source_receipt": runtime_source_receipt,
        "runtime_environment": runtime_environment,
        "final": final,
        "stop_reason": "max_cycles" if final else "running",
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
    if train_dataset.dataset_contract != visual_cli._image_split_contract(
        dataset_report,
        "train",
    ):
        raise ValueError("train dataset changed after global validation")
    if val_dataset.dataset_contract != visual_cli._image_split_contract(
        dataset_report,
        "sft_validation",
    ):
        raise ValueError("SFT validation dataset changed after global validation")
    visual_cli._require_image_manifests_unchanged(args, dataset_report)
    excluded_documents, excluded_ngrams, _ = _load_exclusions(args.reviewed_exclusions)
    text_common = {
        "native_encoder": native_encoder,
        "bos_id": BOS_ID,
        "eos_id": EOS_ID,
        "max_seq_len": int(preprocess["budgets"]["context"]["max_sequence_tokens"]),
        "exclusion_document_ids": excluded_documents,
        "exclusion_ngram_sha256": excluded_ngrams,
    }
    text_partition = TextReplayPartition(args.text_replay, **text_common)
    text_train_dataset = text_partition.dataset("train")
    text_validation_dataset = text_partition.dataset("sft_validation")

    visual_result, _ = _strict_json(
        args.visual_stage_result,
        where="VISUAL_STAGE_RESULT",
    )
    identity = _best_identity(visual_result)
    output = Path(args.output).resolve()
    best_path = Path(identity["path"]).resolve()
    if output == best_path or best_path in output.parents or output in best_path.parents:
        raise ValueError("joint output and visual best checkpoint must be disjoint")
    visual_metadata, _ = load_verified_policy_metadata(
        best_path,
        expected_sha256=identity["metadata_sha256"],
    )
    runtime_source_receipt = _runtime_source_receipt()
    _assert_saved_data_contracts(
        visual_metadata,
        dataset_report=dataset_report,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        text_partition=text_partition,
        preprocess=preprocess,
        tokenizer_contract=token_contract,
        tokenizer_vocab_extent=vocab_extent,
        runtime_source_receipt=runtime_source_receipt,
    )
    prepared = promote_visual_policy_to_joint(
        best_path,
        identity["model_sha256"],
        identity["metadata_sha256"],
        visual_result,
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
            max_decode_pixels=int(preprocess["budgets"]["decode"]["max_pixels_per_asset"]),
        ),
        max_raw_patch_tokens_per_view=int(
            preprocess["budgets"]["patch"]["max_raw_tokens_per_view"]
        ),
        max_seq_len=prepared.rdt_config.max_seq_len,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_id=PAD_ID,
    )
    val_batches = visual_cli._validation_batches(val_dataset, collator)
    replay_validation_batches = visual_cli._text_batches(
        text_validation_dataset,
        pad_id=PAD_ID,
    )
    text_train_collator = TextReplayCollator(pad_id=PAD_ID)
    sampler = OCRQuotaSampler(
        dict(zip(train_dataset.sample_ids, train_dataset.quota_buckets, strict=True)),
        global_batch_size=args.global_batch_size,
        seed=args.seed,
        world_size=1,
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    model = prepared.policy.to(device)
    model.reverse_loss_enabled = False
    configure_joint_stage_trainable(model)
    if _trainability_snapshot(model) != prepared.run_contract["trainability_contract"]:
        raise ValueError("joint CLI trainability differs from promoted run contract")
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
    optimizer_contract = _optimizer_contract(optimizer)
    runtime_environment = visual_cli._runtime_environment(device, args.precision)
    decode = lambda ids: bundle.tokenizer.decode([int(value) for value in ids.tolist()])  # noqa: E731
    max_new_tokens = int(preprocess["budgets"]["output"]["recommended_max_new_tokens"])
    historical_baseline = copy.deepcopy(visual_metadata["last_validation"])
    baseline = evaluate_ocr_joint(
        model,
        val_batches,
        decode,
        max_new_tokens=max_new_tokens,
        device=device,
        expected_image_dataset_contract_sha256=val_dataset.contract_sha256,
        text_replay_batches=replay_validation_batches,
        expected_text_replay_contract_sha256=(
            text_validation_dataset.contract_sha256
        ),
    )
    historical_text_nll = float(
        historical_baseline["text_replay"]["token_nll"]
    )
    runtime_text_nll = float(baseline["text_replay"]["token_nll"])
    if (
        not math.isfinite(historical_text_nll)
        or historical_text_nll <= 0
        or not math.isfinite(runtime_text_nll)
        or runtime_text_nll <= 0
    ):
        raise ValueError("historical and runtime text baselines must be positive")
    baseline_comparison = {
        "historical_sha256": visual_cli._json_sha256(historical_baseline),
        "joint_runtime_sha256": visual_cli._json_sha256(baseline),
        "historical_runtime_environment": visual_metadata.get(
            "runtime_environment"
        ),
        "joint_runtime_environment": runtime_environment,
        "deployment_weighted_cer_delta": (
            float(baseline["deployment_weighted_cer"])
            - float(historical_baseline["deployment_weighted_cer"])
        ),
        "text_token_nll_relative_delta": (
            runtime_text_nll / historical_text_nll - 1.0
        ),
    }
    historical_bucket_cer = {
        bucket: float(
            historical_baseline["real"]["buckets"][bucket][
                "raw_grapheme_cer"
            ]
        )
        for bucket in DEPLOYMENT_BUCKET_WEIGHTS
    }
    historical_gate = joint_eval_eligibility(
        baseline,
        baseline_bucket_cer=historical_bucket_cer,
        baseline_text_token_nll=historical_text_nll,
        min_relative_cer_improvement=0.0,
    )
    baseline_comparison["historical_non_regression_gate"] = historical_gate
    if historical_gate["eligible"] is not True:
        raise ValueError(
            "current runtime baseline regresses from visual best: "
            f"{historical_gate['reasons']}"
        )

    output.mkdir(parents=True, exist_ok=False)
    visual_cli._atomic_json(output / "DATA_ADMISSION.json", dataset_report)
    visual_cli._atomic_json(
        output / "HISTORICAL_VISUAL_BEST_VALIDATION.json",
        historical_baseline,
    )
    visual_cli._atomic_json(output / "JOINT_RUNTIME_BASELINE.json", baseline)
    visual_cli._atomic_json(output / "BASELINE_COMPARISON.json", baseline_comparison)
    text_cursor = 0
    last_eval = baseline
    best_eval = None
    best_checkpoint = None
    best_cer = float("inf")
    bad_evals = 0
    completed_cycle = 0
    stop_reason = "max_cycles"
    for cycle in range(1, args.cycles + 1):
        ocr_batches, staged_sampler_state, _sample_ids = _prepare_ocr_microbatches(
            sampler,
            train_dataset,
            collator,
        )
        text_batch, staged_text_cursor, _indices = _text_batch_at_cursor(
            text_train_dataset,
            text_train_collator,
            cursor=text_cursor,
            batch_size=args.text_batch_size,
        )
        metrics = train_joint_cycle(
            model,
            ocr_batches,
            text_batch,
            optimizer,
            text_weight=0.2,
            device=device,
            scheduler=scheduler,
            scaler=scaler,
            precision=args.precision,
            grad_clip=args.grad_clip,
            loss_chunk_size=4096,
        )
        if not metrics["stepped"]:
            raise RuntimeError("joint optimizer cycle was skipped")
        sampler.load_state_dict(staged_sampler_state)
        text_cursor = staged_text_cursor
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
                text_replay_batches=replay_validation_batches,
                expected_text_replay_contract_sha256=(
                    text_validation_dataset.contract_sha256
                ),
            )
            eligibility = dual_baseline_joint_eligibility(
                last_eval,
                runtime_baseline=baseline,
                historical_visual_baseline=historical_baseline,
            )
            last_eval["eligibility"] = eligibility
            current_cer = float(last_eval["deployment_weighted_cer"])
            improved = visual_cli._strict_relative_improvement(
                current_cer,
                best_cer,
            )
            if eligibility["eligible"] and improved:
                best_cer = current_cer
                best_eval = copy.deepcopy(last_eval)
                bad_evals = 0
                visual_cli._require_runtime_source_unchanged(
                    runtime_source_receipt,
                    extra_scripts=("scripts/train_ocr_anyres_joint_sft.py",),
                )
                metadata = _metadata(
                    prepared,
                    cycle=cycle,
                    args=args,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    text_train_dataset=text_train_dataset,
                    text_validation_dataset=text_validation_dataset,
                    text_partition=text_partition,
                    sampler=sampler,
                    text_cursor=text_cursor,
                    dataset_report=dataset_report,
                    historical_baseline=historical_baseline,
                    baseline=baseline,
                    last_validation=last_eval,
                    optimizer_contract=optimizer_contract,
                    runtime_source_receipt=runtime_source_receipt,
                    runtime_environment=runtime_environment,
                    final=False,
                )
                best_checkpoint = save_checkpoint(
                    output / "best", cycle, model, optimizer, scheduler,
                    metadata=metadata, keep_last_n=3, scaler=scaler,
                )
                if best_checkpoint is None:
                    raise RuntimeError("eligible joint best checkpoint was not saved")
            elif math.isfinite(best_cer):
                bad_evals += 1
                if bad_evals >= args.early_stop_patience:
                    stop_reason = "validation_plateau"
        if cycle % args.save_every == 0:
            visual_cli._require_runtime_source_unchanged(
                runtime_source_receipt,
                extra_scripts=("scripts/train_ocr_anyres_joint_sft.py",),
            )
            metadata = _metadata(
                prepared,
                cycle=cycle,
                args=args,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                text_train_dataset=text_train_dataset,
                text_validation_dataset=text_validation_dataset,
                text_partition=text_partition,
                sampler=sampler,
                text_cursor=text_cursor,
                dataset_report=dataset_report,
                historical_baseline=historical_baseline,
                baseline=baseline,
                last_validation=last_eval,
                optimizer_contract=optimizer_contract,
                runtime_source_receipt=runtime_source_receipt,
                runtime_environment=runtime_environment,
                final=False,
            )
            save_checkpoint(
                output / "last", cycle, model, optimizer, scheduler,
                metadata=metadata, keep_last_n=3, scaler=scaler,
            )
        print(json.dumps({"cycle": cycle, **metrics}, sort_keys=True), flush=True)
        if stop_reason == "validation_plateau":
            break

    visual_cli._require_runtime_source_unchanged(
        runtime_source_receipt,
        extra_scripts=("scripts/train_ocr_anyres_joint_sft.py",),
    )
    final_metadata = _metadata(
        prepared,
        cycle=completed_cycle,
        args=args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        text_train_dataset=text_train_dataset,
        text_validation_dataset=text_validation_dataset,
        text_partition=text_partition,
        sampler=sampler,
        text_cursor=text_cursor,
        dataset_report=dataset_report,
        historical_baseline=historical_baseline,
        baseline=baseline,
        last_validation=last_eval,
        optimizer_contract=optimizer_contract,
        runtime_source_receipt=runtime_source_receipt,
        runtime_environment=runtime_environment,
        final=True,
    )
    final_metadata["stop_reason"] = stop_reason
    final_checkpoint = save_checkpoint(
        output / "final", completed_cycle, model, optimizer, scheduler,
        metadata=final_metadata, keep_last_n=3, scaler=scaler,
    )
    if final_checkpoint is None:
        raise RuntimeError("joint final checkpoint was not saved")
    result = {
        "schema_version": 1,
        "kind": JOINT_STAGE_RESULT_KIND,
        "run_contract_sha256": prepared.run_contract["contract_canonical_sha256"],
        "parent_anyres_checkpoint": prepared.metadata_template[
            "parent_anyres_checkpoint"
        ],
        "final_checkpoint": visual_cli._checkpoint_identity(final_checkpoint),
        "best_eligible_checkpoint": (
            None if best_checkpoint is None else visual_cli._checkpoint_identity(best_checkpoint)
        ),
        "baseline_validation_sha256": visual_cli._json_sha256(baseline),
        "historical_visual_validation_sha256": visual_cli._json_sha256(
            historical_baseline
        ),
        "final_validation_sha256": visual_cli._json_sha256(last_eval),
        "best_validation_sha256": (
            None if best_eval is None else visual_cli._json_sha256(best_eval)
        ),
        "last_eligibility": last_eval.get("eligibility"),
        "grpo_promotion_allowed": best_checkpoint is not None,
        "completed_cycles": completed_cycle,
        "stop_reason": stop_reason,
        "optimizer_contract_sha256": optimizer_contract["canonical_sha256"],
        "quota_sampler_state_sha256": canonical_json_sha256(sampler.state_dict()),
        "text_replay_cursor_sha256": _text_cursor_state(
            text_train_dataset,
            cursor=text_cursor,
            batch_size=args.text_batch_size,
        )["canonical_sha256"],
        "runtime_source_receipt": runtime_source_receipt,
        "runtime_environment": runtime_environment,
    }
    result["canonical_sha256"] = canonical_json_sha256(result)
    visual_cli._atomic_json(output / "JOINT_STAGE_RESULT.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
