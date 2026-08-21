#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run the one-shot sealed AnyRes OCR benchmark after formal selection."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Model.config import BOS_ID, EOS_ID, PAD_ID, OMVTConfig  # noqa: E402
from Model.ocr.anyres_preprocess_contract import (  # noqa: E402
    validate_anyres_preprocess_contract,
)
from Model.ocr.tokenization import (  # noqa: E402
    canonical_json_sha256,
    make_ocr_target_encoder,
    native_tokenization_contract,
)
from Model.posttrain.checkpointing import (  # noqa: E402
    load_verified_checkpoint_metadata_envelope,
    reconstruct_policy_from_checkpoint,
    verified_file_sha256,
)
from Model.posttrain.ocr_anyres_collator import AnyresOCRSFTCollator  # noqa: E402
from Model.posttrain.ocr_anyres_grpo_run import (  # noqa: E402
    validate_anyres_grpo_progress,
    validate_anyres_grpo_run_contract,
)
from Model.posttrain.ocr_anyres_reward import (  # noqa: E402
    build_anyres_ocr_reward_adapter,
)
from Model.posttrain.ocr_joint_eval import evaluate_ocr_joint  # noqa: E402
from Model.posttrain.ocr_locked_contract import (  # noqa: E402
    claim_locked_benchmark_once,
    preclaim_locked_benchmark,
    write_claimed_evaluation_incomplete_stub,
)
from Model.posttrain.ocr_locked_data import (  # noqa: E402
    load_locked_benchmark_after_claim,
)
from Model.posttrain.ocr_locked_eval import (  # noqa: E402
    build_locked_evaluation_batches,
    build_locked_evaluation_report,
    finalize_locked_evaluation_report,
)
from Model.posttrain.ocr_selection import (  # noqa: E402
    validate_anyres_grpo_selection_receipt,
)
from Tokenizer.multimodal import NativeImageProcessorV2, PILImageProcessor  # noqa: E402
from Tokenizer.unified.bundle import TokenizerBundle  # noqa: E402
from scripts import train_ocr_anyres_sft as visual_cli  # noqa: E402
from scripts.train_ocr_anyres_grpo import (  # noqa: E402
    GRPO_PROGRESS_METADATA_KEY,
    GRPO_RUN_METADATA_KEY,
    _strict_json_file,
    select_authenticated_pilots,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--selection-receipt", required=True)
    parser.add_argument("--pilot-result", action="append", required=True)
    parser.add_argument("--kl-selection-receipt", required=True)
    parser.add_argument("--joint-stage-result", required=True)
    parser.add_argument("--dataset-admission", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--golden-anchor", required=True)
    parser.add_argument("--build-receipt", required=True)
    parser.add_argument("--sealed-root", required=True)
    parser.add_argument("--ledger-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--text-batch-size", type=int, default=4)
    return parser.parse_args(argv)


def _checkpoint_identity(path: Path, metadata_sha256: str, model_sha256: str):
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "model_sha256": verified_file_sha256(
            resolved / "model.pt", expected_sha256=model_sha256
        ),
        "metadata_sha256": verified_file_sha256(
            resolved / "meta.pt", expected_sha256=metadata_sha256
        ),
    }


def _dtype(precision: str) -> torch.dtype:
    return torch.float32 if precision == "fp32" else torch.bfloat16


def _current_evaluator_source() -> dict:
    return visual_cli._runtime_source_receipt(
        extra_scripts=(
            "scripts/train_ocr_anyres_grpo.py",
            "scripts/eval_ocr_anyres_locked.py",
        )
    )


def _preclaim(args: argparse.Namespace) -> dict:
    if len(args.pilot_result) != 3:
        raise ValueError("locked evaluation requires exactly three PILOT_RESULT files")
    if args.text_batch_size <= 0:
        raise ValueError("--text-batch-size must be positive")
    output = Path(args.out).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("--out must not already exist")
    if output.parent.is_symlink() or not output.parent.is_dir():
        raise ValueError("--out parent must be an existing real directory")
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    selected_envelope, metadata_sha = load_verified_checkpoint_metadata_envelope(
        checkpoint / "meta.pt"
    )
    metadata = selected_envelope["metadata"]
    selection_value = _strict_json_file(
        args.selection_receipt,
        where="AnyRes formal selection receipt",
    )
    terminal_value = selection_value.get("terminal_checkpoint")
    if not isinstance(terminal_value, dict):
        raise ValueError("selection receipt has no terminal checkpoint")
    terminal_envelope, _ = load_verified_checkpoint_metadata_envelope(
        Path(str(terminal_value.get("path"))) / "meta.pt",
        expected_sha256=terminal_value.get("metadata_sha256"),
    )
    terminal_metadata = terminal_envelope["metadata"]
    run = validate_anyres_grpo_run_contract(
        terminal_metadata.get(GRPO_RUN_METADATA_KEY)
    )
    progress = validate_anyres_grpo_progress(
        terminal_metadata.get(GRPO_PROGRESS_METADATA_KEY),
        run_contract=run,
    )
    selection = validate_anyres_grpo_selection_receipt(
        selection_value,
        formal_run_contract=run,
        terminal_progress=progress,
    )
    if selection["selected_checkpoint"]["path"] != str(checkpoint):
        raise ValueError("--checkpoint is not the formal selected checkpoint")
    selected_identity = _checkpoint_identity(
        checkpoint,
        selection["selected_checkpoint"]["metadata_sha256"],
        selection["selected_checkpoint"]["model_sha256"],
    )
    if metadata_sha != selected_identity["metadata_sha256"]:
        raise ValueError("selected metadata changed during preclaim")
    kl_receipt = select_authenticated_pilots(
        args.pilot_result,
        existing_receipt=args.kl_selection_receipt,
    )
    if kl_receipt != run["formal_selection_receipt"]:
        raise ValueError("formal run KL receipt differs from full pilot recomputation")
    joint_result = _strict_json_file(
        args.joint_stage_result,
        where="JOINT_STAGE_RESULT",
    )
    if joint_result.get("canonical_sha256") != run[
        "parent_joint_stage_result_sha256"
    ]:
        raise ValueError("JOINT_STAGE_RESULT differs from formal lineage")
    dataset_admission = _strict_json_file(
        args.dataset_admission,
        where="public AnyRes DATA_ADMISSION",
    )
    if canonical_json_sha256(dataset_admission) != run[
        "dataset_ready_admission_sha256"
    ]:
        raise ValueError("public dataset admission differs from formal run")
    bundle = TokenizerBundle.from_dir(args.tokenizer)
    issues = bundle.validate()
    if issues:
        raise ValueError("invalid tokenizer bundle: " + "; ".join(issues))
    tokenizer_contract = native_tokenization_contract(
        bundle.tokenizer,
        args.tokenizer,
    )
    if canonical_json_sha256(tokenizer_contract) != run["admission"][
        "tokenizer_contract_sha256"
    ]:
        raise ValueError("locked evaluator tokenizer differs from formal admission")
    preclaim = preclaim_locked_benchmark(args.golden_anchor, args.build_receipt)
    if preclaim.anchor_canonical_sha256 != selection[
        "locked_golden_anchor_sha256"
    ]:
        raise ValueError("locked anchor differs from formal selection receipt")
    if preclaim.anchor["preprocess_contract_sha256"] != run["admission"][
        "preprocess_contract_sha256"
    ] or preclaim.anchor["tokenizer_contract_sha256"] != run["admission"][
        "tokenizer_contract_sha256"
    ]:
        raise ValueError("locked anchor preprocess/tokenizer differs from formal run")
    selected = reconstruct_policy_from_checkpoint(
        checkpoint,
        require_vision=True,
        metadata_override=metadata,
        expected_metadata_sha256=selected_identity["metadata_sha256"],
        expected_model_sha256=selected_identity["model_sha256"],
    )
    reference_identity = copy.deepcopy(run["parent_joint_checkpoint"])
    reference = reconstruct_policy_from_checkpoint(
        reference_identity["path"],
        require_vision=True,
        expected_metadata_sha256=reference_identity["metadata_sha256"],
        expected_model_sha256=reference_identity["model_sha256"],
    )
    if (
        selected.rdt_config != reference.rdt_config
        or selected.omvt_config != reference.omvt_config
        or selected.native_detail_config != reference.native_detail_config
        or selected.vision_cross_attention_config
        != reference.vision_cross_attention_config
        or selected.native_migration_receipt != reference.native_migration_receipt
    ):
        raise ValueError("selected/reference AnyRes geometry differs")
    evaluator_source = _current_evaluator_source()
    return {
        "output": output,
        "checkpoint": checkpoint,
        "metadata": metadata,
        "run": run,
        "progress": progress,
        "selection": selection,
        "selection_file_sha256": verified_file_sha256(args.selection_receipt),
        "kl_receipt": kl_receipt,
        "kl_file_sha256": verified_file_sha256(args.kl_selection_receipt),
        "joint_result": joint_result,
        "joint_file_sha256": verified_file_sha256(args.joint_stage_result),
        "dataset_admission": dataset_admission,
        "bundle": bundle,
        "preclaim": preclaim,
        "selected": selected,
        "selected_identity": selected_identity,
        "reference": reference,
        "reference_identity": reference_identity,
        "evaluator_source": evaluator_source,
    }


def _run(args: argparse.Namespace) -> int:
    state = _preclaim(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    claim_payload = {
        "locked_golden_anchor_sha256": state["preclaim"].anchor_canonical_sha256,
        "selected_model_sha256": state["selected_identity"]["model_sha256"],
        "selected_metadata_sha256": state["selected_identity"]["metadata_sha256"],
        "reference_model_sha256": state["reference_identity"]["model_sha256"],
        "reference_metadata_sha256": state["reference_identity"]["metadata_sha256"],
        "selection_receipt_sha256": state["selection_file_sha256"],
        "formal_run_sha256": state["run"]["canonical_sha256"],
        "kl_selection_receipt_sha256": state["kl_file_sha256"],
        "joint_stage_result_sha256": state["joint_result"]["canonical_sha256"],
        "training_source_closure_sha256": state["run"]["source_closure_sha256"],
        "evaluator_source_closure_sha256": state["evaluator_source"][
            "canonical_sha256"
        ],
        "device": str(device),
        "precision": args.precision,
        "text_batch_size": args.text_batch_size,
        "output": str(state["output"]),
    }
    claim = claim_locked_benchmark_once(
        args.ledger_dir,
        state["preclaim"].build_receipt_sha256,
        claim_payload,
    )
    write_claimed_evaluation_incomplete_stub(
        state["output"],
        claim,
        {
            "selection_receipt_sha256": state["selection_file_sha256"],
            "formal_run_sha256": state["run"]["canonical_sha256"],
        },
    )
    encoder = make_ocr_target_encoder(state["bundle"].tokenizer, mode="native")
    reward_adapter = build_anyres_ocr_reward_adapter(state["bundle"])
    source = load_locked_benchmark_after_claim(
        args.sealed_root,
        state["preclaim"],
        claim,
        encode_reference=encoder,
        decode_ids=reward_adapter.decode_completion,
    )
    metadata = state["metadata"]
    preprocess = validate_anyres_preprocess_contract(
        metadata["anyres_preprocess_contract"]
    )
    expected_max_new_tokens = int(
        preprocess["budgets"]["output"]["recommended_max_new_tokens"]
    )
    locked_max_new_tokens = int(
        state["preclaim"].build_receipt["token_stats"][
            "recommended_max_new_tokens"
        ]
    )
    if locked_max_new_tokens != expected_max_new_tokens:
        raise ValueError("locked decode budget differs from formal preprocess contract")
    omvt_cfg = OMVTConfig(**metadata["omvt_config"])
    collator = AnyresOCRSFTCollator(
        encode_reference=encoder,
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
        max_seq_len=int(state["selected"].rdt_config.max_seq_len),
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_id=PAD_ID,
    )
    image_batches, text_batches = build_locked_evaluation_batches(
        source,
        image_collator=collator,
        text_batch_size=args.text_batch_size,
    )
    evaluation_args = {
        "val_batches": image_batches,
        "decode": reward_adapter.decode_completion,
        "max_new_tokens": locked_max_new_tokens,
        "device": device,
        "expected_image_dataset_contract_sha256": (
            source.image_dataset_contract_sha256
        ),
        "text_replay_batches": text_batches,
        "expected_text_replay_contract_sha256": source.text_dataset_contract_sha256,
    }
    selected_model = state["selected"].model.to(
        device=device, dtype=_dtype(args.precision)
    )
    selected_model.reverse_loss_enabled = False
    selected_report = evaluate_ocr_joint(model=selected_model, **evaluation_args)
    selected_model.to("cpu")
    del selected_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    reference_model = state["reference"].model.to(
        device=device, dtype=_dtype(args.precision)
    )
    reference_model.reverse_loss_enabled = False
    reference_model.requires_grad_(False)
    reference_model.eval()
    reference_report = evaluate_ocr_joint(model=reference_model, **evaluation_args)
    reference_model.to("cpu")
    del reference_model
    report = build_locked_evaluation_report(
        selected_report=selected_report,
        reference_report=reference_report,
        selected_checkpoint=state["selected_identity"],
        reference_checkpoint=state["reference_identity"],
        selection_receipt_sha256=state["selection_file_sha256"],
        kl_selection_receipt_sha256=state["kl_file_sha256"],
        joint_stage_result_sha256=state["joint_result"]["canonical_sha256"],
        source_closure_sha256=state["run"]["source_closure_sha256"],
        locked_anchor_sha256=state["preclaim"].anchor_canonical_sha256,
        locked_build_receipt_sha256=state["preclaim"].build_receipt_sha256,
        claim_marker_sha256=claim.marker_sha256,
    )
    if _current_evaluator_source() != state["evaluator_source"]:
        raise RuntimeError("evaluator source changed after the one-shot claim")
    finalize_locked_evaluation_report(state["output"], claim, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0 if report["comparison"]["production_eligible"] else 2


def main(argv: Sequence[str] | None = None) -> int:
    return _run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
