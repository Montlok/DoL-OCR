# -*- coding: utf-8 -*-

"""RDT GRPO (Group Relative Policy Optimization) training entry point.

Production-usable online RL with **verifiable** rewards (no reward model). Loads
a policy from an SFT/DPO checkpoint, builds a frozen reference, samples a group
of responses per prompt via the native ``generate()``, scores them with
rule-based rewards (exact/numeric match, Mongolian script purity, ``<think>``
format), normalizes advantages within each group, and takes a clipped
policy-gradient step with a KL penalty. Externally injected ``<tool_result>``
spans are excluded from the objective.

Usage::

    python -m scripts.train_grpo --smoke
    python -m scripts.train_grpo --config two_stage_pretrain \\
        --tokenizer path/to/bundle --data prompts/*.jsonl \\
        --init-checkpoint outputs/sft/latest --output outputs/grpo \\
        --group-size 8 --numeric-weight 1.0 --purity-weight 0.3 --format-weight 0.2
"""

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Model.config import (  # noqa: E402
    BOS_ID,
    EOS_ID,
    IMAGE_END_ID,
    IMAGE_PATCH_ID,
    IMAGE_START_ID,
    PAD_ID,
    OMVTConfig,
    RDTConfig,
    TrainingConfig,
    base_config,
    pretrain_config,
    segmented_pretrain_config,
    segmented_tiny_config,
    small_config,
    tiny_config,
    two_stage_pretrain_config,
    two_stage_tiny_config,
)
from Model.model import RDTForCausalLM  # noqa: E402
from Model.posttrain.checkpointing import (  # noqa: E402
    OCR_GRPO_CONTRACT_VERSION,
    reconstruct_policy_from_checkpoint,
)
from Model.posttrain.grpo import (  # noqa: E402
    GRPOConfig,
    grpo_compute_loss,
)
from Model.posttrain.ocr_eval import (  # noqa: E402
    build_ocr_pixel_batch as _ocr_pixel_batch,
    evaluate_ocr_manifest as _evaluate_ocr_validation,
)
from Model.posttrain.ocr_decode import decode_ocr_completion  # noqa: E402
from Model.posttrain.preference_data import OCRPromptDataset, PromptDataset  # noqa: E402
from Model.posttrain.rewards import (  # noqa: E402
    RewardConfig,
    compute_rewards_with_breakdown,
)
from Model.training import (  # noqa: E402
    RankZeroLogger,
    TrainState,
    apply_parallelism,
    build_optimizer,
    build_scheduler,
    clip_or_check_grad_norm,
    destroy_distributed,
    init_distributed,
    is_main_process,
    resolve_checkpoint_dir,
    resume_state,
    save_checkpoint,
)

CONFIG_CHOICES = {
    "tiny": tiny_config,
    "small": small_config,
    "base": base_config,
    "pretrain": pretrain_config,
    "two_stage_tiny": two_stage_tiny_config,
    "two_stage_pretrain": two_stage_pretrain_config,
    "segmented_tiny": segmented_tiny_config,
    "segmented_pretrain": segmented_pretrain_config,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO-align RDT with verifiable rewards")
    p.add_argument("--config", choices=list(CONFIG_CHOICES), default="two_stage_tiny")
    p.add_argument("--task", choices=["text", "ocr"], default="text",
                   help="ocr enables strict image-conditioned GRPO")
    p.add_argument("--tokenizer", default="", help="TokenizerBundle dir (real runs)")
    p.add_argument("--data", default="", help="prompt JSONL path")
    p.add_argument("--init-checkpoint", default="",
                   help="checkpoint dir to initialize policy + reference")
    p.add_argument("--output", default="outputs/grpo")
    p.add_argument("--resume", default="")
    p.add_argument("--dist", choices=["single", "ddp", "fsdp"], default="single")
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--prompts-per-step", type=int, default=2)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--max-prompt-len", type=int, default=512)
    p.add_argument("--temperature", type=float, default=None,
                   help="must remain 1.0 for exact on-policy scoring")
    p.add_argument("--top-p", type=float, default=None,
                   help="reserved; truncated sampling is rejected by GRPO")
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--keep-last-n", type=int, default=3)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--kl-coef", type=float, default=0.04)
    p.add_argument("--exact-weight", type=float, default=0.0)
    p.add_argument("--numeric-weight", type=float, default=0.0)
    p.add_argument("--purity-weight", type=float, default=0.0)
    p.add_argument("--purity-min-ratio", type=float, default=0.0)
    p.add_argument("--format-weight", type=float, default=0.0)
    p.add_argument("--grapheme-cer-weight", type=float, default=None,
                   help="dense -grapheme-CER reward; OCR default 1.0")
    p.add_argument("--grapheme-cer-cap", type=float, default=None,
                   help="optional CER tail cap; uncapped preserves dense ordering")
    p.add_argument("--normalized-cer-reward", action="store_true",
                   help="fold nominal Unicode before CER (raw grapheme CER is default)")
    p.add_argument("--cer-backend", choices=["auto", "python", "rust"], default="auto")
    p.add_argument("--empty-penalty-weight", type=float, default=None,
                   help="OCR default 0.1")
    p.add_argument("--invalid-token-penalty-weight", type=float, default=None,
                   help="penalize reserved control ids hidden by normal decode; OCR default 0.1")
    p.add_argument("--length-penalty-weight", type=float, default=None,
                   help="OCR default 0.1")
    p.add_argument("--max-length-ratio", type=float, default=2.0)
    p.add_argument("--recurrent-steps", type=int, default=None,
                   help="latent depth for sampling + scoring; defaults to config")
    p.add_argument("--dist-backend", choices=["nccl", "gloo"], default="nccl")
    p.add_argument("--image-root", default="",
                   help="base directory for relative OCR image paths")
    p.add_argument("--required-split", default="rl_train",
                   help="OCR rows must carry this split label")
    p.add_argument("--golden-manifest", default="",
                   help="locked golden JSONL; any id/image overlap aborts")
    p.add_argument("--validation-manifest", default="",
                   help="held-out rl_val JSONL used for checkpoint selection")
    p.add_argument("--validation-split", default="rl_val")
    p.add_argument("--golden-split", default="golden")
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--min-validation-eos-rate", type=float, default=0.99)
    p.add_argument("--max-validation-invalid-rate", type=float, default=0.0)
    p.add_argument("--early-stop-patience", type=int, default=5,
                   help="stop after this many validation checks without CER improvement")
    p.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    p.add_argument("--train-scope", choices=["all", "vision", "projector"],
                   default=None,
                   help="OCR default vision (language frozen); text default all")
    p.add_argument("--max-degenerate-steps", type=int, default=20,
                   help="abort after this many all-zero-advantage steps")
    p.add_argument("--max-consecutive-optimizer-skips", type=int, default=20,
                   help="abort fp16 training after this many scaler-skipped updates")
    p.add_argument("--kl-abort-threshold", type=float, default=1.0)
    p.add_argument("--kl-abort-patience", type=int, default=5)
    p.add_argument("--log-ratio-clip", type=float, default=20.0)
    p.add_argument("--min-visual-logit-delta", type=float, default=1e-6,
                   help="abort OCR RL if real-vs-blank image logits are insensitive")
    p.add_argument("--min-baseline-visual-cer-gap", type=float, default=0.0,
                   help="require blank validation CER - real validation CER above this")
    p.add_argument("--no-tool-result-mask", action="store_true",
                   help="disable masking of externally-injected <tool_result> "
                        "spans from the RL objective (on by default)")
    p.add_argument("--smoke", action="store_true", help="run 4 in-memory steps")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def _apply_task_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.train_scope is None:
        args.train_scope = "vision" if args.task == "ocr" else "all"
    if args.temperature is None:
        args.temperature = 1.0
    return args


def _build_model_cfg(args: argparse.Namespace) -> RDTConfig:
    cfg = CONFIG_CHOICES[args.config]()
    if args.recurrent_steps is not None:
        if args.recurrent_steps <= 0:
            raise ValueError("--recurrent-steps must be positive")
        cfg.recurrent_steps = args.recurrent_steps
    return cfg


def _build_train_cfg(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        train_data=args.data,
        micro_batch_size=args.prompts_per_step,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        precision=args.precision,
        parallel=args.dist,
        dist_backend=args.dist_backend,
        fsdp_mixed_precision=args.precision,
        output_dir=args.output,
        save_every=args.save_every,
        keep_last_n=args.keep_last_n,
        log_every=args.log_every,
        resume=args.resume,
        tensorboard=False,
        seed=args.seed,
    )


def _reward_cfg(args: argparse.Namespace) -> RewardConfig:
    cer_weight = (
        1.0 if args.task == "ocr" else 0.0
    ) if args.grapheme_cer_weight is None else args.grapheme_cer_weight
    empty_weight = (
        0.1 if args.task == "ocr" else 0.0
    ) if args.empty_penalty_weight is None else args.empty_penalty_weight
    length_weight = (
        0.1 if args.task == "ocr" else 0.0
    ) if args.length_penalty_weight is None else args.length_penalty_weight
    invalid_weight = (
        0.1 if args.task == "ocr" else 0.0
    ) if args.invalid_token_penalty_weight is None else args.invalid_token_penalty_weight
    return RewardConfig(
        exact_match_weight=args.exact_weight,
        numeric_match_weight=args.numeric_weight,
        purity_weight=args.purity_weight,
        purity_min_ratio=args.purity_min_ratio,
        format_weight=args.format_weight,
        grapheme_cer_weight=cer_weight,
        grapheme_cer_cap=args.grapheme_cer_cap,
        grapheme_cer_normalize=args.normalized_cer_reward,
        grapheme_cer_backend=args.cer_backend,
        empty_response_penalty_weight=empty_weight,
        invalid_token_penalty_weight=invalid_weight,
        length_excess_penalty_weight=length_weight,
        max_length_ratio=args.max_length_ratio,
    )


def _smoke_encode(text: str) -> list[int]:
    return [(ord(c) % (RDTConfig().vocab_size - 1000)) + 1000 for c in text]


def _smoke_decode(ids: torch.Tensor) -> str:
    return "".join(chr((int(i) % 90) + 33) for i in ids.tolist() if int(i) != PAD_ID)


def _grpo_config(
    args: argparse.Namespace,
    model_cfg: RDTConfig,
    tool_result_open_ids: Sequence[int] | None = None,
    tool_result_close_ids: Sequence[int] | None = None,
) -> GRPOConfig:
    return GRPOConfig(
        clip_eps=args.clip_eps,
        kl_coef=args.kl_coef,
        recurrent_steps=model_cfg.recurrent_steps,
        group_size=args.group_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        tool_result_open_ids=list(tool_result_open_ids or []),
        tool_result_close_ids=list(tool_result_close_ids or []),
        log_ratio_clip=args.log_ratio_clip,
    )


def _validate_args(args: argparse.Namespace) -> int:
    finite_args = {
        "learning-rate": args.learning_rate,
        "weight-decay": args.weight_decay,
        "grad-clip": args.grad_clip,
        "clip-eps": args.clip_eps,
        "kl-coef": args.kl_coef,
        "kl-abort-threshold": args.kl_abort_threshold,
        "log-ratio-clip": args.log_ratio_clip,
        "min-visual-logit-delta": args.min_visual_logit_delta,
        "min-baseline-visual-cer-gap": args.min_baseline_visual_cer_gap,
        "early-stop-min-delta": args.early_stop_min_delta,
        "min-validation-eos-rate": args.min_validation_eos_rate,
        "max-validation-invalid-rate": args.max_validation_invalid_rate,
    }
    nonfinite = [name for name, value in finite_args.items() if not math.isfinite(value)]
    if nonfinite:
        print(
            "[error] numeric arguments must be finite: " + ", ".join(nonfinite),
            file=sys.stderr,
        )
        return 2
    if args.group_size <= 1:
        print("[error] --group-size must be at least 2", file=sys.stderr)
        return 2
    if args.prompts_per_step <= 0:
        print("[error] --prompts-per-step must be positive", file=sys.stderr)
        return 2
    if args.max_new_tokens <= 0 and not args.smoke:
        print("[error] --max-new-tokens must be positive", file=sys.stderr)
        return 2
    if args.max_prompt_len <= 0:
        print("[error] --max-prompt-len must be positive", file=sys.stderr)
        return 2
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.grad_clip <= 0:
        print(
            "[error] learning-rate/grad-clip must be positive and weight-decay "
            "must be non-negative",
            file=sys.stderr,
        )
        return 2
    if not 0.0 < args.clip_eps < 1.0 or args.kl_coef < 0:
        print("[error] clip-eps must be in (0,1) and kl-coef non-negative", file=sys.stderr)
        return 2
    if args.log_ratio_clip <= 0:
        print("[error] --log-ratio-clip must be positive", file=sys.stderr)
        return 2
    if args.resume and args.init_checkpoint:
        print("[error] use --resume or --init-checkpoint, not both", file=sys.stderr)
        return 2
    if args.max_degenerate_steps <= 0:
        print("[error] --max-degenerate-steps must be positive", file=sys.stderr)
        return 2
    if args.max_consecutive_optimizer_skips <= 0:
        print(
            "[error] --max-consecutive-optimizer-skips must be positive",
            file=sys.stderr,
        )
        return 2
    if args.kl_abort_threshold <= 0 or args.kl_abort_patience <= 0:
        print("[error] KL abort threshold/patience must be positive", file=sys.stderr)
        return 2
    if args.min_visual_logit_delta <= 0:
        print("[error] --min-visual-logit-delta must be positive", file=sys.stderr)
        return 2
    if args.min_baseline_visual_cer_gap < 0:
        print(
            "[error] --min-baseline-visual-cer-gap must be non-negative",
            file=sys.stderr,
        )
        return 2
    if args.eval_every <= 0 or args.eval_batch_size <= 0:
        print("[error] eval interval/batch size must be positive", file=sys.stderr)
        return 2
    if not 0.0 <= args.min_validation_eos_rate <= 1.0:
        print("[error] --min-validation-eos-rate must be in [0,1]", file=sys.stderr)
        return 2
    if not 0.0 <= args.max_validation_invalid_rate <= 1.0:
        print("[error] --max-validation-invalid-rate must be in [0,1]", file=sys.stderr)
        return 2
    if args.early_stop_patience <= 0 or args.early_stop_min_delta < 0:
        print("[error] early-stop patience must be positive and min-delta non-negative", file=sys.stderr)
        return 2
    try:
        reward_cfg = _reward_cfg(args)
    except ValueError as exc:
        print(f"[error] invalid reward config: {exc}", file=sys.stderr)
        return 2
    if not args.smoke:
        if not args.tokenizer:
            print("[error] --tokenizer is required (or pass --smoke)", file=sys.stderr)
            return 2
        if not args.data:
            print("[error] --data is required (or pass --smoke)", file=sys.stderr)
            return 2
        reward_weights = (
            reward_cfg.exact_match_weight,
            reward_cfg.numeric_match_weight,
            reward_cfg.purity_weight,
            reward_cfg.format_weight,
            reward_cfg.grapheme_cer_weight,
            reward_cfg.empty_response_penalty_weight,
            reward_cfg.invalid_token_penalty_weight,
            reward_cfg.length_excess_penalty_weight,
        )
        if not any(weight > 0 for weight in reward_weights):
            print("[error] at least one reward weight must be > 0", file=sys.stderr)
            return 2
        if args.temperature != 1.0 or args.top_p is not None:
            print(
                "[error] GRPO requires --temperature 1.0 and no --top-p so "
                "rollout and scored policy distributions match",
                file=sys.stderr,
            )
            return 2
        if args.task == "ocr":
            if not (args.init_checkpoint or args.resume):
                print("[error] OCR GRPO requires a multimodal checkpoint", file=sys.stderr)
                return 2
            if not args.golden_manifest:
                print("[error] OCR GRPO requires --golden-manifest leakage gate", file=sys.stderr)
                return 2
            if not args.validation_manifest:
                print(
                    "[error] OCR GRPO requires --validation-manifest for "
                    "checkpoint selection/early stopping",
                    file=sys.stderr,
                )
                return 2
            if reward_cfg.grapheme_cer_weight <= 0:
                print("[error] OCR GRPO requires --grapheme-cer-weight > 0", file=sys.stderr)
                return 2
            if reward_cfg.grapheme_cer_normalize and args.cer_backend == "auto":
                print(
                    "[error] normalized OCR reward requires an explicit "
                    "--cer-backend python|rust for resume-stable semantics",
                    file=sys.stderr,
                )
                return 2
            if reward_cfg.format_weight:
                print("[error] OCR GRPO must not reward <think> formatting", file=sys.stderr)
                return 2
    return 0


def _prepare_reference_model(
    reference: RDTForCausalLM,
    train_cfg: TrainingConfig,
    local_rank: int,
    device: torch.device,
) -> torch.nn.Module:
    reference.reverse_loss_enabled = False
    reference.eval()
    for param in reference.parameters():
        param.requires_grad_(False)

    if (
        train_cfg.parallel == "fsdp"
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        return apply_parallelism(reference, train_cfg, local_rank).eval()

    # DDP does not shard parameters and can reject modules with no trainable
    # parameters, so a frozen reference only benefits from FSDP wrapping.
    return reference.to(device).eval()


def _build_reference_model(
    policy: RDTForCausalLM,
    model_cfg: RDTConfig,
    train_cfg: TrainingConfig,
    local_rank: int,
    device: torch.device,
) -> torch.nn.Module:
    del model_cfg  # retained for compatibility with the existing helper API
    # deepcopy preserves a pre-installed OMVT injector and its exact geometry.
    return _prepare_reference_model(
        copy.deepcopy(policy), train_cfg, local_rank, device
    )


def _load_immutable_reference(
    checkpoint: str,
    source_checkpoint: str,
    model_cfg: RDTConfig,
    omvt_cfg: OMVTConfig | None,
    train_cfg: TrainingConfig,
    local_rank: int,
    device: torch.device,
    *,
    require_vision: bool,
) -> tuple[torch.nn.Module, Path]:
    restored = reconstruct_policy_from_checkpoint(
        checkpoint,
        require_vision=require_vision,
    )
    if not (restored.checkpoint_dir / "COMPLETE").is_file():
        raise ValueError("immutable reference checkpoint has no COMPLETE marker")
    if asdict(restored.rdt_config) != asdict(model_cfg):
        raise ValueError("immutable reference RDT config differs from resumed policy")
    restored_omvt = asdict(restored.omvt_config) if restored.omvt_config else None
    current_omvt = asdict(omvt_cfg) if omvt_cfg else None
    if restored_omvt != current_omvt:
        raise ValueError("immutable reference OMVT config differs from resumed policy")
    if restored.metadata.get("phase") != "grpo_reference":
        raise ValueError("reference checkpoint is not an immutable GRPO reference")
    if restored.metadata.get("contract_version") != OCR_GRPO_CONTRACT_VERSION:
        raise ValueError("immutable reference uses a different OCR GRPO contract")
    if restored.metadata.get("immutable") is not True:
        raise ValueError("reference checkpoint is not marked immutable")
    if restored.metadata.get("source_checkpoint") != source_checkpoint:
        raise ValueError("reference checkpoint points to a different source policy")
    reference = _prepare_reference_model(
        restored.model,
        train_cfg,
        local_rank,
        device,
    )
    return reference, restored.checkpoint_dir


def _configure_trainable_scope(policy: RDTForCausalLM, scope: str) -> list[str]:
    if scope == "all":
        for param in policy.parameters():
            param.requires_grad_(True)
        # The legacy MLP encoder is not on the OMVT OCR forward path.  Keeping
        # random, permanently zero-gradient fallback parameters in the optimizer
        # obscures trainable counts and can create optimizer-state surprises.
        if policy.vision.omvt is not None:
            for param in policy.vision.encoder.parameters():
                param.requires_grad_(False)
    else:
        if policy.vision.omvt is None:
            raise ValueError(f"train scope {scope!r} requires an installed OMVT module")
        for param in policy.parameters():
            param.requires_grad_(False)
        target = policy.vision.omvt if scope == "vision" else policy.vision.omvt.projector
        for param in target.parameters():
            param.requires_grad_(True)
    names = [name for name, param in policy.named_parameters() if param.requires_grad]
    if not names:
        raise ValueError("train scope selected zero parameters")
    return names


def _precision_context(precision: str, device: torch.device):
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _distributed_mean_metrics(
    metrics: dict[str, float],
    device: torch.device,
) -> dict[str, float]:
    """Return identical rank-mean health metrics on every worker."""

    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return dict(metrics)
    keys = sorted(metrics)
    values = torch.tensor(
        [float(metrics[key]) for key in keys],
        dtype=torch.float64,
        device=device,
    )
    torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
    values /= torch.distributed.get_world_size()
    return {key: float(value) for key, value in zip(keys, values.tolist())}


def _distributed_max(value: float, device: torch.device) -> float:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return float(value)
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return float(tensor)


def _all_ranks_true(value: bool, device: torch.device) -> bool:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return bool(value)
    tensor = torch.tensor(int(bool(value)), dtype=torch.int32, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN)
    return bool(tensor.item())


def _consensus_bool(value: bool, device: torch.device, name: str) -> bool:
    """Require a mixed-precision control decision to agree on every rank."""

    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return bool(value)
    tensor = torch.tensor(int(bool(value)), dtype=torch.int32, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    votes = int(tensor.item())
    world_size = torch.distributed.get_world_size()
    if votes not in {0, world_size}:
        raise RuntimeError(
            f"distributed workers disagreed on {name}: {votes}/{world_size} true"
        )
    return votes == world_size


def _validate_resumable_checkpoint_files(
    checkpoint: str | Path,
    *,
    require_scaler: bool,
) -> Path:
    """Reject model-only or interrupted artifacts passed as ``--resume``."""

    checkpoint_dir = resolve_checkpoint_dir(checkpoint)
    required = {
        "COMPLETE",
        "model.pt",
        "optimizer.pt",
        "scheduler.pt",
        "rng.pt",
        "meta.pt",
    }
    if require_scaler:
        required.add("scaler.pt")
    missing = sorted(name for name in required if not (checkpoint_dir / name).is_file())
    if missing:
        raise ValueError(
            "--resume requires a complete resumable GRPO checkpoint; missing: "
            + ", ".join(missing)
        )
    return checkpoint_dir


def _make_grad_scaler(
    precision: str,
    device: torch.device,
    model: torch.nn.Module | None = None,
):
    enabled = precision == "fp16" and device.type == "cuda"
    if enabled and model is not None and hasattr(model, "clip_grad_norm_"):
        try:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            return ShardedGradScaler()
        except (ImportError, RuntimeError):  # pragma: no cover - old torch fallback
            pass
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - old torch fallback
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _dataset_exclusions(
    dataset: OCRPromptDataset,
) -> tuple[set[str], set[str], set[str], set[str]]:
    """Extract identity keys used to prove split disjointness."""

    ids: set[str] = set()
    images: set[str] = set()
    hashes: set[str] = set()
    groups: set[str] = set()
    for idx in range(len(dataset)):
        row = dataset[idx]
        ids.add(row["id"])
        groups.add(row["group_id"])
        images.add(row["image"])
        if row.get("sha256"):
            hashes.add(row["sha256"])
    return ids, images, hashes, groups


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distributed_file_sha256(path: str | Path, rank: int) -> str:
    """Hash a shared artifact once, then broadcast the digest to every rank."""

    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return _file_sha256(path)
    payload = [_file_sha256(path) if rank == 0 else None]
    torch.distributed.broadcast_object_list(payload, src=0)
    if not isinstance(payload[0], str):
        raise RuntimeError("rank 0 failed to broadcast artifact SHA-256")
    return payload[0]


def _data_contract(
    args: argparse.Namespace,
    bundle,
) -> dict[str, str]:
    """Immutable data/tokenizer fingerprint persisted across resumes."""

    vocab_payload = json.dumps(
        sorted((str(token), int(idx)) for token, idx in bundle.tokenizer.vocab.items()),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    contract = {
        "train_manifest_sha256": _file_sha256(args.data),
        "tokenizer_vocab_sha256": hashlib.sha256(vocab_payload).hexdigest(),
        "tokenizer_manifest_sha256": _file_sha256(
            Path(args.tokenizer) / "manifest.json"
        ),
        "image_root": str(Path(args.image_root).absolute()) if args.image_root else "",
        "required_split": args.required_split,
    }
    if args.task == "ocr":
        contract.update(
            {
                "validation_manifest_sha256": _file_sha256(args.validation_manifest),
                "golden_manifest_sha256": _file_sha256(args.golden_manifest),
                "validation_split": args.validation_split,
                "golden_split": args.golden_split,
            }
        )
    return contract


def _ocr_pixel_values(
    rows: list[dict],
    processor,
    omvt_cfg: OMVTConfig,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    packed = _ocr_pixel_batch(rows, processor, omvt_cfg, device)
    out: list[dict[str, torch.Tensor]] = []
    for idx in range(len(rows)):
        one: dict[str, torch.Tensor] = {}
        for key, value in packed.items():
            # bbox geometry is shared [N,4]; images/patches carry leading B.
            one[key] = (
                value if key.endswith("_bbox") else value[idx : idx + 1]
            )
        out.append(one)
    return out


@torch.no_grad()
def _visual_conditioning_delta(
    policy: torch.nn.Module,
    prompt: torch.Tensor,
    pixel_values: dict[str, torch.Tensor],
    precision: str,
    device: torch.device,
) -> float:
    """Max next-token-logit change between a real and blank visual payload."""

    blank = {
        key: value if key.endswith("_bbox") else torch.zeros_like(value)
        for key, value in pixel_values.items()
    }
    was_training = policy.training
    policy.eval()
    try:
        ids = prompt.unsqueeze(0)
        with _precision_context(precision, device):
            real = policy(
                input_ids=ids,
                pixel_values=pixel_values,
                return_logits=True,
            )["logits"][:, -1, :].float()
            empty = policy(
                input_ids=ids,
                pixel_values=blank,
                return_logits=True,
            )["logits"][:, -1, :].float()
        return float((real - empty).abs().max())
    finally:
        if was_training:
            policy.train()


def _checkpoint_metadata(
    args: argparse.Namespace,
    train_cfg: TrainingConfig,
    model_cfg: RDTConfig,
    omvt_cfg: OMVTConfig | None,
    reward_cfg: RewardConfig,
    grpo_cfg: GRPOConfig,
    source_checkpoint: str,
    reference_checkpoint: str,
    *,
    data_contract: dict[str, str] | None = None,
    health_state: dict[str, object] | None = None,
    final: bool = False,
) -> dict:
    return {
        "phase": "grpo",
        "contract_version": OCR_GRPO_CONTRACT_VERSION,
        "task": args.task,
        "config": args.config,
        "rdt_config": asdict(model_cfg),
        "omvt_config": asdict(omvt_cfg) if omvt_cfg is not None else None,
        "training_config": asdict(train_cfg),
        "grpo_config": asdict(grpo_cfg),
        "reward_config": asdict(reward_cfg),
        "train_scope": args.train_scope,
        "max_consecutive_optimizer_skips": args.max_consecutive_optimizer_skips,
        "source_checkpoint": source_checkpoint,
        "reference_checkpoint": reference_checkpoint,
        "golden_manifest": args.golden_manifest,
        "validation_manifest": args.validation_manifest,
        "required_split": args.required_split,
        "validation_split": args.validation_split,
        "golden_split": args.golden_split,
        "eval_config": {
            "eval_every": args.eval_every,
            "eval_batch_size": args.eval_batch_size,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            "min_baseline_visual_cer_gap": args.min_baseline_visual_cer_gap,
            "min_validation_eos_rate": args.min_validation_eos_rate,
            "max_validation_invalid_rate": args.max_validation_invalid_rate,
        },
        "data_config": {
            "max_prompt_len": args.max_prompt_len,
        },
        "safety_config": {
            "max_degenerate_steps": args.max_degenerate_steps,
            "kl_abort_threshold": args.kl_abort_threshold,
            "kl_abort_patience": args.kl_abort_patience,
            "max_consecutive_optimizer_skips": (
                args.max_consecutive_optimizer_skips
            ),
            "min_visual_logit_delta": args.min_visual_logit_delta,
        },
        "data_contract": dict(data_contract or {}),
        "health_state": dict(health_state or {}),
        "final": bool(final),
    }


def _reference_metadata(
    model_cfg: RDTConfig,
    omvt_cfg: OMVTConfig | None,
    source_checkpoint: str,
) -> dict:
    return {
        "phase": "grpo_reference",
        "contract_version": OCR_GRPO_CONTRACT_VERSION,
        "rdt_config": asdict(model_cfg),
        "omvt_config": asdict(omvt_cfg) if omvt_cfg is not None else None,
        "source_checkpoint": source_checkpoint,
        "immutable": True,
    }


_RESUME_MUTABLE_FIELDS = {
    "output_dir",
    "save_every",
    "keep_last_n",
    "resume",
    "log_every",
    "tensorboard",
    "wandb_project",
}


def _validate_resume_contract(
    metadata: dict,
    args: argparse.Namespace,
    train_cfg: TrainingConfig,
    reward_cfg: RewardConfig,
    grpo_cfg: GRPOConfig,
    data_contract: dict[str, str],
) -> None:
    """Refuse resumes that would mix optimizer, reward, or policy semantics."""

    if metadata.get("phase") != "grpo":
        raise ValueError("--resume checkpoint is not a GRPO checkpoint")
    expected_top = {
        "contract_version": OCR_GRPO_CONTRACT_VERSION,
        "task": args.task,
        "train_scope": args.train_scope,
        "max_consecutive_optimizer_skips": args.max_consecutive_optimizer_skips,
        "required_split": args.required_split,
        "validation_split": args.validation_split,
        "golden_split": args.golden_split,
    }
    conflicts: list[str] = []
    for key, current in expected_top.items():
        if metadata.get(key) != current:
            conflicts.append(f"{key}: checkpoint={metadata.get(key)!r} current={current!r}")

    for key, current in (
        ("reward_config", asdict(reward_cfg)),
        ("grpo_config", asdict(grpo_cfg)),
        ("data_contract", data_contract),
        (
            "eval_config",
            {
                "eval_every": args.eval_every,
                "eval_batch_size": args.eval_batch_size,
                "early_stop_patience": args.early_stop_patience,
                "early_stop_min_delta": args.early_stop_min_delta,
                "min_baseline_visual_cer_gap": args.min_baseline_visual_cer_gap,
                "min_validation_eos_rate": args.min_validation_eos_rate,
                "max_validation_invalid_rate": args.max_validation_invalid_rate,
            },
        ),
        ("data_config", {"max_prompt_len": args.max_prompt_len}),
        (
            "safety_config",
            {
                "max_degenerate_steps": args.max_degenerate_steps,
                "kl_abort_threshold": args.kl_abort_threshold,
                "kl_abort_patience": args.kl_abort_patience,
                "max_consecutive_optimizer_skips": (
                    args.max_consecutive_optimizer_skips
                ),
                "min_visual_logit_delta": args.min_visual_logit_delta,
            },
        ),
    ):
        saved = metadata.get(key)
        if saved != current:
            conflicts.append(f"{key}: checkpoint and current configuration differ")

    saved_train = metadata.get("training_config")
    if not isinstance(saved_train, dict):
        conflicts.append("training_config: missing from checkpoint")
    else:
        for key, current in asdict(train_cfg).items():
            if key in _RESUME_MUTABLE_FIELDS:
                continue
            if saved_train.get(key) != current:
                conflicts.append(
                    f"training_config.{key}: checkpoint={saved_train.get(key)!r} "
                    f"current={current!r}"
                )
    if args.task == "ocr":
        health = metadata.get("health_state")
        required_health = {
            "best_val_grapheme_cer",
            "best_val_step",
            "bad_eval_count",
            "eval_count",
            "best_checkpoint",
            "degenerate_streak",
            "high_kl_streak",
            "baseline_visual_cer_gap",
            "batches_consumed",
            "optimizer_skip_count",
            "consecutive_optimizer_skips",
            "best_val_eligible",
        }
        if not isinstance(health, dict):
            conflicts.append("health_state: missing from OCR GRPO checkpoint")
        else:
            missing = sorted(required_health - set(health))
            if missing:
                conflicts.append("health_state: missing " + ", ".join(missing))
    if conflicts:
        raise ValueError("unsafe GRPO resume configuration:\n  - " + "\n  - ".join(conflicts))


def _iter_prompt_batches(
    dataset,
    batch_size: int,
    *,
    rank: int = 0,
    world_size: int = 1,
    shuffle: bool = False,
    seed: int = 42,
    start_batch: int = 0,
):
    """Yield rank-sharded prompt rows, cycling forever.

    When the prompt set is smaller than the process count, every rank reads the
    full set to avoid distributed collectives hanging on ranks with no batches.
    """
    n = len(dataset)
    if n <= 0:
        raise ValueError("prompt dataset is empty")
    if world_size > 1 and n >= world_size:
        indices = list(range(rank, n, world_size))
    else:
        indices = list(range(n))
    if not indices:
        raise ValueError(f"rank={rank} received no prompts from dataset of size {n}")
    if start_batch < 0:
        raise ValueError("start_batch must be non-negative")
    rng = random.Random(seed + rank)
    order = list(indices)
    if shuffle:
        rng.shuffle(order)
    i = 0
    remaining = start_batch * batch_size
    while remaining:
        available = len(order) - i
        take = min(remaining, available)
        i += take
        remaining -= take
        if i >= len(order) and remaining:
            i = 0
            order = list(indices)
            if shuffle:
                rng.shuffle(order)
    while True:
        batch = []
        for _ in range(batch_size):
            if i >= len(order):
                i = 0
                order = list(indices)
                if shuffle:
                    rng.shuffle(order)
            batch.append(dataset[order[i]])
            i += 1
        yield batch


def _smoke_dataset():
    prompts = [
        {"prompt_ids": _smoke_encode("2+2? "), "reference": "4"},
        {"prompt_ids": _smoke_encode("sain uu "), "reference": None},
    ]
    return prompts


def main(argv: list[str] | None = None) -> int:
    args = _apply_task_defaults(parse_args(argv))
    rc = _validate_args(args)
    if rc != 0:
        return rc

    fallback_model_cfg = _build_model_cfg(args)
    train_cfg = _build_train_cfg(args)
    rank, world_size, local_rank = init_distributed(backend=train_cfg.dist_backend)
    rank_seed = args.seed + rank
    torch.manual_seed(rank_seed)
    random.seed(rank_seed)
    try:
        import numpy as np

        np.random.seed(rank_seed % (2**32))
    except ImportError:  # pragma: no cover - optional dependency
        pass
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if args.resume:
        _validate_resumable_checkpoint_files(
            args.resume,
            require_scaler=(train_cfg.precision == "fp16" and device.type == "cuda"),
        )

    output_path = Path(train_cfg.output_dir)
    if not args.resume and not args.smoke and output_path.exists():
        occupied = (
            (output_path / "latest").exists()
            or (output_path / "latest").is_symlink()
            or (output_path / "reference").exists()
            or (output_path / "best").exists()
            or any(output_path.glob("step_*"))
            or any(output_path.glob(".step_*.tmp-*"))
            or any(output_path.glob(".step_*.backup"))
        )
        if occupied:
            raise FileExistsError(
                f"output directory already contains a training run: {output_path}; "
                "use --resume or a new --output"
            )
    if is_main_process():
        output_path.mkdir(parents=True, exist_ok=True)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    load_checkpoint_path = args.resume or args.init_checkpoint
    source_checkpoint = ""
    reference_checkpoint = ""
    reference_artifact_dir = ""
    source_metadata: dict = {}
    if load_checkpoint_path:
        restored = reconstruct_policy_from_checkpoint(
            load_checkpoint_path,
            fallback_rdt_config=fallback_model_cfg,
            require_vision=args.task == "ocr",
        )
        policy = restored.model
        model_cfg = restored.rdt_config
        omvt_cfg = restored.omvt_config
        source_metadata = restored.metadata
        if args.recurrent_steps is not None:
            model_cfg.recurrent_steps = args.recurrent_steps
        if args.resume:
            source_checkpoint = str(source_metadata.get("source_checkpoint", ""))
            reference_checkpoint = str(
                source_metadata.get("reference_checkpoint", "")
            )
            if not source_checkpoint:
                raise ValueError("GRPO resume checkpoint has no original source_checkpoint")
            if not reference_checkpoint:
                raise ValueError("GRPO resume checkpoint has no immutable reference_checkpoint")
        else:
            source_checkpoint = str(restored.checkpoint_dir.absolute())
    else:
        model_cfg = fallback_model_cfg
        omvt_cfg = None
        policy = RDTForCausalLM(model_cfg)
        source_checkpoint = "random_init"
    policy.reverse_loss_enabled = False
    trainable_names = _configure_trainable_scope(policy, args.train_scope)
    if args.resume:
        reference, restored_reference_dir = _load_immutable_reference(
            reference_checkpoint,
            source_checkpoint,
            model_cfg,
            omvt_cfg,
            train_cfg,
            local_rank,
            device,
            require_vision=args.task == "ocr",
        )
        reference_artifact_dir = str(restored_reference_dir)
    else:
        reference = _build_reference_model(
            policy, model_cfg, train_cfg, local_rank, device
        )
    policy = policy.to(device)

    if is_main_process():
        n_trainable = sum(param.numel() for param in policy.parameters() if param.requires_grad)
        n_total = sum(param.numel() for param in policy.parameters())
        print(
            f"[train] task={args.task} scope={args.train_scope} "
            f"trainable={n_trainable:,}/{n_total:,} tensors={len(trainable_names)}",
            flush=True,
        )

    optimizer = build_optimizer(policy, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)
    policy = apply_parallelism(policy, train_cfg, local_rank)
    # Generation is necessarily dropout-free.  Keep differentiable scoring in
    # the same mode so detached old_logp is the exact rollout-policy snapshot,
    # even if a future RDT/OMVT checkpoint enables dropout.
    policy.eval()
    scaler = _make_grad_scaler(train_cfg.precision, device, policy)

    tool_open_ids: list[int] = []
    tool_close_ids: list[int] = []
    processor = None
    validation_dataset = None
    golden_dataset = None
    run_data_contract: dict[str, str] = {}
    if args.smoke:
        dataset = _smoke_dataset()
        decode = _smoke_decode
    else:
        from Tokenizer.unified.bundle import TokenizerBundle

        from Model.posttrain.chat_template import (
            TOOL_RESULT_CLOSE,
            TOOL_RESULT_OPEN,
        )

        bundle = TokenizerBundle.from_dir(args.tokenizer)
        bundle_issues = bundle.validate()
        if bundle_issues:
            raise ValueError(
                "invalid tokenizer bundle:\n  - " + "\n  - ".join(bundle_issues)
            )
        if args.task == "ocr":
            from scripts.build_ocr_data import make_ocr_target_encoder

            encode_reference = make_ocr_target_encoder(bundle.tokenizer)
            decode = lambda ids: decode_ocr_completion(  # noqa: E731
                ids,
                bundle.tokenizer.decode,
                eos_id=EOS_ID,
                valid_token_ids=bundle.tokenizer.id_to_token,
                require_eos=True,
            )
            if omvt_cfg is None:
                raise RuntimeError("OCR checkpoint reconstruction produced no OMVT config")
            from Tokenizer.multimodal import PILImageProcessor

            dataset_kwargs = dict(
                encode=lambda text: bundle.encode(
                    text, add_bos=False, add_eos=False
                ),
                n_image_tokens=omvt_cfg.compress_to,
                bos_id=BOS_ID,
                image_start_id=IMAGE_START_ID,
                image_patch_id=IMAGE_PATCH_ID,
                image_end_id=IMAGE_END_ID,
                encode_reference=encode_reference,
                image_root=args.image_root or None,
                max_prompt_len=args.max_prompt_len,
                max_completion_len=args.max_new_tokens,
                max_seq_len=model_cfg.max_seq_len,
                validate_images=True,
                verify_image_decode=True,
                require_sha256=True,
                require_group_id=True,
                require_domain=True,
                verify_sha256=True,
            )
            golden_dataset_kwargs = dict(dataset_kwargs)
            # The locked golden labels are not a source of generation-length
            # or reward hyperparameters.  Training may inspect schema and
            # identity keys only, so even reference token counts stay hidden.
            golden_dataset_kwargs.update(
                max_completion_len=None,
                inspect_reference_tokens=False,
                retain_reference=False,
                inspect_prompt_tokens=False,
            )
            golden_dataset = OCRPromptDataset(
                args.golden_manifest,
                required_split=args.golden_split,
                **golden_dataset_kwargs,
            )
            (
                golden_ids,
                golden_images,
                golden_sha256,
                golden_groups,
            ) = _dataset_exclusions(golden_dataset)
            validation_dataset = OCRPromptDataset(
                args.validation_manifest,
                required_split=args.validation_split,
                excluded_ids=golden_ids,
                excluded_images=golden_images,
                excluded_sha256=golden_sha256,
                excluded_groups=golden_groups,
                **dataset_kwargs,
            )
            val_ids, val_images, val_sha256, val_groups = _dataset_exclusions(
                validation_dataset
            )
            dataset = OCRPromptDataset(
                args.data,
                required_split=args.required_split or None,
                excluded_ids=golden_ids | val_ids,
                excluded_images=golden_images | val_images,
                excluded_sha256=golden_sha256 | val_sha256,
                excluded_groups=golden_groups | val_groups,
                **dataset_kwargs,
            )
            processor = PILImageProcessor(
                image_size=omvt_cfg.image_size,
                in_channels=omvt_cfg.in_channels,
            )
        else:
            decode = lambda ids: bundle.tokenizer.decode(  # noqa: E731
                [int(i) for i in ids.tolist() if int(i) != PAD_ID]
            )
            dataset = PromptDataset(
                args.data,
                encode=lambda t: bundle.encode(t),
                bos_id=BOS_ID,
                max_prompt_len=args.max_prompt_len,
            )
        if args.task == "text" and not args.no_tool_result_mask:
            tool_open_ids = list(bundle.encode(TOOL_RESULT_OPEN, add_bos=False, add_eos=False))
            tool_close_ids = list(bundle.encode(TOOL_RESULT_CLOSE, add_bos=False, add_eos=False))
        run_data_contract = _data_contract(args, bundle)

    if not args.resume and not args.smoke:
        reference_root = (Path(train_cfg.output_dir) / "reference").absolute()
        reference_checkpoint = str(reference_root / "step_00000000")
        reference_artifact_dir = reference_checkpoint
        if args.task != "ocr":
            save_checkpoint(
                reference_root,
                0,
                reference,
                optimizer=None,
                scheduler=None,
                metadata=_reference_metadata(model_cfg, omvt_cfg, source_checkpoint),
                keep_last_n=1,
            )
    if not args.smoke and (args.resume or args.task != "ocr"):
        run_data_contract["reference_model_sha256"] = _distributed_file_sha256(
            Path(reference_artifact_dir) / "model.pt",
            rank,
        )

    reward_cfg = _reward_cfg(args)
    grpo_cfg = _grpo_config(args, model_cfg, tool_open_ids, tool_close_ids)
    if args.resume:
        _validate_resume_contract(
            source_metadata,
            args,
            train_cfg,
            reward_cfg,
            grpo_cfg,
            run_data_contract,
        )
    state = TrainState()
    if train_cfg.resume:
        state.step = resume_state(
            train_cfg.resume, policy, optimizer, scheduler, state=state
        )
        pending_scaler = state.extra.pop("grad_scaler_state", None)
        if pending_scaler is not None and scaler.is_enabled():
            scaler.load_state_dict(pending_scaler)
    restored_health = source_metadata.get("health_state", {}) if args.resume else {}
    degenerate_streak = int(restored_health.get("degenerate_streak", 0))
    high_kl_streak = int(restored_health.get("high_kl_streak", 0))
    best_val_cer = float(restored_health.get("best_val_grapheme_cer", float("inf")))
    best_val_step = int(restored_health.get("best_val_step", -1))
    bad_eval_count = int(restored_health.get("bad_eval_count", 0))
    eval_count = int(restored_health.get("eval_count", 0))
    best_checkpoint = str(restored_health.get("best_checkpoint", ""))
    best_val_eligible = bool(restored_health.get("best_val_eligible", False))
    baseline_visual_cer_gap = float(
        restored_health.get("baseline_visual_cer_gap", float("nan"))
    )
    batches_consumed = int(restored_health.get("batches_consumed", state.step))
    optimizer_skip_count = int(restored_health.get("optimizer_skip_count", 0))
    consecutive_optimizer_skips = int(
        restored_health.get("consecutive_optimizer_skips", 0)
    )
    batches = _iter_prompt_batches(
        dataset,
        args.prompts_per_step,
        rank=rank,
        world_size=world_size,
        shuffle=not args.smoke,
        seed=args.seed,
        start_batch=batches_consumed,
    )

    logger = RankZeroLogger(train_cfg.output_dir, enable_tensorboard=False)
    t0 = time.time()
    completed = False

    def health_payload() -> dict[str, object]:
        return {
            "degenerate_streak": degenerate_streak,
            "high_kl_streak": high_kl_streak,
            "best_val_grapheme_cer": best_val_cer,
            "best_val_step": best_val_step,
            "bad_eval_count": bad_eval_count,
            "eval_count": eval_count,
            "best_checkpoint": best_checkpoint,
            "best_val_eligible": best_val_eligible,
            "baseline_visual_cer_gap": baseline_visual_cer_gap,
            "batches_consumed": batches_consumed,
            "optimizer_skip_count": optimizer_skip_count,
            "consecutive_optimizer_skips": consecutive_optimizer_skips,
        }

    try:
        if args.task == "ocr":
            assert processor is not None and omvt_cfg is not None
            first_row = dataset[0]
            first_prompt = torch.tensor(
                first_row["prompt_ids"], dtype=torch.long, device=device
            )
            first_pixels = _ocr_pixel_values(
                [first_row], processor, omvt_cfg, device
            )[0]
            delta = _visual_conditioning_delta(
                policy,
                first_prompt,
                first_pixels,
                train_cfg.precision,
                device,
            )
            if (
                not torch.isfinite(torch.tensor(delta))
                or delta < args.min_visual_logit_delta
            ):
                raise RuntimeError(
                    "visual conditioning gate failed: real-vs-blank "
                    f"max_logit_delta={delta:.6g} < "
                    f"{args.min_visual_logit_delta:.6g}"
                )
            if is_main_process():
                print(f"[gate] visual_logit_delta={delta:.6g}", flush=True)

        if args.task == "ocr" and not args.resume:
            assert validation_dataset is not None
            assert processor is not None and omvt_cfg is not None
            baseline = _evaluate_ocr_validation(
                policy,
                validation_dataset,
                processor,
                omvt_cfg,
                decode,
                batch_size=args.eval_batch_size,
                max_new_tokens=args.max_new_tokens,
                recurrent_steps=grpo_cfg.recurrent_steps,
                precision=train_cfg.precision,
                device=device,
                cer_backend=reward_cfg.grapheme_cer_backend,
            )
            blank_baseline = _evaluate_ocr_validation(
                policy,
                validation_dataset,
                processor,
                omvt_cfg,
                decode,
                batch_size=args.eval_batch_size,
                max_new_tokens=args.max_new_tokens,
                recurrent_steps=grpo_cfg.recurrent_steps,
                precision=train_cfg.precision,
                device=device,
                cer_backend=reward_cfg.grapheme_cer_backend,
                blank_visual=True,
            )
            baseline_visual_cer_gap = (
                blank_baseline["grapheme_cer"] - baseline["grapheme_cer"]
            )
            if baseline_visual_cer_gap <= args.min_baseline_visual_cer_gap:
                raise RuntimeError(
                    "baseline visual CER gate failed: blank-real gap "
                    f"{baseline_visual_cer_gap:.6f} <= "
                    f"{args.min_baseline_visual_cer_gap:.6f}; refusing RL on a "
                    "policy that does not outperform its blank-image ablation"
                )
            reference_root = Path(reference_checkpoint).parent
            save_checkpoint(
                reference_root,
                0,
                reference,
                optimizer=None,
                scheduler=None,
                metadata=_reference_metadata(model_cfg, omvt_cfg, source_checkpoint),
                keep_last_n=1,
            )
            run_data_contract["reference_model_sha256"] = _distributed_file_sha256(
                Path(reference_artifact_dir) / "model.pt",
                rank,
            )
            best_val_cer = baseline["grapheme_cer"]
            best_val_step = state.step
            best_val_eligible = bool(
                baseline["eos_rate"] >= args.min_validation_eos_rate
                and baseline["invalid_output_rate"]
                <= args.max_validation_invalid_rate
            )
            bad_eval_count = 0
            eval_count = 1
            best_root = (Path(train_cfg.output_dir) / "best").absolute()
            best_checkpoint = str(best_root / f"step_{state.step:08d}")
            save_checkpoint(
                best_root,
                state.step,
                policy,
                optimizer=None,
                scheduler=None,
                metadata=_checkpoint_metadata(
                    args,
                    train_cfg,
                    model_cfg,
                    omvt_cfg,
                    reward_cfg,
                    grpo_cfg,
                    source_checkpoint,
                    reference_checkpoint,
                    data_contract=run_data_contract,
                    health_state=health_payload(),
                ),
                keep_last_n=1,
            )
            # Establish a fully resumable step-0 anchor immediately. Waiting
            # for the first periodic/evaluation save would leave the beginning
            # of a long run without optimizer/RNG recovery after an outage.
            save_checkpoint(
                train_cfg.output_dir,
                state.step,
                policy,
                optimizer,
                scheduler,
                metadata=_checkpoint_metadata(
                    args,
                    train_cfg,
                    model_cfg,
                    omvt_cfg,
                    reward_cfg,
                    grpo_cfg,
                    source_checkpoint,
                    reference_checkpoint,
                    data_contract=run_data_contract,
                    health_state=health_payload(),
                ),
                keep_last_n=train_cfg.keep_last_n,
                scaler=scaler if scaler.is_enabled() else None,
            )
            logger.log(
                state.step,
                {
                    **{f"val_{name}": value for name, value in baseline.items()},
                    **{
                        f"val_blank_{name}": value
                        for name, value in blank_baseline.items()
                    },
                    "val_visual_grapheme_cer_gap": baseline_visual_cer_gap,
                },
            )
            if is_main_process():
                print(
                    f"[val] baseline grapheme_cer={best_val_cer:.6f} "
                    f"blank_gap={baseline_visual_cer_gap:.6f} "
                    f"eligible={int(best_val_eligible)} "
                    f"n={int(baseline['samples'])}",
                    flush=True,
                )
        elif args.task == "ocr":
            if not baseline_visual_cer_gap > args.min_baseline_visual_cer_gap:
                raise RuntimeError(
                    "resume checkpoint did not pass the original visual CER gate"
                )
            if not best_checkpoint or not Path(best_checkpoint).is_dir():
                raise FileNotFoundError(
                    "resume metadata points to a missing best checkpoint: "
                    f"{best_checkpoint!r}"
                )

        resume_already_plateaued = bool(
            args.resume
            and args.task == "ocr"
            and bad_eval_count >= args.early_stop_patience
        )
        if resume_already_plateaued and is_main_process():
            print(
                "[early-stop] resume checkpoint already reached validation "
                f"patience; keeping selected {best_checkpoint}",
                flush=True,
            )

        last_logged_step = state.step
        while not resume_already_plateaued and state.step < train_cfg.max_steps:
            rows = next(batches)
            batches_consumed += 1
            prompts = [torch.tensor(r["prompt_ids"], dtype=torch.long, device=device)
                       for r in rows]
            references = [r.get("reference") for r in rows]
            if args.task == "ocr":
                assert processor is not None and omvt_cfg is not None
                pixels = _ocr_pixel_values(rows, processor, omvt_cfg, device)
            else:
                pixels = None

            reward_parts: dict[str, float] = {}

            def reward_fn(responses, idx, _refs=references):
                refs = [_refs[idx]] * len(responses)
                rewards, parts = compute_rewards_with_breakdown(
                    responses, refs, reward_cfg
                )
                for name, value in parts.items():
                    reward_parts[name] = reward_parts.get(name, 0.0) + value / len(rows)
                return rewards

            optimizer.zero_grad(set_to_none=True)
            with _precision_context(train_cfg.precision, device):
                loss, metrics = grpo_compute_loss(
                    policy, reference, prompts, reward_fn, decode,
                    cfg=grpo_cfg, eos_id=EOS_ID, pad_id=PAD_ID,
                    pixel_values=pixels,
                )
            loss_is_finite = bool(torch.isfinite(loss.detach()))
            if not _all_ranks_true(loss_is_finite, device):
                raise FloatingPointError(f"non-finite GRPO loss at step {state.step}")
            metrics.update({f"reward_{name}": value for name, value in reward_parts.items()})
            rank_max_kl = _distributed_max(metrics.get("kl", 0.0), device)
            metrics = _distributed_mean_metrics(metrics, device)
            metrics["kl_rank_max"] = rank_max_kl

            degenerate_streak = (
                degenerate_streak + 1
                if metrics.get("degenerate_group", 0.0) >= 1.0 - 1e-9
                else 0
            )
            if degenerate_streak >= args.max_degenerate_steps:
                raise RuntimeError(
                    "all prompt groups produced zero reward variance for "
                    f"{degenerate_streak} consecutive steps; refusing zero-advantage RL"
                )
            high_kl_streak = (
                high_kl_streak + 1
                if rank_max_kl > args.kl_abort_threshold
                else 0
            )
            if high_kl_streak >= args.kl_abort_patience:
                raise RuntimeError(
                    f"KL exceeded {args.kl_abort_threshold} for "
                    f"{high_kl_streak} consecutive steps"
                )

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            metrics["grad_norm"] = clip_or_check_grad_norm(
                policy,
                train_cfg.grad_clip,
                step=state.step,
                allow_nonfinite=scaler.is_enabled(),
            )
            if scaler.is_enabled():
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                stepped = scaler.get_scale() >= scale_before
            else:
                optimizer.step()
                stepped = True
            stepped = _consensus_bool(stepped, device, "GradScaler optimizer step")
            if stepped:
                consecutive_optimizer_skips = 0
                scheduler.step()
                state.step += 1
            else:
                optimizer_skip_count += 1
                consecutive_optimizer_skips += 1
            metrics["lr"] = float(scheduler.get_last_lr()[0])
            metrics["degenerate_streak"] = float(degenerate_streak)
            metrics["high_kl_streak"] = float(high_kl_streak)
            metrics["batches_consumed"] = float(batches_consumed)
            metrics["optimizer_skip_count"] = float(optimizer_skip_count)
            metrics["consecutive_optimizer_skips"] = float(
                consecutive_optimizer_skips
            )
            metrics["optimizer_step_skipped"] = float(not stepped)
            if not all(math.isfinite(float(value)) for value in metrics.values()):
                # fp16 overflow is represented by the scaler skip and a
                # non-finite grad norm; no other health metric may be non-finite.
                allowed = scaler.is_enabled() and not stepped
                finite_without_grad = all(
                    math.isfinite(float(value))
                    for name, value in metrics.items()
                    if name != "grad_norm"
                )
                if not (allowed and finite_without_grad):
                    raise FloatingPointError(
                        f"non-finite GRPO health metric at step {state.step}"
                    )
            if not stepped:
                if is_main_process():
                    print(
                        "[amp-skip] fp16 overflow skipped optimizer update "
                        f"at batch={batches_consumed} "
                        f"streak={consecutive_optimizer_skips}/"
                        f"{args.max_consecutive_optimizer_skips}",
                        flush=True,
                    )
                if (
                    consecutive_optimizer_skips
                    >= args.max_consecutive_optimizer_skips
                ):
                    raise RuntimeError(
                        "fp16 GradScaler skipped too many consecutive optimizer "
                        f"updates ({consecutive_optimizer_skips})"
                    )
                # A scaler skip changes no policy parameter and therefore must
                # not consume max_steps or validation patience.  Its batch/RNG
                # cursor is still persisted with the next successful checkpoint.
                continue

            checkpoint_written = False
            stop_for_plateau = False
            if args.task == "ocr" and state.step % args.eval_every == 0:
                assert validation_dataset is not None
                assert processor is not None and omvt_cfg is not None
                eval_started = time.time()
                validation = _evaluate_ocr_validation(
                    policy,
                    validation_dataset,
                    processor,
                    omvt_cfg,
                    decode,
                    batch_size=args.eval_batch_size,
                    max_new_tokens=args.max_new_tokens,
                    recurrent_steps=grpo_cfg.recurrent_steps,
                    precision=train_cfg.precision,
                    device=device,
                    cer_backend=reward_cfg.grapheme_cer_backend,
                )
                eval_count += 1
                current_cer = validation["grapheme_cer"]
                validation_eligible = bool(
                    validation["eos_rate"] >= args.min_validation_eos_rate
                    and validation["invalid_output_rate"]
                    <= args.max_validation_invalid_rate
                )
                improved = (
                    validation_eligible
                    and current_cer < best_val_cer - args.early_stop_min_delta
                )
                if improved:
                    best_val_cer = current_cer
                    best_val_step = state.step
                    best_val_eligible = True
                    bad_eval_count = 0
                    best_root = (Path(train_cfg.output_dir) / "best").absolute()
                    best_checkpoint = str(best_root / f"step_{state.step:08d}")
                    save_checkpoint(
                        best_root,
                        state.step,
                        policy,
                        optimizer=None,
                        scheduler=None,
                        metadata=_checkpoint_metadata(
                            args,
                            train_cfg,
                            model_cfg,
                            omvt_cfg,
                            reward_cfg,
                            grpo_cfg,
                            source_checkpoint,
                            reference_checkpoint,
                            data_contract=run_data_contract,
                            health_state=health_payload(),
                        ),
                        keep_last_n=1,
                    )
                else:
                    bad_eval_count += 1
                metrics.update(
                    {f"val_{name}": value for name, value in validation.items()}
                )
                metrics["val_best_grapheme_cer"] = best_val_cer
                metrics["val_best_step"] = float(best_val_step)
                metrics["val_bad_eval_count"] = float(bad_eval_count)
                metrics["val_eligible"] = float(validation_eligible)
                metrics["val_eval_time_s"] = time.time() - eval_started
                stop_for_plateau = bad_eval_count >= args.early_stop_patience

                # Every validation decision is resumable.  Otherwise a power
                # loss between sparse periodic saves can erase plateau state
                # and silently extend training beyond the configured patience.
                if (
                    not args.smoke
                    and not stop_for_plateau
                    and state.step < train_cfg.max_steps
                ):
                    save_checkpoint(
                        train_cfg.output_dir,
                        state.step,
                        policy,
                        optimizer,
                        scheduler,
                        metadata=_checkpoint_metadata(
                            args,
                            train_cfg,
                            model_cfg,
                            omvt_cfg,
                            reward_cfg,
                            grpo_cfg,
                            source_checkpoint,
                            reference_checkpoint,
                            data_contract=run_data_contract,
                            health_state=health_payload(),
                        ),
                        keep_last_n=train_cfg.keep_last_n,
                        scaler=scaler if scaler.is_enabled() else None,
                    )
                    checkpoint_written = True
                if is_main_process():
                    print(
                        f"[val] step={state.step} grapheme_cer={current_cer:.6f} "
                        f"eligible={int(validation_eligible)} "
                        f"best={best_val_cer:.6f}@{best_val_step} "
                        f"bad={bad_eval_count}/{args.early_stop_patience}",
                        flush=True,
                    )

            if state.step % train_cfg.log_every == 0 or args.smoke:
                dt = max(1e-6, time.time() - t0)
                logged_steps = max(1, state.step - last_logged_step)
                logger.log(
                    state.step,
                    {
                        **metrics,
                        "step_time_s": round(dt / logged_steps, 3),
                        "log_interval_s": round(dt, 3),
                    },
                )
                t0 = time.time()
                last_logged_step = state.step
            if (
                train_cfg.save_every
                and state.step % train_cfg.save_every == 0
                and not args.smoke
                and not checkpoint_written
                and state.step < train_cfg.max_steps
            ):
                save_checkpoint(
                    train_cfg.output_dir, state.step, policy, optimizer, scheduler,
                    metadata=_checkpoint_metadata(
                        args,
                        train_cfg,
                        model_cfg,
                        omvt_cfg,
                        reward_cfg,
                        grpo_cfg,
                        source_checkpoint,
                        reference_checkpoint,
                        data_contract=run_data_contract,
                        health_state=health_payload(),
                    ),
                    keep_last_n=train_cfg.keep_last_n,
                    scaler=scaler if scaler.is_enabled() else None,
                )
            if args.smoke and state.step >= 4:
                break
            if stop_for_plateau:
                if is_main_process():
                    print(
                        "[early-stop] validation grapheme CER did not improve "
                        f"for {bad_eval_count} checks; selected "
                        f"{best_checkpoint}",
                        flush=True,
                    )
                break
        completed = True
    finally:
        try:
            logger.close()
            if completed and not args.smoke:
                save_checkpoint(
                    train_cfg.output_dir, state.step, policy, optimizer, scheduler,
                    metadata=_checkpoint_metadata(
                        args,
                        train_cfg,
                        model_cfg,
                        omvt_cfg,
                        reward_cfg,
                        grpo_cfg,
                        source_checkpoint,
                        reference_checkpoint,
                        data_contract=run_data_contract,
                        health_state=health_payload(),
                        final=True,
                    ),
                    keep_last_n=train_cfg.keep_last_n,
                    scaler=scaler if scaler.is_enabled() else None,
                )
        finally:
            destroy_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
