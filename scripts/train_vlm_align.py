# -*- coding: utf-8 -*-

"""VLM alignment: OMVT vision tower → projector → RDT.

Runs a synthetic end-to-end forward/backward where the RDT LM consumes
``<image_patch>`` slots filled by the OMVT compressed tokens.  The LM head
is fine-tuned by default; pass ``--freeze-rdt`` to train only the
projector/tower.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import asdict, fields, replace
from pathlib import Path

import torch

from Model.config import (
    BOS_ID,
    EOS_ID,
    IMAGE_PATCH_ID,
    OMVTConfig,
    PAD_ID,
    RDTConfig,
    TrainingConfig,
)
from Model.model import RDTForCausalLM
from Model.omvt import OMVTInjector
from Model.omvt.patcher import collate_omvt_batch
from Model.training import (
    EarlyStoppingConfig,
    LossPlateauStopper,
    RankZeroLogger,
    TrainState,
    build_dataloader,
    build_optimizer,
    build_scheduler,
    clip_or_check_grad_norm,
    load_checkpoint_metadata,
    resolve_checkpoint_dir,
    resume_state,
    save_checkpoint,
    train_one_step,
)
from Model.training.multimodal_cli import make_omvt_cfg
from Model.training.omvt_checkpoint import (
    load_omvt_payload,
    tower_state_from_payload,
)
from Tokenizer.multimodal import PILImageProcessor
from scripts.train_rdt import CONFIG_CHOICES, _resolve_mamba_backend


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=list(CONFIG_CHOICES), default="tiny")
    p.add_argument(
        "--mamba",
        choices=["auto", "official", "naive"],
        default="auto",
        help=(
            "Mamba backend for the RDT side. auto uses official CUDA Mamba on "
            "CUDA/Linux and NaiveSSM on macOS/CPU."
        ),
    )
    p.add_argument("--steps", type=int, default=4)
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="smoothed-loss observations without improvement (0 disables)",
    )
    p.add_argument(
        "--early-stop-min-steps",
        type=int,
        default=0,
        help="minimum optimizer step before plateau patience is counted",
    )
    p.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="minimum absolute smoothed-loss decrease considered an improvement",
    )
    p.add_argument(
        "--early-stop-smoothing",
        choices=("ema", "window"),
        default="ema",
        help="smoother used for training-loss plateau detection",
    )
    p.add_argument(
        "--early-stop-ema-alpha",
        type=float,
        default=0.01,
        help="new-observation weight for EMA smoothing",
    )
    p.add_argument(
        "--early-stop-window-size",
        type=int,
        default=100,
        help="rolling-mean width when --early-stop-smoothing=window",
    )
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--image-size", type=int, default=56)
    p.add_argument("--seq-len", type=int, default=24)
    p.add_argument("--n-image-tokens", type=int, default=None)
    p.add_argument(
        "--recurrent-steps",
        type=int,
        default=None,
        help="fixed RDT refinement depth; defaults to the init/resume "
        "checkpoint value (or the selected config when no checkpoint exists)",
    )
    p.add_argument("--freeze-rdt", action="store_true")
    p.add_argument(
        "--frozen-vision",
        action="store_true",
        help="freeze the OMVT tower as well (useful for projector-only ablations)",
    )
    p.add_argument(
        "--data",
        default="",
        help="JSONL spec for real multimodal pretraining (rows must carry an 'images' field)",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--output", default="outputs/vlm_align")
    p.add_argument("--init-rdt-checkpoint", default="")
    p.add_argument("--init-omvt-checkpoint", default="")
    p.add_argument(
        "--use-ema-tower",
        action="store_true",
        help="when --init-omvt-checkpoint carries 'tower_ema', overlay the EMA "
        "weights on the tower state before loading",
    )
    p.add_argument("--resume", default="")
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument(
        "--keep-last-n",
        type=int,
        default=0,
        help="delete older step checkpoints as new ones are saved (0 = keep all)",
    )
    p.add_argument(
        "--no-resume-skip-data",
        action="store_true",
        help="do not fast-forward the deterministic data stream on resume",
    )
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--precision",
        choices=("auto", "fp32", "bf16"),
        default="fp32",
        help="'auto' = bf16 on cuda, fp32 elsewhere; default keeps the legacy "
        "fp32 smoke behavior",
    )
    p.add_argument("--warmup-steps", type=int, default=1)
    p.add_argument(
        "--d-vision",
        type=int,
        default=64,
        help="OMVT tower width for from-scratch towers; ignored when "
        "--init-omvt-checkpoint provides its own omvt_config",
    )
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="'auto' = cuda if available else cpu (legacy behavior)",
    )
    p.add_argument(
        "--patch-preset",
        choices=("derived", "prod"),
        default="derived",
        help="for from-scratch towers: 'derived' keeps the legacy smoke "
        "geometry (patches scaled from --image-size); 'prod' uses the "
        "OMVTConfig dataclass multi-scale defaults (32x8 / 8x32 / 16x16 / "
        "56x56). Ignored when --init-omvt-checkpoint provides omvt_config.",
    )
    p.add_argument(
        "--grad-ckpt",
        action="store_true",
        help="enable RDT gradient checkpointing (grad_ckpt_recurrent + "
        "grad_ckpt_prelude_coda) to fit long sequences on small GPUs",
    )
    return p.parse_args(argv)


def _early_stopping_config(args) -> EarlyStoppingConfig:
    return EarlyStoppingConfig(
        patience=args.early_stop_patience,
        min_steps=args.early_stop_min_steps,
        min_delta=args.early_stop_min_delta,
        smoothing=args.early_stop_smoothing,
        ema_alpha=args.early_stop_ema_alpha,
        window_size=args.early_stop_window_size,
    )


def _build_omvt_cfg(args, checkpoint_metadata: dict | None = None) -> OMVTConfig:
    if args.init_omvt_checkpoint:
        # The checkpoint's own omvt_config is the only authoritative source of
        # tower geometry: building a CLI-derived config here and loading a
        # prod-geometry tower (e.g. d_vision=512, dataclass patch shapes) into
        # it would fail on shape mismatch.
        payload = load_omvt_payload(args.init_omvt_checkpoint, weights_only=False)
        if isinstance(payload, dict) and "omvt_config" in payload:
            cfg = OMVTConfig(**payload["omvt_config"])
            if args.image_size != cfg.image_size:
                print(
                    f"[init] --image-size {args.image_size} -> {cfg.image_size} "
                    "(from OMVT checkpoint)",
                )
            return cfg
    checkpoint_metadata = checkpoint_metadata or {}
    if isinstance(checkpoint_metadata.get("omvt_config"), dict):
        cfg = OMVTConfig(**checkpoint_metadata["omvt_config"])
        if args.image_size != cfg.image_size:
            print(
                f"[init] --image-size {args.image_size} -> {cfg.image_size} "
                "(from RDT/VLM checkpoint)",
            )
        return cfg
    return make_omvt_cfg(
        args.image_size,
        args.d_vision,
        args.n_image_tokens,
        preset=getattr(args, "patch_preset", "derived"),
    )


def _make_text_batch(args, vocab_floor=300, vocab_ceil=320):
    B, L, N = args.batch_size, args.seq_len, args.n_image_tokens
    rng = torch.Generator().manual_seed(args.seed)
    # layout: [BOS] <image_patch>*N <random text...> [EOS]
    text_len = L - N - 2
    if text_len <= 0:
        raise ValueError("seq_len must be greater than 2 + n_image_tokens")
    text_ids = torch.randint(vocab_floor, vocab_ceil, (B, text_len), generator=rng)
    input_ids = torch.full((B, L), 0, dtype=torch.long)
    input_ids[:, 0] = BOS_ID
    input_ids[:, 1 : 1 + N] = IMAGE_PATCH_ID
    input_ids[:, 1 + N : 1 + N + text_len] = text_ids
    input_ids[:, -1] = EOS_ID
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    return input_ids, attention_mask, labels


def _checkpoint_metadata(path: str) -> dict:
    if not path:
        return {}
    p = Path(path)
    # Legacy callers may pass an arbitrary standalone state-dict file.  Such a
    # file has no neighbouring checkpoint contract, so fall back to CLI config.
    if p.is_file() and p.name != "model.pt":
        return {}
    return load_checkpoint_metadata(path)


def _build_rdt_cfg(args, metadata: dict, device: torch.device) -> RDTConfig:
    raw = metadata.get("rdt_config")
    if isinstance(raw, dict):
        rdt_cfg = RDTConfig(**raw)
        print("[init] using RDTConfig from checkpoint metadata")
    else:
        rdt_cfg = CONFIG_CHOICES[args.config]()
    rdt_cfg = replace(rdt_cfg, max_seq_len=args.seq_len)
    if args.recurrent_steps is not None:
        rdt_cfg = replace(rdt_cfg, recurrent_steps=args.recurrent_steps)
    if args.grad_ckpt:
        rdt_cfg = replace(
            rdt_cfg, grad_ckpt_recurrent=True, grad_ckpt_prelude_coda=True
        )
    return _resolve_mamba_backend(
        rdt_cfg,
        args.mamba,
        device=device,
        context="scripts.train_vlm_align",
    )


def _resolve_model_state(path: str):
    p = Path(path)
    if p.is_file():
        state = torch.load(p, map_location="cpu", weights_only=False)
    else:
        ckpt_dir = resolve_checkpoint_dir(p)
        # Initialization only needs model weights.  Avoid loading the optimizer
        # state, which is substantially larger than the frozen language model.
        state = torch.load(
            ckpt_dir / "model.pt", map_location="cpu", weights_only=False
        )
    if isinstance(state, dict) and "model" in state and "embed.weight" not in state:
        state = state["model"]
    return state


def _load_rdt_init(model: RDTForCausalLM, path: str) -> None:
    if not path:
        return
    state = _resolve_model_state(path)
    checkpoint_has_omvt = any(k.startswith("vision.omvt.") for k in state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = (
        [k for k in missing if k.startswith("vision.omvt.")]
        if not checkpoint_has_omvt
        else []
    )
    bad_missing = [k for k in missing if k not in allowed_missing]
    if bad_missing or unexpected:
        detail = (
            f"missing={bad_missing[:8]} unexpected={list(unexpected)[:8]}"
        )
        raise RuntimeError(
            f"RDT checkpoint {path} does not exactly match the language model; "
            f"refusing to freeze partially loaded weights ({detail})"
        )
    if allowed_missing:
        print(
            f"[init] loaded text-only RDT checkpoint {path}; "
            f"initialized {len(allowed_missing)} vision.omvt tensors separately"
        )
    else:
        print(f"[init] loaded full RDT/VLM checkpoint {path} (exact key match)")


def _configure_trainable_modules(model: RDTForCausalLM, args) -> list[str]:
    if args.freeze_rdt:
        for param in model.parameters():
            param.requires_grad_(False)
        for param in model.vision.omvt.parameters():
            param.requires_grad_(True)
    if args.frozen_vision:
        for param in model.vision.omvt.tower.parameters():
            param.requires_grad_(False)

    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    if args.freeze_rdt:
        leaked = [name for name in trainable_names if not name.startswith("vision.omvt.")]
        if leaked:
            raise RuntimeError(
                "--freeze-rdt left non-OMVT parameters trainable: "
                + ", ".join(leaked[:8])
            )
    if not trainable_names:
        raise ValueError("model has no trainable parameters")
    return trainable_names


def _alignment_metadata(
    args,
    rdt_cfg: RDTConfig,
    omvt_cfg: OMVTConfig,
    train_cfg: TrainingConfig,
    *,
    inherited: dict | None = None,
    early_stopper: LossPlateauStopper | None = None,
    stop_reason: str = "",
    final: bool = False,
) -> dict:
    inherited = inherited or {}
    return {
        "phase": "vlm_align",
        "config": inherited.get("config", args.config) if args.resume else args.config,
        "rdt_config": asdict(rdt_cfg),
        "omvt_config": asdict(omvt_cfg),
        "training_config": asdict(train_cfg),
        "freeze_rdt": bool(args.freeze_rdt),
        "frozen_vision": bool(args.frozen_vision),
        "recurrent_steps": int(rdt_cfg.recurrent_steps),
        "mamba_backend": "official" if rdt_cfg.use_official_mamba else "naive",
        "source_rdt_checkpoint": (
            args.init_rdt_checkpoint
            or inherited.get("source_rdt_checkpoint", "")
        ),
        "source_omvt_checkpoint": (
            args.init_omvt_checkpoint
            or inherited.get("source_omvt_checkpoint", "")
        ),
        "use_ema_tower": bool(
            args.use_ema_tower or inherited.get("use_ema_tower", False)
        ),
        "early_stopping": (
            early_stopper.metadata_dict()
            if early_stopper is not None
            else inherited.get("early_stopping", {})
        ),
        "stop_reason": str(stop_reason),
        "final": bool(final),
    }


_RESUME_MUTABLE_TRAINING_FIELDS = {
    "output_dir",
    "save_every",
    "keep_last_n",
    "resume",
    "resume_skip_data",
    "log_every",
    "eval_every",
    "eval_max_batches",
    "tensorboard",
    "wandb_project",
}


def _resume_training_conflicts(
    train_cfg: TrainingConfig,
    checkpoint_metadata: dict,
) -> list[str]:
    """Return continuation-critical config differences for a resume.

    Optimizer and scheduler state are only meaningful under the configuration
    that produced them.  Operational settings such as checkpoint retention and
    logging may change; data geometry, batch shape, optimizer, LR schedule,
    precision, recurrence curriculum, distributed mode, and seed may not.
    """

    saved = checkpoint_metadata.get("training_config")
    if not isinstance(saved, dict):
        return [
            "checkpoint has no training_config metadata; use it as "
            "--init-rdt-checkpoint for a new optimizer run instead of --resume"
        ]
    current = asdict(train_cfg)
    conflicts: list[str] = []
    for field in fields(TrainingConfig):
        name = field.name
        if name in _RESUME_MUTABLE_TRAINING_FIELDS:
            continue
        if name not in saved:
            conflicts.append(f"{name}: missing from checkpoint metadata")
        elif saved[name] != current[name]:
            conflicts.append(
                f"{name}: checkpoint={saved[name]!r} current={current[name]!r}"
            )
    return conflicts


def _fast_forward_stream(batch_iter, resumed_step: int, train_cfg: TrainingConfig) -> None:
    skip = resumed_step * train_cfg.grad_accum_steps
    if skip <= 0:
        return
    t0 = time.time()
    for done in range(skip):
        next(batch_iter)
        if (done + 1) % 5000 == 0:
            print(
                f"scripts.train_vlm_align: resume fast-forward "
                f"{done + 1}/{skip} batches ({time.time() - t0:.0f}s elapsed)",
                flush=True,
            )
    print(
        f"scripts.train_vlm_align: resume fast-forwarded {skip} batches in "
        f"{time.time() - t0:.0f}s",
        flush=True,
    )


def _resolve_omvt_state(path: str, use_ema: bool = False):
    payload = load_omvt_payload(path, weights_only=False)
    state = tower_state_from_payload(payload, use_ema=use_ema)
    if use_ema and isinstance(payload, dict) and payload.get("tower_ema"):
        print("[init] using EMA tower weights")
    return state


def _load_omvt_init(model: RDTForCausalLM, path: str, use_ema: bool = False) -> None:
    if not path:
        return
    if model.vision.omvt is None:
        raise ValueError("OMVT injector must be installed before loading tower weights")
    model.vision.omvt.tower.load_state_dict(_resolve_omvt_state(path, use_ema=use_ema))


def main(argv=None):
    args = parse_args(argv)
    # Fast-fail validation **before** any device alloc / model construction.
    # Mirrors the train_rdt CLI pattern: misconfigured runs should not pay the
    # cost of building the model only to crash inside the first step.
    if args.steps <= 0:
        print("scripts.train_vlm_align: --steps must be positive", file=sys.stderr)
        return 2
    try:
        early_stop_config = _early_stopping_config(args)
    except ValueError as exc:
        print(f"scripts.train_vlm_align: {exc}", file=sys.stderr)
        return 2
    if early_stop_config.min_steps > args.steps:
        print(
            "scripts.train_vlm_align: --early-stop-min-steps cannot exceed --steps",
            file=sys.stderr,
        )
        return 2
    if args.image_size <= 0 or args.image_size % 4 != 0:
        print(
            "scripts/train_vlm_align: --image-size must be a positive multiple of 4",
            file=sys.stderr,
        )
        return 2
    if args.resume and (args.init_rdt_checkpoint or args.init_omvt_checkpoint):
        print(
            "scripts/train_vlm_align: --resume cannot be combined with "
            "--init-rdt-checkpoint or --init-omvt-checkpoint",
            file=sys.stderr,
        )
        return 2
    if args.recurrent_steps is not None and args.recurrent_steps <= 0:
        print(
            "scripts/train_vlm_align: --recurrent-steps must be positive",
            file=sys.stderr,
        )
        return 2
    if args.keep_last_n < 0:
        print(
            "scripts/train_vlm_align: --keep-last-n must be non-negative",
            file=sys.stderr,
        )
        return 2

    source_path = args.resume or args.init_rdt_checkpoint
    try:
        source_metadata = _checkpoint_metadata(source_path)
        omvt_cfg = _build_omvt_cfg(args, source_metadata)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        print(f"scripts/train_vlm_align: {exc}", file=sys.stderr)
        return 2
    if args.n_image_tokens is None:
        args.n_image_tokens = omvt_cfg.compress_to
    elif args.n_image_tokens != omvt_cfg.compress_to:
        print(
            "scripts/train_vlm_align: --n-image-tokens does not match checkpoint "
            f"OMVT config ({args.n_image_tokens} != {omvt_cfg.compress_to})",
            file=sys.stderr,
        )
        return 2
    args.image_size = omvt_cfg.image_size
    if args.seq_len <= args.n_image_tokens + 2:
        print(
            "scripts/train_vlm_align: --seq-len must be > --n-image-tokens + 2 "
            f"(got seq_len={args.seq_len}, n_image_tokens={args.n_image_tokens})",
            file=sys.stderr,
        )
        return 2
    if args.resume and source_metadata:
        for key in ("freeze_rdt", "frozen_vision"):
            if (
                key in source_metadata
                and bool(source_metadata[key]) != bool(getattr(args, key))
            ):
                print(
                    f"scripts/train_vlm_align: resume {key}={getattr(args, key)} "
                    f"conflicts with checkpoint value {source_metadata[key]}",
                    file=sys.stderr,
                )
                return 2
        saved_rdt = source_metadata.get("rdt_config")
        if (
            args.recurrent_steps is not None
            and isinstance(saved_rdt, dict)
            and int(saved_rdt.get("recurrent_steps", args.recurrent_steps))
            != args.recurrent_steps
        ):
            print(
                "scripts/train_vlm_align: --recurrent-steps conflicts with the "
                "resume checkpoint",
                file=sys.stderr,
            )
            return 2

    torch.manual_seed(args.seed)
    if getattr(args, "device", "auto") == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    try:
        rdt_cfg = _build_rdt_cfg(args, source_metadata, device)
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.precision == "auto":
        precision = "bf16" if device.type == "cuda" else "fp32"
    else:
        precision = args.precision
    train_cfg = TrainingConfig(
        train_data=args.data,
        seq_len=args.seq_len,
        micro_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.05,
        max_steps=args.steps,
        warmup_steps=max(1, args.warmup_steps),
        precision=precision,
        output_dir=args.output,
        save_every=args.save_every,
        resume=args.resume,
        keep_last_n=args.keep_last_n,
        resume_skip_data=not args.no_resume_skip_data,
    )
    if args.resume:
        conflicts = _resume_training_conflicts(train_cfg, source_metadata)
        if conflicts:
            print(
                "scripts/train_vlm_align: resume training config conflicts with "
                "the checkpoint; refusing mixed optimizer/scheduler semantics:",
                file=sys.stderr,
            )
            for conflict in conflicts[:12]:
                print(f"  - {conflict}", file=sys.stderr)
            if len(conflicts) > 12:
                print(f"  - ... and {len(conflicts) - 12} more", file=sys.stderr)
            return 2

    try:
        early_stopper = LossPlateauStopper.from_metadata(
            early_stop_config,
            source_metadata.get("early_stopping") if args.resume else None,
            require_state=bool(args.resume),
        )
    except (TypeError, ValueError) as exc:
        print(f"scripts.train_vlm_align: {exc}", file=sys.stderr)
        return 2
    if args.resume and source_metadata.get("stop_reason") == "loss_plateau":
        print(
            "scripts.train_vlm_align: checkpoint already stopped on a loss "
            "plateau; use it as an initialization checkpoint for a new run",
            file=sys.stderr,
        )
        return 2

    model = RDTForCausalLM(rdt_cfg).to(device)
    # plug in matching-size OMVT injector (otherwise dispatcher would build
    # a default-sized one on first forward and fail on tiny synthetic inputs).
    model.vision._omvt_cfg = omvt_cfg
    model.vision.omvt = OMVTInjector(rdt_cfg, omvt_cfg).to(device)
    _load_rdt_init(model, args.init_rdt_checkpoint)
    _load_omvt_init(model, args.init_omvt_checkpoint, use_ema=args.use_ema_tower)

    trainable_names = _configure_trainable_modules(model, args)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"[train] recurrent_steps={rdt_cfg.recurrent_steps} "
        f"trainable={n_trainable:,}/{n_total:,} tensors={len(trainable_names)}"
    )

    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)
    state = TrainState()
    if args.resume:
        state.step = resume_state(
            args.resume,
            model,
            optimizer,
            scheduler,
            state=state,
        )
        if early_stopper.enabled and early_stopper.last_step != state.step:
            print(
                "scripts.train_vlm_align: checkpoint early-stop state does not "
                f"match checkpoint step ({early_stopper.last_step} != {state.step})",
                file=sys.stderr,
            )
            return 2

    Path(args.output).mkdir(parents=True, exist_ok=True)
    logger = RankZeroLogger(args.output, enable_tensorboard=False)

    t0 = time.time()
    completed = False
    stop_reason = ""

    def _save(*, final: bool = False) -> None:
        save_checkpoint(
            args.output,
            state.step,
            model,
            optimizer,
            scheduler,
            metadata=_alignment_metadata(
                args,
                rdt_cfg,
                omvt_cfg,
                train_cfg,
                inherited=source_metadata,
                early_stopper=early_stopper,
                stop_reason=stop_reason,
                final=final,
            ),
            keep_last_n=args.keep_last_n,
        )

    def _record_completed_step(metrics: dict[str, float]) -> bool:
        nonlocal stop_reason
        loss_value = float(metrics["loss"])
        should_stop = early_stopper.observe(loss_value, state.step)

        logged = {"loss": loss_value}
        for key in ("grad_norm", "lr", "tokens", "rec_steps"):
            value = metrics.get(key)
            if value is not None and math.isfinite(float(value)):
                logged[key] = float(value)
        if early_stopper.enabled:
            if early_stopper.smoothed_loss is not None:
                logged["early_stop_smoothed_loss"] = early_stopper.smoothed_loss
            if early_stopper.best_loss is not None:
                logged["early_stop_best_loss"] = early_stopper.best_loss
            logged["early_stop_bad_steps"] = float(early_stopper.bad_steps)
        logger.log(state.step, logged)

        if should_stop:
            stop_reason = "loss_plateau"
            print(
                f"[early-stop] loss plateau at step {state.step}: "
                f"smoothed={early_stopper.smoothed_loss:.8f} "
                f"best={early_stopper.best_loss:.8f} "
                f"bad_steps={early_stopper.bad_steps}/"
                f"{early_stopper.config.patience}",
                flush=True,
            )
        elif state.step >= args.steps:
            stop_reason = "max_steps"

        terminal_step = should_stop or state.step >= args.steps
        if (
            args.save_every
            and state.step % args.save_every == 0
            and (not terminal_step or args.smoke)
        ):
            _save()
        return should_stop

    try:
        if args.data:
            # Real-data path: pull pixel-aware batches from the streaming
            # JSONL dataloader and reuse the canonical train_one_step so
            # CLI behaviour matches train_rdt.
            dataloader = build_dataloader(
                args.data,
                train_cfg,
                world_size=1,
                rank=0,
                pad_id=PAD_ID,
                image_processor=PILImageProcessor(image_size=args.image_size),
                omvt_cfg=omvt_cfg,
            )
            batch_iter = iter(dataloader)
            if args.resume and train_cfg.resume_skip_data and state.step > 0:
                _fast_forward_stream(batch_iter, state.step, train_cfg)
            while state.step < args.steps:
                metrics = train_one_step(
                    model,
                    batch_iter,
                    optimizer,
                    scheduler,
                    train_cfg,
                    state,
                    device=device,
                )
                if _record_completed_step(metrics):
                    break
        else:
            while state.step < args.steps:
                step = state.step + 1
                input_ids, attention_mask, labels = _make_text_batch(args)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                images = torch.randn(
                    args.batch_size,
                    omvt_cfg.in_channels,
                    omvt_cfg.image_size,
                    omvt_cfg.image_size,
                    device=device,
                )
                batch = collate_omvt_batch(images, omvt_cfg)

                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    pixel_values=dict(batch),
                    steps=rdt_cfg.recurrent_steps,
                )
                loss = out["loss"]
                if not bool(torch.isfinite(loss.detach())):
                    raise FloatingPointError(
                        f"non-finite VLM align loss at step {state.step}"
                    )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = clip_or_check_grad_norm(model, 1.0, step=step)
                optimizer.step()
                scheduler.step()
                state.step = step

                metrics = {
                    "loss": float(loss.detach()),
                    "grad_norm": grad_norm,
                    "lr": float(scheduler.get_last_lr()[0]),
                }
                if _record_completed_step(metrics):
                    break
        if not stop_reason:
            stop_reason = "max_steps"
        completed = True
    finally:
        logger.close()
        if completed and not args.smoke:
            _save(final=True)
    mode = "real-data" if args.data else "smoke"
    print(
        f"VLM align {mode} run OK in {time.time() - t0:.1f}s "
        f"(step={state.step}, stop_reason={stop_reason})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
