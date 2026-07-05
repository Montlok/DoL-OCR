# -*- coding: utf-8 -*-

"""RDT text pretraining entry point.

Usage:

    python -m scripts.train_rdt --config tiny --smoke
    torchrun --nproc_per_node=2 -m scripts.train_rdt --config small \\
        --dist ddp --data path/to/shards/*.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

# Allow `torchrun scripts/train_rdt.py` (direct script invocation) to find
# the repository-root packages without requiring an editable install.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Model.config import (  # noqa: E402
    IGNORE_INDEX,
    PAD_ID,
    RDTConfig,
    TrainingConfig,
    base_config,
    mol_tiny_config,
    pretrain_config,
    segmented_pretrain_config,
    segmented_tiny_config,
    small_config,
    tiny_config,
    two_stage_pretrain_config,
    two_stage_tiny_config,
)
from Model.model import RDTForCausalLM  # noqa: E402
from Model.layers.mamba3_layer import official_available  # noqa: E402
from Model.training import (  # noqa: E402
    PretrainingCollator,
    RankZeroLogger,
    TrainState,
    add_multimodal_args,
    apply_parallelism,
    build_dataloader,
    build_image_processor,
    build_omvt_cfg,
    build_optimizer,
    build_scheduler,
    destroy_distributed,
    evaluate,
    init_distributed,
    is_main_process,
    resume_state,
    save_checkpoint,
    throughput_str,
    train_one_step,
)
from Model.training.status import StatusReporter  # noqa: E402


CONFIG_CHOICES = {
    "tiny": tiny_config,
    "small": small_config,
    "base": base_config,
    "pretrain": pretrain_config,
    "mol_tiny": mol_tiny_config,
    "two_stage_tiny": two_stage_tiny_config,
    "two_stage_pretrain": two_stage_pretrain_config,
    "segmented_tiny": segmented_tiny_config,
    "segmented_pretrain": segmented_pretrain_config,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pretrain RDT on text data")
    p.add_argument("--config", choices=list(CONFIG_CHOICES), default="tiny")
    p.add_argument("--data", default="")
    p.add_argument("--eval-data", default="")
    p.add_argument(
        "--tokenizer-bundle",
        default="",
        help="optional tokenizer bundle dir; recorded in checkpoint metadata",
    )
    p.add_argument("--output", default="outputs/rdt")
    p.add_argument("--resume", default="")
    p.add_argument(
        "--no-resume-skip-data",
        action="store_true",
        help=(
            "do not fast-forward the data stream to the checkpointed step on "
            "--resume (the stream then restarts at row 0; fine for smoke "
            "resumes, wrong for real continuation)"
        ),
    )
    p.add_argument("--dist", choices=["single", "ddp", "fsdp"], default="single")
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument(
        "--mamba",
        choices=["auto", "official", "naive"],
        default="auto",
        help=(
            "Mamba backend selection. 'auto' (default) uses official CUDA "
            "Mamba on CUDA/Linux and NaiveSSM on macOS/CPU; 'official' fails "
            "fast unless a CUDA-matched mamba_ssm is usable; 'naive' forces "
            "the fallback for macOS or cached decoding."
        ),
    )
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    p.add_argument("--adam-use-atan2", action="store_true")
    p.add_argument("--muon-momentum", type=float, default=0.95)
    p.add_argument("--muon-ns-steps", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=100_000)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--lr-decay-steps", type=int, default=None)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--lr-schedule", choices=["cosine", "wsd"], default="cosine")
    p.add_argument("--wsd-stable-ratio", type=float, default=0.8)
    p.add_argument(
        "--wsd-decay-shape",
        choices=["1-sqrt", "linear", "cosine"],
        default="1-sqrt",
    )
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-max-batches", type=int, default=32)
    p.add_argument(
        "--min-supervised-rate",
        type=float,
        default=0.01,
        help=(
            "minimum supervised label ratio required in prebuilt JSONL shards; "
            "set 0 only to explicitly disable this safety gate"
        ),
    )
    p.add_argument(
        "--data-gate-rows",
        type=int,
        default=10_000,
        help="number of JSONL rows sampled for pretraining data gate; 0 scans all rows",
    )
    p.add_argument("--bptt-window", type=int, default=None)
    p.add_argument(
        "--rec-steps-sampling",
        choices=["fixed", "poisson"],
        default="poisson",
        help=(
            "poisson = sample the recurrent depth per optimizer step "
            "(log-normal Poisson around the target) so the model stays "
            "usable at non-default depths; fixed = always cfg.recurrent_steps"
        ),
    )
    p.add_argument("--rec-steps-min", type=int, default=1)
    p.add_argument(
        "--rec-steps-max",
        type=int,
        default=None,
        help="cap for sampled depth (default: 2x target steps)",
    )
    p.add_argument("--rec-steps-sigma", type=float, default=0.5)
    p.add_argument("--recurrent-steps-start", type=int, default=None)
    p.add_argument("--recurrent-steps-ramp", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--dist-backend", choices=["nccl", "gloo"], default="nccl")
    p.add_argument(
        "--grad-ckpt-recurrent",
        choices=["auto", "on", "off"],
        default="auto",
        help="auto = inherit from RDTConfig; on/off override the model setting",
    )
    p.add_argument(
        "--grad-ckpt-prelude-coda",
        choices=["auto", "on", "off"],
        default="auto",
        help="auto = inherit from RDTConfig; on/off override the model setting",
    )
    p.add_argument("--smoke", action="store_true", help="run 4 in-memory steps")
    p.add_argument("--seed", type=int, default=42)
    add_multimodal_args(p)
    p.add_argument(
        "--patch-preset",
        choices=("derived", "prod"),
        default="derived",
        help="OMVT tower patch geometry (see train_vlm_align)",
    )
    p.add_argument(
        "--mix-data",
        default="",
        help="second JSONL stream (e.g. multimodal OCR rows) interleaved with "
        "--data at optimizer-step granularity; batches stay stream-pure so "
        "the collator's one-image-per-row invariant holds",
    )
    p.add_argument(
        "--mix-every",
        type=int,
        default=0,
        help="with --mix-data: every Nth optimizer step draws from the mix "
        "stream (e.g. 3 = two text steps then one mix step); 0 disables",
    )
    p.add_argument(
        "--init-omvt-checkpoint",
        default="",
        help="SSL tower checkpoint to initialize vision.omvt.tower from "
        "(requires --multimodal)",
    )
    p.add_argument(
        "--use-ema-tower",
        action="store_true",
        help="overlay tower_ema weights when --init-omvt-checkpoint has them",
    )
    return p.parse_args(argv)


def _tri_to_bool(value: str) -> bool | None:
    if value == "auto":
        return None
    return value == "on"


def _official_mamba_usable(device: str | torch.device | None = None) -> tuple[bool, str]:
    """Return whether this host should run the official CUDA Mamba backend."""

    if device is not None:
        try:
            device_type = torch.device(device).type
        except (RuntimeError, TypeError) as exc:
            return False, f"invalid target device {device!r}: {exc}"
        if device_type != "cuda":
            return False, f"target device is {device_type}, not cuda"
    host_os = platform.system()
    if host_os == "Darwin":
        return False, "macOS uses the NaiveSSM fallback"
    if host_os != "Linux":
        return False, f"{host_os} is not Linux"
    if not torch.cuda.is_available():
        return False, "CUDA is not available"
    if not official_available():
        return False, "mamba_ssm is not importable"
    return True, "official CUDA Mamba is available"


def _resolve_mamba_backend(
    cfg: RDTConfig,
    mode: str,
    *,
    use_cache: bool = False,
    device: str | torch.device | None = None,
    context: str = "scripts.train_rdt",
) -> RDTConfig:
    """Apply the ``--mamba`` override using deployment-oriented defaults.

    * ``official`` forces ``use_official_mamba=True`` (model construction will
      raise a clear error if the kernel is unusable).
    * ``naive`` forces the pure-PyTorch fallback, which is the expected macOS
      path and the only backend that supports incremental decode cache.
    * ``auto`` means production by default: use official Mamba on CUDA/Linux,
      otherwise use NaiveSSM. This deliberately overrides tiny configs on CUDA
      so smoke tests exercise the same backend as pretraining.
    """

    if use_cache and mode != "naive":
        raise ValueError(
            f"{context}: --use-cache requires --mamba naive and a NaiveSSM "
            "checkpoint; official Mamba kernels are not steppable by the decode "
            "cache, and official/naive checkpoints are not interchangeable."
        )

    if mode == "official":
        usable, reason = _official_mamba_usable(device=device)
        if not usable:
            raise RuntimeError(
                f"{context}: --mamba official requested but official Mamba is "
                f"not usable on this host ({reason}). Use --mamba naive on "
                "macOS/CPU, or install a CUDA-matched mamba-ssm wheel."
            )
        return replace(cfg, use_official_mamba=True)
    if mode == "naive":
        return replace(cfg, use_official_mamba=False)

    usable, reason = _official_mamba_usable(device=device)
    if usable:
        return replace(cfg, use_official_mamba=True)

    if cfg.use_official_mamba:
        sys.stderr.write(
            "WARNING: --mamba=auto falling back to NaiveSSM because "
            f"{reason}. This is expected for local fallback runs (macOS, "
            "CPU-only/CPU-target, or unsupported OSes); CUDA/Linux training "
            "should use --mamba=official to fail fast if the official backend "
            "is missing.\n"
        )
    return replace(cfg, use_official_mamba=False)


def _build_model_cfg(args: argparse.Namespace) -> RDTConfig:
    cfg = CONFIG_CHOICES[args.config]()
    if args.seq_len is not None:
        cfg = replace(cfg, max_seq_len=args.seq_len)
    if args.smoke and args.seq_len is None:
        cfg = replace(cfg, max_seq_len=64)
    cfg = _resolve_mamba_backend(cfg, args.mamba)
    return cfg


def _apply_train_overrides(model_cfg: RDTConfig, train_cfg: TrainingConfig) -> RDTConfig:
    """Project ``TrainingConfig`` knobs that live on ``RDTConfig`` into the
    model config so they take effect at construction time."""

    overrides: dict = {}
    if train_cfg.grad_ckpt_recurrent is not None:
        overrides["grad_ckpt_recurrent"] = bool(train_cfg.grad_ckpt_recurrent)
    if train_cfg.grad_ckpt_prelude_coda is not None:
        overrides["grad_ckpt_prelude_coda"] = bool(train_cfg.grad_ckpt_prelude_coda)
    if overrides:
        model_cfg = replace(model_cfg, **overrides)
    return model_cfg


def _build_train_cfg(args: argparse.Namespace, model_cfg: RDTConfig) -> TrainingConfig:
    seq_len = args.seq_len if args.seq_len is not None else model_cfg.max_seq_len
    return TrainingConfig(
        train_data=args.data,
        eval_data=args.eval_data,
        seq_len=seq_len,
        micro_batch_size=args.micro_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        num_workers=args.num_workers if not args.smoke else 0,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_use_atan2=args.adam_use_atan2,
        muon_momentum=args.muon_momentum,
        muon_ns_steps=args.muon_ns_steps,
        max_steps=4 if args.smoke else args.max_steps,
        warmup_steps=1 if args.smoke else args.warmup_steps,
        lr_decay_steps=args.lr_decay_steps,
        min_lr_ratio=args.min_lr_ratio,
        lr_schedule=args.lr_schedule,
        wsd_stable_ratio=args.wsd_stable_ratio,
        wsd_decay_shape=args.wsd_decay_shape,
        precision=args.precision,
        grad_clip=args.grad_clip,
        parallel=args.dist,
        dist_backend=args.dist_backend,
        grad_ckpt_recurrent=_tri_to_bool(args.grad_ckpt_recurrent),
        grad_ckpt_prelude_coda=_tri_to_bool(args.grad_ckpt_prelude_coda),
        output_dir=args.output,
        save_every=args.save_every,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_max_batches=args.eval_max_batches,
        bptt_window=args.bptt_window,
        recurrent_steps_sampling=args.rec_steps_sampling,
        recurrent_steps_min=args.rec_steps_min,
        recurrent_steps_max=args.rec_steps_max,
        recurrent_steps_sigma=args.rec_steps_sigma,
        recurrent_steps_start=args.recurrent_steps_start,
        recurrent_steps_ramp=args.recurrent_steps_ramp,
        seed=args.seed,
        resume=args.resume,
        resume_skip_data=not args.no_resume_skip_data,
    )


def _interleaved_batches(main_iter, mix_iter, *, grad_accum_steps: int, mix_every: int):
    """Deterministic block-level interleave of two batch streams.

    One optimizer step consumes ``grad_accum_steps`` consecutive micro
    batches (see ``train_one_step``), so stream switching happens on block
    boundaries: every ``mix_every``-th block comes from ``mix_iter``, the
    rest from ``main_iter``. Batches stay stream-pure, which preserves the
    collator's all-or-none images-per-batch invariant. The schedule is a
    pure function of the block index, so resume fast-forward through this
    iterator advances both underlying streams by exactly their share.
    """

    if mix_every < 2:
        raise ValueError("mix_every must be >= 2 (1 would starve the main stream)")
    block = 0
    while True:
        src = mix_iter if block % mix_every == mix_every - 1 else main_iter
        for _ in range(grad_accum_steps):
            yield next(src)
        block += 1


def _load_omvt_tower_init(model, path: str, use_ema: bool, rank: int) -> None:
    """Load SSL tower weights into the pre-installed OMVT injector."""

    if not path:
        return
    if model.vision.omvt is None:
        raise ValueError(
            "--init-omvt-checkpoint requires --multimodal (no OMVT injector installed)"
        )
    from Model.training.omvt_checkpoint import (
        load_omvt_payload,
        tower_state_from_payload,
    )

    payload = load_omvt_payload(path, weights_only=False)
    state = tower_state_from_payload(payload, use_ema=use_ema)
    model.vision.omvt.tower.load_state_dict(state)
    if rank == 0:
        ema = " (EMA)" if use_ema and isinstance(payload, dict) and payload.get("tower_ema") else ""
        print(f"[init] loaded OMVT tower from {path}{ema}", flush=True)


def _fast_forward_stream(batch_iter, resumed_step: int, train_cfg, rank: int) -> None:
    """Skip the batches a resumed run already consumed.

    The stream is deterministic (seeded shuffle, fixed shard order), so a
    rebuilt iterator restarts at row 0 while ``state.step`` keeps counting;
    consuming ``step * grad_accum_steps`` batches realigns the data with the
    schedule. Read+collate only — no forward pass, so it is IO-bound.
    """
    skip = resumed_step * train_cfg.grad_accum_steps
    if skip <= 0:
        return
    t0 = time.time()
    for done in range(skip):
        next(batch_iter)
        if rank == 0 and (done + 1) % 5000 == 0:
            print(
                f"scripts/train_rdt: resume fast-forward {done + 1}/{skip} "
                f"batches ({time.time() - t0:.0f}s elapsed)",
                flush=True,
            )
    if rank == 0:
        print(
            f"scripts/train_rdt: resume fast-forwarded {skip} batches in "
            f"{time.time() - t0:.0f}s (use --no-resume-skip-data to disable)",
            flush=True,
        )


def _target_recurrent_steps_for_train(
    model_cfg: RDTConfig,
    train_cfg: TrainingConfig,
) -> int | None:
    """Return an explicit depth only when the training curriculum needs one.

    Passing ``steps`` unconditionally disables ``SegmentedCore`` random-r
    sampling. Let the model resolve its own depth unless a depth ramp is active.
    """

    if train_cfg.recurrent_steps_start is None or train_cfg.recurrent_steps_ramp <= 0:
        return None
    return model_cfg.recurrent_steps


def _smoke_batches(model_cfg: RDTConfig, train_cfg: TrainingConfig):
    rng = torch.Generator().manual_seed(train_cfg.seed)
    ids = torch.randint(
        300,
        320,
        (train_cfg.seq_len - 2,),
        generator=rng,
    ).tolist()
    seq = [model_cfg.bos_id] + ids + [model_cfg.eos_id]
    row = {"input_ids": seq, "attention_mask": [1] * len(seq), "labels": seq}
    collator = PretrainingCollator(max_seq_len=train_cfg.seq_len)
    while True:
        yield collator([row] * train_cfg.micro_batch_size)


def _sample_supervision_metrics(shards: list[Path], max_rows: int) -> dict[str, float]:
    rows_seen = 0
    active_tokens = 0
    supervised_tokens = 0
    for shard in shards:
        with shard.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in {shard}:{line_no}: {exc}") from exc
                labels = row.get("labels")
                if not isinstance(labels, list):
                    raise ValueError(f"missing labels list in {shard}:{line_no}")
                mask = row.get("attention_mask")
                if mask is None:
                    mask = [1] * len(labels)
                if not isinstance(mask, list) or len(mask) != len(labels):
                    raise ValueError(
                        f"attention_mask must align with labels in {shard}:{line_no}"
                    )
                for label, active in zip(labels, mask):
                    if int(active):
                        active_tokens += 1
                        if int(label) != IGNORE_INDEX:
                            supervised_tokens += 1
                rows_seen += 1
                if max_rows > 0 and rows_seen >= max_rows:
                    return {
                        "rows": float(rows_seen),
                        "active_tokens": float(active_tokens),
                        "supervised_tokens": float(supervised_tokens),
                        "supervised_rate": (
                            supervised_tokens / active_tokens if active_tokens else 0.0
                        ),
                    }
    return {
        "rows": float(rows_seen),
        "active_tokens": float(active_tokens),
        "supervised_tokens": float(supervised_tokens),
        "supervised_rate": supervised_tokens / active_tokens if active_tokens else 0.0,
    }


def _validate_args(args: argparse.Namespace) -> int:
    """Argument-level validation.

    Runs **before** any expensive setup (distributed init, model alloc,
    output-dir creation) so misconfigured invocations fail fast with a
    clear stderr message and a non-zero exit code, instead of OOMing
    halfway through model construction or leaving stale ``outputs/`` dirs.
    Returns 0 on success, a non-zero code on failure.
    """

    if not args.smoke and not args.data:
        print(
            "scripts/train_rdt: --data is required for non-smoke runs; "
            "pass --smoke to run on synthetic tokens explicitly.",
            file=sys.stderr,
        )
        return 2
    if args.resume and not Path(args.resume).exists():
        print(
            f"scripts/train_rdt: --resume path does not exist: {args.resume}",
            file=sys.stderr,
        )
        return 2
    if args.tokenizer_bundle and not Path(args.tokenizer_bundle).is_dir():
        print(
            "scripts/train_rdt: --tokenizer-bundle must be an existing directory: "
            f"{args.tokenizer_bundle}",
            file=sys.stderr,
        )
        return 2
    if not (0.0 <= args.min_supervised_rate <= 1.0):
        print(
            "scripts/train_rdt: --min-supervised-rate must be in [0, 1]",
            file=sys.stderr,
        )
        return 2
    if args.data_gate_rows < 0:
        print(
            "scripts/train_rdt: --data-gate-rows must be non-negative",
            file=sys.stderr,
        )
        return 2
    if bool(args.mix_data) != bool(args.mix_every):
        print(
            "scripts/train_rdt: --mix-data and --mix-every must be set together",
            file=sys.stderr,
        )
        return 2
    if args.mix_every and args.mix_every < 2:
        print(
            "scripts/train_rdt: --mix-every must be >= 2",
            file=sys.stderr,
        )
        return 2
    if args.init_omvt_checkpoint and not getattr(args, "multimodal", False):
        print(
            "scripts/train_rdt: --init-omvt-checkpoint requires --multimodal",
            file=sys.stderr,
        )
        return 2
    # Resolve the data spec **here** rather than waiting for build_dataloader
    # so an empty glob (typo'd shard pattern) fails *before* we allocate a
    # multi-billion-parameter model and initialize the process group.
    if not args.smoke and args.data:
        from Model.training.data import _resolve_shards

        shards = _resolve_shards(args.data)
        if not shards:
            print(
                f"scripts/train_rdt: --data resolved zero shards: {args.data!r}",
                file=sys.stderr,
            )
            return 2
        if args.mix_data:
            mix_shards = _resolve_shards(args.mix_data)
            if not mix_shards:
                print(
                    f"scripts/train_rdt: --mix-data resolved zero shards: "
                    f"{args.mix_data!r}",
                    file=sys.stderr,
                )
                return 2
            if args.min_supervised_rate > 0:
                try:
                    mix_metrics = _sample_supervision_metrics(
                        mix_shards, args.data_gate_rows
                    )
                except ValueError as exc:
                    print(
                        f"scripts/train_rdt: mix data gate failed: {exc}",
                        file=sys.stderr,
                    )
                    return 2
                if mix_metrics["supervised_rate"] < args.min_supervised_rate:
                    print(
                        "scripts/train_rdt: mix supervised_rate "
                        f"{mix_metrics['supervised_rate']:.6f} is below "
                        f"{args.min_supervised_rate:.6f}",
                        file=sys.stderr,
                    )
                    return 2
        if args.min_supervised_rate > 0:
            try:
                metrics = _sample_supervision_metrics(shards, args.data_gate_rows)
            except ValueError as exc:
                print(f"scripts/train_rdt: data gate failed: {exc}", file=sys.stderr)
                return 2
            if metrics["rows"] <= 0:
                print(
                    f"scripts/train_rdt: data gate found no JSONL rows: {args.data!r}",
                    file=sys.stderr,
                )
                return 2
            if metrics["active_tokens"] <= 0:
                print(
                    "scripts/train_rdt: data gate found zero active tokens in "
                    f"{int(metrics['rows'])} sampled rows",
                    file=sys.stderr,
                )
                return 2
            if metrics["supervised_rate"] < args.min_supervised_rate:
                print(
                    "scripts/train_rdt: supervised_rate "
                    f"{metrics['supervised_rate']:.6f} is below "
                    f"{args.min_supervised_rate:.6f} over "
                    f"{int(metrics['rows'])} sampled rows; check labels or pass "
                    "--min-supervised-rate 0 only for an intentional unsupervised run.",
                    file=sys.stderr,
                )
                return 2
        if args.eval_data:
            eval_shards = _resolve_shards(args.eval_data)
            if not eval_shards:
                print(
                    f"scripts/train_rdt: --eval-data resolved zero shards: "
                    f"{args.eval_data!r}",
                    file=sys.stderr,
                )
                return 2
    return 0


def _evaluate_distributed(
    model,
    eval_loader,
    train_cfg: TrainingConfig,
    *,
    device: torch.device,
) -> dict[str, float]:
    metrics = evaluate(
        model,
        eval_loader,
        train_cfg,
        device=device,
        max_batches=train_cfg.eval_max_batches,
    )
    tokens = float(metrics["eval_tokens"])
    loss_sum = float(metrics["eval_loss"]) * tokens
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        packed = torch.tensor([loss_sum, tokens], device=device, dtype=torch.float64)
        torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
        loss_sum = float(packed[0].item())
        tokens = float(packed[1].item())
    return {
        "eval_loss": loss_sum / max(1.0, tokens),
        "eval_tokens": tokens,
    }


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _tokenizer_bundle_metadata(path: str) -> dict:
    if not path:
        return {}
    root = Path(path)
    files: dict[str, str] = {}
    for name in (
        "config.json",
        "morphbpe.json",
        "general.json",
        "vocab.json",
        "manifest.json",
    ):
        candidate = root / name
        if candidate.exists():
            files[name] = _file_sha256(candidate)
    manifest = {}
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    return {
        "path": str(root),
        "files": files,
        "manifest": manifest,
    }


def _git_metadata() -> dict:
    def _run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=_REPO_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return ""

    return {
        "commit": _run("rev-parse", "HEAD"),
        "branch": _run("branch", "--show-current"),
        "dirty": bool(_run("status", "--short")),
    }


def _run_metadata(
    args: argparse.Namespace,
    model_cfg: RDTConfig,
    train_cfg: TrainingConfig,
    omvt_cfg,
) -> dict:
    return {
        "config_name": args.config,
        "rdt_config": asdict(model_cfg),
        "training_config": asdict(train_cfg),
        "omvt_config": asdict(omvt_cfg) if omvt_cfg is not None else None,
        "mix": (
            {"mix_data": args.mix_data, "mix_every": args.mix_every}
            if args.mix_data
            else None
        ),
        "tokenizer_bundle": _tokenizer_bundle_metadata(args.tokenizer_bundle),
        "git": _git_metadata(),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rc = _validate_args(args)
    if rc != 0:
        return rc
    try:
        model_cfg_preview = _build_model_cfg(args)
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    train_cfg_preview = _build_train_cfg(args, model_cfg_preview)
    omvt_cfg = build_omvt_cfg(args)
    rank, world_size, local_rank = init_distributed(backend=train_cfg_preview.dist_backend)
    if args.dist != "single" and world_size == 1 and not args.smoke:
        print(
            "[warn] --dist != single but world_size=1; running single-process",
            file=sys.stderr,
        )

    torch.manual_seed(args.seed + rank)

    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )

    model_cfg = model_cfg_preview
    train_cfg = train_cfg_preview
    model_cfg = _apply_train_overrides(model_cfg, train_cfg)
    run_metadata = _run_metadata(args, model_cfg, train_cfg, omvt_cfg)

    if is_main_process():
        Path(train_cfg.output_dir).mkdir(parents=True, exist_ok=True)

    model = RDTForCausalLM(model_cfg).to(device)
    if omvt_cfg is not None:
        # Pre-install a matching-size OMVT injector so the dispatcher
        # does not lazily build a default-sized one on first forward
        # (which would mismatch the dataloader's pixel geometry).
        from Model.omvt import OMVTInjector

        model.vision._omvt_cfg = omvt_cfg
        model.vision.omvt = OMVTInjector(model_cfg, omvt_cfg).to(device)
    if args.init_omvt_checkpoint and not train_cfg.resume:
        _load_omvt_tower_init(model, args.init_omvt_checkpoint, args.use_ema_tower, rank)

    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)

    # NOTE: optimizer is built on the pre-wrap parameters. Under FSDP this is
    # only safe because we wrap with ``use_orig_params=True`` (see
    # ``apply_parallelism``), which keeps the original ``nn.Parameter`` objects
    # the optimizer references. If that flag ever changes, the optimizer must be
    # built *after* ``apply_parallelism`` instead. Verify on the GPU cluster.
    model = apply_parallelism(model, train_cfg, local_rank)

    state = TrainState()
    if train_cfg.resume:
        state.step = resume_state(
            train_cfg.resume, model, optimizer, scheduler, state=state
        )

    image_processor = build_image_processor(args)

    if args.smoke:
        batch_iter = _smoke_batches(model_cfg, train_cfg)
        eval_loader = None
    else:
        dataloader = build_dataloader(
            train_cfg.train_data,
            train_cfg,
            world_size=world_size,
            rank=rank,
            pad_id=PAD_ID,
            image_processor=image_processor,
            omvt_cfg=omvt_cfg,
        )
        batch_iter = iter(dataloader)
        if args.mix_data:
            mix_loader = build_dataloader(
                args.mix_data,
                train_cfg,
                world_size=world_size,
                rank=rank,
                pad_id=PAD_ID,
                image_processor=image_processor,
                omvt_cfg=omvt_cfg,
            )
            batch_iter = _interleaved_batches(
                batch_iter,
                iter(mix_loader),
                grad_accum_steps=train_cfg.grad_accum_steps,
                mix_every=args.mix_every,
            )
        if train_cfg.resume and train_cfg.resume_skip_data and state.step > 0:
            _fast_forward_stream(batch_iter, state.step, train_cfg, rank)
        eval_loader = None
        if train_cfg.eval_data:
            eval_loader = build_dataloader(
                train_cfg.eval_data,
                replace(train_cfg, shuffle_buffer=0),
                world_size=world_size,
                rank=rank,
                pad_id=PAD_ID,
                infinite=False,
                drop_last=False,
                image_processor=build_image_processor(args),
                omvt_cfg=omvt_cfg,
            )

    logger = RankZeroLogger(train_cfg.output_dir, enable_tensorboard=train_cfg.tensorboard)
    reporter = None
    if is_main_process():
        reporter = StatusReporter(
            train_cfg.output_dir,
            run_metadata={
                "config_name": args.config,
                "world_size": world_size,
                "git": run_metadata["git"],
            },
            max_steps=train_cfg.max_steps,
        )
    stop_requested = False
    t0 = time.time()
    tokens_window = 0
    completed = False
    try:
        while state.step < train_cfg.max_steps and not stop_requested:
            metrics = train_one_step(
                model,
                batch_iter,
                optimizer,
                scheduler,
                train_cfg,
                state,
                device=device,
                target_recurrent_steps=_target_recurrent_steps_for_train(
                    model_cfg,
                    train_cfg,
                ),
            )
            tokens_window += int(metrics["tokens"])

            if state.step % train_cfg.log_every == 0 or args.smoke:
                dt = max(1e-6, time.time() - t0)
                tp = throughput_str(tokens_window, dt)
                logger.log(state.step, {**metrics, "throughput": tp})
                if reporter is not None:
                    reporter.update(state.step, {**metrics, "throughput": tp})
                t0 = time.time()
                tokens_window = 0

            # Control plane: rank0 reads commands, all ranks act in lockstep
            # (save/eval are collectives under FSDP and must run everywhere).
            commands = reporter.poll_control() if reporter is not None else []
            if world_size > 1:
                box = [commands]
                torch.distributed.broadcast_object_list(box, src=0)
                commands = box[0]

            want_eval = (
                eval_loader is not None
                and train_cfg.eval_every
                and state.step % train_cfg.eval_every == 0
            ) or ("eval" in commands and eval_loader is not None)
            if want_eval:
                eval_metrics = _evaluate_distributed(
                    model,
                    eval_loader,
                    train_cfg,
                    device=device,
                )
                logger.log(state.step, eval_metrics)
                if reporter is not None:
                    reporter.update(state.step, eval_metrics, state="running")

            want_save = (
                train_cfg.save_every
                and state.step % train_cfg.save_every == 0
                and not args.smoke
            ) or ("save" in commands and not args.smoke)
            if want_save:
                save_checkpoint(
                    train_cfg.output_dir,
                    state.step,
                    model,
                    optimizer,
                    scheduler,
                    metadata={**run_metadata, "checkpoint": "periodic"},
                    keep_last_n=train_cfg.keep_last_n,
                    scaler=state.extra.get("grad_scaler"),
                )

            if "stop" in commands:
                stop_requested = True
        completed = True
    finally:
        try:
            logger.close()
            if reporter is not None:
                reporter.finish(
                    state="stopped" if stop_requested else "finished",
                    step=state.step,
                )
            # NOTE: `save_checkpoint` is a collective under FSDP (the
            # `state_dict(FullStateDictConfig)` call gathers from every rank),
            # so we must enter it on every rank; the helper internally limits
            # the actual file write to rank 0. Guarding the call with
            # `is_main_process()` would deadlock rank 0 at shutdown.
            if completed and not args.smoke:
                save_checkpoint(
                    train_cfg.output_dir,
                    state.step,
                    model,
                    optimizer,
                    scheduler,
                    metadata={**run_metadata, "checkpoint": "final"},
                    keep_last_n=train_cfg.keep_last_n,
                    scaler=state.extra.get("grad_scaler"),
                )
        finally:
            destroy_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
