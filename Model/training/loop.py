# -*- coding: utf-8 -*-

"""Single-step train / eval primitives."""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from Model.config import IGNORE_INDEX, TrainingConfig
from Model.training.optim import recurrent_steps_for_step


@dataclass
class TrainState:
    step: int = 0
    tokens_seen: int = 0
    last_loss: float = float("nan")
    extra: dict[str, Any] = field(default_factory=dict)


def _autocast_ctx(precision: str, device_type: str):
    if precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device_type, dtype=dtype)


def _new_cuda_grad_scaler():
    """Build a CUDA ``GradScaler`` using the non-deprecated API when available."""

    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


def _grad_scaler_for(model: nn.Module, cfg: TrainingConfig, device_type: str):
    """Return a fp16 ``GradScaler`` (sharded under FSDP) or ``None``.

    fp16 autocast without loss scaling silently underflows small gradients to
    zero and can diverge to NaN. bf16/fp32 have enough exponent range and need
    no scaler. CPU has no fp16 GradScaler support, so we leave it unscaled.
    """

    if cfg.precision != "fp16" or device_type != "cuda":
        return None
    # FSDP shards gradients across ranks; a plain GradScaler.unscale_ would see
    # only the local shard, so use the sharded variant when the model is FSDP.
    if hasattr(model, "clip_grad_norm_"):
        try:
            from torch.distributed.fsdp.sharded_grad_scaler import (
                ShardedGradScaler,
            )

            return ShardedGradScaler()
        except Exception:  # pragma: no cover - older torch without ShardedGradScaler
            return _new_cuda_grad_scaler()
    return _new_cuda_grad_scaler()


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            # Pixel batches arrive as ``{"images": ..., "vertical_patches": ...}``;
            # we have to recurse so every leaf tensor lands on the model's
            # device. Non-tensor leaves (rare) pass through untouched.
            out[k] = {
                kk: vv.to(device, non_blocking=True) if isinstance(vv, torch.Tensor) else vv
                for kk, vv in v.items()
            }
        else:
            out[k] = v
    return out


def _local_grad_norm(parameters) -> torch.Tensor:
    """Return a local L2 grad norm without mutating gradients."""

    total: torch.Tensor | None = None
    for param in parameters:
        grad = param.grad
        if grad is None:
            continue
        grad = grad.detach()
        if grad.is_sparse:
            grad = grad.coalesce().values()
        grad_sq = grad.float().pow(2).sum()
        total = grad_sq if total is None else total + grad_sq
    if total is None:
        return torch.tensor(0.0)
    return total.sqrt()


def clip_or_check_grad_norm(
    model: nn.Module,
    max_norm: float | None,
    *,
    step: int,
    allow_nonfinite: bool = False,
) -> float:
    """Clip gradients when requested and fail fast on non-finite norms.

    ``allow_nonfinite=True`` is intended for fp16 ``GradScaler`` paths, where
    scaler.step() already skips overflow updates. bf16/fp32 paths do not have
    that protection, so non-finite gradients must never reach optimizer.step().
    """

    if max_norm is not None and max_norm > 0:
        if hasattr(model, "clip_grad_norm_"):
            grad_norm = model.clip_grad_norm_(max_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    else:
        grad_norm = _local_grad_norm(model.parameters())

    grad_norm_val = float(grad_norm)
    if not allow_nonfinite and not math.isfinite(grad_norm_val):
        raise FloatingPointError(
            f"non-finite gradients at step {step} (grad_norm={grad_norm_val})"
        )
    return grad_norm_val


def train_one_step(
    model: nn.Module,
    batch_iter,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    cfg: TrainingConfig,
    state: TrainState,
    *,
    device: torch.device,
    target_recurrent_steps: int | None = None,
) -> dict[str, float]:
    """Run one optimizer step (covers ``grad_accum_steps`` micro batches)."""

    model.train()
    optimizer.zero_grad(set_to_none=True)

    device_type = device.type if device.type in ("cuda", "cpu") else "cpu"
    loss_terms: list[torch.Tensor] = []
    token_count = 0

    # fp16 needs loss scaling; persist the scaler across steps via state.extra
    # so its dynamic scale factor is preserved (and can be checkpointed).
    scaler = state.extra.get("grad_scaler")
    if scaler is None and cfg.precision == "fp16":
        scaler = _grad_scaler_for(model, cfg, device_type)
        if scaler is not None:
            pending = state.extra.pop("grad_scaler_state", None)
            if pending is not None:
                scaler.load_state_dict(pending)
            state.extra["grad_scaler"] = scaler

    rec_steps = None
    schedule_active = cfg.recurrent_steps_sampling == "poisson" or (
        cfg.recurrent_steps_start is not None and cfg.recurrent_steps_ramp > 0
    )
    if target_recurrent_steps is not None and schedule_active:
        rec_steps = recurrent_steps_for_step(state.step, cfg, target_recurrent_steps)

    is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)
    for micro in range(cfg.grad_accum_steps):
        batch = next(batch_iter)
        # Count tokens on the CPU mask **before** moving to device so the
        # ``.sum()`` does not force a host-sync against the GPU stream.
        cpu_mask = batch.get("attention_mask")
        if isinstance(cpu_mask, torch.Tensor):
            token_count += int(cpu_mask.sum().item())
        batch = _to_device(batch, device)

        # DDP all-reduces gradients during every backward; with accumulation
        # only the final micro-batch's reduction is needed. ``no_sync`` must
        # wrap the forward too — DDP samples the sync flag at forward time.
        # FSDP is left to sync every micro-batch on purpose: its no_sync keeps
        # full unsharded grads alive and trades the bandwidth for peak memory.
        sync_ctx = (
            model.no_sync()
            if is_ddp and micro < cfg.grad_accum_steps - 1
            else contextlib.nullcontext()
        )
        with sync_ctx:
            with _autocast_ctx(cfg.precision, device_type):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch["labels"],
                    word_pos=batch.get("word_pos"),
                    morph_depth=batch.get("morph_depth"),
                    position_contract=batch.get("position_contract"),
                    pixel_values=batch.get("pixel_values"),
                    steps=rec_steps,
                    bptt_window=cfg.bptt_window,
                    return_logits=not cfg.use_loss_chunking,
                )
            loss = out["loss"] / cfg.grad_accum_steps
            if not bool(torch.isfinite(loss.detach())):
                raise FloatingPointError(
                    f"non-finite training loss at step {state.step}"
                )
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        # Keep the per-microstep loss as a detached tensor; we only sync
        # to host once per optimizer step to avoid stalling the training
        # loop on the device queue.
        loss_terms.append(loss.detach())

    if scaler is not None:
        # Unscale before clipping so grad_clip is applied to true gradients.
        scaler.unscale_(optimizer)

    grad_norm_val = clip_or_check_grad_norm(
        model,
        cfg.grad_clip,
        step=state.step,
        allow_nonfinite=scaler is not None,
    )

    if scaler is not None:
        # GradScaler.step() may SKIP the underlying optimizer update when the
        # unscaled grads contain inf/NaN. An overflow step lowers the dynamic
        # scale on update(); a real step keeps or grows it. Only advance the LR
        # scheduler when the optimizer actually stepped, so fp16 overflow steps
        # don't silently consume warmup/decay slots and desync the schedule.
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        stepped = scaler.get_scale() >= scale_before
    else:
        optimizer.step()
        stepped = True
    if stepped:
        scheduler.step()

    if loss_terms:
        # One host-sync per optimizer step instead of per micro-step.
        loss_sum = torch.stack(loss_terms).sum().item() * cfg.grad_accum_steps
    else:
        loss_sum = 0.0

    state.step += 1
    state.tokens_seen += token_count
    state.last_loss = loss_sum / max(1, cfg.grad_accum_steps)
    return {
        "loss": state.last_loss,
        "grad_norm": grad_norm_val,
        "lr": float(scheduler.get_last_lr()[0]),
        "tokens": float(token_count),
        "rec_steps": float(rec_steps) if rec_steps is not None else float("nan"),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    batches,
    cfg: TrainingConfig,
    *,
    device: torch.device,
    max_batches: int = 32,
) -> dict[str, float]:
    model.eval()
    device_type = device.type if device.type in ("cuda", "cpu") else "cpu"
    # The model reduces its loss using its *own* ``ignore_index`` (RDTConfig),
    # which may be overridden away from the module-level default. Read it from
    # the unwrapped module so the eval denominator counts exactly the positions
    # the model treated as targets (DDP/FSDP expose the real module via
    # ``.module``).
    core = model
    while hasattr(core, "module"):
        core = core.module
    ignore_index = getattr(getattr(core, "cfg", None), "ignore_index", IGNORE_INDEX)

    total_loss = 0.0
    total_targets = 0
    seen = 0
    for batch in batches:
        if seen >= max_batches:
            break
        # Count every consumed batch toward the limit *before* any skip so an
        # infinite streaming dataloader yielding all-ignore rows can't spin
        # forever — ``max_batches`` must bound batches read, not just averaged.
        seen += 1
        batch = _to_device(batch, device)
        with _autocast_ctx(cfg.precision, device_type):
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch["labels"],
                word_pos=batch.get("word_pos"),
                morph_depth=batch.get("morph_depth"),
                position_contract=batch.get("position_contract"),
                pixel_values=batch.get("pixel_values"),
                return_logits=not cfg.use_loss_chunking,
            )
        # Report the **forward (causal) LM loss** — the perplexity-relevant
        # term — weighted by its own valid next-token count. Labels are shifted
        # internally (logits[:, :-1] vs labels[:, 1:]); image/pad positions are
        # ignore_index. Using the forward part keeps the cross-batch average
        # exact regardless of the reverse/ponder terms a bidirectional training
        # objective folds into ``out["loss"]`` (those use a different shift and
        # token count, which would bias a combined-loss weighting).
        labels = batch["labels"]
        n_targets = int((labels[:, 1:] != ignore_index).sum().item())
        if n_targets == 0:
            continue
        loss_parts = out.get("loss_parts") or {}
        batch_loss = loss_parts.get("forward")
        if batch_loss is None:
            batch_loss = float(out["loss"].item())
        total_loss += float(batch_loss) * n_targets
        total_targets += n_targets
    avg = total_loss / max(1, total_targets)
    return {"eval_loss": avg, "eval_tokens": float(total_targets)}


__all__ = ["TrainState", "evaluate", "train_one_step"]
