# -*- coding: utf-8 -*-

"""Single-optimizer-cycle primitives for anyres OCR and text replay SFT.

This module intentionally owns no dataloader, sampler, checkpoint, or CLI
state.  Callers assemble one complete cycle in memory, configure the intended
trainable scope before building the optimizer, and then invoke one of the two
cycle functions below.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from Model.ocr.position_contract import BOUNDARY_V1
from Model.posttrain.ocr_joint_forward import forward_anyres_ocr_batch


JOINT_OCR_MICROBATCHES = 4
DEFAULT_TEXT_WEIGHT = 0.2
DEFAULT_LOSS_CHUNK_SIZE = 4096


def configure_visual_stage_trainable(model: nn.Module) -> tuple[str, ...]:
    """Freeze everything except the native detail tower and ragged bridge."""

    core = _training_core(model)
    native_tower = _required_module(
        core,
        "vision.native_detail_tower",
        stage="visual",
    )
    bridge = _required_module(core, "vision_cross_attention", stage="visual")
    core.requires_grad_(False)
    native_tower.requires_grad_(True)
    bridge.requires_grad_(True)
    return _trainable_names(core)


def configure_joint_stage_trainable(model: nn.Module) -> tuple[str, ...]:
    """Train LM plus v1/native visual paths while freezing the legacy MLP.

    The required v1 OMVT tower/projector, native detail tower, and bridge are
    checked explicitly.  Future native projectors remain trainable because the
    joint stage enables the complete model before freezing only
    ``vision.encoder``, the unused legacy MLP fallback.
    """

    core = _training_core(model)
    for path in (
        "vision.omvt.tower",
        "vision.omvt.projector",
        "vision.native_detail_tower",
        "vision_cross_attention",
    ):
        _required_module(core, path, stage="joint")
    legacy_encoder = _required_module(core, "vision.encoder", stage="joint")
    core.requires_grad_(True)
    legacy_encoder.requires_grad_(False)
    return _trainable_names(core)


def train_visual_cycle(
    model: nn.Module,
    ocr_batches: Sequence[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    precision: str = "fp32",
    grad_clip: float | None = 1.0,
    loss_chunk_size: int = DEFAULT_LOSS_CHUNK_SIZE,
) -> dict[str, float | int | bool]:
    """Average one or more OCR microbatches and take one optimizer step."""

    batches = _validate_ocr_batches(ocr_batches, exact_count=None)
    return _train_cycle(
        model,
        batches,
        optimizer,
        text_batch=None,
        text_weight=0.0,
        device=_resolve_device(model, device),
        scheduler=scheduler,
        scaler=scaler,
        precision=precision,
        grad_clip=grad_clip,
        loss_chunk_size=loss_chunk_size,
    )


def train_joint_cycle(
    model: nn.Module,
    ocr_batches: Sequence[Mapping[str, Any]],
    text_batch: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    *,
    text_weight: float = DEFAULT_TEXT_WEIGHT,
    device: torch.device | str | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    precision: str = "fp32",
    grad_clip: float | None = 1.0,
    loss_chunk_size: int = DEFAULT_LOSS_CHUNK_SIZE,
) -> dict[str, float | int | bool]:
    """Run exactly ``4 OCR + 1 text`` forwards in one optimizer cycle.

    Each OCR loss contributes ``1/4`` and the causal text CE contributes
    ``text_weight`` (default ``0.2``).  Each weighted microbatch is backpropagated
    immediately so its graph can be released; the complete cycle still has one
    unscale/clip decision and at most one optimizer/scheduler step.
    """

    batches = _validate_ocr_batches(
        ocr_batches,
        exact_count=JOINT_OCR_MICROBATCHES,
    )
    if not isinstance(text_batch, Mapping):
        raise TypeError("text_batch must be a mapping")
    if (
        isinstance(text_weight, bool)
        or not isinstance(text_weight, (int, float))
        or not math.isfinite(float(text_weight))
        or float(text_weight) < 0.0
    ):
        raise ValueError("text_weight must be a finite non-negative number")
    return _train_cycle(
        model,
        batches,
        optimizer,
        text_batch=text_batch,
        text_weight=float(text_weight),
        device=_resolve_device(model, device),
        scheduler=scheduler,
        scaler=scaler,
        precision=precision,
        grad_clip=grad_clip,
        loss_chunk_size=loss_chunk_size,
    )


def _train_cycle(
    model: nn.Module,
    ocr_batches: tuple[Mapping[str, Any], ...],
    optimizer: torch.optim.Optimizer,
    *,
    text_batch: Mapping[str, Any] | None,
    text_weight: float,
    device: torch.device,
    scheduler: Any | None,
    scaler: Any | None,
    precision: str,
    grad_clip: float | None,
    loss_chunk_size: int,
) -> dict[str, float | int | bool]:
    _validate_precision(precision, device)
    if grad_clip is not None and (
        isinstance(grad_clip, bool)
        or not isinstance(grad_clip, (int, float))
        or not math.isfinite(float(grad_clip))
        or float(grad_clip) <= 0.0
    ):
        raise ValueError("grad_clip must be None or a finite positive number")
    if (
        isinstance(loss_chunk_size, bool)
        or not isinstance(loss_chunk_size, int)
        or loss_chunk_size <= 0
    ):
        raise ValueError("loss_chunk_size must be a positive integer")
    if getattr(_training_core(model), "reverse_loss_enabled", None) is not False:
        raise RuntimeError(
            "OCR SFT optimizer cycles require reverse_loss_enabled=False"
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    ocr_losses: list[torch.Tensor] = []
    ocr_weight = 1.0 / len(ocr_batches)
    try:
        for batch in ocr_batches:
            with _autocast_context(precision, device):
                output = forward_anyres_ocr_batch(
                    model,
                    batch,
                    device=device,
                    return_logits=False,
                    loss_chunk_size=loss_chunk_size,
                )
                ocr_loss = _scalar_loss(output, "OCR")
                weighted_ocr_loss = ocr_loss * ocr_weight
            _require_finite_loss(ocr_loss)
            if scaler is None:
                weighted_ocr_loss.backward()
            else:
                scaler.scale(weighted_ocr_loss).backward()
            ocr_losses.append(ocr_loss.detach())

        ocr_mean = torch.stack(ocr_losses).mean()
        if text_batch is None:
            text_ce = ocr_mean.new_zeros(())
        else:
            with _autocast_context(precision, device):
                live_text_ce = _forward_text_ce(
                    model,
                    text_batch,
                    device=device,
                    loss_chunk_size=loss_chunk_size,
                )
                weighted_text_ce = live_text_ce * text_weight
            _require_finite_loss(live_text_ce)
            if scaler is None:
                weighted_text_ce.backward()
            else:
                scaler.scale(weighted_text_ce).backward()
            text_ce = live_text_ce.detach()
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise

    total_loss = ocr_mean + text_weight * text_ce
    if scaler is not None:
        scaler.unscale_(optimizer)

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters or not any(
        parameter.grad is not None for parameter in trainable_parameters
    ):
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError("optimizer cycle produced no trainable gradients")
    grad_norm = torch.nn.utils.clip_grad_norm_(
        trainable_parameters,
        float(grad_clip) if grad_clip is not None else float("inf"),
        error_if_nonfinite=False,
    )
    grad_is_finite = bool(torch.isfinite(grad_norm.detach()))
    if not grad_is_finite and scaler is None:
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("non-finite gradients in OCR joint optimizer cycle")

    stepped = False
    if scaler is not None:
        if grad_is_finite:
            scale_before = _scaler_scale(scaler)
            scaler.step(optimizer)
            scaler.update()
            scale_after = _scaler_scale(scaler)
            stepped = (
                True
                if scale_before is None or scale_after is None
                else scale_after >= scale_before
            )
        else:
            # ``unscale_`` has already registered the overflow. Updating the
            # scaler lowers its scale without ever calling optimizer.step().
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
    else:
        optimizer.step()
        stepped = True
    if stepped and scheduler is not None:
        scheduler.step()

    metric_values = torch.stack(
        (
            total_loss.detach().float(),
            ocr_mean.detach().float(),
            text_ce.detach().float(),
            grad_norm.detach().to(device=total_loss.device, dtype=torch.float32),
        )
    ).cpu().tolist()
    return {
        "loss": metric_values[0],
        "ocr_loss": metric_values[1],
        "text_loss": metric_values[2],
        "weighted_text_loss": metric_values[2] * text_weight,
        "grad_norm": metric_values[3],
        "lr": float(optimizer.param_groups[0]["lr"]),
        "ocr_microbatches": len(ocr_batches),
        "text_weight": text_weight,
        "stepped": stepped,
    }


def _require_finite_loss(loss: torch.Tensor) -> None:
    if not bool(torch.isfinite(loss.detach())):
        raise FloatingPointError("non-finite loss in OCR joint optimizer cycle")


def _forward_text_ce(
    model: nn.Module,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
    loss_chunk_size: int,
) -> torch.Tensor:
    declared = batch.get("position_contract")
    if declared is not None and declared != BOUNDARY_V1:
        raise ValueError("text replay batch must use boundary_v1")
    if "word_pos" in batch or "morph_depth" in batch:
        raise ValueError(
            "text replay must derive positions from boundary_v1, not tensors"
        )
    required = {"input_ids", "labels"}
    if not required.issubset(batch):
        raise ValueError("text replay batch requires input_ids and labels")
    input_ids = _batch_tensor(batch, "input_ids", device)
    labels = _batch_tensor(batch, "labels", device)
    attention_mask = (
        _batch_tensor(batch, "attention_mask", device)
        if "attention_mask" in batch
        else None
    )
    if labels.shape != input_ids.shape:
        raise ValueError("text replay labels must match input_ids")
    core = _training_core(model)
    if getattr(core, "reverse_loss_enabled", None) is not False:
        raise RuntimeError(
            "joint text replay requires reverse_loss_enabled=False"
        )
    if bool(getattr(getattr(core, "cfg", None), "use_act", False)):
        raise RuntimeError("joint text replay requires use_act=False for pure CE")
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        position_contract=BOUNDARY_V1,
        return_logits=False,
        loss_chunk_size=loss_chunk_size,
    )
    if not isinstance(output, Mapping):
        raise TypeError("text replay forward must return a mapping")
    if output.get("logits") is not None:
        raise RuntimeError("chunked text replay forward must not return logits")
    return _scalar_loss(output, "text replay")


def _scalar_loss(output: object, kind: str) -> torch.Tensor:
    loss = output.get("loss") if isinstance(output, Mapping) else None
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
        raise TypeError(f"{kind} forward must return one scalar loss tensor")
    return loss


def _validate_ocr_batches(
    batches: Sequence[Mapping[str, Any]],
    *,
    exact_count: int | None,
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(batches, Sequence) or isinstance(batches, (str, bytes)):
        raise TypeError("ocr_batches must be a finite sequence")
    result = tuple(batches)
    if not result:
        raise ValueError("ocr_batches must not be empty")
    if exact_count is not None and len(result) != exact_count:
        raise ValueError(f"joint cycle requires exactly {exact_count} OCR batches")
    for index, batch in enumerate(result):
        if not isinstance(batch, Mapping):
            raise TypeError("every OCR batch must be a mapping")
        if batch.get("position_contract") != BOUNDARY_V1:
            raise ValueError(
                f"ocr_batches[{index}] must declare boundary_v1"
            )
        forbidden = sorted(
            {"word_pos", "morph_depth", "token_offsets"}.intersection(batch)
        )
        if forbidden:
            raise ValueError(
                f"ocr_batches[{index}] must not materialize position fields: "
                f"{forbidden}"
            )
    return result


def _required_module(model: nn.Module, path: str, *, stage: str) -> nn.Module:
    current: object = model
    for component in path.split("."):
        current = getattr(current, component, None)
        if current is None:
            raise RuntimeError(f"{stage} stage requires installed {path}")
    if not isinstance(current, nn.Module):
        raise TypeError(f"{path} must be an nn.Module")
    return current


def _trainable_names(model: nn.Module) -> tuple[str, ...]:
    names = tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    if not names:
        raise RuntimeError("training stage selected zero parameters")
    return names


def _training_core(model: nn.Module) -> nn.Module:
    core = model
    while hasattr(core, "module"):
        candidate = getattr(core, "module")
        if not isinstance(candidate, nn.Module):
            break
        core = candidate
    return core


def _resolve_device(
    model: nn.Module,
    requested: torch.device | str | None,
) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    parameter = next(model.parameters(), None)
    if parameter is None:
        raise ValueError("cannot infer device from a parameterless model")
    return parameter.device


def _autocast_context(precision: str, device: torch.device):
    if precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _validate_precision(precision: str, device: torch.device) -> None:
    if precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError("precision must be fp32, bf16, or fp16")
    if precision != "fp32" and device.type not in {"cpu", "cuda"}:
        raise ValueError("autocast is supported only on CPU or CUDA here")


def _batch_tensor(
    batch: Mapping[str, Any],
    key: str,
    device: torch.device,
) -> torch.Tensor:
    value = batch.get(key)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"text replay {key} must be a tensor")
    return value.to(device, non_blocking=True)


def _scaler_scale(scaler: Any) -> float | None:
    getter = getattr(scaler, "get_scale", None)
    return float(getter()) if callable(getter) else None


__all__ = [
    "DEFAULT_LOSS_CHUNK_SIZE",
    "DEFAULT_TEXT_WEIGHT",
    "JOINT_OCR_MICROBATCHES",
    "configure_joint_stage_trainable",
    "configure_visual_stage_trainable",
    "train_joint_cycle",
    "train_visual_cycle",
]
