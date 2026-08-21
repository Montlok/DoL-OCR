# -*- coding: utf-8 -*-

"""One in-memory optimizer cycle for anyres OCR GRPO plus text replay.

This module deliberately owns no dataloader, sampler, checkpoint, journal,
file, or CLI state.  The caller supplies one admitted OCR batch, one selected
KL trial, and an already constructed optimizer (including the repository's
four learning-rate roles).  Exactly one GRPO objective is evaluated per call;
an optional causal text CE is then backpropagated separately so neither graph
is retained until the optimizer step.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from Model.ocr.position_contract import BOUNDARY_V1
from Model.posttrain.grpo import GRPOConfig
from Model.posttrain.ocr_anyres_grpo import (
    AnyresGRPOAdmission,
    anyres_grpo_compute_loss,
)
from Model.posttrain.ocr_anyres_reward import AnyresOCRRewardAdapter
from Model.posttrain.ocr_joint_trainer import (
    DEFAULT_LOSS_CHUNK_SIZE,
    DEFAULT_TEXT_WEIGHT,
    _autocast_context,
    _forward_text_ce,
    _scaler_scale,
    _validate_precision,
)
from Model.posttrain.text_replay import canonical_json_sha256


@dataclass(frozen=True)
class AnyresGRPOKLAblationTrial:
    """Immutable record selecting one member of a bounded KL ablation.

    The complete candidate set is recorded for auditability, but
    ``selected_index`` binds this optimizer cycle to exactly one coefficient.
    Running the other candidates requires separate calls, optimizer state, and
    experiment bookkeeping; this primitive never multiplexes them internally.
    """

    candidate_kl_coefs: tuple[float, ...] = (0.04, 0.01, 0.0)
    selected_index: int = 0
    trial_steps: int = 200
    experiment_id: str = "ocr_anyres_kl_ablation_v1"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.candidate_kl_coefs, tuple)
            or not self.candidate_kl_coefs
        ):
            raise ValueError("candidate_kl_coefs must be a non-empty tuple")
        normalized: list[float] = []
        for value in self.candidate_kl_coefs:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("candidate KL coefficients must be finite and non-negative")
            value = float(value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("candidate KL coefficients must be finite and non-negative")
            normalized.append(value)
        if len(set(normalized)) != len(normalized):
            raise ValueError("candidate KL coefficients must be unique")
        if (
            isinstance(self.selected_index, bool)
            or not isinstance(self.selected_index, int)
            or not 0 <= self.selected_index < len(normalized)
        ):
            raise ValueError("selected_index is outside candidate_kl_coefs")
        if (
            isinstance(self.trial_steps, bool)
            or not isinstance(self.trial_steps, int)
            or self.trial_steps <= 0
        ):
            raise ValueError("trial_steps must be a positive integer")
        if (
            not isinstance(self.experiment_id, str)
            or not self.experiment_id
            or self.experiment_id != self.experiment_id.strip()
            or any(ord(character) < 0x20 for character in self.experiment_id)
        ):
            raise ValueError("experiment_id must be a non-empty safe string")

    @property
    def selected_kl_coef(self) -> float:
        return float(self.candidate_kl_coefs[self.selected_index])

    @property
    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "dol_ocr_anyres_grpo_kl_ablation_trial_v1",
            "experiment_id": self.experiment_id,
            "candidate_kl_coefs": [
                float(value) for value in self.candidate_kl_coefs
            ],
            "selected_index": self.selected_index,
            "selected_kl_coef": self.selected_kl_coef,
            "trial_steps": self.trial_steps,
            "execution_contract": "one_selected_kl_per_optimizer_cycle",
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_json_sha256(self.canonical_payload)

    def require_matches(self, cfg: GRPOConfig) -> None:
        if not isinstance(cfg, GRPOConfig):
            raise TypeError("grpo_config must be a GRPOConfig")
        if float(cfg.kl_coef) != self.selected_kl_coef:
            raise ValueError(
                "GRPO kl_coef differs from the selected KL ablation trial: "
                f"{cfg.kl_coef!r} != {self.selected_kl_coef!r}"
            )


def train_anyres_grpo_cycle(
    policy: nn.Module,
    reference: nn.Module | None,
    ocr_batch: Mapping[str, Any],
    reward_adapter: AnyresOCRRewardAdapter,
    admission: AnyresGRPOAdmission,
    grpo_config: GRPOConfig,
    optimizer: torch.optim.Optimizer,
    *,
    kl_trial: AnyresGRPOKLAblationTrial,
    text_batch: Mapping[str, Any] | None = None,
    text_weight: float = DEFAULT_TEXT_WEIGHT,
    device: torch.device | str | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    precision: str = "fp32",
    grad_clip: float | None = 1.0,
    loss_chunk_size: int = DEFAULT_LOSS_CHUNK_SIZE,
) -> dict[str, Any]:
    """Backpropagate one GRPO loss and optional text CE, then step once.

    The GRPO graph is released before the optional text forward.  All
    pre-step exceptions and all non-finite losses/gradients clear policy
    gradients and leave both optimizer and scheduler untouched.
    """

    resolved_device = _preflight_cycle(
        policy,
        reference,
        ocr_batch,
        text_batch,
        reward_adapter,
        admission,
        grpo_config,
        optimizer,
        kl_trial=kl_trial,
        text_weight=text_weight,
        device=device,
        precision=precision,
        grad_clip=grad_clip,
        loss_chunk_size=loss_chunk_size,
    )

    policy.train()
    optimizer.zero_grad(set_to_none=True)
    grpo_metrics: dict[str, float]
    try:
        with _autocast_context(precision, resolved_device):
            live_grpo_loss, grpo_metrics = anyres_grpo_compute_loss(
                policy,
                reference,
                ocr_batch,
                reward_adapter,
                admission,
                grpo_config,
                resolved_device,
            )
        _require_scalar_finite_loss(live_grpo_loss, "anyres GRPO")
        active_group_fraction = grpo_metrics.get("active_group_frac")
        if (
            isinstance(active_group_fraction, bool)
            or not isinstance(active_group_fraction, (int, float))
            or not math.isfinite(float(active_group_fraction))
            or not 0.0 <= float(active_group_fraction) <= 1.0
        ):
            raise ValueError("GRPO metrics have no valid active_group_frac")
        if float(active_group_fraction) == 0.0:
            optimizer.zero_grad(set_to_none=True)
            return {
                "loss": float(live_grpo_loss.detach()),
                "grpo_loss": float(live_grpo_loss.detach()),
                "text_loss": 0.0,
                "weighted_text_loss": 0.0,
                "grad_norm": 0.0,
                "text_weight": float(text_weight),
                "used_text_replay": False,
                "optimizer_steps": 0,
                "scheduler_steps": 0,
                "grpo_compute_calls": 1,
                "stepped": False,
                "skip_reason": "no_active_reward_groups",
                "grpo_metrics": dict(grpo_metrics),
                "optimizer_groups": _optimizer_group_records(optimizer),
                "kl_ablation": {
                    **kl_trial.canonical_payload,
                    "canonical_sha256": kl_trial.canonical_sha256,
                },
            }
        _backward(live_grpo_loss, scaler)
        grpo_loss_value = live_grpo_loss.detach()
        del live_grpo_loss
        _require_reference_grad_free(reference)

        if text_batch is None:
            text_loss_value = grpo_loss_value.new_zeros(())
        else:
            with _autocast_context(precision, resolved_device):
                live_text_loss = _forward_text_ce(
                    policy,
                    text_batch,
                    device=resolved_device,
                    loss_chunk_size=loss_chunk_size,
                )
                weighted_text_loss = live_text_loss * float(text_weight)
            _require_scalar_finite_loss(live_text_loss, "text replay")
            _backward(weighted_text_loss, scaler)
            text_loss_value = live_text_loss.detach()
            del live_text_loss, weighted_text_loss
            _require_reference_grad_free(reference)
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise

    total_loss_value = grpo_loss_value + float(text_weight) * text_loss_value
    if scaler is not None:
        scaler.unscale_(optimizer)

    trainable_parameters = [
        parameter for parameter in policy.parameters() if parameter.requires_grad
    ]
    if not any(parameter.grad is not None for parameter in trainable_parameters):
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError("anyres GRPO cycle produced no policy gradients")
    grad_norm = torch.nn.utils.clip_grad_norm_(
        trainable_parameters,
        float(grad_clip) if grad_clip is not None else float("inf"),
        error_if_nonfinite=False,
    )
    if not bool(torch.isfinite(grad_norm.detach())):
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.update()
        raise FloatingPointError("non-finite gradients in anyres GRPO cycle")
    _require_reference_grad_free(reference)

    try:
        if scaler is None:
            optimizer.step()
        else:
            scale_before = _scaler_scale(scaler)
            scaler.step(optimizer)
            scaler.update()
            scale_after = _scaler_scale(scaler)
            if (
                scale_before is not None
                and scale_after is not None
                and scale_after < scale_before
            ):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    "gradient scaler skipped the anyres GRPO optimizer step"
                )
        if scheduler is not None:
            scheduler.step()
    except BaseException:
        # This cannot roll back an optimizer that itself raised after mutating
        # state, but it guarantees no stale gradient is reused by a caller.
        optimizer.zero_grad(set_to_none=True)
        raise

    # One batched host transfer owns trainer-level scalar metrics.  GRPO's
    # internal diagnostic dictionary is already detached by its own contract.
    metric_values = torch.stack(
        (
            total_loss_value.detach().float(),
            grpo_loss_value.detach().float(),
            text_loss_value.detach().float(),
            grad_norm.detach().to(
                device=total_loss_value.device,
                dtype=torch.float32,
            ),
        )
    ).cpu().tolist()
    return {
        "loss": metric_values[0],
        "grpo_loss": metric_values[1],
        "text_loss": metric_values[2],
        "weighted_text_loss": metric_values[2] * float(text_weight),
        "grad_norm": metric_values[3],
        "text_weight": float(text_weight),
        "used_text_replay": text_batch is not None,
        "optimizer_steps": 1,
        "scheduler_steps": 1 if scheduler is not None else 0,
        "grpo_compute_calls": 1,
        "stepped": True,
        "skip_reason": None,
        "grpo_metrics": dict(grpo_metrics),
        "optimizer_groups": _optimizer_group_records(optimizer),
        "kl_ablation": {
            **kl_trial.canonical_payload,
            "canonical_sha256": kl_trial.canonical_sha256,
        },
    }


def _preflight_cycle(
    policy: nn.Module,
    reference: nn.Module | None,
    ocr_batch: Mapping[str, Any],
    text_batch: Mapping[str, Any] | None,
    reward_adapter: AnyresOCRRewardAdapter,
    admission: AnyresGRPOAdmission,
    grpo_config: GRPOConfig,
    optimizer: torch.optim.Optimizer,
    *,
    kl_trial: AnyresGRPOKLAblationTrial,
    text_weight: float,
    device: torch.device | str | None,
    precision: str,
    grad_clip: float | None,
    loss_chunk_size: int,
) -> torch.device:
    if not isinstance(policy, nn.Module):
        raise TypeError("policy must be a torch module")
    if isinstance(getattr(policy, "module", None), nn.Module):
        raise ValueError("anyres GRPO trainer supports only one unwrapped process")
    if getattr(policy, "reverse_loss_enabled", None) is not False:
        raise RuntimeError("anyres GRPO requires reverse_loss_enabled=False")
    if not isinstance(admission, AnyresGRPOAdmission):
        raise TypeError("admission must be an AnyresGRPOAdmission")
    if not isinstance(reward_adapter, AnyresOCRRewardAdapter):
        raise TypeError("reward_adapter must be an AnyresOCRRewardAdapter")
    if admission.position_contract != BOUNDARY_V1:
        raise ValueError("anyres GRPO admission must bind boundary_v1")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    if not isinstance(kl_trial, AnyresGRPOKLAblationTrial):
        raise TypeError("kl_trial must be an AnyresGRPOKLAblationTrial")
    kl_trial.require_matches(grpo_config)
    if not isinstance(ocr_batch, Mapping):
        raise TypeError("ocr_batch must be a mapping")
    _require_boundary_batch(ocr_batch, "OCR GRPO")
    if text_batch is not None:
        if not isinstance(text_batch, Mapping):
            raise TypeError("text_batch must be a mapping")
        _require_boundary_batch(text_batch, "text replay")
        if text_batch.get("dataset_contract_sha256") != (
            admission.text_replay_train_contract_sha256
        ):
            raise ValueError(
                "text replay batch differs from admitted train contract"
            )
        _validate_text_replay_batch(policy, text_batch)
    _validate_nonnegative_finite("text_weight", text_weight)
    if grad_clip is not None:
        _validate_positive_finite("grad_clip", grad_clip)
    if (
        isinstance(loss_chunk_size, bool)
        or not isinstance(loss_chunk_size, int)
        or loss_chunk_size <= 0
    ):
        raise ValueError("loss_chunk_size must be a positive integer")

    resolved_device = _resolve_device(policy, device)
    _validate_precision(precision, resolved_device)
    _validate_reference(
        policy,
        reference,
        grpo_config,
        device=resolved_device,
    )
    _validate_optimizer_ownership(policy, reference, optimizer)
    return resolved_device


def _require_boundary_batch(batch: Mapping[str, Any], kind: str) -> None:
    if batch.get("position_contract") != BOUNDARY_V1:
        raise ValueError(f"{kind} batch must explicitly declare boundary_v1")
    forbidden = sorted(
        {"word_pos", "morph_depth", "token_offsets"}.intersection(batch)
    )
    if forbidden:
        raise ValueError(
            f"{kind} batch must not materialize position fields: {forbidden}"
        )


def _validate_text_replay_batch(
    policy: nn.Module,
    batch: Mapping[str, Any],
) -> None:
    tensors: dict[str, torch.Tensor] = {}
    for key in ("input_ids", "attention_mask", "labels"):
        value = batch.get(key)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"text replay {key} must be a tensor")
        tensors[key] = value
    input_ids = tensors["input_ids"]
    attention = tensors["attention_mask"]
    labels = tensors["labels"]
    if (
        input_ids.ndim != 2
        or attention.shape != input_ids.shape
        or labels.shape != input_ids.shape
    ):
        raise ValueError("text replay tensors must have aligned [B, T] shapes")
    if input_ids.shape[0] == 0 or input_ids.shape[1] < 2:
        raise ValueError("text replay batch must contain non-empty causal rows")
    if input_ids.dtype != torch.long or labels.dtype != torch.long:
        raise ValueError("text replay input_ids and labels must use torch.long")
    if attention.dtype != torch.long:
        raise ValueError("text replay attention_mask must use torch.long")
    if not bool(((attention == 0) | (attention == 1)).all()):
        raise ValueError("text replay attention_mask must be binary")
    if attention.shape[1] > 1 and bool(
        ((attention[:, 1:] - attention[:, :-1]) > 0).any()
    ):
        raise ValueError("text replay padding must be a right-side suffix")
    if not bool((attention.sum(dim=1) >= 2).all()):
        raise ValueError("text replay rows must contain at least BOS and EOS")
    ignore_index = int(getattr(getattr(policy, "cfg", None), "ignore_index", -100))
    active = attention.bool()
    if not bool((labels[active] == input_ids[active]).all()):
        raise ValueError("text replay must supervise every active sequence token")
    if bool((labels[~active] != ignore_index).any()):
        raise ValueError("text replay padding labels must use ignore_index")

    cfg = getattr(policy, "cfg", None)
    bos_id = getattr(cfg, "bos_id", None)
    eos_id = getattr(cfg, "eos_id", None)
    if bos_id is not None and not bool((input_ids[:, 0] == int(bos_id)).all()):
        raise ValueError("text replay rows must start with model BOS")
    if eos_id is not None:
        lengths = attention.sum(dim=1).to(dtype=torch.long)
        row_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
        if not bool((input_ids[row_indices, lengths - 1] == int(eos_id)).all()):
            raise ValueError("text replay rows must end with model EOS")


def _validate_reference(
    policy: nn.Module,
    reference: nn.Module | None,
    cfg: GRPOConfig,
    *,
    device: torch.device,
) -> None:
    if cfg.kl_coef > 0 and reference is None:
        raise ValueError("positive KL requires a frozen reference model")
    if cfg.kl_coef == 0 and reference is not None:
        raise ValueError("KL=0 must not construct or pass an unused reference")
    if reference is None:
        return
    if not isinstance(reference, nn.Module):
        raise TypeError("reference must be a torch module or None")
    if isinstance(getattr(reference, "module", None), nn.Module):
        raise ValueError("anyres GRPO reference must be unwrapped")
    if reference is policy:
        raise ValueError("policy and reference must be distinct models")
    if getattr(reference, "reverse_loss_enabled", None) is not False:
        raise RuntimeError("anyres GRPO reference requires reverse_loss_enabled=False")
    if any(parameter.requires_grad for parameter in reference.parameters()):
        raise ValueError("anyres GRPO reference parameters must be frozen")
    reference_device = _single_module_device(reference, "reference")
    if reference_device != device:
        raise ValueError(
            f"reference device differs from policy: {reference_device} != {device}"
        )
    policy_parameters = dict(policy.named_parameters())
    reference_parameters = dict(reference.named_parameters())
    if set(policy_parameters) != set(reference_parameters):
        raise ValueError("policy/reference parameter names differ")
    for name, parameter in policy_parameters.items():
        other = reference_parameters[name]
        if parameter.shape != other.shape or parameter.dtype != other.dtype:
            raise ValueError(
                f"policy/reference parameter contract differs for {name}"
            )
    policy_buffers = dict(policy.named_buffers())
    reference_buffers = dict(reference.named_buffers())
    if set(policy_buffers) != set(reference_buffers):
        raise ValueError("policy/reference buffer names differ")
    for name, buffer in policy_buffers.items():
        other = reference_buffers[name]
        if buffer.shape != other.shape or buffer.dtype != other.dtype:
            raise ValueError(f"policy/reference buffer contract differs for {name}")
    _require_reference_grad_free(reference)


def _validate_optimizer_ownership(
    policy: nn.Module,
    reference: nn.Module | None,
    optimizer: torch.optim.Optimizer,
) -> None:
    optimizer_parameters: list[nn.Parameter] = []
    if not optimizer.param_groups:
        raise ValueError("optimizer must have at least one parameter group")
    for group_index, group in enumerate(optimizer.param_groups):
        params = group.get("params")
        if not isinstance(params, list):
            params = list(params)
        for parameter in params:
            if not isinstance(parameter, nn.Parameter):
                raise TypeError(
                    f"optimizer group {group_index} contains a non-Parameter"
                )
            optimizer_parameters.append(parameter)
    optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
    if len(optimizer_ids) != len(set(optimizer_ids)):
        raise ValueError("optimizer parameter groups must be disjoint")
    policy_trainable = {
        id(parameter): parameter
        for parameter in policy.parameters()
        if parameter.requires_grad
    }
    if not policy_trainable:
        raise ValueError("policy has no trainable parameters")
    if set(optimizer_ids) != set(policy_trainable):
        missing = set(policy_trainable) - set(optimizer_ids)
        extra = set(optimizer_ids) - set(policy_trainable)
        raise ValueError(
            "optimizer must exactly cover trainable policy parameters: "
            f"missing={len(missing)}, extra_or_frozen={len(extra)}"
        )
    if reference is not None:
        reference_ids = {id(parameter) for parameter in reference.parameters()}
        if reference_ids.intersection(optimizer_ids):
            raise ValueError("frozen reference parameters must not enter optimizer")


def _require_reference_grad_free(reference: nn.Module | None) -> None:
    if reference is not None and any(
        parameter.grad is not None for parameter in reference.parameters()
    ):
        raise RuntimeError("frozen reference accumulated a gradient")


def _backward(loss: torch.Tensor, scaler: Any | None) -> None:
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()


def _require_scalar_finite_loss(loss: object, kind: str) -> None:
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
        raise TypeError(f"{kind} must return one scalar loss tensor")
    if not bool(torch.isfinite(loss.detach())):
        raise FloatingPointError(f"non-finite {kind} loss")


def _resolve_device(
    policy: nn.Module,
    requested: torch.device | str | None,
) -> torch.device:
    policy_device = _single_module_device(policy, "policy")
    result = policy_device if requested is None else torch.device(requested)
    if result.type == "cuda" and result.index is None:
        result = torch.device("cuda", torch.cuda.current_device())
    if policy_device != result:
        raise ValueError(
            "requested device differs from policy parameters: "
            f"{result} != {policy_device}"
        )
    return result


def _single_module_device(module: nn.Module, role: str) -> torch.device:
    tensors = [*module.parameters(), *module.buffers()]
    if not tensors:
        raise ValueError(f"cannot infer device from parameterless {role}")
    devices = {tensor.device for tensor in tensors}
    if len(devices) != 1:
        raise ValueError(f"{role} parameters/buffers span devices: {devices}")
    return next(iter(devices))


def _validate_nonnegative_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and non-negative")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _validate_positive_finite(name: str, value: object) -> float:
    result = _validate_nonnegative_finite(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _optimizer_group_records(
    optimizer: torch.optim.Optimizer,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, group in enumerate(optimizer.param_groups):
        record: dict[str, Any] = {
            "index": index,
            "lr": float(group["lr"]),
            "parameter_tensors": len(group["params"]),
        }
        for source, target in (
            ("ocr_joint_role", "role"),
            ("ocr_joint_decay", "decay"),
            ("ocr_joint_base_lr", "base_lr"),
        ):
            if source in group:
                value = group[source]
                record[target] = float(value) if source.endswith("_lr") else value
        records.append(record)
    return records


__all__ = [
    "AnyresGRPOKLAblationTrial",
    "train_anyres_grpo_cycle",
]
