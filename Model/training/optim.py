# -*- coding: utf-8 -*-

"""Optimizer + LR schedule builders.

Optimizers:
* ``adamw`` (default) — torch AdamW, optionally with the **Adam-atan2** update
  (arXiv:2407.05872): the eps-guarded ratio ``m / (sqrt(v) + eps)`` is replaced
  by ``atan2(m, b*sqrt(v))``, which is scale-invariant and immune to bf16
  underflow in the denominator. Drop-in; ``adam_eps`` becomes unused.
* ``muon`` — **experimental**, Moonlight/Muon (arXiv:2502.16982): orthogonalize
  the momentum of 2-D weight matrices via a Newton–Schulz iteration; everything
  else (embeddings, lm_head, norms, biases) stays on AdamW. Caveat for this
  model: shared recurrent weights receive ~r× amplified gradients, so Muon's
  interaction with the RDT loop is an open question — opt-in only.

Schedules: ``cosine`` (default) and ``wsd`` (warmup-stable-decay,
MiniCPM arXiv:2404.06395).
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable

import torch
import torch.nn as nn

from Model.config import TrainingConfig

try:  # optional: keep optim usable even if rmsnorm import fails for any reason
    from Model.layers.rmsnorm import RMSNorm as _RMSNorm
except Exception:  # pragma: no cover - defensive
    _RMSNorm = None

_NORM_TYPES: tuple[type, ...] = (
    (nn.LayerNorm, nn.GroupNorm, _RMSNorm) if _RMSNorm is not None else (nn.LayerNorm, nn.GroupNorm)
)


def _split_params_by_decay(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Return ``(decay, no_decay)`` parameter lists using the repo policy."""

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    seen: set[int] = set()

    for module in model.modules():
        is_norm = isinstance(module, _NORM_TYPES)

        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            key = id(param)
            if key in seen:
                continue
            seen.add(key)

            no_wd = getattr(param, "_no_weight_decay", False)
            if (
                no_wd
                or is_norm
                or name.endswith("bias")
                or isinstance(module, nn.Embedding)
                or param.ndim <= 1
            ):
                no_decay.append(param)
            else:
                decay.append(param)

    return decay, no_decay


def param_groups_with_no_decay(
    model: nn.Module,
    weight_decay: float,
) -> list[dict]:
    """Split parameters into decayed / non-decayed groups.

    Norms, biases, embeddings, and tensors marked with ``_no_weight_decay``
    (e.g. mamba ``dt_bias``, ``A_log``, ``D``) skip weight decay.
    """

    decay, no_decay = _split_params_by_decay(model)

    groups: list[dict] = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def build_optimizer(
    model: nn.Module,
    cfg: TrainingConfig,
) -> torch.optim.Optimizer:
    groups = param_groups_with_no_decay(model, cfg.weight_decay)
    if not groups:
        raise ValueError("model has no trainable parameters")

    name = cfg.optimizer.lower()

    if name == "adamw":
        return _build_adamw(groups, cfg)

    if name == "muon":
        return _build_muon(model, cfg)

    raise ValueError(f"unsupported optimizer: {cfg.optimizer}")


def build_ocr_joint_adamw(
    model: nn.Module,
    cfg: TrainingConfig,
    lm_lr: float,
    tower_lr: float,
    projector_lr: float,
    bridge_lr: float,
) -> torch.optim.AdamW:
    """Build fail-closed AdamW groups for joint OCR/text optimization.

    The v1 OMVT tower, visual projectors, native detail tower, and ragged
    bridge retain distinct base learning rates. Every group is then split by
    the repository's existing decay policy. The legacy MLP vision encoder is
    always frozen before coverage is checked.
    """

    if cfg.optimizer.lower() != "adamw" or cfg.adam_use_atan2:
        raise ValueError(
            "build_ocr_joint_adamw requires optimizer='adamw' and "
            "adam_use_atan2=False"
        )

    role_lrs = {
        "lm": _validate_joint_lr("lm_lr", lm_lr),
        "tower": _validate_joint_lr("tower_lr", tower_lr),
        "projector": _validate_joint_lr("projector_lr", projector_lr),
        "bridge": _validate_joint_lr("bridge_lr", bridge_lr),
    }

    legacy = _module_at_path(model, "vision.encoder")
    legacy_ids: set[int] = set()
    if legacy is not None:
        legacy_ids = {id(param) for param in legacy.parameters()}
        legacy.requires_grad_(False)

    roles: dict[int, str] = {}

    def claim(module: nn.Module | None, role: str, source: str) -> None:
        if module is None:
            return
        for param in module.parameters():
            param_id = id(param)
            if param_id in legacy_ids:
                raise ValueError(
                    f"legacy vision parameter is shared with {source}; refusing "
                    "to unfreeze the legacy encoder"
                )
            if not param.requires_grad:
                continue
            previous = roles.get(param_id)
            if previous is not None and previous != role:
                raise ValueError(
                    f"parameter is shared across OCR joint roles: "
                    f"{previous!r} and {role!r} ({source})"
                )
            roles[param_id] = role

    claim(_module_at_path(model, "vision.omvt.tower"), "tower", "vision.omvt.tower")
    claim(
        _module_at_path(model, "vision.omvt.projector"),
        "projector",
        "vision.omvt.projector",
    )

    tower_paths = (
        "native_detail_tower",
        "detail_tower",
        "vision.native_detail_tower",
        "vision.native_tower",
    )
    projector_paths = (
        "native_projector",
        "detail_projector",
        "vision.native_projector",
        "vision.detail_projector",
        "vision.omvt_v2.projector",
    )
    bridge_paths = (
        "vision_cross_attention",
        "native_bridge",
        "detail_bridge",
        "vision.native_bridge",
        "vision.detail_bridge",
    )
    for path in tower_paths:
        claim(_module_at_path(model, path), "tower", path)
    for path in projector_paths:
        claim(_module_at_path(model, path), "projector", path)
    for path in bridge_paths:
        claim(_module_at_path(model, path), "bridge", path)

    from Model.layers.vision_cross_attention import RaggedVisionCrossAttention
    from Model.omvt.native_tower import NativeOMVTDetailTower

    for module_name, module in model.named_modules():
        if isinstance(module, NativeOMVTDetailTower):
            claim(module, "tower", module_name or "<root native detail tower>")
        elif isinstance(module, RaggedVisionCrossAttention):
            claim(module, "bridge", module_name or "<root vision bridge>")

    trainable = [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad
    ]
    if not trainable:
        raise ValueError("model has no trainable parameters after freezing legacy vision")

    for name, param in trainable:
        param_id = id(param)
        if param_id in roles:
            continue
        if _looks_visual_parameter(name):
            raise ValueError(
                f"unclassified trainable visual parameter {name!r}; add an "
                "explicit joint-optimizer role instead of routing it to the LM"
            )
        roles[param_id] = "lm"

    trainable_ids = {id(param) for _, param in trainable}
    if set(roles) != trainable_ids:
        missing = trainable_ids - set(roles)
        extra = set(roles) - trainable_ids
        raise RuntimeError(
            "OCR joint parameter-role coverage failed: "
            f"missing={len(missing)}, non_trainable_claims={len(extra)}"
        )

    decay, no_decay = _split_params_by_decay(model)
    decay_ids = {id(param) for param in decay}
    no_decay_ids = {id(param) for param in no_decay}
    if decay_ids & no_decay_ids or (decay_ids | no_decay_ids) != trainable_ids:
        raise RuntimeError("OCR joint decay policy does not cover trainable parameters")

    ordered = [param for _, param in trainable]
    groups: list[dict] = []
    for role in ("lm", "tower", "projector", "bridge"):
        for decay_name, eligible, weight_decay in (
            ("decay", decay_ids, cfg.weight_decay),
            ("no_decay", no_decay_ids, 0.0),
        ):
            params = [
                param
                for param in ordered
                if roles[id(param)] == role and id(param) in eligible
            ]
            if params:
                groups.append(
                    {
                        "params": params,
                        "lr": role_lrs[role],
                        "weight_decay": weight_decay,
                        "ocr_joint_role": role,
                        "ocr_joint_decay": decay_name,
                        "ocr_joint_base_lr": role_lrs[role],
                    }
                )

    grouped_ids = [id(param) for group in groups for param in group["params"]]
    if len(grouped_ids) != len(set(grouped_ids)) or set(grouped_ids) != trainable_ids:
        raise RuntimeError("OCR joint optimizer groups are not disjoint and complete")

    return torch.optim.AdamW(
        groups,
        lr=role_lrs["lm"],
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
    )


def _validate_joint_lr(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _module_at_path(model: nn.Module, path: str) -> nn.Module | None:
    current: object = model
    for component in path.split("."):
        if not hasattr(current, component):
            return None
        current = getattr(current, component)
        if current is None:
            return None
    if not isinstance(current, nn.Module):
        raise TypeError(f"{path} must be an nn.Module when present")
    return current


def _looks_visual_parameter(name: str) -> bool:
    root = name.split(".", 1)[0]
    return root.startswith("vision") or root.startswith("native") or root in {
        "detail_tower",
        "detail_projector",
        "detail_bridge",
    }


def _build_adamw(groups: list[dict], cfg: TrainingConfig) -> torch.optim.Optimizer:
    if cfg.adam_use_atan2:
        return AdamAtan2(
            groups,
            lr=cfg.learning_rate,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
        )
    return torch.optim.AdamW(
        groups,
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
    )


def _build_muon(model: nn.Module, cfg: TrainingConfig) -> torch.optim.Optimizer:
    """Hybrid Muon: 2-D hidden weights on Muon, the rest on AdamW(/atan2).

    The routing preserves :func:`param_groups_with_no_decay`: tensors that would
    skip weight decay stay on AdamW with ``weight_decay=0``, and non-2-D tensors
    that should decay stay on AdamW with the configured decay. Muon only gets
    2-D decay-eligible hidden matrices; embeddings and output heads stay on the
    adaptive optimizer.
    """

    muon_params: list[nn.Parameter] = []
    adam_decay: list[nn.Parameter] = []
    adam_no_decay: list[nn.Parameter] = []

    head_ids: set[int] = set()
    for attr in ("lm_head", "reverse_head"):
        head = getattr(model, attr, None)
        if head is not None:
            head_ids |= {id(p) for p in head.parameters(recurse=False)}
    embed_ids = {
        id(p)
        for m in model.modules()
        if isinstance(m, nn.Embedding)
        for p in m.parameters(recurse=False)
    }

    decay, no_decay = _split_params_by_decay(model)
    for param in decay:
        if param.ndim == 2 and id(param) not in embed_ids and id(param) not in head_ids:
            muon_params.append(param)
        else:
            adam_decay.append(param)
    adam_no_decay.extend(no_decay)

    sub: list[torch.optim.Optimizer] = []
    if muon_params:
        sub.append(
            Muon(
                muon_params,
                lr=cfg.learning_rate,
                momentum=cfg.muon_momentum,
                weight_decay=cfg.weight_decay,
                ns_steps=cfg.muon_ns_steps,
            )
        )
    adam_groups: list[dict] = []
    if adam_decay:
        adam_groups.append({"params": adam_decay, "weight_decay": cfg.weight_decay})
    if adam_no_decay:
        adam_groups.append({"params": adam_no_decay, "weight_decay": 0.0})
    if adam_groups:
        sub.append(_build_adamw(adam_groups, cfg))

    if not sub:
        raise ValueError("model has no trainable parameters")
    return CombinedOptimizer(sub)


class AdamAtan2(torch.optim.Optimizer):
    """AdamW with the atan2 update (arXiv:2407.05872), decoupled weight decay.

    ``a``/``b`` are the paper's shape constants; ``a=1.27`` keeps the update
    magnitude close to Adam in the small-gradient regime while ``atan2`` removes
    the eps hyper-parameter and the bf16 underflow failure mode.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.0,
        a: float = 1.27,
        b: float = 1.0,
    ):
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError("betas must be in [0, 1)")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, a=a, b=b)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            wd = group["weight_decay"]
            a, b = group["a"], group["b"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.to(dtype=torch.float32)
                state = self.state[p]

                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)

                m, v = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bc1 = 1 - beta1 ** t
                bc2 = 1 - beta2 ** t
                m_hat = m / bc1
                denom = (v / bc2).sqrt_()

                if wd != 0:
                    p.mul_(1 - lr * wd)
                update = torch.atan2(m_hat, b * denom).to(dtype=p.dtype)
                p.add_(update, alpha=-lr * a)

        return loss


def zeropower_via_newtonschulz5(g: torch.Tensor, steps: int) -> torch.Tensor:
    """Quintic Newton–Schulz orthogonalization (Keller Jordan / Moonlight)."""

    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.to(dtype=torch.float32)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.t()
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        aa = x @ x.t()
        bb = b * aa + c * (aa @ aa)
        x = a * x + bb @ x
    if transposed:
        x = x.t()
    return x


class Muon(torch.optim.Optimizer):
    """Muon for 2-D weight matrices (experimental — see module docstring)."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
    ):
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not (0.0 <= momentum < 1.0):
            raise ValueError("momentum must be in [0, 1)")
        if ns_steps <= 0:
            raise ValueError("ns_steps must be positive")
        defaults = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError("Muon only supports 2-D weight matrices")
                grad = p.grad.to(dtype=torch.float32)
                state = self.state[p]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(
                        p, dtype=torch.float32
                    )
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)

                ortho = zeropower_via_newtonschulz5(buf, ns_steps).to(dtype=p.dtype)
                # RMS-matching scale so the effective step size is comparable
                # across differently-shaped matrices (Moonlight).
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5

                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.add_(ortho, alpha=-lr * scale)

        return loss


class CombinedOptimizer(torch.optim.Optimizer):
    """Drives several optimizers as one (shared ``param_groups`` view).

    Exposes the concatenation of the sub-optimizers' ``param_groups`` (the same
    dict objects), so ``LambdaLR`` scales every group and checkpoint
    save/restore round-trips through ``state_dict``.
    """

    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        if not optimizers:
            raise ValueError("CombinedOptimizer needs at least one optimizer")
        self.optimizers = optimizers
        groups = [g for opt in optimizers for g in opt.param_groups]
        super().__init__(groups, {})
        self._sync_param_groups()
        self._sync_state()

    def _sync_param_groups(self) -> None:
        self.param_groups = [
            group for opt in self.optimizers for group in opt.param_groups
        ]

    def _sync_state(self) -> None:
        self.state.clear()
        for opt in self.optimizers:
            self.state.update(opt.state)

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for opt in self.optimizers:
            opt.step()
        self._sync_state()
        return loss

    def state_dict(self):
        self._sync_state()
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict):
        for opt, sub in zip(self.optimizers, state_dict["optimizers"]):
            opt.load_state_dict(sub)
        self._sync_param_groups()
        self._sync_state()


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: TrainingConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    """LR schedule: warmup then ``cosine`` or ``wsd`` decay to min_lr_ratio."""

    warmup = max(0, cfg.warmup_steps)
    decay_total = max(1, cfg.lr_decay_steps or cfg.max_steps)
    min_ratio = cfg.min_lr_ratio

    if cfg.lr_schedule == "wsd":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, _wsd_lambda(warmup, decay_total, min_ratio, cfg)
        )

    def cosine_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(max(1, warmup))
        progress = (step - warmup) / max(1, decay_total - warmup)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_lambda)


def _wsd_lambda(warmup: int, decay_total: int, min_ratio: float, cfg: TrainingConfig):
    """Warmup → stable plateau (lr×1) → short decay tail to min_ratio."""

    span = max(1, decay_total - warmup)
    decay_start = warmup + int(round(span * cfg.wsd_stable_ratio))
    decay_span = max(1, decay_total - decay_start)
    shape = cfg.wsd_decay_shape

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(max(1, warmup))
        if step < decay_start:
            return 1.0
        progress = min(1.0, max(0.0, (step - decay_start) / decay_span))
        if shape == "linear":
            factor = 1.0 - progress
        elif shape == "cosine":
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:  # "1-sqrt" (MiniCPM)
            factor = 1.0 - math.sqrt(progress)
        return min_ratio + (1.0 - min_ratio) * factor

    return lr_lambda


def recurrent_steps_for_step(
    step: int,
    cfg: TrainingConfig,
    target_steps: int,
) -> int:
    """Recurrent depth for one optimizer step.

    Applies the optional ramp curriculum (start → target), then — when
    ``cfg.recurrent_steps_sampling == "poisson"`` — replaces the fixed depth
    with a log-normal Poisson draw centered on the ramped target (Geiping et
    al. 2025, "Scaling up Test-Time Compute with Latent Reasoning"):

        tau ~ Normal(log(t - 1) - sigma^2 / 2, sigma)
        r = 1 + Poisson(exp(tau)),  clamped to [min, max]

    so ``E[r] ≈ t``. Training across depths is what makes the model usable
    at depths other than the default at inference time; fixed-depth training
    measurably degrades both shallower and deeper evaluation.

    The draw is seeded from ``(cfg.seed, step)`` only — never the rank — so
    every rank unrolls the same depth. Under FSDP each recurrent step issues
    its own all-gathers; rank-divergent depths would deadlock collectives.
    """

    target = _ramp_target(step, cfg, target_steps)

    if cfg.recurrent_steps_sampling != "poisson":
        return target

    lo = cfg.recurrent_steps_min
    hi = cfg.recurrent_steps_max
    if hi is None:
        hi = max(2 * target_steps, lo)

    sigma = cfg.recurrent_steps_sigma
    mu = max(target - 1, 1)

    rng = random.Random((cfg.seed << 32) ^ (step * 0x9E3779B97F4A7C15))
    tau = rng.gauss(math.log(mu) - 0.5 * sigma * sigma, sigma)
    r = 1 + _poisson_sample(rng, math.exp(tau))

    return max(lo, min(r, hi))


def _ramp_target(step: int, cfg: TrainingConfig, target_steps: int) -> int:
    """Optional recurrent-depth curriculum: ramp from start → target."""

    start = cfg.recurrent_steps_start
    if start is None or cfg.recurrent_steps_ramp <= 0 or start >= target_steps:
        return target_steps
    progress = min(1.0, max(0.0, step / cfg.recurrent_steps_ramp))
    return int(round(start + (target_steps - start) * progress))


def _poisson_sample(rng: random.Random, lam: float) -> int:
    """Knuth Poisson sampler; fine for the small lambdas used here."""

    if lam <= 0.0:
        return 0
    threshold = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        p *= rng.random()
        if p <= threshold:
            return k
        k += 1


def _unused_iterable(_: Iterable[nn.Parameter]) -> None:  # pragma: no cover
    return None


__all__ = [
    "AdamAtan2",
    "CombinedOptimizer",
    "Muon",
    "build_ocr_joint_adamw",
    "build_optimizer",
    "build_scheduler",
    "param_groups_with_no_decay",
    "recurrent_steps_for_step",
    "zeropower_via_newtonschulz5",
]
