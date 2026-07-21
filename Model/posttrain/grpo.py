# -*- coding: utf-8 -*-

"""Group Relative Policy Optimization (GRPO) for the RDT model.

GRPO (Shao et al., 2024) replaces PPO's value model with a *group-relative*
baseline: for each prompt, sample a group of responses, score them with a
(here verifiable) reward, and normalize rewards within the group to form
advantages. The token-level surrogate is PPO-style with ratio clipping plus a
k3 KL penalty toward a frozen reference.

Rigor rules enforced here:
- Advantages are normalized **within each prompt group** (zero mean / unit std).
- The loss is reduced over **completion tokens only** (token-aligned mask); the
  prompt and any externally injected ``<tool_result>`` spans never contribute,
  matching the SFT masking contract.
- Policy, ``old`` (sampling), and reference log-probs must all be produced at
  the same latent depth (``recurrent_steps``) so ratios and KL are coherent.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch

from Model.model import RDTForCausalLM
from Model.posttrain.logprobs import token_logprobs_with_mask
from Model.posttrain.masking import completion_mask_excluding_tool_results

PixelValues = torch.Tensor | Mapping[str, torch.Tensor]


def _unwrap_model(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def _is_fsdp(model) -> bool:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        return isinstance(model, FSDP)
    except ImportError:  # pragma: no cover
        return False


def generate_sequences(model, *args, **kwargs):
    """Call native generation through DDP/FSDP wrappers safely."""

    if _is_fsdp(model):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        with FSDP.summon_full_params(
            model, recurse=True, writeback=False
        ):
            return _unwrap_model(model).generate(*args, **kwargs)
    return _unwrap_model(model).generate(*args, **kwargs)


def group_normalized_advantages(
    rewards: torch.Tensor,
    group_size: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize rewards within each prompt group to zero mean / unit std.

    Args:
        rewards: ``[N]`` where ``N = num_prompts * group_size``, group-contiguous.
        group_size: responses sampled per prompt.

    Returns:
        ``[N]`` advantages.
    """
    if group_size <= 1:
        raise ValueError("group_size must be at least 2 for group-relative rewards")
    if rewards.numel() % group_size != 0:
        raise ValueError("rewards length must be divisible by group_size")
    groups = rewards.view(-1, group_size)
    mean = groups.mean(dim=1, keepdim=True)
    std = groups.std(dim=1, keepdim=True, unbiased=False)
    adv = (groups - mean) / (std + eps)
    return adv.reshape(-1)


@dataclass
class GRPOConfig:
    clip_eps: float = 0.2
    kl_coef: float = 0.04
    recurrent_steps: int | None = None
    group_size: int = 4
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_p: float | None = None
    tool_result_open_ids: list[int] | None = None
    tool_result_close_ids: list[int] | None = None
    log_ratio_clip: float = 20.0

    def __post_init__(self) -> None:
        if not 0.0 < self.clip_eps < 1.0:
            raise ValueError("clip_eps must be in (0, 1)")
        if not math.isfinite(self.kl_coef) or self.kl_coef < 0:
            raise ValueError("kl_coef must be non-negative")
        if self.group_size <= 1:
            raise ValueError("group_size must be at least 2")
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if self.temperature != 1.0 or self.top_p is not None:
            raise ValueError(
                "GRPO requires temperature=1.0 and top_p=None until sampling-"
                "transformed log-probs are implemented; otherwise rollout and "
                "scoring distributions differ"
            )
        if not math.isfinite(self.log_ratio_clip) or self.log_ratio_clip <= 0:
            raise ValueError("log_ratio_clip must be positive")


def repeat_pixel_values(
    pixel_values: PixelValues | None,
    repeats: int,
) -> PixelValues | None:
    """Repeat one prompt's visual payload across a GRPO sample group.

    OMVT bounding boxes are shared ``[N, 4]`` geometry and therefore must not be
    repeated.  Image/patch tensors carry a leading batch dimension and are
    expanded from one to ``repeats``.  Already-grouped payloads are accepted so
    callers can pre-batch explicitly.
    """

    if pixel_values is None:
        return None
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    def _repeat_tensor(value: torch.Tensor, *, shared: bool = False) -> torch.Tensor:
        if shared:
            return value
        if value.ndim == 0:
            raise ValueError("pixel tensor must have at least one dimension")
        if value.shape[0] == repeats:
            return value
        if value.shape[0] != 1:
            raise ValueError(
                "pixel payload batch must be one prompt or one full group; "
                f"got leading dimension {value.shape[0]} for group={repeats}"
            )
        return value.expand(repeats, *value.shape[1:])

    if isinstance(pixel_values, Mapping):
        return {
            key: _repeat_tensor(value, shared=key.endswith("_bbox"))
            for key, value in pixel_values.items()
        }
    # Legacy MLP pixels may be a single unbatched ``[N, P]`` tensor.  A leading
    # singleton is unambiguous for the OCR path and preserves the old B=1 API.
    value = pixel_values
    if value.ndim == 2:
        value = value.unsqueeze(0)
    return _repeat_tensor(value)


def combine_pixel_values(
    payloads: Sequence[PixelValues | None],
) -> PixelValues | None:
    """Concatenate already-grouped visual payloads for one scoring forward."""

    if not payloads or all(payload is None for payload in payloads):
        return None
    if any(payload is None for payload in payloads):
        raise ValueError("cannot combine a mixture of visual and text-only payloads")
    concrete = [payload for payload in payloads if payload is not None]
    first = concrete[0]
    if isinstance(first, Mapping):
        if not all(isinstance(payload, Mapping) for payload in concrete):
            raise TypeError("pixel payload types differ across prompts")
        keys = set(first)
        if any(set(payload) != keys for payload in concrete):
            raise ValueError("OMVT pixel payload keys differ across prompts")
        combined: dict[str, torch.Tensor] = {}
        for key in first:
            values = [payload[key] for payload in concrete]
            if key.endswith("_bbox"):
                anchor = values[0]
                if any(value is not anchor and not torch.equal(value, anchor) for value in values[1:]):
                    raise ValueError(f"shared OMVT geometry differs for {key}")
                combined[key] = anchor
            else:
                combined[key] = torch.cat(values, dim=0)
        return combined
    if any(isinstance(payload, Mapping) for payload in concrete):
        raise TypeError("pixel payload types differ across prompts")
    return torch.cat([payload for payload in concrete if isinstance(payload, torch.Tensor)], dim=0)


def _completion_action_mask(
    sequences: torch.Tensor,
    prompt_length: int,
    eos_id: int | None,
) -> torch.Tensor:
    mask = torch.zeros_like(sequences, dtype=torch.float32)
    completion = sequences[:, prompt_length:]
    if eos_id is None:
        mask[:, prompt_length:] = 1.0
    else:
        is_eos = completion == eos_id
        eos_count = is_eos.cumsum(dim=-1)
        mask[:, prompt_length:] = (
            (eos_count == 0) | (is_eos & (eos_count == 1))
        ).float()
    return mask


def grpo_loss(
    policy_token_logp: torch.Tensor,
    old_token_logp: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    ref_token_logp: torch.Tensor | None = None,
    cfg: GRPOConfig | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Token-level GRPO surrogate loss.

    Args:
        policy_token_logp: ``[B, L]`` current-policy per-token log-probs.
        old_token_logp: ``[B, L]`` sampling-policy log-probs (no grad).
        advantages: ``[B]`` per-sequence advantages (broadcast over tokens).
        completion_mask: ``[B, L]`` 1 on response tokens to optimize.
        ref_token_logp: ``[B, L]`` frozen reference log-probs for the KL term.

    Returns:
        ``(loss, metrics)``.
    """
    cfg = cfg or GRPOConfig()
    if cfg.group_size <= 1:
        raise ValueError("cfg.group_size must be at least 2")
    mask = completion_mask.to(
        device=policy_token_logp.device, dtype=policy_token_logp.dtype
    )
    advantages = advantages.to(
        device=policy_token_logp.device,
        dtype=policy_token_logp.dtype,
    )

    if cfg.log_ratio_clip <= 0:
        raise ValueError("cfg.log_ratio_clip must be positive")
    raw_log_ratio = policy_token_logp - old_token_logp
    bounded_log_ratio = raw_log_ratio.clamp(
        -cfg.log_ratio_clip, cfg.log_ratio_clip
    )
    log_ratio = raw_log_ratio + (bounded_log_ratio - raw_log_ratio).detach()
    ratio = torch.exp(log_ratio)
    adv = advantages.unsqueeze(1)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
    pg = -torch.min(surr1, surr2)

    if ref_token_logp is not None and cfg.kl_coef:
        # k3 (unbiased, non-negative) KL estimator.
        raw_diff = ref_token_logp - policy_token_logp
        bounded_diff = raw_diff.clamp(-cfg.log_ratio_clip, cfg.log_ratio_clip)
        # Bound the forward exponential without creating a zero-gradient dead
        # zone outside the numeric range. Gradient clipping in the trainer then
        # controls the finite straight-through corrective signal.
        diff = raw_diff + (bounded_diff - raw_diff).detach()
        kl = torch.expm1(diff) - diff
        kl_numeric_clipped = (raw_diff != bounded_diff).to(mask.dtype)
    else:
        kl = torch.zeros_like(pg)
        kl_numeric_clipped = torch.zeros_like(pg)
    per_token = pg + cfg.kl_coef * kl
    per_sequence_denom = mask.sum(dim=-1).clamp(min=1.0)
    per_sequence_loss = (per_token * mask).sum(dim=-1) / per_sequence_denom
    loss = per_sequence_loss.mean()

    def _masked_sequence_mean(values: torch.Tensor) -> torch.Tensor:
        per_sequence = (values * mask).sum(dim=-1) / per_sequence_denom
        return per_sequence.mean()

    clipped = ((ratio < 1.0 - cfg.clip_eps) | (ratio > 1.0 + cfg.clip_eps)).to(
        mask.dtype
    )
    numeric_clipped = (raw_log_ratio != bounded_log_ratio).to(mask.dtype)

    metrics = {
        "loss": float(loss.detach()),
        "pg": float(_masked_sequence_mean(pg).detach()),
        "kl": float(_masked_sequence_mean(kl).detach()),
        "ratio_mean": float(_masked_sequence_mean(ratio).detach()),
        "adv_mean": float(advantages.mean().detach()),
        "adv_abs_mean": float(advantages.abs().mean().detach()),
        "clip_frac": float(_masked_sequence_mean(clipped).detach()),
        "numeric_clip_frac": float(_masked_sequence_mean(numeric_clipped).detach()),
        "kl_numeric_clip_frac": float(
            _masked_sequence_mean(kl_numeric_clipped).detach()
        ),
        "completion_tokens": float(mask.sum().detach()),
    }
    return loss, metrics


def sample_group(
    model: RDTForCausalLM,
    prompt_ids: torch.Tensor,
    cfg: GRPOConfig,
    eos_id: int | None = None,
    pad_id: int | None = None,
    pixel_values: PixelValues | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``cfg.group_size`` continuations for a single prompt.

    Args:
        prompt_ids: ``[P]`` 1-D prompt token ids.

    Returns:
        ``(sequences[G, P+n], completion_mask[G, P+n])`` where the mask is 1 on
        sampled tokens (everything after the prompt that is not padding).
    """
    if cfg.group_size <= 1:
        raise ValueError("cfg.group_size must be at least 2")
    if prompt_ids.dim() != 1:
        raise ValueError("prompt_ids must be 1-D")
    p = prompt_ids.shape[0]
    batch = prompt_ids.unsqueeze(0).expand(cfg.group_size, -1).contiguous()
    grouped_pixels = repeat_pixel_values(pixel_values, cfg.group_size)
    seqs = generate_sequences(
        model,
        batch,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        eos_id=eos_id,
        pad_id=pad_id,
        recurrent_steps=cfg.recurrent_steps,
        pixel_values=grouped_pixels,
    )
    # A sampled PAD is still a policy action. It may only be ignored after an
    # EOS has finished the row; masking by token value lets the policy hide
    # arbitrary actions from both the objective and OCR reward.
    mask = _completion_action_mask(seqs, p, eos_id)
    return seqs, mask


def _grpo_compute_loss_impl(
    policy: RDTForCausalLM,
    reference: RDTForCausalLM | None,
    prompts: Sequence[torch.Tensor],
    reward_fn: Callable[[Sequence[str], int], torch.Tensor],
    decode: Callable[[torch.Tensor], str],
    cfg: GRPOConfig | None = None,
    eos_id: int | None = None,
    pad_id: int | None = None,
    pixel_values: Sequence[PixelValues | None] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Sample groups and build the differentiable GRPO loss for a prompt batch.

    For each prompt: sample a group, snapshot the sampling log-probs (``old``),
    score the group with ``reward_fn`` (decoded via ``decode``), normalize
    advantages within the group, and accumulate the clipped surrogate + KL
    toward ``reference`` (frozen). The optimizer step is left to the caller so
    trainers can own grad accumulation / clipping / scheduling.

    ``reward_fn(responses, prompt_index) -> [group_size]`` keeps reward logic
    (references, verifiers) outside the optimizer.

    Returns ``(loss, metrics)`` with ``loss`` averaged over prompts.
    """
    cfg = cfg or GRPOConfig()
    if cfg.group_size <= 1:
        raise ValueError("cfg.group_size must be at least 2")
    if not prompts:
        raise ValueError("prompts cannot be empty")
    if pixel_values is None:
        pixel_values = [None] * len(prompts)
    elif len(pixel_values) != len(prompts):
        raise ValueError(
            "prompts/pixel_values length mismatch: "
            f"{len(prompts)} != {len(pixel_values)}"
        )
    _unwrap_model(policy).reverse_loss_enabled = False

    grouped_payloads = [
        repeat_pixel_values(payload, cfg.group_size) for payload in pixel_values
    ]
    prompt_lengths = {int(prompt.shape[0]) for prompt in prompts}
    rollout_groups: list[tuple[torch.Tensor, torch.Tensor]] | None = None
    if len(prompt_lengths) == 1:
        prompt_length = next(iter(prompt_lengths))
        prompt_batch = torch.stack(list(prompts), dim=0).repeat_interleave(
            cfg.group_size, dim=0
        )
        rollout_pixels = combine_pixel_values(grouped_payloads)
        rollout_sequences = generate_sequences(
            policy,
            prompt_batch,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            eos_id=eos_id,
            pad_id=pad_id,
            recurrent_steps=cfg.recurrent_steps,
            pixel_values=rollout_pixels,
        )
        rollout_mask = _completion_action_mask(
            rollout_sequences, prompt_length, eos_id
        )
        rollout_groups = [
            (
                rollout_sequences[
                    idx * cfg.group_size : (idx + 1) * cfg.group_size
                ],
                rollout_mask[idx * cfg.group_size : (idx + 1) * cfg.group_size],
            )
            for idx in range(len(prompts))
        ]

    agg: dict[str, float] = {}
    sequence_groups: list[torch.Tensor] = []
    objective_masks: list[torch.Tensor] = []
    attention_masks: list[torch.Tensor] = []
    advantage_groups: list[torch.Tensor] = []
    grouped_pixel_payloads: list[PixelValues | None] = []

    for idx, prompt in enumerate(prompts):
        grouped_pixels = grouped_payloads[idx]
        if rollout_groups is not None:
            seqs, mask = rollout_groups[idx]
        else:
            seqs, mask = sample_group(
                policy,
                prompt,
                cfg,
                eos_id=eos_id,
                pad_id=pad_id,
                pixel_values=grouped_pixels,
            )
        # Prompt positions and every sampled action through EOS are valid model
        # context, including a PAD id sampled before EOS. Only generation-added
        # post-EOS fill is right padding. Tool-result masking below affects the
        # objective, not the causal context seen by subsequent tokens.
        score_attention = torch.ones_like(seqs, dtype=torch.long)
        score_attention[:, prompt.shape[0]:] = mask[:, prompt.shape[0]:].long()
        if cfg.tool_result_open_ids and cfg.tool_result_close_ids:
            # Externally injected tool-result tokens were not sampled by the
            # policy -> exclude them from the RL objective.
            mask = completion_mask_excluding_tool_results(
                mask, seqs, cfg.tool_result_open_ids, cfg.tool_result_close_ids
            )
        responses = [decode(seqs[g, prompt.shape[0]:]) for g in range(seqs.shape[0])]
        agg["unique_response_frac"] = agg.get("unique_response_frac", 0.0) + (
            len(set(responses)) / cfg.group_size
        )
        agg["completion_length"] = agg.get("completion_length", 0.0) + float(
            mask.sum(dim=-1).mean()
        )
        if eos_id is not None:
            completion_ids = seqs[:, prompt.shape[0]:]
            agg["eos_rate"] = agg.get("eos_rate", 0.0) + float(
                (completion_ids == eos_id).any(dim=-1).float().mean()
            )
        rewards = reward_fn(responses, idx).to(device=seqs.device, dtype=torch.float32)
        if rewards.numel() != cfg.group_size:
            raise ValueError(
                "reward_fn must return one scalar per sampled response "
                f"(got {rewards.numel()} for group_size={cfg.group_size})"
            )
        adv = group_normalized_advantages(rewards, cfg.group_size)
        reward_std = rewards.std(unbiased=False)
        agg["reward_mean"] = agg.get("reward_mean", 0.0) + float(rewards.mean())
        agg["reward_std"] = agg.get("reward_std", 0.0) + float(reward_std)
        agg["reward_min"] = agg.get("reward_min", 0.0) + float(rewards.min())
        agg["reward_max"] = agg.get("reward_max", 0.0) + float(rewards.max())
        agg["degenerate_group"] = agg.get("degenerate_group", 0.0) + float(
            reward_std <= 1e-6
        )
        sequence_groups.append(seqs)
        objective_masks.append(mask)
        attention_masks.append(score_attention)
        advantage_groups.append(adv)
        grouped_pixel_payloads.append(grouped_pixels)

    # Rollouts can have different lengths because each prompt group stops when
    # all its responses emit EOS. Right-pad once, then score every prompt/group
    # in one policy forward and one reference forward. This preserves the exact
    # per-sequence objective while avoiding prompts_per_step repeated 1B-model
    # passes and repeated OMVT tower launches.
    score_pad_id = (
        int(pad_id)
        if pad_id is not None
        else int(getattr(_unwrap_model(policy).cfg, "pad_id", 0))
    )
    max_len = max(group.shape[1] for group in sequence_groups)

    def _right_pad(group: torch.Tensor, value: int | float) -> torch.Tensor:
        width = max_len - group.shape[1]
        if width == 0:
            return group
        tail = torch.full(
            (group.shape[0], width),
            value,
            dtype=group.dtype,
            device=group.device,
        )
        return torch.cat([group, tail], dim=1)

    sequences = torch.cat(
        [_right_pad(group, score_pad_id) for group in sequence_groups], dim=0
    )
    objective_mask = torch.cat(
        [_right_pad(group, 0.0) for group in objective_masks], dim=0
    )
    score_attention = torch.cat(
        [_right_pad(group, 0) for group in attention_masks], dim=0
    )
    advantages = torch.cat(advantage_groups, dim=0)
    score_pixels = combine_pixel_values(grouped_pixel_payloads)

    # One fresh online update is taken per rollout batch, so the sampling policy
    # and current policy are identical before the optimizer step. Detaching the
    # differentiable score is the exact old-policy snapshot and avoids a second
    # redundant policy forward.
    policy_logp, shifted_mask = token_logprobs_with_mask(
        policy,
        sequences,
        objective_mask,
        attention_mask=score_attention,
        recurrent_steps=cfg.recurrent_steps,
        pixel_values=score_pixels,
    )
    old_logp = policy_logp.detach()
    with torch.no_grad():
        if reference is not None:
            ref_logp, _ = token_logprobs_with_mask(
                reference,
                sequences,
                objective_mask,
                attention_mask=score_attention,
                recurrent_steps=cfg.recurrent_steps,
                pixel_values=score_pixels,
            )
        else:
            ref_logp = None
    total, loss_metrics = grpo_loss(
        policy_logp,
        old_logp,
        advantages,
        shifted_mask,
        ref_token_logp=ref_logp,
        cfg=cfg,
    )
    n = len(prompts)
    out = {k: v / n for k, v in agg.items()}
    out.update(loss_metrics)
    out["completion_tokens_total"] = loss_metrics["completion_tokens"]
    out["completion_tokens"] = loss_metrics["completion_tokens"] / n
    out["loss"] = float(total.detach())
    return total, out


def grpo_compute_loss(
    policy: RDTForCausalLM,
    reference: RDTForCausalLM | None,
    prompts: Sequence[torch.Tensor],
    reward_fn: Callable[[Sequence[str], int], torch.Tensor],
    decode: Callable[[torch.Tensor], str],
    cfg: GRPOConfig | None = None,
    eos_id: int | None = None,
    pad_id: int | None = None,
    pixel_values: Sequence[PixelValues | None] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute GRPO in the same dropout-free mode used by rollout sampling.

    ``generate()`` samples in evaluation mode. Scoring the sampled actions in
    training mode would make ``old_logp = policy_logp.detach()`` incorrect for
    any checkpoint with text/vision dropout enabled. Evaluation mode does not
    disable autograd, so the policy score remains fully differentiable; the
    caller's original module modes are restored before returning.
    """

    policy_was_training = policy.training
    reference_was_training = reference.training if reference is not None else False
    policy.eval()
    if reference is not None:
        reference.eval()
    try:
        return _grpo_compute_loss_impl(
            policy,
            reference,
            prompts,
            reward_fn,
            decode,
            cfg=cfg,
            eos_id=eos_id,
            pad_id=pad_id,
            pixel_values=pixel_values,
        )
    finally:
        policy.train(policy_was_training)
        if reference is not None:
            reference.train(reference_was_training)


def grpo_step(
    policy: RDTForCausalLM,
    reference: RDTForCausalLM | None,
    prompts: Sequence[torch.Tensor],
    reward_fn: Callable[[Sequence[str], int], torch.Tensor],
    decode: Callable[[torch.Tensor], str],
    optimizer: torch.optim.Optimizer,
    cfg: GRPOConfig | None = None,
    eos_id: int | None = None,
    pad_id: int | None = None,
    grad_clip: float | None = None,
    pixel_values: Sequence[PixelValues | None] | None = None,
) -> dict[str, float]:
    """Convenience one-call GRPO update: compute loss + optimizer step.

    Wraps :func:`grpo_compute_loss`; trainers that need grad accumulation or
    custom scheduling should call :func:`grpo_compute_loss` directly and own the
    optimizer step. ``grad_clip`` clips the global grad norm before stepping.
    """
    total, out = grpo_compute_loss(
        policy, reference, prompts, reward_fn, decode,
        cfg=cfg, eos_id=eos_id, pad_id=pad_id, pixel_values=pixel_values,
    )
    optimizer.zero_grad()
    total.backward()
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
    optimizer.step()
    return out


__all__ = [
    "GRPOConfig",
    "grpo_compute_loss",
    "grpo_loss",
    "grpo_step",
    "generate_sequences",
    "combine_pixel_values",
    "group_normalized_advantages",
    "repeat_pixel_values",
    "sample_group",
]
