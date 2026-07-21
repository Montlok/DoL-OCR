# -*- coding: utf-8 -*-

"""Direct Preference Optimization (DPO) for the RDT model.

Offline preference alignment: no rollouts, just four log-probabilities per
example (policy/reference x chosen/rejected). The objective is the standard
DPO loss (Rafailov et al., 2023):

    loss = -E[ log sigmoid( beta * ( (pi_c - pi_r) - (ref_c - ref_r) ) ) ]

Rigor rules enforced here:
- All log-probs are completion-only and float32 (see
  :mod:`Model.posttrain.logprobs`): prompt, padding, and externally injected
  ``<tool_result>`` spans are masked out by ``completion_mask``.
- The reference model must be frozen and scored at the **same latent depth**
  (``recurrent_steps``) as the policy; otherwise the implicit KL is meaningless.
- Optional length normalization (length-averaged log-probs) counters the
  well-known DPO length bias.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from Model.model import RDTForCausalLM

from .logprobs import completion_logprobs


@dataclass
class DPOConfig:
    beta: float = 0.1
    length_normalize: bool = False
    recurrent_steps: int | None = None


def _sequence_score(
    model: RDTForCausalLM,
    input_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    attention_mask: torch.Tensor | None,
    recurrent_steps: int | None,
    length_normalize: bool,
    pixel_values: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    summed, token_logp = completion_logprobs(
        model,
        input_ids,
        completion_mask,
        attention_mask=attention_mask,
        recurrent_steps=recurrent_steps,
        pixel_values=pixel_values,
    )
    if not length_normalize:
        return summed
    # token_logp is already masked; recover the per-row completion length.
    lengths = completion_mask[:, 1:].to(summed.dtype).sum(dim=-1).clamp(min=1.0)
    return summed / lengths


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Pure DPO loss from four sequence log-probs.

    Returns ``(loss, metrics)`` where metrics include implicit rewards and
    preference accuracy.
    """
    pi_logratios = policy_chosen_logp - policy_rejected_logp
    ref_logratios = ref_chosen_logp - ref_rejected_logp
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits).mean()

    chosen_reward = beta * (policy_chosen_logp - ref_chosen_logp).detach()
    rejected_reward = beta * (policy_rejected_logp - ref_rejected_logp).detach()
    metrics = {
        "loss": float(loss.detach()),
        "chosen_reward": float(chosen_reward.mean()),
        "rejected_reward": float(rejected_reward.mean()),
        "reward_margin": float((chosen_reward - rejected_reward).mean()),
        "accuracy": float((chosen_reward > rejected_reward).float().mean()),
    }
    return loss, metrics


def dpo_step(
    policy: RDTForCausalLM,
    reference: RDTForCausalLM,
    chosen_ids: torch.Tensor,
    chosen_mask: torch.Tensor,
    rejected_ids: torch.Tensor,
    rejected_mask: torch.Tensor,
    cfg: DPOConfig,
    chosen_attn: torch.Tensor | None = None,
    rejected_attn: torch.Tensor | None = None,
    chosen_pixel_values: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    rejected_pixel_values: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the DPO loss for one preference batch.

    ``*_ids`` are full prompt+response sequences; ``*_mask`` mark the response
    (completion) tokens to score. The reference model is scored under
    ``torch.no_grad`` at the same latent depth as the policy.
    """
    policy_chosen = _sequence_score(
        policy, chosen_ids, chosen_mask, chosen_attn, cfg.recurrent_steps,
        cfg.length_normalize, chosen_pixel_values,
    )
    policy_rejected = _sequence_score(
        policy, rejected_ids, rejected_mask, rejected_attn, cfg.recurrent_steps,
        cfg.length_normalize, rejected_pixel_values,
    )
    with torch.no_grad():
        ref_chosen = _sequence_score(
            reference, chosen_ids, chosen_mask, chosen_attn, cfg.recurrent_steps,
            cfg.length_normalize, chosen_pixel_values,
        )
        ref_rejected = _sequence_score(
            reference, rejected_ids, rejected_mask, rejected_attn,
            cfg.recurrent_steps, cfg.length_normalize, rejected_pixel_values,
        )
    return dpo_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=cfg.beta
    )


__all__ = ["DPOConfig", "dpo_loss", "dpo_step"]
