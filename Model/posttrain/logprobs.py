# -*- coding: utf-8 -*-

"""Per-token log-probability helpers for preference / RL post-training.

Correctness is the whole point here: every alignment objective (DPO, KTO,
GRPO, PPO) reduces to log-probabilities of realized tokens under a policy and
a reference model. Two rules are enforced throughout:

1. **fp32 reduction.** ``log_softmax`` is computed in float32 regardless of the
   model's compute dtype, so preference gaps and KL terms are numerically
   stable.
2. **Sampling/scoring self-consistency.** The log-probs produced here (full
   forward) must match those implied by the incremental decode cache used for
   sampling. ``test_logprobs`` pins this against the bit-exact cache contract;
   if it ever drifts, advantages and KL are silently wrong.

The model argument is the native :class:`~Model.model.RDTForCausalLM`, so RL can
control the latent-depth knob (``recurrent_steps``) explicitly and keep it
identical between sampling, policy scoring, and reference scoring.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F

from Model.model import RDTForCausalLM


def logits_to_token_logprobs(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    """Log-prob of each ``target_ids`` token under ``logits``.

    Args:
        logits: ``[B, T, V]`` next-token logits.
        target_ids: ``[B, T]`` realized tokens aligned to ``logits`` positions.

    Returns:
        ``[B, T]`` log-probabilities, reduced in float32.
    """
    logp = F.log_softmax(logits.float(), dim=-1)
    return logp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)


def sequence_logprobs(
    model: RDTForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    recurrent_steps: int | None = None,
    pixel_values: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    morphology_track_table: torch.Tensor | None = None,
    completion_start: int | None = None,
    pixel_repeats: int = 1,
    position_contract: str | None = None,
    visual_features: torch.Tensor | None = None,
    detail_memory: torch.Tensor | None = None,
    detail_cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-token log-probs of the realized next token.

    Returns ``logp`` of shape ``[B, T-1]`` where
    ``logp[:, t] = log p(input_ids[:, t+1] | input_ids[:, : t+1])``,
    computed from a single full forward in float32.
    """
    if completion_start is not None:
        if type(completion_start) is not int:
            raise TypeError("completion_start must be an integer or None")
        max_start = input_ids.shape[1] - 1
        if completion_start < 0 or completion_start > max_start:
            raise ValueError(
                "completion_start must satisfy "
                f"0 <= completion_start <= {max_start}"
            )

    out = model(
        input_ids,
        attention_mask=attention_mask,
        steps=recurrent_steps,
        return_logits=True,
        pixel_values=pixel_values,
        morphology_track_table=morphology_track_table,
        pixel_repeats=pixel_repeats,
        position_contract=position_contract,
        visual_features=visual_features,
        detail_memory=detail_memory,
        detail_cu_seqlens=detail_cu_seqlens,
    )
    logits = out["logits"]
    if completion_start is None:
        return logits_to_token_logprobs(logits[:, :-1, :], input_ids[:, 1:])

    scored = logits_to_token_logprobs(
        logits[:, completion_start:-1, :],
        input_ids[:, completion_start + 1 :],
    )
    prefix = scored.new_zeros((scored.shape[0], completion_start))
    return torch.cat((prefix, scored), dim=1)


def completion_logprobs(
    model: RDTForCausalLM,
    input_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    recurrent_steps: int | None = None,
    pixel_values: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    morphology_track_table: torch.Tensor | None = None,
    position_contract: str | None = None,
    visual_features: torch.Tensor | None = None,
    detail_memory: torch.Tensor | None = None,
    detail_cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Summed and per-token log-probs over completion (response) tokens only.

    Args:
        input_ids: ``[B, T]``.
        completion_mask: ``[B, T]`` with 1 on tokens whose log-prob should be
            counted (the response / generated tokens). Prompt, padding, and
            externally injected ``<tool_result>`` spans must be 0 — those are
            never produced by the policy and must not enter the objective.

    Returns:
        ``(summed_logp[B], token_logp[B, T-1])`` where ``token_logp`` is already
        masked (0 outside the completion). The mask is shifted to align with the
        next-token prediction positions.
    """
    token_logp = sequence_logprobs(
        model,
        input_ids,
        attention_mask=attention_mask,
        recurrent_steps=recurrent_steps,
        pixel_values=pixel_values,
        morphology_track_table=morphology_track_table,
        position_contract=position_contract,
        visual_features=visual_features,
        detail_memory=detail_memory,
        detail_cu_seqlens=detail_cu_seqlens,
    )
    # A token at position t+1 is predicted from position t, so the mask for the
    # predicted token lives at index t+1 -> drop the first column to align.
    target_mask = completion_mask[:, 1:].to(token_logp.dtype)
    masked = token_logp * target_mask
    return masked.sum(dim=-1), masked


def token_logprobs_with_mask(
    model: RDTForCausalLM,
    input_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    recurrent_steps: int | None = None,
    pixel_values: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    morphology_track_table: torch.Tensor | None = None,
    completion_start: int | None = None,
    pixel_repeats: int = 1,
    position_contract: str | None = None,
    visual_features: torch.Tensor | None = None,
    detail_memory: torch.Tensor | None = None,
    detail_cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unmasked per-token log-probs plus the aligned completion mask.

    Unlike :func:`completion_logprobs`, the returned ``token_logp`` is **not**
    zeroed outside the completion — GRPO needs the raw policy/old/ref log-probs
    to form importance ratios, and applies the mask only at the final reduction.

    Returns:
        ``(token_logp[B, T-1], shifted_mask[B, T-1])``.
    """
    token_logp = sequence_logprobs(
        model,
        input_ids,
        attention_mask=attention_mask,
        recurrent_steps=recurrent_steps,
        pixel_values=pixel_values,
        morphology_track_table=morphology_track_table,
        completion_start=completion_start,
        pixel_repeats=pixel_repeats,
        position_contract=position_contract,
        visual_features=visual_features,
        detail_memory=detail_memory,
        detail_cu_seqlens=detail_cu_seqlens,
    )
    shifted_mask = completion_mask[:, 1:].to(token_logp.dtype)
    return token_logp, shifted_mask
