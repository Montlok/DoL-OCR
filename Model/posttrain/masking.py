# -*- coding: utf-8 -*-

"""Masking utilities for think / tool spans in SFT and RL rollouts.

Supervised OCR/text-replay builders already mask the ``tool``
*role* message because it is laid out at data-construction time. RL is harder:
during an agentic rollout the policy generates autoregressively and a
``<tool_result>...</tool_result>`` payload is **spliced in externally** between
the model's ``<tool_call>`` and its resumed generation. Those tool-result tokens
were never sampled by the policy, so they must be removed from the completion
mask before computing log-probs / advantages — otherwise GRPO scores tokens
the policy did not choose, corrupting ratios and KL.

These helpers are tokenizer-agnostic: the caller passes the *encoded* marker ids
(e.g. ``encode("<tool_result>")``), so the same logic works for any bundle.
Inline ``<think>`` and ``<tool_call>`` spans are **not** masked here — those are
emitted by the policy and remain supervised / optimized.
"""

from __future__ import annotations

import torch


def _find_subsequence(row: list[int], pattern: list[int], start: int) -> int:
    """Index of the first occurrence of ``pattern`` in ``row`` at/after ``start``.

    Returns -1 if not found. Empty patterns return -1 (nothing to match).
    """
    n, m = len(row), len(pattern)
    if m == 0 or m > n:
        return -1
    for i in range(start, n - m + 1):
        if row[i : i + m] == pattern:
            return i
    return -1


def span_mask(
    input_ids: torch.Tensor,
    marker_pairs: list[tuple[list[int], list[int]]],
) -> torch.Tensor:
    """Mask (0) every ``open...close`` span (inclusive) for each marker pair.

    Generic over span *types* so future externally-injected spans (retrieved
    context, system notes, additional tool channels) can be masked without an
    API change — just pass more ``(open_ids, close_ids)`` pairs. Spans are
    matched independently per pair and unioned into the mask.

    Args:
        input_ids: ``[B, T]`` or ``[T]`` token ids.
        marker_pairs: list of ``(open_ids, close_ids)`` encoded marker id lists.

    Returns:
        Float mask of the same shape, 1.0 except inside any matched span (0.0).
        A dangling open marker masks to the end of the row (defensive).
    """
    squeeze = input_ids.dim() == 1
    ids = input_ids.unsqueeze(0) if squeeze else input_ids
    mask = torch.ones_like(ids, dtype=torch.float32)

    for b in range(ids.shape[0]):
        row = ids[b].tolist()
        for open_ids, close_ids in marker_pairs:
            cursor = 0
            while True:
                o = _find_subsequence(row, open_ids, cursor)
                if o < 0:
                    break
                c = _find_subsequence(row, close_ids, o + len(open_ids))
                if c < 0:
                    mask[b, o:] = 0.0
                    break
                end = c + len(close_ids)
                mask[b, o:end] = 0.0
                cursor = end

    return mask.squeeze(0) if squeeze else mask


def tool_result_span_mask(
    input_ids: torch.Tensor,
    open_ids: list[int],
    close_ids: list[int],
) -> torch.Tensor:
    """Mask (0) every ``<tool_result>...</tool_result>`` span, inclusive of markers.

    Thin convenience wrapper over :func:`span_mask` for the single most common
    case. See :func:`span_mask` for the generic multi-marker version.

    Args:
        input_ids: ``[B, T]`` or ``[T]`` token ids.
        open_ids: encoded ``<tool_result>`` marker ids.
        close_ids: encoded ``</tool_result>`` marker ids.

    Returns:
        Float mask of the same shape, 1.0 everywhere except inside tool-result
        spans (which are 0.0). If an open marker has no matching close, the span
        is masked to the end of the row (defensive: never optimize a dangling,
        externally injected payload).
    """
    return span_mask(input_ids, [(open_ids, close_ids)])


def completion_mask_excluding_tool_results(
    base_completion_mask: torch.Tensor,
    input_ids: torch.Tensor,
    open_ids: list[int],
    close_ids: list[int],
) -> torch.Tensor:
    """Combine a base completion mask with tool-result exclusion.

    ``base_completion_mask`` marks the policy-generated region (1 after the
    prompt). The returned mask additionally zeros any externally injected
    tool-result span, yielding the tokens the policy actually produced.
    """
    span = tool_result_span_mask(input_ids, open_ids, close_ids)
    return base_completion_mask.to(span.dtype) * span


__all__ = [
    "completion_mask_excluding_tool_results",
    "span_mask",
    "tool_result_span_mask",
]
