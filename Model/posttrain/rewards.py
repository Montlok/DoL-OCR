# -*- coding: utf-8 -*-

"""Minimal OCR reward primitives shared by the AnyRes GRPO verifier."""

from __future__ import annotations

import math

from Model.ocr.metrics import cer, grapheme_clusters


def grapheme_cer_value(
    response: str,
    reference: str,
    *,
    normalize: bool = False,
    backend: str = "auto",
) -> float:
    """Return per-sample grapheme CER for an OCR transcript.

    normalize=False matches the locked evaluator's headline raw-grapheme
    metric. The backend argument remains compatible with the metrics API,
    whose only supported implementation is deterministic Python folding.
    """

    return float(
        cer(
            [response],
            [reference],
            normalize=normalize,
            backend=backend,
            unit="grapheme",
        )
    )


def length_excess_penalty(
    response: str,
    reference: str,
    *,
    max_ratio: float = 2.0,
) -> float:
    """Bound reward-hacking through overlong continuations to [0, 1]."""

    if not math.isfinite(max_ratio):
        raise ValueError("max length ratio must be finite")
    if max_ratio < 1.0:
        raise ValueError("max length ratio must be at least 1")
    pred_len = len(grapheme_clusters(response))
    ref_len = max(1, len(grapheme_clusters(reference)))
    excess = pred_len / ref_len - max_ratio
    return min(1.0, max(0.0, excess / max_ratio))


__all__ = ["grapheme_cer_value", "length_excess_penalty"]
