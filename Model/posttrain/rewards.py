# -*- coding: utf-8 -*-

"""Verifiable / rule-based rewards for GRPO.

Mongolian + STEM alignment is a great fit for *verifiable* rewards: no learned
reward model is needed, which removes a whole class of reward-hacking and
distribution-shift failures. Rewards here are pure functions of the decoded
response text (and an optional reference answer):

- ``grapheme_cer_reward``: dense OCR correctness against a parallel transcript.
- ``exact_match_reward``: STEM answer correctness (normalized string / numeric).
- ``mongolian_script_ratio`` / ``language_purity_reward``: fraction of script
  that is Mongolian (Cyrillic + traditional Mongolian block), penalizing
  code-switching / script leakage.
- ``format_reward``: well-formed ``<think>...</think>`` then a final answer.

``RewardConfig`` linearly combines components; ``compute_rewards`` returns a
per-response tensor ready for group normalization.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from Model.ocr.metrics import cer, grapheme_clusters
from Model.posttrain.ocr_decode import INVALID_OCR_TOKEN

# Cyrillic (Mongolian uses Cyrillic) + traditional Mongolian script block.
_CYRILLIC = (0x0400, 0x04FF)
_MONGOLIAN = (0x1800, 0x18AF)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _in_range(ch: str, rng: tuple[int, int]) -> bool:
    return rng[0] <= ord(ch) <= rng[1]


def mongolian_script_ratio(text: str) -> float:
    """Fraction of letters that are Mongolian (Cyrillic or traditional)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    mong = sum(
        1 for c in letters if _in_range(c, _CYRILLIC) or _in_range(c, _MONGOLIAN)
    )
    return mong / len(letters)


def language_purity_reward(text: str, min_ratio: float = 0.0) -> float:
    """Reward in ``[0, 1]`` equal to the Mongolian script ratio.

    ``min_ratio`` hard-zeros responses below a purity floor (useful to strongly
    discourage script leakage).
    """
    if not math.isfinite(min_ratio) or not 0.0 <= min_ratio <= 1.0:
        raise ValueError("min_ratio must be finite and in [0, 1]")
    ratio = mongolian_script_ratio(text)
    return 0.0 if ratio < min_ratio else ratio


def _normalize_answer(text: str) -> str:
    return " ".join(text.strip().lower().split())


def exact_match_reward(response: str, reference: str) -> float:
    """1.0 if normalized strings match, else 0.0."""
    return 1.0 if _normalize_answer(response) == _normalize_answer(reference) else 0.0


def numeric_match_reward(response: str, reference: str, tol: float = 1e-6) -> float:
    """1.0 if the last number in the response equals the reference number."""
    resp_nums = _NUM_RE.findall(response)
    ref_nums = _NUM_RE.findall(reference)
    if not resp_nums or not ref_nums:
        return 0.0
    return 1.0 if abs(float(resp_nums[-1]) - float(ref_nums[-1])) <= tol else 0.0


def format_reward(text: str) -> float:
    """1.0 if there is a single well-formed think block followed by content."""
    blocks = _THINK_RE.findall(text)
    if len(blocks) != 1:
        return 0.0
    after = _THINK_RE.sub("", text, count=1).strip()
    return 1.0 if after else 0.0


def grapheme_cer_value(
    response: str,
    reference: str,
    *,
    normalize: bool = False,
    backend: str = "auto",
) -> float:
    """Per-sample grapheme CER used by OCR reinforcement learning.

    ``normalize=False`` is intentional: it matches the repository's headline
    ``OCRReport.grapheme_cer`` metric, which clusters the raw transcript and
    therefore still charges rendering-variant mistakes.  A caller that wants a
    nominal-Unicode objective may opt into folding explicitly.
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


def grapheme_cer_reward(
    response: str,
    reference: str,
    *,
    cap: float | None = None,
    normalize: bool = False,
    backend: str = "auto",
) -> float:
    """Dense OCR reward ``-grapheme_CER`` with an optional tail cap.

    GRPO standardizes rewards within each sample group, and generation has a
    hard token budget, so the uncapped default remains numerically bounded and
    preserves ordering even while the policy is still poor. A finite cap is an
    explicit opt-in for unusually noisy corpora.
    """

    if cap is not None:
        if not math.isfinite(cap):
            raise ValueError("grapheme CER cap must be finite")
        if cap <= 0:
            raise ValueError("grapheme CER cap must be positive")
    value = grapheme_cer_value(
        response,
        reference,
        normalize=normalize,
        backend=backend,
    )
    return -value if cap is None else -min(value, float(cap))


def empty_response_penalty(response: str) -> float:
    """One for an empty/whitespace-only response, otherwise zero."""

    return 1.0 if not response.strip() else 0.0


def invalid_token_penalty(response: str) -> float:
    """One when reward-safe decoding observed any reserved control token."""

    return 1.0 if INVALID_OCR_TOKEN in response else 0.0


def length_excess_penalty(
    response: str,
    reference: str,
    *,
    max_ratio: float = 2.0,
) -> float:
    """Bounded penalty for reward-hacking through very long continuations.

    The value is zero while the prediction is at most ``max_ratio`` times the
    reference grapheme length, then rises linearly and caps at one.
    """

    if not math.isfinite(max_ratio):
        raise ValueError("max length ratio must be finite")
    if max_ratio < 1.0:
        raise ValueError("max length ratio must be at least 1")
    pred_len = len(grapheme_clusters(response))
    ref_len = max(1, len(grapheme_clusters(reference)))
    excess = pred_len / ref_len - max_ratio
    return min(1.0, max(0.0, excess / max_ratio))


@dataclass
class RewardConfig:
    exact_match_weight: float = 0.0
    numeric_match_weight: float = 0.0
    purity_weight: float = 0.0
    purity_min_ratio: float = 0.0
    format_weight: float = 0.0
    grapheme_cer_weight: float = 0.0
    grapheme_cer_cap: float | None = None
    grapheme_cer_normalize: bool = False
    grapheme_cer_backend: str = "auto"
    empty_response_penalty_weight: float = 0.0
    invalid_token_penalty_weight: float = 0.0
    length_excess_penalty_weight: float = 0.0
    max_length_ratio: float = 2.0

    def __post_init__(self) -> None:
        weights = {
            "exact_match_weight": self.exact_match_weight,
            "numeric_match_weight": self.numeric_match_weight,
            "purity_weight": self.purity_weight,
            "format_weight": self.format_weight,
            "grapheme_cer_weight": self.grapheme_cer_weight,
            "empty_response_penalty_weight": self.empty_response_penalty_weight,
            "invalid_token_penalty_weight": self.invalid_token_penalty_weight,
            "length_excess_penalty_weight": self.length_excess_penalty_weight,
        }
        negative = [name for name, value in weights.items() if value < 0]
        if negative:
            raise ValueError("reward weights must be non-negative: " + ", ".join(negative))
        nonfinite = [name for name, value in weights.items() if not math.isfinite(value)]
        if nonfinite:
            raise ValueError("reward weights must be finite: " + ", ".join(nonfinite))
        if not math.isfinite(self.purity_min_ratio):
            raise ValueError("purity_min_ratio must be finite")
        if not 0.0 <= self.purity_min_ratio <= 1.0:
            raise ValueError("purity_min_ratio must be in [0, 1]")
        if self.grapheme_cer_cap is not None:
            if not math.isfinite(self.grapheme_cer_cap):
                raise ValueError("grapheme_cer_cap must be finite")
            if self.grapheme_cer_cap <= 0:
                raise ValueError("grapheme_cer_cap must be positive")
        if self.grapheme_cer_backend not in {"auto", "python", "rust"}:
            raise ValueError("grapheme_cer_backend must be auto, python, or rust")
        if not math.isfinite(self.max_length_ratio):
            raise ValueError("max_length_ratio must be finite")
        if self.max_length_ratio < 1.0:
            raise ValueError("max_length_ratio must be at least 1")

    @property
    def requires_reference(self) -> bool:
        return bool(
            self.exact_match_weight
            or self.numeric_match_weight
            or self.grapheme_cer_weight
            or self.length_excess_penalty_weight
        )


def reward_breakdown_for(
    response: str,
    reference: str | None,
    cfg: RewardConfig,
) -> dict[str, float]:
    """Weighted reward components for one decoded completion."""

    if cfg.requires_reference and reference is None:
        raise ValueError("the configured reward requires a reference transcript")

    out: dict[str, float] = {}
    if cfg.grapheme_cer_weight and reference is not None:
        out["grapheme_cer"] = cfg.grapheme_cer_weight * grapheme_cer_reward(
            response,
            reference,
            cap=cfg.grapheme_cer_cap,
            normalize=cfg.grapheme_cer_normalize,
            backend=cfg.grapheme_cer_backend,
        )
    if cfg.exact_match_weight and reference is not None:
        out["exact_match"] = cfg.exact_match_weight * exact_match_reward(
            response, reference
        )
    if cfg.numeric_match_weight and reference is not None:
        out["numeric_match"] = cfg.numeric_match_weight * numeric_match_reward(
            response, reference
        )
    if cfg.purity_weight:
        out["purity"] = cfg.purity_weight * language_purity_reward(
            response, cfg.purity_min_ratio
        )
    if cfg.format_weight:
        out["format"] = cfg.format_weight * format_reward(response)
    if cfg.empty_response_penalty_weight:
        out["empty_penalty"] = -cfg.empty_response_penalty_weight * empty_response_penalty(
            response
        )
    if cfg.invalid_token_penalty_weight:
        out["invalid_token_penalty"] = (
            -cfg.invalid_token_penalty_weight * invalid_token_penalty(response)
        )
    if cfg.length_excess_penalty_weight and reference is not None:
        out["length_penalty"] = -cfg.length_excess_penalty_weight * length_excess_penalty(
            response,
            reference,
            max_ratio=cfg.max_length_ratio,
        )
    return out


def reward_for(
    response: str,
    reference: str | None,
    cfg: RewardConfig,
) -> float:
    return sum(reward_breakdown_for(response, reference, cfg).values())


def compute_rewards(
    responses: Sequence[str],
    references: Sequence[str | None] | None,
    cfg: RewardConfig,
) -> torch.Tensor:
    """Per-response scalar rewards as a float tensor ``[N]``."""
    responses = list(responses)
    if references is None:
        references = [None] * len(responses)
    else:
        references = list(references)
    if len(references) != len(responses):
        raise ValueError(
            "responses/references length mismatch: "
            f"{len(responses)} != {len(references)}"
        )
    rewards, _ = compute_rewards_with_breakdown(responses, references, cfg)
    return rewards


def compute_rewards_with_breakdown(
    responses: Sequence[str],
    references: Sequence[str | None] | None,
    cfg: RewardConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute scalar rewards and mean components in one verifier pass."""

    responses = list(responses)
    references = [None] * len(responses) if references is None else list(references)
    if len(responses) != len(references):
        raise ValueError(
            "responses/references length mismatch: "
            f"{len(responses)} != {len(references)}"
        )
    totals: list[float] = []
    sums: dict[str, float] = {}
    for response, reference in zip(responses, references):
        components = reward_breakdown_for(response, reference, cfg)
        totals.append(sum(components.values()))
        for name, value in components.items():
            sums[name] = sums.get(name, 0.0) + value
    denom = max(1, len(responses))
    means = {name: value / denom for name, value in sums.items()}
    return torch.tensor(totals, dtype=torch.float32), means


def reward_component_means(
    responses: Sequence[str],
    references: Sequence[str | None] | None,
    cfg: RewardConfig,
) -> dict[str, float]:
    """Mean weighted component values for reward-health dashboards."""

    _, means = compute_rewards_with_breakdown(responses, references, cfg)
    return means


__all__ = [
    "RewardConfig",
    "compute_rewards",
    "compute_rewards_with_breakdown",
    "empty_response_penalty",
    "exact_match_reward",
    "numeric_match_reward",
    "format_reward",
    "grapheme_cer_reward",
    "grapheme_cer_value",
    "invalid_token_penalty",
    "language_purity_reward",
    "length_excess_penalty",
    "mongolian_script_ratio",
    "reward_breakdown_for",
    "reward_component_means",
    "reward_for",
]
