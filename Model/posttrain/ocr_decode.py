# -*- coding: utf-8 -*-

"""Reward-safe decoding for OCR policy completions.

The unified tokenizer intentionally hides several control tokens when decoding
(``<pad>``, ``<unk>``, ``<bos>``, ``<eos>``, and the morphology boundary).
That behaviour is convenient for inference, but unsafe for reinforcement
learning: a policy could emit hidden controls and receive the same CER as if it
had produced nothing.  This module makes every reserved token before EOS
visible to the reward while retaining the normal tokenizer for content tokens.
"""

from __future__ import annotations

from collections.abc import Callable, Container, Iterable

import torch

from Model.config import EOS_ID, SEGMENT, WORD_BOUNDARY_ID

INVALID_OCR_TOKEN = "\ufffd"


def decode_ocr_completion(
    ids: torch.Tensor | Iterable[int],
    decode_content: Callable[[list[int]], str],
    *,
    eos_id: int = EOS_ID,
    word_boundary_id: int = WORD_BOUNDARY_ID,
    valid_token_ids: Container[int] | None = None,
    require_eos: bool = False,
) -> str:
    """Decode one OCR completion without silently hiding invalid controls.

    The first EOS terminates the completion and is not rendered.  Every other
    id in the reserved special segment is replaced by ``INVALID_OCR_TOKEN``,
    except the word-boundary id whose public tokenizer semantics are a space.
    When ``valid_token_ids`` is supplied, unassigned vocabulary ids are also
    exposed rather than being silently decoded to an empty string. Content
    spans are delegated to the production tokenizer so byte fallback and both
    language tracks retain their exact decoding behaviour. With
    ``require_eos=True``, exhausting the sampled budget without EOS appends the
    same visible invalid sentinel, so correct text with an invalid termination
    cannot receive a perfect OCR reward.
    """

    values = ids.detach().cpu().tolist() if isinstance(ids, torch.Tensor) else list(ids)
    special_lo, special_hi = SEGMENT["special"]
    parts: list[str] = []
    content: list[int] = []
    saw_eos = False

    def flush() -> None:
        if content:
            parts.append(decode_content(content))
            content.clear()

    for raw_id in values:
        token_id = int(raw_id)
        if token_id == eos_id:
            saw_eos = True
            break
        invalid_reserved = (
            special_lo <= token_id < special_hi and token_id != word_boundary_id
        )
        invalid_unassigned = (
            valid_token_ids is not None and token_id not in valid_token_ids
        )
        if invalid_reserved or invalid_unassigned:
            flush()
            parts.append(INVALID_OCR_TOKEN)
            continue
        content.append(token_id)
    flush()
    if require_eos and not saw_eos:
        parts.append(INVALID_OCR_TOKEN)
    return "".join(parts)


__all__ = ["INVALID_OCR_TOKEN", "decode_ocr_completion"]
