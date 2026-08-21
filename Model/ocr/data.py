# -*- coding: utf-8 -*-

"""OCR training-row construction (torch-free, render-free).

The generative OCR path (A architecture) trains on the same pre-tokenized JSONL
schema the VLM dataloader already consumes (see :mod:`Model.training.data`):
each row carries ``input_ids``/``attention_mask``/``labels`` plus a one-element
``images`` list. The image is represented in the token stream by exactly
``n_image_tokens`` ``<image_patch>`` slots — the OMVT injector asserts this
one-for-one with the compressed visual tokens, so the count must equal the
tower's ``compress_to``.

Token layout per row::

    [BOS] <image_start> <image_patch> * N <image_end> <instruction...> <target...> [EOS]

Only the transcription target (and the terminal EOS) is supervised; the image
slots and the instruction are masked with ``ignore_index`` so the loss measures
recognition, not prompt memorization.

This module deliberately knows nothing about rendering or a concrete tokenizer,
so the row contract can be unit-tested without fonts, libraqm, torch, or a
trained tokenizer bundle. :mod:`scripts.build_ocr_data` wires it to real
rendering + a :class:`TokenizerBundle`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from Model.ocr.alignment_contract import validate_ocr_image_binding_fields
from Tokenizer.pretraining.morphology import (
    MORPH_TRACK_RESET,
    derive_morph_info_from_track_ids,
)


def build_ocr_row(
    target_ids: Sequence[int],
    n_image_tokens: int,
    image_ref: Any,
    *,
    bos_id: int,
    image_start_id: int,
    image_patch_id: int,
    image_end_id: int,
    eos_id: int,
    instruction_ids: Sequence[int] = (),
    target_track_ids: Sequence[int] | None = None,
    instruction_track_ids: Sequence[int] | None = None,
    image_sha256: str | None = None,
    image_size_bytes: int | None = None,
    add_eos: bool = True,
    ignore_index: int = -100,
) -> dict[str, Any]:
    """Build one pre-tokenized OCR training row.

    Args:
        target_ids: token ids of the ground-truth transcription (supervised).
        n_image_tokens: number of ``<image_patch>`` slots; must equal the OMVT
            tower ``compress_to`` for the image payload.
        image_ref: opaque per-row image reference passed through in ``images``
            (e.g. a file path); one image per row.
        bos_id/image_start_id/image_patch_id/image_end_id/eos_id: special ids.
        instruction_ids: optional prompt tokens placed after ``<image_end>`` and
            before the target (masked from the loss).
        target_track_ids/instruction_track_ids: canonical morphology-route ids
            aligned with their token sequences. When supplied, the row also
            carries the exact ``word_pos``/``morph_depth`` representation used
            by language pretraining. Strict frozen-LM OCR builders require
            both; legacy/smoke callers may omit both.
        image_sha256/image_size_bytes: expected bytes of ``image_ref``. They
            must be supplied together for native frozen-LM OCR data; the
            strict collator verifies them while reading the image.
        add_eos: append ``eos_id`` to the target and supervise it.
        ignore_index: label value for masked (unsupervised) positions.

    Returns:
        A dict with ``input_ids``, ``attention_mask``, ``labels`` (aligned), and
        a single-element ``images`` list.
    """
    if n_image_tokens < 1:
        raise ValueError("n_image_tokens must be >= 1")
    target_ids = [int(t) for t in target_ids]
    if not target_ids:
        raise ValueError("target_ids must be non-empty")
    instruction_ids = [int(t) for t in instruction_ids]
    if (target_track_ids is None) != (instruction_track_ids is None):
        raise ValueError(
            "target_track_ids and instruction_track_ids must be supplied together"
        )
    target_tracks = (
        [int(value) for value in target_track_ids]
        if target_track_ids is not None
        else None
    )
    instruction_tracks = (
        [int(value) for value in instruction_track_ids]
        if instruction_track_ids is not None
        else None
    )
    if target_tracks is not None and len(target_tracks) != len(target_ids):
        raise ValueError("target_track_ids must align with target_ids")
    if (
        instruction_tracks is not None
        and len(instruction_tracks) != len(instruction_ids)
    ):
        raise ValueError("instruction_track_ids must align with instruction_ids")

    prompt = (
        [bos_id, image_start_id]
        + [image_patch_id] * n_image_tokens
        + [image_end_id]
        + instruction_ids
    )
    supervised = target_ids + ([eos_id] if add_eos else [])

    input_ids = prompt + supervised
    labels = [ignore_index] * len(prompt) + supervised
    attention_mask = [1] * len(input_ids)

    if len(input_ids) != len(labels) or len(input_ids) != len(attention_mask):
        raise RuntimeError("OCR row fields must have aligned lengths")
    if input_ids.count(image_patch_id) != n_image_tokens:
        raise ValueError(
            "OCR row must contain exactly n_image_tokens image_patch slots; "
            "instruction_ids and target_ids must not contain image_patch_id"
        )

    row = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "images": [image_ref],
    }
    if (image_sha256 is None) != (image_size_bytes is None):
        raise ValueError(
            "image_sha256 and image_size_bytes must be supplied together"
        )
    if image_sha256 is not None and image_size_bytes is not None:
        row["image_sha256"] = image_sha256
        row["image_size_bytes"] = image_size_bytes
        validate_ocr_image_binding_fields(row)
    if target_tracks is not None and instruction_tracks is not None:
        prompt_tracks = (
            [MORPH_TRACK_RESET] * (2 + n_image_tokens + 1)
            + instruction_tracks
        )
        all_tracks = (
            prompt_tracks
            + target_tracks
            + ([MORPH_TRACK_RESET] if add_eos else [])
        )
        word_pos, morph_depth = derive_morph_info_from_track_ids(all_tracks)
        if len(word_pos) != len(input_ids):
            raise RuntimeError("OCR morphology fields must align with input_ids")
        row["word_pos"] = word_pos
        row["morph_depth"] = morph_depth
    return row


def split_ocr_row(
    row: dict[str, Any],
    *,
    ignore_index: int = -100,
    eos_id: int | None = None,
) -> tuple[list[int], list[int], Any]:
    """Invert :func:`build_ocr_row`: recover ``(prompt, target, image_ref)``.

    The prompt is the leading run of positions whose label is ``ignore_index``
    (BOS + image slots + instruction); the supervised tail is the reference
    transcription. With ``eos_id`` set, one trailing EOS is stripped from the
    target so the reference matches the transcription text exactly.

    Generative evaluation feeds the prompt to ``generate`` and scores the
    sampled continuation against the returned target.
    """
    input_ids = [int(t) for t in row["input_ids"]]
    labels = [int(t) for t in row["labels"]]
    if len(input_ids) != len(labels):
        raise ValueError("input_ids and labels must have aligned lengths")
    split = 0
    while split < len(labels) and labels[split] == ignore_index:
        split += 1
    if split == 0 or split == len(labels):
        raise ValueError(
            "OCR row must start with a masked prompt followed by a "
            "supervised target"
        )
    if any(t == ignore_index for t in labels[split:]):
        raise ValueError("supervised target must be a contiguous tail")
    target = input_ids[split:]
    if eos_id is not None and target and target[-1] == eos_id:
        target = target[:-1]
    if not target:
        raise ValueError("target is empty after stripping EOS")
    images = row.get("images") or []
    image_ref = images[0] if images else None
    return input_ids[:split], target, image_ref


__all__ = ["build_ocr_row", "split_ocr_row"]
