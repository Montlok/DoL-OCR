# -*- coding: utf-8 -*-
"""Model-side morphology features derived from tokenizer spans."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class TokenLike(Protocol):
    track: str
    start: int
    end: int


WORD_TRACKS = {"mn", "general"}
MORPH_TRACK_RESET = 0
MORPH_TRACK_MONGOLIAN = 1
MORPH_TRACK_GENERAL = 2


def derive_morph_info_from_tokens(
    tokens: Sequence[TokenLike],
) -> tuple[list[int], list[int]]:
    """Derive word position and intra-word subtoken depth from encoded tokens.

    Mongolian MorphBPE pieces that are contiguous in the original text share a
    word position and advance morph_depth. General byte-level BPE pieces use the
    same convention. Only ``mn`` and ``general`` are word tracks; punctuation is
    routed to a separate ``general_punct`` track (see
    ``dual_tokenizer._general_piece_track``), so spaces, specials, and
    punctuation all reset the current word -- ``hello,world`` and ``这。图`` keep
    one word per side of the punctuation instead of gluing them together.
    """

    word_positions: list[int] = []
    morph_depths: list[int] = []

    cur_word = -1
    cur_depth = 0
    prev_track: str | None = None
    prev_end: int | None = None

    for token in tokens:
        track = token.track
        start = int(token.start)
        end = int(token.end)

        if start < 0 or end < 0:
            word_positions.append(max(cur_word, 0))
            morph_depths.append(0)
            prev_track = None
            prev_end = None
            continue

        if track not in WORD_TRACKS:
            word_positions.append(max(cur_word, 0))
            morph_depths.append(0)
            prev_track = None
            prev_end = None
            continue

        same_word = prev_track == track and prev_end is not None and start == prev_end
        if same_word:
            cur_depth += 1
        else:
            cur_word += 1
            cur_depth = 0

        word_positions.append(cur_word)
        morph_depths.append(cur_depth)
        prev_track = track
        prev_end = end

    return word_positions, morph_depths


def derive_morph_info_from_track_ids(
    track_ids: Sequence[int],
) -> tuple[list[int], list[int]]:
    """Derive the pretrained morphology features from canonical route ids.

    This is the offset-free contract used during autoregressive decoding, when
    only generated token ids are available.  ``0`` is a reset/non-word token,
    ``1`` is a Mongolian word piece, and ``2`` is a general-script word piece.
    Consecutive pieces on the same word track share ``word_pos`` and advance
    ``morph_depth``; a reset or track transition starts the next word.

    Strict OCR producers compare this result with
    :func:`derive_morph_info_from_tokens` for every reference and reject any
    route whose token id is ambiguous.  That makes this runtime representation
    identical to the span-aware pretraining representation for all admitted
    OCR text.
    """

    valid_word_tracks = {MORPH_TRACK_MONGOLIAN, MORPH_TRACK_GENERAL}
    word_positions: list[int] = []
    morph_depths: list[int] = []
    cur_word = -1
    cur_depth = 0
    previous = MORPH_TRACK_RESET

    for raw_track in track_ids:
        track = int(raw_track)
        if track not in {
            MORPH_TRACK_RESET,
            MORPH_TRACK_MONGOLIAN,
            MORPH_TRACK_GENERAL,
        }:
            raise ValueError(f"unknown morphology track id: {track}")
        if track not in valid_word_tracks:
            word_positions.append(max(cur_word, 0))
            morph_depths.append(0)
            previous = MORPH_TRACK_RESET
            continue
        if previous == track:
            cur_depth += 1
        else:
            cur_word += 1
            cur_depth = 0
        word_positions.append(cur_word)
        morph_depths.append(cur_depth)
        previous = track
    return word_positions, morph_depths


def derive_morph_info_from_offsets(
    token_offsets: Sequence[tuple[int, int] | list[int]],
) -> tuple[list[int], list[int]]:
    """Best-effort fallback for legacy encoded rows without token tracks."""

    word_positions: list[int] = []
    morph_depths: list[int] = []

    cur_word = -1
    cur_depth = 0
    prev_end: int | None = None

    for offset in token_offsets:
        start, end = int(offset[0]), int(offset[1])
        if start < 0 or end < 0:
            word_positions.append(max(cur_word, 0))
            morph_depths.append(0)
            prev_end = None
            continue

        if prev_end is None or start != prev_end:
            cur_word += 1
            cur_depth = 0
        else:
            cur_depth += 1

        word_positions.append(cur_word)
        morph_depths.append(cur_depth)
        prev_end = end

    return word_positions, morph_depths


def derive_morph_info_from_boundary_ids(
    input_ids,
    word_boundary_id: int,
    morpheme_boundary_id: int,
    special_id_range: tuple = (0, 256),
    max_depth: int | None = None,
) -> tuple[list[int], list[int]]:
    """Derive ``(word_pos, morph_depth)`` from token IDs at runtime.

    Fallback used when a batch lacks precomputed morphology fields. Rules:

    * ``<word_boundary>`` (``▁``) opens a new word (``word_pos`` += 1)
      at depth 0. The very first ``word_boundary`` either anchors word 0
      (if no content has been seen yet) or word 1 (if content tokens
      preceded it; those tokens then implicitly belong to word 0).
    * ``<morpheme>`` (``◈``) stays inside the current word and increments
      ``morph_depth``.
    * Other specials in ``special_id_range`` (BOS/EOS/image/etc.) reset
      depth to 0; their ``word_pos`` is the current word (or 0 before any
      word has opened). They do **not** advance ``word_pos``.
    * Other tokens inherit ``(word_pos, depth)``.

    Output positions are non-negative. ``max_depth`` (if given and
    non-negative) is the largest representable depth, so clipping is
    inclusive.  Model runtime paths normally leave this uncapped and let
    morphological RoPE own the single, identical clipping operation for both
    precomputed and dynamically derived features.
    """

    lo, hi = special_id_range
    word_positions: list[int] = []
    morph_depths: list[int] = []

    cap = max_depth if (max_depth is not None and max_depth >= 0) else None

    cur_word = -1
    cur_depth = 0

    for token_id in input_ids:
        tid = int(token_id)
        is_special = lo <= tid < hi

        if is_special and tid == word_boundary_id:
            cur_word += 1
            cur_depth = 0
            word_positions.append(max(cur_word, 0))
            morph_depths.append(0)
            continue

        if is_special and tid == morpheme_boundary_id:
            cur_depth += 1
            if cap is not None:
                cur_depth = min(cur_depth, cap)
            word_positions.append(max(cur_word, 0))
            morph_depths.append(cur_depth)
            continue

        if is_special:
            cur_depth = 0
            word_positions.append(max(cur_word, 0))
            morph_depths.append(0)
            continue

        # content token: opens word 0 if none has been opened yet
        if cur_word < 0:
            cur_word = 0
            cur_depth = 0
        word_positions.append(cur_word)
        morph_depths.append(cur_depth)

    return word_positions, morph_depths
