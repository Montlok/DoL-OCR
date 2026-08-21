# -*- coding: utf-8 -*-
"""Dual-track tokenizer with one unified id space.

Tracks:
    mn       -> MorphBPE (morphology-aware traditional Mongolian)
    general  -> byte-level BPE (Chinese / English / Japanese / Cyrillic /
                digits / punctuation / symbols), lossless, never ``<unk>``
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any

try:
    from ..generic_bpe import encode_byte_fallback, is_byte_token
    from .encoded import DualTrackResult, EncodedToken
    from .vocab import (
        SEGMENT,
        SPECIAL_TOKENS,
        build_unified_vocab,
    )
except ImportError:  # pragma: no cover - supports direct script execution.
    from Tokenizer.generic_bpe import encode_byte_fallback, is_byte_token
    from Tokenizer.unified.encoded import DualTrackResult, EncodedToken
    from Tokenizer.unified.vocab import (
        SEGMENT,
        SPECIAL_TOKENS,
        build_unified_vocab,
    )

SPECIAL_TOKEN_TEXTS = tuple(sorted(SPECIAL_TOKENS, key=len, reverse=True))

# Fast gate for ``special_at``: a special token can only begin at a character
# that is the first character of some special token. Checking membership in
# this small set lets the hot per-character scan skip the full startswith loop
# for the overwhelming majority of text characters.
_SPECIAL_FIRST_CHARS = frozenset(t[0] for t in SPECIAL_TOKEN_TEXTS if t)

MONGOLIAN_RANGES = [
    (0x1800, 0x18AF),
    (0x11660, 0x1167F),
]

MONGOLIAN_PUNCTUATION = {
    "\u1800",
    "\u1801",
    "\u1802",
    "\u1803",
    "\u1804",
    "\u1805",
    "\u1806",
    "\u1807",
    "\u1808",
    "\u1809",
}

SPACE_CHARS = {
    " ",
    "\u00a0",
    "\u202f",
}
NNBSP = "\u202f"


@dataclass(frozen=True)
class Span:
    lang: str
    text: str
    start: int
    end: int


def in_ranges(cp: int, ranges: list[tuple[int, int]]) -> bool:
    return any(lo <= cp <= hi for lo, hi in ranges)


def char_lang(ch: str) -> str:
    if ch in SPACE_CHARS:
        return "space"
    # Mongolian punctuation lives inside the Mongolian block but is not part of
    # word morphology, so it goes to the general track (not MorphBPE).
    if ch in MONGOLIAN_PUNCTUATION:
        return "general"
    if in_ranges(ord(ch), MONGOLIAN_RANGES):
        return "mn"
    return "general"


def _is_mongolian_char(ch: str) -> bool:
    return in_ranges(ord(ch), MONGOLIAN_RANGES)


def contextual_char_lang(text: str, index: int) -> str:
    ch = text[index]
    if ch == NNBSP:
        prev_is_mn = index > 0 and _is_mongolian_char(text[index - 1])
        next_is_mn = index + 1 < len(text) and _is_mongolian_char(text[index + 1])
        if prev_is_mn and next_is_mn:
            return "mn"
    return char_lang(ch)


def special_at(text: str, start: int) -> str | None:
    if start >= len(text) or text[start] not in _SPECIAL_FIRST_CHARS:
        return None
    for token in SPECIAL_TOKEN_TEXTS:
        if text.startswith(token, start):
            return token
    return None


def segment_by_language(text: str) -> list[Span]:
    if not text:
        return []

    spans: list[Span] = []
    start = 0
    cur_lang = ""
    i = 0

    while i < len(text):
        special = special_at(text, i)
        if special is not None:
            if cur_lang and start < i:
                spans.append(Span(cur_lang, text[start:i], start, i))
            end = i + len(special)
            spans.append(Span("special", text[i:end], i, end))
            i = end
            start = i
            cur_lang = ""
            continue

        lang = contextual_char_lang(text, i)
        if not cur_lang:
            cur_lang = lang
            start = i
        elif lang != cur_lang:
            spans.append(Span(cur_lang, text[start:i], start, i))
            start = i
            cur_lang = lang
        i += 1

    if cur_lang and start < len(text):
        spans.append(Span(cur_lang, text[start:], start, len(text)))

    return spans


def _general_piece_track(surface: str) -> str:
    """Classify a general-track piece as a word piece or a punctuation piece.

    Punctuation/symbol pieces use the ``general_punct`` track so the model-side
    morphology derivation resets ``word_pos``/``morph_depth`` around them (the
    same boundary behavior the retired ``misc`` track provided), while real
    word pieces (containing any letter or number) keep grouping under
    ``general`` so contiguous subtokens of one word share a word position.
    """

    if surface and any(unicodedata.category(ch)[0] in ("L", "N") for ch in surface):
        return "general"
    return "general_punct"


class DualTrackTokenizer:
    def __init__(
        self,
        unified_vocab: dict[str, int],
        morphbpe: Any,
        general: Any,
    ):
        self.vocab = unified_vocab
        self.id_to_token = {idx: tok for tok, idx in unified_vocab.items()}
        self.unk_id = unified_vocab["<unk>"]
        self.morphbpe = morphbpe
        self.general = general

        self.mn_local_to_global = {
            local_id: unified_vocab[token]
            for token, local_id in morphbpe.vocab.items()
            if token in unified_vocab
        }
        # Reverse map used only by the no-offset MorphBPE fallback to recover
        # each piece's surface text and assign monotonic, non-overlapping spans.
        self.mn_local_id_to_text = {
            local_id: token for token, local_id in morphbpe.vocab.items()
        }

        mn_lo, _mn_hi = SEGMENT["mongolian"]
        gen_lo, gen_hi = SEGMENT["general"]
        self.general_local_to_global: dict[int, int] = {}
        self.general_global_to_local: dict[int, int] = {}
        for token, local_id in general.get_vocab().items():
            global_id = unified_vocab.get(token)
            if global_id is None:
                continue
            # A surface string present in BOTH tracks (ASCII letters, digits,
            # punctuation picked up by MorphBPE from Latin fragments in the
            # Mongolian corpus) owns a single unified id in the MorphBPE
            # segment. The general track must still map onto that shared id:
            # gating the forward map on the general segment range turned every
            # such piece — including the byte-level fallback atoms — into
            # <unk>, destroying ~27% of English and ~29% of code tokens.
            self.general_local_to_global[local_id] = global_id
            if (
                gen_lo <= global_id < gen_hi
                or (
                    global_id >= mn_lo
                    and token not in SPECIAL_TOKENS
                    and not is_byte_token(token)
                )
            ):
                # Shared general pieces live in the MorphBPE id segment, but
                # still carry ByteLevel semantics at decode time. ASCII
                # surfaces happen to decode correctly as raw unified strings;
                # non-ASCII byte atoms do not. For example U+FE31 ``︱`` ends
                # in the ByteLevel atom ``±``. If that atom shares a MorphBPE
                # id, flushing it as the literal U+00B1 corrupts the sequence
                # to ``�±``. Inverting every non-special/non-byte fallback
                # forward mapping keeps adjacent byte atoms in one general
                # decoder buffer and exactly reconstructs the source UTF-8.
                #
                # Explicit ``<0xNN>`` ids remain on the byte-fallback path
                # below; reserved ids (notably ▁/◈) retain their documented
                # unified semantics.
                self.general_global_to_local[global_id] = local_id

    @property
    def vocab_size(self) -> int:
        return max(self.vocab.values()) + 1

    def encode(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> list[int]:
        return self.encode_with_spans(text, add_bos=add_bos, add_eos=add_eos).input_ids

    def encode_plain_text(self, text: str) -> list[int]:
        """Encode literal user/label text without interpreting control strings.

        OCR transcriptions are plain text. A visible ``▁``/``◈`` glyph or a
        literal substring such as ``<bos>`` must not become a structural model
        token (which decode intentionally folds or drops). Special-looking
        spans therefore use the same pretrained general ByteLevel track as
        other non-Mongolian text. The method never adds BOS/EOS; callers add
        structural tokens explicitly through their row format.
        """

        return self.encode_plain_text_with_spans(text).input_ids

    def encode_plain_text_with_spans(self, text: str) -> DualTrackResult:
        """Plain-text encoding with route metadata retained for auditing."""

        return self.encode_with_spans(
            text,
            add_bos=False,
            add_eos=False,
            interpret_special_tokens=False,
        )

    @staticmethod
    def _mark_mongolian_fallback(
        tokens: list[EncodedToken],
    ) -> list[EncodedToken]:
        """Make a MorphBPE miss observable to strict downstream contracts."""

        return [
            EncodedToken(
                token.id,
                token.token,
                "mn_general_fallback",
                token.start,
                token.end,
                surface=token.surface,
                metadata=token.metadata,
            )
            for token in tokens
        ]

    def encode_with_spans(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        interpret_special_tokens: bool = True,
    ) -> DualTrackResult:
        spans = segment_by_language(text)
        tokens: list[EncodedToken] = []

        if add_bos:
            tokens.append(EncodedToken(self.vocab["<bos>"], "<bos>", "special", -1, -1))

        for span in spans:
            if span.lang == "special":
                if interpret_special_tokens:
                    tokens.extend(self._encode_special(span))
                else:
                    tokens.extend(
                        self._encode_general(
                            Span("general", span.text, span.start, span.end)
                        )
                    )
            elif span.lang == "mn":
                tokens.extend(self._encode_mongolian(span))
            elif span.lang == "space":
                tokens.extend(self._encode_space(span))
            else:
                tokens.extend(self._encode_general(span))

        if add_eos:
            tokens.append(EncodedToken(self.vocab["<eos>"], "<eos>", "special", -1, -1))

        return DualTrackResult(
            input_ids=[token.id for token in tokens], tokens=tokens, spans=spans
        )

    def _encode_special(self, span: Span) -> list[EncodedToken]:
        token_id = self.vocab.get(span.text, self.unk_id)
        return [EncodedToken(token_id, span.text, "special", span.start, span.end)]

    def _encode_mongolian(self, span: Span) -> list[EncodedToken]:
        if hasattr(self.morphbpe, "encode_with_offsets"):
            local_tokens = self.morphbpe.encode_with_offsets(span.text)
            tokens: list[EncodedToken] = []
            for item in local_tokens:
                if isinstance(item, dict):
                    local_id = item["id"]
                    text = item["token"]
                    start = item["start"]
                    end = item["end"]
                else:
                    local_id = item.id
                    text = item.token
                    start = item.start
                    end = item.end
                global_id = self.mn_local_to_global.get(local_id, self.unk_id)
                if global_id == self.unk_id:
                    # MorphBPE is morphology-aware but its finite seed
                    # alphabet does not necessarily contain every scalar in
                    # the Mongolian and Mongolian Supplement blocks. Keeping
                    # <unk> would make an OCR label unrecoverable. Fall back
                    # only for the uncovered source slice while surrounding
                    # covered morphemes retain their pretrained ids.
                    raw_start = max(0, int(start))
                    raw_end = min(len(span.text), int(end))
                    raw = span.text[raw_start:raw_end]
                    if not raw:
                        raise ValueError(
                            "MorphBPE produced <unk> without a recoverable "
                            f"source span for {span.text!r}"
                        )
                    tokens.extend(
                        self._mark_mongolian_fallback(
                            self._encode_general(
                                Span(
                                    "general",
                                    raw,
                                    span.start + raw_start,
                                    span.start + raw_end,
                                )
                            )
                        )
                    )
                    continue
                tokens.append(
                    EncodedToken(
                        global_id,
                        text,
                        "mn",
                        span.start + start,
                        span.start + end,
                    )
                )
            return tokens

        local_ids = self.morphbpe.encode(span.text)
        if any(
            self.mn_local_to_global.get(local_id, self.unk_id) == self.unk_id
            for local_id in local_ids
        ):
            # Legacy MorphBPE fixtures do not expose offsets, so a precise
            # slice fallback is impossible. Preserve the text for permissive
            # callers, but mark the route so strict OCR contracts reject it.
            return self._mark_mongolian_fallback(
                self._encode_general(
                    Span(
                        "general",
                        span.text,
                        span.start,
                        span.end,
                    )
                )
            )
        # No per-token offsets available: recover each piece's surface from the
        # reverse vocab and walk a cursor through the span so offsets stay
        # monotonic and non-overlapping.
        tokens = []
        cursor = 0
        text = span.text
        for local_id in local_ids:
            piece = self.mn_local_id_to_text.get(local_id, "")
            found = text.find(piece, cursor) if piece else -1
            if found >= 0:
                start, end = found, found + len(piece)
                surface = piece
                cursor = end
            else:
                start = end = cursor
                surface = piece
            tokens.append(
                EncodedToken(
                    self.mn_local_to_global.get(local_id, self.unk_id),
                    surface,
                    "mn",
                    span.start + start,
                    span.start + end,
                )
            )
        return tokens

    def _encode_general(self, span: Span) -> list[EncodedToken]:
        pieces = self.general.encode_pieces(span.text)
        if not pieces and span.text:
            return encode_byte_fallback(
                span.text, self.vocab, self.unk_id, span.start, "general"
            )
        tokens: list[EncodedToken] = []
        for local_id, token, start, end in pieces:
            tokens.append(
                EncodedToken(
                    self.general_local_to_global.get(local_id, self.unk_id),
                    token,
                    _general_piece_track(span.text[start:end]),
                    span.start + start,
                    span.start + end,
                )
            )
        return tokens

    def _encode_space(self, span: Span) -> list[EncodedToken]:
        space_id = self.vocab.get("\u2581", self.unk_id)
        return [
            EncodedToken(space_id, "\u2581", "space", pos, pos + 1)
            for pos in range(span.start, span.end)
        ]

    def decode(self, ids: list[int]) -> str:
        parts: list[str] = []
        byte_buf: list[int] = []
        gen_buf: list[int] = []

        def flush_bytes() -> None:
            if byte_buf:
                parts.append(bytes(byte_buf).decode("utf-8", errors="replace"))
                byte_buf.clear()

        def flush_general() -> None:
            if gen_buf:
                parts.append(self.general.decode(list(gen_buf)))
                gen_buf.clear()

        for idx in ids:
            local = self.general_global_to_local.get(idx)
            if local is not None:
                flush_bytes()
                gen_buf.append(local)
                continue

            token = self.id_to_token.get(idx, "")

            if is_byte_token(token):
                flush_general()
                try:
                    byte_buf.append(int(token[3:5], 16))
                    continue
                except ValueError:
                    pass

            flush_general()
            flush_bytes()

            if token in {"<pad>", "<unk>", "<bos>", "<eos>", "<img>"}:
                continue
            if token == "\u2581":
                parts.append(" ")
            elif token == "\u25c8":
                continue
            else:
                parts.append(token)

        flush_general()
        flush_bytes()
        return "".join(parts)


def run_segment(text: str) -> None:
    for span in segment_by_language(text):
        print(f"[{span.lang}] {span.start}:{span.end} {span.text!r}")


def main() -> None:
    import sys

    if len(sys.argv) < 3 or sys.argv[1] != "segment":
        print("Usage: python -m Tokenizer.unified.dual_tokenizer segment <text>")
        return
    run_segment(sys.argv[2])


if __name__ == "__main__":
    main()


__all__ = [
    "DualTrackTokenizer",
    "Span",
    "SEGMENT",
    "SPECIAL_TOKENS",
    "build_unified_vocab",
    "char_lang",
    "contextual_char_lang",
    "segment_by_language",
]
