# -*- coding: utf-8 -*-
"""Morphology-constrained BPE tokenizer for traditional Mongolian."""

from __future__ import annotations

from typing import Any

from Tokenizer.traditional_mongolian.stemmer import MongolStemmer
from Tokenizer.traditional_mongolian.unicode_norm import CTRL_ALL, strip_all_with_map

from . import serialization
from .offsets import MorphToken, Piece, split_on_ascii_space


class MorphBPETokenizer:
    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        merges: dict[tuple[str, str], tuple[str, int]] | None = None,
        stemmer: MongolStemmer | None = None,
        min_boundary_confidence: float = 0.60,
        seed_alphabet: bool = False,
    ):
        self.vocab = vocab or {"<unk>": 0}
        self.merges = merges or {}
        self.stemmer = stemmer
        self.min_boundary_confidence = min_boundary_confidence
        self.seed_alphabet = seed_alphabet
        self._id_to_token = {i: tok for tok, i in self.vocab.items()}
        # MVS/FVS preservation is gated on the vocab: vocabularies that carry
        # the control characters as tokens (v3+ builds) keep them in the id
        # stream; older vocabularies fold them exactly as before, so v2
        # bundles stay bit-identical.
        self._preserve_controls = all(ch in self.vocab for ch in CTRL_ALL)

    # ---- io ----

    @classmethod
    def from_file(
        cls, path: str, stemmer: MongolStemmer | None = None
    ) -> "MorphBPETokenizer":
        payload = serialization.load(path)
        merges = serialization.merges_from_payload(payload)
        return cls(
            vocab={str(k): int(v) for k, v in payload.get("vocab", {}).items()},
            merges=merges,
            stemmer=stemmer,
            min_boundary_confidence=float(
                payload.get("config", {}).get("min_boundary_confidence", 0.60)
            ),
            seed_alphabet=bool(payload.get("config", {}).get("seed_alphabet", False)),
        )

    def save(self, path: str, extra_config: dict[str, Any] | None = None) -> None:
        serialization.dump(self, path, extra_config=extra_config)

    # ---- encode ----

    def encode(self, text: str) -> list[int]:
        return [tok.id for tok in self.encode_with_offsets(text)]

    def encode_with_offsets(self, text: str, base_start: int = 0) -> list[MorphToken]:
        tokens: list[MorphToken] = []
        for start, end in split_on_ascii_space(text):
            for tok in self.tokenize_word(text[start:end], base_start + start):
                tokens.append(tok)
        return tokens

    def tokenize_word(self, word: str, base_start: int = 0) -> list[MorphToken]:
        if not self._preserve_controls:
            return self._tokenize_clean_word(word, base_start)

        # Controls are hard segmentation boundaries: each control character
        # becomes its own token at its original position and BPE runs on the
        # control-free segments between them. MVS only ever marks the
        # stem/separated-suffix junction, so splitting there is exactly the
        # morpheme boundary MorphBPE wants anyway; FVS/NIRUGU are rare (2.4‰)
        # word-internal marks where the fertility cost is negligible. Within
        # a clean segment skeleton offsets equal original offsets, which also
        # sidesteps the skeleton-vs-original coordinate drift the legacy path
        # has for multi-boundary words with embedded controls.
        tokens: list[MorphToken] = []
        seg_start = 0
        for i, ch in enumerate(word):
            if ch not in CTRL_ALL:
                continue
            if i > seg_start:
                tokens.extend(
                    self._tokenize_clean_word(word[seg_start:i], base_start + seg_start)
                )
            tokens.append(
                MorphToken(ch, self.vocab[ch], base_start + i, base_start + i + 1)
            )
            seg_start = i + 1
        if seg_start < len(word):
            tokens.extend(
                self._tokenize_clean_word(word[seg_start:], base_start + seg_start)
            )
        return tokens

    def _tokenize_clean_word(self, word: str, base_start: int = 0) -> list[MorphToken]:
        skeleton, boundary_map = strip_all_with_map(word)
        if not skeleton:
            return []

        analysis = self.stemmer.analyze(word) if self.stemmer else None
        forbidden: set[int] = set()
        if analysis is not None and analysis.confidence >= self.min_boundary_confidence:
            forbidden = set(analysis.skeleton_boundaries[1:-1])

        pieces = [
            Piece(ch, boundary_map[i], boundary_map[i + 1])
            for i, ch in enumerate(skeleton)
        ]
        pieces = self._apply_merges(pieces, forbidden)
        return [
            MorphToken(
                p.text,
                self._piece_id(p.text),
                base_start + p.start,
                base_start + p.end,
            )
            for p in pieces
        ]

    def _apply_merges(self, pieces: list[Piece], forbidden: set[int]) -> list[Piece]:
        if not self.merges:
            return pieces
        changed = True
        while changed:
            changed = False
            best_index: int | None = None
            best_rank: int | None = None
            best_text = ""
            for i in range(len(pieces) - 1):
                if pieces[i].end in forbidden:
                    continue
                merge = self.merges.get((pieces[i].text, pieces[i + 1].text))
                if merge is None:
                    continue
                merged, rank = merge
                if best_rank is None or rank < best_rank:
                    best_index = i
                    best_rank = rank
                    best_text = merged
            if best_index is not None:
                left = pieces[best_index]
                right = pieces[best_index + 1]
                pieces[best_index : best_index + 2] = [
                    Piece(best_text, left.start, right.end)
                ]
                changed = True
        return pieces

    def _piece_id(self, piece: str) -> int:
        if piece in self.vocab:
            return self.vocab[piece]
        return self.vocab.get("<unk>", 0)

    # ---- decode ----

    def decode(self, ids: list[int]) -> str:
        if not self._id_to_token:
            self._id_to_token = {i: tok for tok, i in self.vocab.items()}
        out: list[str] = []
        for i in ids:
            tok = self._id_to_token.get(i, "")
            if tok == "<unk>":
                out.append("\ufffd")
                continue
            out.append(tok)
        return "".join(out)
