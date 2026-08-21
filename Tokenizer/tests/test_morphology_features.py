# -*- coding: utf-8 -*-

"""Regression tests for ``derive_morph_info_from_tokens``.

These pin the contract that only ``mn`` and ``general`` are word tracks, while
``general_punct``, ``space`` and ``special`` reset the running word. After the
two-track redesign punctuation is routed to a dedicated ``general_punct`` track
(not ``general``), so common punctuation-heavy text such as ``hello,world`` or
``这。图`` keeps clean per-word morphology features instead of gluing words
together through contiguous offsets.
"""

import unicodedata
import unittest
from dataclasses import dataclass

from Tokenizer.pretraining.morphology import (
    WORD_TRACKS,
    derive_morph_info_from_tokens,
)

try:  # the end-to-end test needs the real byte-level BPE backend
    import tokenizers as _tokenizers  # noqa: F401

    _HAS_TOKENIZERS = True
except Exception:  # pragma: no cover - environment without tokenizers
    _HAS_TOKENIZERS = False


@dataclass
class FakeToken:
    track: str
    start: int
    end: int


class WordTracksTest(unittest.TestCase):
    def test_word_tracks_exclude_general_punct(self):
        self.assertIn("mn", WORD_TRACKS)
        self.assertIn("general", WORD_TRACKS)
        self.assertNotIn("general_punct", WORD_TRACKS)
        self.assertNotIn("space", WORD_TRACKS)
        self.assertNotIn("special", WORD_TRACKS)


class DeriveMorphInfoTest(unittest.TestCase):
    def test_contiguous_general_pieces_share_word_and_bump_depth(self):
        # "hello" as five contiguous general byte pieces -> one word, depth 0..4.
        tokens = [FakeToken("general", i, i + 1) for i in range(5)]
        word_pos, depth = derive_morph_info_from_tokens(tokens)
        self.assertEqual(word_pos, [0, 0, 0, 0, 0])
        self.assertEqual(depth, [0, 1, 2, 3, 4])

    def test_general_punct_resets_word(self):
        # "hello,world": comma on the general_punct track must separate words.
        tokens = (
            [FakeToken("general", i, i + 1) for i in range(5)]
            + [FakeToken("general_punct", 5, 6)]
            + [FakeToken("general", 6 + i, 7 + i) for i in range(5)]
        )
        word_pos, depth = derive_morph_info_from_tokens(tokens)
        # hello -> word 0; comma -> reset (word 0, depth 0); world -> word 1.
        self.assertEqual(word_pos, [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        self.assertEqual(depth, [0, 1, 2, 3, 4, 0, 0, 1, 2, 3, 4])

    def test_punct_between_cjk_words_resets(self):
        # 这。图 : letter / punct / letter -> two distinct words around the period.
        tokens = [
            FakeToken("general", 0, 1),
            FakeToken("general_punct", 1, 2),
            FakeToken("general", 2, 3),
        ]
        word_pos, depth = derive_morph_info_from_tokens(tokens)
        self.assertEqual(word_pos, [0, 0, 1])
        self.assertEqual(depth, [0, 0, 0])

    def test_space_and_special_reset(self):
        tokens = [
            FakeToken("general", 0, 1),
            FakeToken("space", 1, 2),
            FakeToken("general", 2, 3),
            FakeToken("special", -1, -1),
            FakeToken("general", 3, 4),
        ]
        word_pos, depth = derive_morph_info_from_tokens(tokens)
        self.assertEqual(word_pos, [0, 0, 1, 1, 2])
        self.assertEqual(depth, [0, 0, 0, 0, 0])

    def test_non_contiguous_general_pieces_start_new_word(self):
        # A gap in offsets (e.g. a dropped separator) opens a new word.
        tokens = [FakeToken("general", 0, 1), FakeToken("general", 5, 6)]
        word_pos, depth = derive_morph_info_from_tokens(tokens)
        self.assertEqual(word_pos, [0, 1])
        self.assertEqual(depth, [0, 0])

    def test_track_switch_starts_new_word(self):
        # Adjacent offsets but different word tracks must not merge.
        tokens = [FakeToken("mn", 0, 1), FakeToken("general", 1, 2)]
        word_pos, depth = derive_morph_info_from_tokens(tokens)
        self.assertEqual(word_pos, [0, 1])
        self.assertEqual(depth, [0, 0])


@unittest.skipUnless(_HAS_TOKENIZERS, "requires the 'tokenizers' package")
class RealBPEPunctuationTest(unittest.TestCase):
    """End-to-end: a *real* trained byte-level BPE never glues punctuation
    into a word piece, so punctuation always resets word_pos/morph_depth.

    This locks the guarantee a reviewer cannot eyeball from the merge table:
    the GPT-2 ``ByteLevel(use_regex=True)`` pre-tokenizer splits ``\\p{L}+``,
    ``\\p{N}+`` and punctuation into separate chunks *before* any merge, so no
    learned merge can ever span a letter<->punctuation boundary. Hence
    ``_general_piece_track`` classifies every punctuation piece as
    ``general_punct`` (a non-word track) even after aggressive training on
    punctuation-heavy text.
    """

    @classmethod
    def setUpClass(cls):
        from Tokenizer.generic_bpe.general_bpe import GeneralBPETrainer

        # Punctuation-heavy corpus repeated to force merges that *would* glue
        # punctuation to letters if the pre-tokenizer did not separate them.
        corpus = (
            ["hello,world"] * 200
            + ["这。图书馆"] * 200
            + ["foo.bar,baz!qux"] * 200
            + ["a,b,c,d,e"] * 200
        )
        cls.model = GeneralBPETrainer(vocab_size=400, min_frequency=1).train(corpus)

    @staticmethod
    def _piece_track(surface: str) -> str:
        from Tokenizer.unified.dual_tokenizer import _general_piece_track

        return _general_piece_track(surface)

    def _tokens(self, text: str):
        tokens = []
        for _lid, _tok, start, end in self.model.encode_pieces(text):
            surface = text[start:end]
            tokens.append(FakeToken(self._piece_track(surface), start, end))
        return tokens

    def test_no_piece_mixes_letter_and_punct(self):
        for text in ["hello,world", "这。图", "foo.bar,baz!qux", "a,b,c"]:
            for _lid, _tok, start, end in self.model.encode_pieces(text):
                surface = text[start:end]
                has_word = any(
                    unicodedata.category(ch)[0] in ("L", "N") for ch in surface
                )
                has_punct = any(
                    unicodedata.category(ch)[0] not in ("L", "N", "Z", "C")
                    for ch in surface
                )
                self.assertFalse(
                    has_word and has_punct,
                    f"piece {surface!r} mixes letters and punctuation",
                )

    def test_punctuation_pieces_route_to_general_punct(self):
        for _lid, _tok, start, end in self.model.encode_pieces("foo.bar,baz!qux"):
            surface = "foo.bar,baz!qux"[start:end]
            if surface in {".", ",", "!"}:
                self.assertEqual(self._piece_track(surface), "general_punct")

    def test_hello_comma_world_resets_word(self):
        tokens = self._tokens("hello,world")
        word_pos, _depth = derive_morph_info_from_tokens(tokens)
        # The comma sits on general_punct and must split "hello" from "world".
        self.assertEqual(max(word_pos), 1)
        self.assertEqual(word_pos[0], 0)
        self.assertEqual(word_pos[-1], 1)

    def test_cjk_period_resets_word(self):
        text = "这。图"
        pieces = self.model.encode_pieces(text)
        tokens = self._tokens(text)
        word_pos, _depth = derive_morph_info_from_tokens(tokens)
        # The period (general_punct) must separate the two CJK characters: every
        # piece left of the period belongs to a strictly lower word than every
        # piece right of it. (One CJK char may split into several byte pieces at
        # overlapping offsets, so we assert a boundary, not an exact count.)
        left = [wp for (_l, _t, s, _e), wp in zip(pieces, word_pos) if s < 1]
        right = [wp for (_l, _t, s, _e), wp in zip(pieces, word_pos) if s >= 2]
        self.assertTrue(left and right)
        self.assertLess(max(left), min(right))


if __name__ == "__main__":
    unittest.main()
