# -*- coding: utf-8 -*-

"""Unit tests for the pure helpers of ``scripts.render_mn_pages``.

``_to_presentation`` encodes the linguistic rule the whole synthetic render
set depends on: nominal-form MVS (U+180E) is a *suffix separator* that
real-world fonts key on NNBSP (U+202F), except before a genuinely separated
final vowel (a single word-final ᠠ/ᠡ), which must keep its MVS. Getting this
wrong silently corrupts every suffix glyph in the rendered pages while the
OCR labels (which stay nominal) keep looking correct.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.render_mn_pages import (  # noqa: E402
    _chunk_words,
    _normalize,
    _page_html,
    _to_presentation,
)

MVS = "᠎"
NNBSP = " "


class ToPresentationTest(unittest.TestCase):
    def test_separated_final_vowel_keeps_mvs(self):
        # ᠭᠠᠵᠠᠷ᠎ᠠ: a single word-final ᠠ after MVS is a true separated vowel.
        word = "ᠭᠠᠵᠠᠷ" + MVS + "ᠠ"
        self.assertEqual(_to_presentation(word), word)
        # …also when a non-letter follows the vowel (space/punctuation).
        self.assertEqual(_to_presentation(word + " x"), word + " x")
        e_word = "ᠦᠨᠡᠭᠡ" + MVS + "ᠡ"
        self.assertEqual(_to_presentation(e_word), e_word)

    def test_suffix_separator_becomes_nnbsp(self):
        # MVS before a multi-letter suffix (ᠠᠴᠠ): nominal suffix separator,
        # must surface as NNBSP or fonts shape every suffix glyph wrong.
        self.assertEqual(
            _to_presentation("ᠭᠠᠵᠠᠷ" + MVS + "ᠠᠴᠠ"),
            "ᠭᠠᠵᠠᠷ" + NNBSP + "ᠠᠴᠠ",
        )
        # MVS before a non-vowel letter is likewise a separator.
        self.assertEqual(
            _to_presentation("ᠭᠠᠵᠠᠷ" + MVS + "ᠲᠠᠢ"),
            "ᠭᠠᠵᠠᠷ" + NNBSP + "ᠲᠠᠢ",
        )

    def test_trailing_mvs_maps_to_nnbsp(self):
        # A dangling MVS (malformed input) is not a separated vowel; pin the
        # fall-through so the choke point stays deterministic.
        self.assertEqual(_to_presentation("ᠠᠪ" + MVS), "ᠠᠪ" + NNBSP)

    def test_text_without_mvs_passes_through(self):
        text = "ᠮᠣᠩᠭᠣᠯ 123 abc" + NNBSP + "ᠳᠤ"
        self.assertEqual(_to_presentation(text), text)


class PageHtmlTest(unittest.TestCase):
    def test_html_carries_presentation_form_not_nominal(self):
        # _page_html is the single choke point between nominal data and
        # rendered pixels: the emitted HTML must hold the presentation form.
        p = {
            "text": "ᠭᠠᠵᠠᠷ" + MVS + "ᠠᠴᠠ",
            "title": "",
            "font_px": 24,
            "margin_px": 50,
            "line_height": 1.5,
            "fg": "#101010",
            "bg": "#ffffff",
        }
        html = _page_html("file:///font.ttf", 800, p)
        self.assertIn("ᠭᠠᠵᠠᠷ" + NNBSP + "ᠠᠴᠠ", html)
        self.assertNotIn(MVS + "ᠠᠴᠠ", html)


class ChunkWordsTest(unittest.TestCase):
    def test_budget_respected_without_losing_or_splitting_words(self):
        words = [f"w{i}" for i in range(20)]
        chunks = _chunk_words(words, budget_chars=10)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 10)
        self.assertEqual(" ".join(chunks).split(" "), words)

    def test_oversized_word_still_emitted_whole(self):
        chunks = _chunk_words(["ᠠ" * 50], budget_chars=10)
        self.assertEqual(chunks, ["ᠠ" * 50])

    def test_nnbsp_suffix_words_stay_glued(self):
        # Chunking splits on plain spaces only; an NNBSP-joined suffix is one
        # word and must never be torn apart at a chunk boundary.
        glued = "ᠭᠠᠵᠠᠷ" + NNBSP + "ᠠᠴᠠ"
        chunks = _chunk_words([glued, "ᠠᠪ"], budget_chars=4)
        self.assertEqual(chunks[0], glued)


class NormalizeTest(unittest.TestCase):
    def test_layout_whitespace_collapses_but_nnbsp_survives(self):
        raw = "ᠨᠢᠭᠡ\n\nᠬᠣᠶᠠᠷ\t ᠭᠤᠷᠪᠠ" + NNBSP + "ᠳᠤ  ᠳᠥᠷᠪᠡ"
        self.assertEqual(
            _normalize(raw),
            "ᠨᠢᠭᠡ ᠬᠣᠶᠠᠷ ᠭᠤᠷᠪᠠ" + NNBSP + "ᠳᠤ ᠳᠥᠷᠪᠡ",
        )

    def test_edges_stripped(self):
        self.assertEqual(_normalize("  ᠠᠪ \n"), "ᠠᠪ")


if __name__ == "__main__":
    unittest.main()
