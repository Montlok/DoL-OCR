# -*- coding: utf-8 -*-

"""Regression tests for ``build_unified_vocab`` segment overflow handling.

Oversized MorphBPE/general vocabularies must fail fast instead of being
silently truncated, since truncated local ids would later be emitted as
``<unk>`` and break the no-``<unk>`` guarantee.
"""

import unittest
from unittest import mock

from Tokenizer.unified import vocab as vocab_mod
from Tokenizer.unified.vocab import build_unified_vocab

# Tiny id layout so we can overflow a segment with a handful of tokens.
_TINY_SEGMENT = {
    "special": (0, 4),
    "mongolian": (4, 8),
    "general": (8, 12),
}


def _toks(prefix: str, n: int) -> dict[str, int]:
    return {f"{prefix}{i}": i for i in range(n)}


class BuildUnifiedVocabOverflowTest(unittest.TestCase):
    def test_fits_within_segments(self):
        with mock.patch.dict(vocab_mod.SEGMENT, _TINY_SEGMENT, clear=True):
            unified = build_unified_vocab(_toks("m", 4), _toks("g", 4))
        # 4 Mongolian + 4 general tokens fit exactly in their tiny segments.
        self.assertEqual(unified["m0"], 4)
        self.assertEqual(unified["m3"], 7)
        self.assertEqual(unified["g0"], 8)
        self.assertEqual(unified["g3"], 11)

    def test_general_overflow_raises(self):
        with mock.patch.dict(vocab_mod.SEGMENT, _TINY_SEGMENT, clear=True):
            with self.assertRaises(ValueError) as ctx:
                build_unified_vocab(_toks("m", 4), _toks("g", 5))
        self.assertIn("general BPE vocabulary overflows", str(ctx.exception))

    def test_mongolian_overflow_raises(self):
        with mock.patch.dict(vocab_mod.SEGMENT, _TINY_SEGMENT, clear=True):
            with self.assertRaises(ValueError) as ctx:
                build_unified_vocab(_toks("m", 5), _toks("g", 1))
        self.assertIn("MorphBPE vocabulary overflows", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
