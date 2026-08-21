# -*- coding: utf-8 -*-

"""MVS/FVS preservation (方案 B): control characters survive encode→decode.

The v2-era encoder folded MVS/FVS/NIRUGU at the vocab layer, so
``ᠠᠴᠠ᠎ᠠ`` (separated suffix) and ``ᠠᠴᠠᠠ`` (joined) encoded to identical
ids and the model could never learn to emit the separation mark — the
GPT-2 Turkish-İ class of normalization loss, at 2.65% of mn corpus chars.
Preservation is gated on the vocab carrying the control characters as
tokens, so legacy (v2) vocabularies keep their old behavior bit-for-bit.
"""

import unittest

from Tokenizer.generic_bpe import GeneralBPEModel
from Tokenizer.morphbpe import MorphBPETokenizer
from Tokenizer.morphbpe.trainer import MorphBPETrainer
from Tokenizer.traditional_mongolian import MVS
from Tokenizer.traditional_mongolian.stemmer import MongolStemmer
from Tokenizer.traditional_mongolian.unicode_norm import CTRL_ALL, FVS1
from Tokenizer.unified.dual_tokenizer import DualTrackTokenizer, build_unified_vocab

BICIG_MERGES = {
    ("ᠪ", "ᠢ"): ("ᠪᠢ", 0),
    ("ᠪᠢ", "ᠴ"): ("ᠪᠢᠴ", 1),
    ("ᠪᠢᠴ", "ᠢ"): ("ᠪᠢᠴᠢ", 2),
    ("ᠪᠢᠴᠢ", "ᠭ"): ("ᠪᠢᠴᠢᠭ", 3),
    ("ᠦ", "ᠨ"): ("ᠦᠨ", 4),
}


def _vocab(with_controls: bool) -> dict[str, int]:
    vocab = {"<unk>": 0, "ᠪᠢᠴᠢᠭ": 1, "ᠦᠨ": 2, "ᠠ": 3, "ᠴ": 4, "ᠠᠴᠠ": 5}
    if with_controls:
        for ch in sorted(CTRL_ALL):
            vocab.setdefault(ch, len(vocab))
    return vocab


class PreservingEncodeTest(unittest.TestCase):
    def setUp(self):
        self.tok = MorphBPETokenizer(
            vocab=_vocab(with_controls=True),
            merges=dict(BICIG_MERGES),
            stemmer=MongolStemmer(),
        )

    def test_mvs_word_emits_control_token_with_exact_offsets(self):
        word = "ᠪᠢᠴᠢᠭ" + MVS + "ᠦᠨ"
        tokens = self.tok.encode_with_offsets(word)
        surfaces = [t.token for t in tokens]
        self.assertEqual(surfaces, ["ᠪᠢᠴᠢᠭ", MVS, "ᠦᠨ"])
        self.assertEqual(
            [(t.start, t.end) for t in tokens], [(0, 5), (5, 6), (6, 8)]
        )
        # The control token must be a real id, not <unk>.
        self.assertNotEqual(tokens[1].id, self.tok.vocab["<unk>"])

    def test_separated_and_joined_suffix_encode_differently(self):
        separated = self.tok.encode("ᠠᠴᠠ" + MVS + "ᠠ")
        joined = self.tok.encode("ᠠᠴᠠᠠ")
        self.assertNotEqual(separated, joined)

    def test_fvs_word_round_trips_at_surface_level(self):
        word = "ᠠ" + FVS1 + "ᠴᠠ"
        tokens = self.tok.encode_with_offsets(word)
        self.assertEqual("".join(t.token for t in tokens), word)

    def test_controls_only_word(self):
        tokens = self.tok.encode_with_offsets(MVS)
        self.assertEqual([t.token for t in tokens], [MVS])


class LegacyVocabCompatTest(unittest.TestCase):
    """Vocabularies without control tokens (v2 bundles) keep folding."""

    def setUp(self):
        self.tok = MorphBPETokenizer(
            vocab=_vocab(with_controls=False),
            merges=dict(BICIG_MERGES),
            stemmer=MongolStemmer(),
        )

    def test_mvs_still_folds_and_ids_match_joined_form(self):
        word = "ᠪᠢᠴᠢᠭ" + MVS + "ᠦᠨ"
        tokens = self.tok.encode_with_offsets(word)
        self.assertEqual([t.token for t in tokens], ["ᠪᠢᠴᠢᠭ", "ᠦᠨ"])
        # Legacy spans swallow the control char (pinned v2 behavior).
        self.assertEqual([(t.start, t.end) for t in tokens], [(0, 5), (5, 8)])


class TrainerRegistersControlsTest(unittest.TestCase):
    def test_fresh_training_yields_preserving_tokenizer(self):
        trainer = MorphBPETrainer(vocab_size=64, min_pair_freq=1)
        tok = trainer.train(["ᠪᠢᠴᠢᠭ" + MVS + "ᠦᠨ", "ᠠᠴᠠ" + MVS + "ᠠ"])
        for ch in CTRL_ALL:
            self.assertIn(ch, tok.vocab)
        ids_sep = tok.encode("ᠠᠴᠠ" + MVS + "ᠠ")
        ids_join = tok.encode("ᠠᠴᠠᠠ")
        self.assertNotEqual(ids_sep, ids_join)


class UnifiedRoundTripTest(unittest.TestCase):
    """End-to-end through the dual-track tokenizer: decode restores marks."""

    def setUp(self):
        morph = MorphBPETokenizer(
            vocab=_vocab(with_controls=True),
            merges=dict(BICIG_MERGES),
            stemmer=MongolStemmer(),
        )
        general = GeneralBPEModel.minimal()
        vocab = build_unified_vocab(
            morphbpe_vocab=morph.vocab,
            general_vocab=general.get_vocab(),
        )
        self.dual = DualTrackTokenizer(vocab, morph, general)

    def test_mvs_round_trips_through_unified_decode(self):
        text = "ᠪᠢᠴᠢᠭ" + MVS + "ᠦᠨ"
        ids = [t.id for t in self.dual.encode_with_spans(text).tokens]
        self.assertEqual(self.dual.decode(ids), text)

    def test_mixed_sentence_round_trips(self):
        text = "ᠠᠴᠠ" + MVS + "ᠠ test ᠠ" + FVS1 + "ᠴᠠ"
        ids = [t.id for t in self.dual.encode_with_spans(text).tokens]
        self.assertEqual(self.dual.decode(ids), text)


if __name__ == "__main__":
    unittest.main()
