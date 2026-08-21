# -*- coding: utf-8 -*-

import unittest

from Tokenizer.evals.mongolian_boundary_recall import compute_metrics
from Tokenizer.evals.tokenizer_hit_rate import (
    build_experimental_tokenizer,
    compute_hit_rate,
)
from Tokenizer.morphbpe.offsets import MorphToken
from Tokenizer.traditional_mongolian import MVS


class BoundaryRecallEvalTest(unittest.TestCase):
    def test_boundary_rate_uses_measured_crossings(self):
        word = "ᠪᠢᠴᠢᠭ" + MVS + "ᠦᠨ"

        class CrossingTokenizer:
            def tokenize_word(self, text):
                return [MorphToken(text, 1, 0, len(text))]

        metrics = compute_metrics([word], CrossingTokenizer())
        self.assertGreater(metrics["morpheme_boundaries"], 0)
        self.assertGreater(metrics["crossed_boundary_count"], 0)
        self.assertLess(metrics["boundary_respecting_rate"], 1.0)

    def test_boundary_rate_is_perfect_for_char_tokens(self):
        word = "ᠪᠢᠴᠢᠭ" + MVS + "ᠦᠨ"
        metrics = compute_metrics([word])
        self.assertGreater(metrics["morpheme_boundaries"], 0)
        self.assertEqual(metrics["crossed_boundary_count"], 0)
        self.assertEqual(metrics["boundary_respecting_rate"], 1.0)

    def test_hit_rate_uses_seeded_mongolian_alphabet(self):
        train = ["ᠮᠣᠩᠭᠣᠯ"]
        eval_texts = ["ᠠᠪᠤ ᠮᠣᠩᠭᠣᠯ"]
        tokenizer = build_experimental_tokenizer(train, vocab_size=64, min_pair_freq=2)

        metrics = compute_hit_rate(eval_texts, tokenizer)

        self.assertEqual(metrics["unk_count"], 0)
        self.assertEqual(metrics["token_hit_rate"], 1.0)
        self.assertEqual(metrics["mongolian_word_hit_rate"], 1.0)
        self.assertEqual(metrics["mongolian_fallback_tokens"], 0)
        self.assertEqual(metrics["mongolian_native_char_rate"], 1.0)

    def test_hit_rate_exposes_lossless_unseeded_alphabet_fallback(self):
        train = ["ᠮᠣᠩᠭᠣᠯ"]
        eval_texts = ["ᠠᠪᠤ"]
        tokenizer = build_experimental_tokenizer(
            train,
            vocab_size=64,
            min_pair_freq=2,
            seed_alphabet=False,
        )

        metrics = compute_hit_rate(eval_texts, tokenizer)

        self.assertEqual(metrics["unk_count"], 0)
        self.assertEqual(metrics["token_hit_rate"], 1.0)
        self.assertEqual(metrics["roundtrip_failures_sample"], [])
        self.assertGreater(metrics["mongolian_fallback_tokens"], 0)
        self.assertEqual(metrics["mongolian_native_char_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
