# -*- coding: utf-8 -*-

import os
import tempfile
import unittest

from Tokenizer.morphbpe import MorphBPETokenizer
from Tokenizer.morphbpe.trainer import MorphBPETrainer
from Tokenizer.traditional_mongolian import MVS
from Tokenizer.traditional_mongolian.stemmer import MongolStemmer


class MorphBPETest(unittest.TestCase):
    def setUp(self):
        self.stemmer = MongolStemmer()

    def test_does_not_merge_across_root_suffix_boundary(self):
        word = "ᠪᠢᠴᠢᠭ" + MVS + "ᠦᠨ"
        vocab = {"<unk>": 0, "ᠪᠢᠴᠢᠭ": 1, "ᠦᠨ": 2, "ᠭᠦ": 3}
        merges = {
            ("ᠪ", "ᠢ"): ("ᠪᠢ", 0),
            ("ᠪᠢ", "ᠴ"): ("ᠪᠢᠴ", 1),
            ("ᠪᠢᠴ", "ᠢ"): ("ᠪᠢᠴᠢ", 2),
            ("ᠪᠢᠴᠢ", "ᠭ"): ("ᠪᠢᠴᠢᠭ", 3),
            ("ᠭ", "ᠦ"): ("ᠭᠦ", 4),
            ("ᠦ", "ᠨ"): ("ᠦᠨ", 5),
        }
        tokenizer = MorphBPETokenizer(vocab=vocab, merges=merges, stemmer=self.stemmer)
        tokens = tokenizer.encode_with_offsets(word)
        surfaces = [token.token for token in tokens]
        self.assertIn("ᠪᠢᠴᠢᠭ", surfaces)
        self.assertIn("ᠦᠨ", surfaces)
        self.assertNotIn("ᠭᠦ", surfaces)

    def test_encode_with_offsets_returns_original_offsets(self):
        word = "ᠪᠢᠴᠢᠭ" + MVS + "ᠦᠨ"
        tokenizer = MorphBPETokenizer(
            vocab={"<unk>": 0, "ᠪᠢᠴᠢᠭ": 1, "ᠦᠨ": 2},
            merges={
                ("ᠪ", "ᠢ"): ("ᠪᠢ", 0),
                ("ᠪᠢ", "ᠴ"): ("ᠪᠢᠴ", 1),
                ("ᠪᠢᠴ", "ᠢ"): ("ᠪᠢᠴᠢ", 2),
                ("ᠪᠢᠴᠢ", "ᠭ"): ("ᠪᠢᠴᠢᠭ", 3),
                ("ᠦ", "ᠨ"): ("ᠦᠨ", 4),
            },
            stemmer=self.stemmer,
        )
        tokens = tokenizer.encode_with_offsets(word)
        self.assertEqual((tokens[0].start, tokens[0].end), (0, 5))
        self.assertEqual((tokens[-1].start, tokens[-1].end), (5, 8))

    def test_save_from_file_round_trip(self):
        tokenizer = MorphBPETokenizer(
            vocab={"<unk>": 0, "ᠪ": 1},
            merges={("ᠪ", "ᠢ"): ("ᠪᠢ", 0)},
            stemmer=self.stemmer,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "morphbpe.json")
            tokenizer.save(path)
            loaded = MorphBPETokenizer.from_file(path, self.stemmer)
        self.assertEqual(loaded.vocab, tokenizer.vocab)
        self.assertEqual(loaded.merges, tokenizer.merges)

    def test_unknown_fallback_does_not_crash(self):
        tokenizer = MorphBPETokenizer(vocab={"<unk>": 0}, stemmer=self.stemmer)
        ids = tokenizer.encode("☃")
        self.assertEqual(ids, [0])

    def test_tokenize_word_offsets_monotonic(self):
        word = "ᠪᠢᠴᠢᠭ" + MVS + "ᠦᠨ"
        tokenizer = MorphBPETokenizer(
            vocab={"<unk>": 0, "ᠪ": 1, "ᠢ": 2, "ᠴ": 3, "ᠭ": 4, "ᠦ": 5, "ᠨ": 6},
            stemmer=self.stemmer,
        )
        tokens = tokenizer.tokenize_word(word)
        last = -1
        for t in tokens:
            self.assertLessEqual(t.start, t.end)
            self.assertGreaterEqual(t.start, last)
            last = t.start
        # Last token end must reach end of word
        self.assertEqual(tokens[-1].end, len(word))

    def test_decode_smoke(self):
        tokenizer = MorphBPETokenizer(
            vocab={"<unk>": 0, "ᠪ": 1, "ᠢ": 2},
            stemmer=self.stemmer,
        )
        self.assertEqual(tokenizer.decode([1, 2]), "ᠪᠢ")
        # <unk> renders as replacement char, never crashes
        self.assertEqual(tokenizer.decode([0]), "\ufffd")

    def test_low_confidence_stemmer_boundary_does_not_block_lexical_merge(self):
        word = "ᠮᠣᠩᠭᠣᠯ"
        tokenizer = MorphBPETokenizer(
            vocab={
                "<unk>": 0,
                "ᠮᠣ": 1,
                "ᠩᠭᠣ": 2,
                "ᠯ": 3,
                "ᠮᠣᠩᠭᠣᠯ": 4,
            },
            merges={
                ("ᠮ", "ᠣ"): ("ᠮᠣ", 0),
                ("ᠩ", "ᠭ"): ("ᠩᠭ", 1),
                ("ᠩᠭ", "ᠣ"): ("ᠩᠭᠣ", 2),
                ("ᠮᠣ", "ᠩᠭᠣ"): ("ᠮᠣᠩᠭᠣ", 3),
                ("ᠮᠣᠩᠭᠣ", "ᠯ"): ("ᠮᠣᠩᠭᠣᠯ", 4),
            },
            stemmer=self.stemmer,
        )
        self.assertEqual(
            [token.token for token in tokenizer.tokenize_word(word)], [word]
        )

    def test_control_chars_do_not_crash(self):
        from Tokenizer.traditional_mongolian import FVS1, NNBSP

        word = "ᠪᠢ" + FVS1 + "ᠴ" + NNBSP + "ᠦᠨ"
        tokenizer = MorphBPETokenizer(
            vocab={"<unk>": 0, "ᠪ": 1, "ᠢ": 2, "ᠴ": 3, "ᠦ": 4, "ᠨ": 5},
            stemmer=self.stemmer,
        )
        # NNBSP splits a word in unicode_norm; we just need no crash + offsets valid.
        tokens = tokenizer.encode_with_offsets(word)
        for t in tokens:
            self.assertGreaterEqual(t.start, 0)
            self.assertLessEqual(t.end, len(word))

    def test_save_load_preserves_encoding(self):
        tokenizer = MorphBPETokenizer(
            vocab={"<unk>": 0, "ᠪ": 1, "ᠢ": 2, "ᠪᠢ": 3},
            merges={("ᠪ", "ᠢ"): ("ᠪᠢ", 0)},
            stemmer=self.stemmer,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mb.json")
            tokenizer.save(path)
            loaded = MorphBPETokenizer.from_file(path, self.stemmer)
        text = "ᠪᠢ"
        self.assertEqual(tokenizer.encode(text), loaded.encode(text))

    def test_trainer_ignores_non_mongolian_words_in_mixed_corpus(self):
        trainer = MorphBPETrainer(stemmer=self.stemmer, vocab_size=64, min_pair_freq=1)
        tokenizer = trainer.train(["ᠮᠣᠩᠭᠣᠯ text 中文 123"])
        self.assertNotIn("t", tokenizer.vocab)
        self.assertNotIn("文", tokenizer.vocab)
        self.assertIn("ᠮ", tokenizer.vocab)

    def test_trainer_seeds_mongolian_alphabet_for_unseen_letters(self):
        trainer = MorphBPETrainer(stemmer=self.stemmer, vocab_size=64, min_pair_freq=2)
        tokenizer = trainer.train(["ᠮᠣᠩᠭᠣᠯ"])

        self.assertIn("ᠠ", tokenizer.vocab)
        self.assertNotEqual(tokenizer.encode("ᠠ"), [tokenizer.vocab["<unk>"]])
        self.assertTrue(tokenizer.seed_alphabet)

    def test_trainer_can_disable_alphabet_seed(self):
        trainer = MorphBPETrainer(
            stemmer=self.stemmer,
            vocab_size=64,
            min_pair_freq=2,
            seed_alphabet=False,
        )
        tokenizer = trainer.train(["ᠮᠣᠩᠭᠣᠯ"])

        self.assertFalse(tokenizer.seed_alphabet)
        self.assertEqual(tokenizer.encode("ᠠ"), [tokenizer.vocab["<unk>"]])


if __name__ == "__main__":
    unittest.main()
