# -*- coding: utf-8 -*-
"""Tests for generic BPE: general byte-level BPE wrapper and byte fallback."""

from __future__ import annotations

import os
import tempfile
import unittest

from Tokenizer.generic_bpe import (
    GeneralBPEModel,
    GeneralBPETrainer,
    decode_bytes,
    encode_byte_fallback,
    is_byte_token,
)


class GeneralBPEModelTests(unittest.TestCase):
    def test_minimal_is_lossless_byte_level(self):
        model = GeneralBPEModel.minimal()
        for text in ("hello", "中文", "日本語", "🙂", "Кириллица"):
            pieces = model.encode_pieces(text)
            self.assertTrue(pieces)
            local_ids = [p[0] for p in pieces]
            self.assertEqual(model.decode(local_ids), text)

    def test_minimal_offsets_are_in_bounds(self):
        model = GeneralBPEModel.minimal()
        text = "中a文"
        pieces = model.encode_pieces(text)
        for _local_id, _tok, start, end in pieces:
            self.assertLessEqual(0, start)
            self.assertLessEqual(start, end)
            self.assertLessEqual(end, len(text))

    def test_train_compresses_and_round_trips(self):
        trainer = GeneralBPETrainer(vocab_size=500, min_frequency=1)
        corpus = ["hello world"] * 50 + ["中文测试"] * 50
        model = trainer.train(corpus)
        self.assertGreaterEqual(model.vocab_size, 256)
        for text in ("hello world", "中文测试", "🙂"):
            pieces = model.encode_pieces(text)
            self.assertEqual(model.decode([p[0] for p in pieces]), text)

    def test_save_and_load_round_trip(self):
        trainer = GeneralBPETrainer(vocab_size=400, min_frequency=1)
        model = trainer.train(["abcabc abc"] * 40)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "general.json")
            model.save(path)
            loaded = GeneralBPEModel.load(path)
        self.assertEqual(loaded.get_vocab(), model.get_vocab())
        pieces = loaded.encode_pieces("abcabc")
        self.assertEqual(loaded.decode([p[0] for p in pieces]), "abcabc")


class ByteFallbackTests(unittest.TestCase):
    def test_emoji_roundtrip(self):
        vocab = {f"<0x{i:02X}>": 1000 + i for i in range(256)}
        vocab["<unk>"] = 0
        tokens = encode_byte_fallback("🙂", vocab, unk_id=0)
        byte_strs = [t.token for t in tokens]
        self.assertTrue(all(is_byte_token(t) for t in byte_strs))
        self.assertEqual(decode_bytes(byte_strs), "🙂")

    def test_cjk_punct_byte_fallback(self):
        vocab = {f"<0x{i:02X}>": 1000 + i for i in range(256)}
        vocab["<unk>"] = 0
        tokens = encode_byte_fallback("，", vocab, unk_id=0)
        self.assertTrue(all(is_byte_token(t.token) for t in tokens))
        # CJK comma is 3 bytes in UTF-8
        self.assertEqual(len(tokens), 3)
        self.assertEqual(decode_bytes([t.token for t in tokens]), "，")

    def test_direct_token_used_when_present(self):
        vocab = {"a": 7, "<unk>": 0}
        tokens = encode_byte_fallback("a", vocab, unk_id=0)
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].id, 7)
        self.assertEqual(tokens[0].token, "a")

    def test_offsets_monotonic(self):
        vocab = {f"<0x{i:02X}>": 1000 + i for i in range(256)}
        vocab["<unk>"] = 0
        tokens = encode_byte_fallback("🙂a，", vocab, unk_id=0, base_start=5)
        last = 5
        for t in tokens:
            self.assertGreaterEqual(t.start, last - 1)
            self.assertLessEqual(t.start, t.end)
            last = t.start

    def test_lone_surrogate_does_not_crash(self):
        vocab = {f"<0x{i:02X}>": 1000 + i for i in range(256)}
        vocab["<unk>"] = 0
        tokens = encode_byte_fallback("a\ud800b", vocab, unk_id=0)
        self.assertTrue(any(is_byte_token(t.token) for t in tokens))
        self.assertTrue(all(isinstance(t.id, int) for t in tokens))


if __name__ == "__main__":
    unittest.main()
