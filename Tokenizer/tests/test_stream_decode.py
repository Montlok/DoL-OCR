# -*- coding: utf-8 -*-

"""StreamDecoder: token-at-a-time decode must never surface transient U+FFFD.

Every Traditional Mongolian letter is a 3-byte UTF-8 sequence, so byte-level
fallback guarantees code points split across token boundaries; the llama.cpp/
HF streaming bug class emits replacement characters mid-word and they vanish
on re-decode — exactly the kind of corruption a reader cannot attribute.
"""

import unittest

from Tokenizer.generic_bpe import GeneralBPEModel
from Tokenizer.unified.dual_tokenizer import DualTrackTokenizer, build_unified_vocab
from Tokenizer.unified.stream_decode import StreamDecoder


class ByteChunkTokenizer:
    """Minimal stand-in: every id maps to a fixed byte string; decode mimics
    the bundle's byte-fallback reassembly (utf-8 with errors='replace')."""

    def __init__(self, chunks):
        self.chunks = chunks

    def decode(self, ids):
        return b"".join(self.chunks[i] for i in ids).decode("utf-8", errors="replace")


class StreamDecoderUnitTest(unittest.TestCase):
    def test_mongolian_bytes_split_across_tokens(self):
        text = "ᠮᠣᠩᠭᠣᠯ ᠪᠢᠴᠢᠭ"
        raw = text.encode("utf-8")
        tok = ByteChunkTokenizer({i: raw[i : i + 1] for i in range(len(raw))})
        sd = StreamDecoder(tok)
        emitted = []
        for i in range(len(raw)):
            piece = sd.push(i)
            self.assertNotIn("�", piece)
            emitted.append(piece)
        emitted.append(sd.flush())
        self.assertEqual("".join(emitted), text)

    def test_uneven_chunks_crossing_codepoints(self):
        text = "蒙文 ᠦᠰᠦᠭ mixed 字"
        raw = text.encode("utf-8")
        sizes = [1, 2, 3, 1, 4, 2]
        chunks, pos, i = {}, 0, 0
        while pos < len(raw):
            n = sizes[i % len(sizes)]
            chunks[i] = raw[pos : pos + n]
            pos += n
            i += 1
        sd = StreamDecoder(ByteChunkTokenizer(chunks))
        pieces = [sd.push(j) for j in range(i)]
        for piece in pieces:
            self.assertNotIn("�", piece)
        self.assertEqual("".join(pieces) + sd.flush(), text)

    def test_genuine_replacement_char_is_delayed_not_lost(self):
        # 0xFF can never start a UTF-8 sequence: the decoder must hold it
        # while it could still be a truncation, then emit it once non-FFFD
        # text follows. Nothing is dropped.
        tok = ByteChunkTokenizer({0: b"ab", 1: b"\xff", 2: b"cd"})
        sd = StreamDecoder(tok)
        out = sd.push(0) + sd.push(1) + sd.push(2) + sd.flush()
        self.assertEqual(out, "ab�cd")

    def test_flush_emits_truncated_tail(self):
        raw = "ᠮ".encode("utf-8")  # 3 bytes, never completed
        sd = StreamDecoder(ByteChunkTokenizer({0: raw[:1], 1: raw[1:2]}))
        self.assertEqual(sd.push(0) + sd.push(1), "")
        self.assertIn("�", sd.flush())

    def test_flush_resets_for_reuse(self):
        raw = "ᠭ".encode("utf-8")
        tok = ByteChunkTokenizer({i: raw[i : i + 1] for i in range(3)})
        sd = StreamDecoder(tok)
        first = "".join(sd.push(i) for i in range(3)) + sd.flush()
        second = "".join(sd.push(i) for i in range(3)) + sd.flush()
        self.assertEqual(first, "ᠭ")
        self.assertEqual(second, "ᠭ")


class FakeMorphBPE:
    # Byte tokens live in the MorphBPE vocab in real bundles (the mn-track
    # byte fallback); mirror that layout so decode exercises the same path.
    vocab = {f"<0x{i:02X}>": i for i in range(256)}
    vocab["ᠮᠣᠩᠭᠣᠯ"] = 256

    def encode(self, text):
        return [self.vocab[text]]


class StreamDecoderBundleTest(unittest.TestCase):
    """Integration against the real DualTrackTokenizer byte-fallback decode."""

    def setUp(self):
        general = GeneralBPEModel.minimal()
        vocab = build_unified_vocab(
            morphbpe_vocab=FakeMorphBPE.vocab,
            general_vocab=general.get_vocab(),
        )
        self.tokenizer = DualTrackTokenizer(vocab, FakeMorphBPE(), general)

    def test_byte_fallback_ids_stream_safely(self):
        char = "中"  # 3 bytes, decoded purely from <0xXX> fallback tokens
        ids = [self.tokenizer.vocab[f"<0x{b:02X}>"] for b in char.encode("utf-8")]
        sd = StreamDecoder(self.tokenizer)
        pieces = [sd.push(i) for i in ids]
        self.assertEqual(pieces[:2], ["", ""])
        self.assertEqual("".join(pieces) + sd.flush(), char)

    def test_mixed_morph_and_byte_tokens(self):
        word_id = self.tokenizer.vocab["ᠮᠣᠩᠭᠣᠯ"]
        byte_ids = [self.tokenizer.vocab[f"<0x{b:02X}>"] for b in "中".encode("utf-8")]
        sd = StreamDecoder(self.tokenizer)
        out = sd.push(word_id)
        for i in byte_ids:
            piece = sd.push(i)
            self.assertNotIn("�", piece)
            out += piece
        out += sd.flush()
        self.assertEqual(out, "ᠮᠣᠩᠭᠣᠯ中")


if __name__ == "__main__":
    unittest.main()
