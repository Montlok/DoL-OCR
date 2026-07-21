# -*- coding: utf-8 -*-

"""OCR target encoding: no <unk>, exact round-trip, byte-fallback safety.

``scripts.build_ocr_data.make_ocr_target_encoder`` builds a closure that
routes OCR supervision targets through the tokenizer's existing byte-fallback
machinery instead of the lossy MorphBPE track. These tests exercise it
against an in-memory fixture (no real tokenizer bundle needed) that
reproduces two failure modes found during development:

- the ByteLevel-alphabet trap, where characters like ``"ä"``/``"Ġ"`` directly
  hit general-track ids that ``DualTrackTokenizer.decode`` reinterprets as
  raw bytes;
- ``"▁"``/``"◈"`` (ids 17/18), which ``decode`` special-cases to a literal
  space / drop rather than round-tripping as themselves.
"""

import unittest

from Tokenizer.generic_bpe import GeneralBPEModel
from Tokenizer.unified.dual_tokenizer import DualTrackTokenizer, build_unified_vocab

from scripts.build_ocr_data import make_ocr_target_encoder

# Traditional Mongolian control characters (see
# Tokenizer/traditional_mongolian/unicode_norm.py for the canonical set).
NIRUGU = "᠊"
FVS1 = "᠋"
FVS2 = "᠌"
FVS3 = "᠍"
MVS = "᠎"
FVS4 = "᠏"
NNBSP = " "

# Seeded single-char Mongolian letters (covered by the fixture vocab).
LETTER_A = "ᠠ"  # MONGOLIAN LETTER A
LETTER_NA = "ᠨ"  # MONGOLIAN LETTER NA
LETTER_GA = "ᠭ"  # MONGOLIAN LETTER GA
LETTER_RA = "ᠷ"  # MONGOLIAN LETTER RA
SEEDED_LETTERS = (LETTER_A, LETTER_NA, LETTER_GA, LETTER_RA)

# Deliberately NOT in the fixture vocab: must fall back to <0xNN> bytes.
UNCOVERED_LETTER = "ᡀ"  # MONGOLIAN LETTER LHA

MULTI_CHAR_WORD = "ᠮᠣᠩᠭᠣᠯ"  # "ᠮᠣᠩᠭᠣᠯ"


class FakeMorphBPE:
    """Minimal MorphBPE stand-in, same shape as test_stream_decode.py's.

    Byte tokens live in the MorphBPE vocab in real bundles (the mn-track
    byte fallback), so the fixture puts them there too.
    """

    vocab: dict[str, int] = {"<unk>": 0}
    vocab.update({f"<0x{i:02X}>": i + 1 for i in range(256)})
    for _i, _ch in enumerate(SEEDED_LETTERS):
        vocab[_ch] = 300 + _i
    for _i, _ch in enumerate((NIRUGU, FVS1, FVS2, FVS3, MVS, FVS4, NNBSP)):
        vocab[_ch] = 400 + _i
    vocab[MULTI_CHAR_WORD] = 500
    del _i, _ch

    def encode(self, text):
        return [self.vocab[text]]


def _make_tokenizer(morphbpe_vocab: dict[str, int]) -> DualTrackTokenizer:
    class _MorphBPE:
        vocab = morphbpe_vocab

        def encode(self, text):
            return [self.vocab[text]]

    general = GeneralBPEModel.minimal()
    vocab = build_unified_vocab(
        morphbpe_vocab=morphbpe_vocab, general_vocab=general.get_vocab()
    )
    return DualTrackTokenizer(vocab, _MorphBPE(), general)


# Adversarial single-character inputs: uncovered Mongolian letter, every
# control character, plain whitespace, a CJK character, an emoji (surrogate
# pair in UTF-16 but a single code point in Python str), and the three
# characters implicated in the two documented traps.
ADVERSARIAL_CHARS = (
    UNCOVERED_LETTER,
    FVS1,
    FVS2,
    FVS3,
    FVS4,
    MVS,
    NNBSP,
    " ",
    "\t",
    "\n",
    "中",  # CJK "middle"
    "\U0001f642",  # slightly smiling face
    "ä",
    "Ġ",
    "▁",  # "▁"
    "◈",  # "◈"
)


class OCRTargetEncodingTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = _make_tokenizer(dict(FakeMorphBPE.vocab))
        self.encode_target = make_ocr_target_encoder(self.tokenizer)

    def test_zero_unk_on_adversarial_inputs(self):
        for ch in ADVERSARIAL_CHARS:
            with self.subTest(ch=repr(ch)):
                ids = self.encode_target(ch)
                self.assertNotIn(self.tokenizer.unk_id, ids)

    def test_byte_exact_roundtrip_on_adversarial_inputs(self):
        for ch in ADVERSARIAL_CHARS:
            with self.subTest(ch=repr(ch)):
                ids = self.encode_target(ch)
                decoded = self.tokenizer.decode(ids)
                self.assertEqual(decoded, ch)
                self.assertEqual(
                    decoded.encode("utf-8", "surrogatepass"),
                    ch.encode("utf-8", "surrogatepass"),
                )

    def test_space_marker_chars_avoid_ids_17_and_18(self):
        # "▁" (id 17) and "◈" (id 18) are special-cased by decode() to " "
        # and dropped, respectively; encoding them must never route through
        # those exact ids, and they must still round-trip to themselves.
        for ch, forbidden_id in (("▁", 17), ("◈", 18)):
            with self.subTest(ch=repr(ch)):
                ids = self.encode_target(ch)
                self.assertNotIn(forbidden_id, ids)
                self.assertEqual(self.tokenizer.decode(ids), ch)

    def test_bytelevel_alphabet_trap_char_avoids_general_segment(self):
        # "ä" direct-hits a general-track id in the raw ByteLevel alphabet;
        # the safe vocab must exclude it so it is forced through <0xNN>.
        ids = self.encode_target("ä")
        for token_id in ids:
            self.assertNotIn(token_id, self.tokenizer.general_global_to_local)
        self.assertEqual(self.tokenizer.decode(ids), "ä")

    def test_seeded_mongolian_letters_are_one_token_each(self):
        for ch in SEEDED_LETTERS:
            with self.subTest(ch=repr(ch)):
                ids = self.encode_target(ch)
                self.assertEqual(len(ids), 1)
                self.assertEqual(self.tokenizer.decode(ids), ch)

    def test_multi_char_word_still_roundtrips_via_bytes(self):
        # The safe vocab only keeps single-char tokens, so a multi-char
        # MorphBPE word token is not directly usable here; the target
        # encoder must still produce a byte-exact round trip for it (each
        # character encoded independently, some via seeded letters/word
        # fragments, the rest via <0xNN>).
        ids = self.encode_target(MULTI_CHAR_WORD)
        self.assertNotIn(self.tokenizer.unk_id, ids)
        self.assertEqual(self.tokenizer.decode(ids), MULTI_CHAR_WORD)

    def test_missing_byte_tokens_raises_with_remedy(self):
        sparse_vocab = {"<unk>": 0, LETTER_A: 300}
        tokenizer = _make_tokenizer(sparse_vocab)
        with self.assertRaisesRegex(ValueError, "byte-fallback tokens"):
            make_ocr_target_encoder(tokenizer)

    def test_lone_surrogate_is_rejected_by_roundtrip_check(self):
        text = "a\ud800b"
        with self.assertRaisesRegex(ValueError, "round-trip mismatch"):
            self.encode_target(text)


if __name__ == "__main__":
    unittest.main()
