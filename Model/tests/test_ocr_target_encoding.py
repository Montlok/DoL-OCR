# -*- coding: utf-8 -*-

"""OCR target encoding: native compatibility, exact round-trip, fallback safety.

The shared encoder prefers the same native token path used for language
pretraining and falls back only when that mode is requested explicitly. These
tests exercise it against an in-memory fixture that reproduces two fallback
failure modes found during development:

- the ByteLevel-alphabet trap, where characters like ``"ä"``/``"Ġ"`` directly
  hit general-track ids that ``DualTrackTokenizer.decode`` reinterprets as
  raw bytes;
- ``"▁"``/``"◈"`` (ids 17/18), which ``decode`` special-cases to a literal
  space / drop rather than round-tripping as themselves.
"""

import json
import tempfile
import unittest
from pathlib import Path

from Tokenizer.generic_bpe import GeneralBPEModel
from Tokenizer.unified.dual_tokenizer import DualTrackTokenizer, build_unified_vocab

from Model.ocr.tokenization import (
    make_ocr_target_encoder,
    ocr_adapter_algorithm_contract,
    tokenizer_manifest_canonical_sha256,
    tokenizer_morphology_track_table,
    tokenizer_vocab_sha256,
)

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
        self.encode_target = make_ocr_target_encoder(
            self.tokenizer,
            mode="native_fallback",
        )

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

    def test_literal_control_token_text_is_rejected_by_pretraining_route(self):
        encode_native = make_ocr_target_encoder(self.tokenizer, mode="native")
        text = "▁◈<bos><image_patch>"
        with self.assertRaisesRegex(ValueError, "round-trip mismatch"):
            encode_native(text)

    def test_bytelevel_alphabet_trap_char_avoids_general_segment(self):
        # "ä" direct-hits a general-track id in the raw ByteLevel alphabet;
        # the explicit legacy byte route must exclude it so it is forced
        # through <0xNN>. Native may reuse a verified pretrained general token.
        encode_legacy = make_ocr_target_encoder(
            self.tokenizer,
            mode="byte_fallback",
        )
        ids = encode_legacy("ä")
        for token_id in ids:
            self.assertNotIn(token_id, self.tokenizer.general_global_to_local)
        self.assertEqual(self.tokenizer.decode(ids), "ä")

    def test_seeded_mongolian_letters_are_one_token_each(self):
        for ch in SEEDED_LETTERS:
            with self.subTest(ch=repr(ch)):
                ids = self.encode_target(ch)
                self.assertEqual(len(ids), 1)
                self.assertEqual(self.tokenizer.decode(ids), ch)

    def test_multi_char_word_reuses_native_morphbpe_token(self):
        ids = self.encode_target(MULTI_CHAR_WORD)
        self.assertEqual(ids, [self.tokenizer.vocab[MULTI_CHAR_WORD]])
        self.assertEqual(self.tokenizer.decode(ids), MULTI_CHAR_WORD)
        self.assertEqual(self.encode_target.stats["native"], 1)
        self.assertEqual(self.encode_target.stats["byte_fallback"], 0)

    def test_native_path_preserves_pretraining_word_boundary(self):
        text = f"{MULTI_CHAR_WORD} {MULTI_CHAR_WORD}"
        ids = self.encode_target(text)
        self.assertEqual(
            ids,
            [
                self.tokenizer.vocab[MULTI_CHAR_WORD],
                self.tokenizer.vocab["▁"],
                self.tokenizer.vocab[MULTI_CHAR_WORD],
            ],
        )
        self.assertEqual(self.tokenizer.decode(ids), text)

    def test_native_features_match_pretraining_morphology(self):
        text = f"{MULTI_CHAR_WORD} {MULTI_CHAR_WORD}"
        encode_native = make_ocr_target_encoder(
            self.tokenizer,
            mode="native",
        )
        features = encode_native.encode_with_features(text)
        self.assertEqual(
            features.input_ids,
            [
                self.tokenizer.vocab[MULTI_CHAR_WORD],
                self.tokenizer.vocab["▁"],
                self.tokenizer.vocab[MULTI_CHAR_WORD],
            ],
        )
        self.assertEqual(features.morphology_track_ids, [1, 0, 1])
        self.assertEqual(features.word_pos, [0, 0, 1])
        self.assertEqual(features.morph_depth, [0, 0, 0])

    def test_strict_native_mode_fails_closed_instead_of_changing_representation(self):
        encode_native = make_ocr_target_encoder(self.tokenizer, mode="native")
        with self.assertRaisesRegex(ValueError, "native OCR target encoding"):
            encode_native(UNCOVERED_LETTER)

    def test_strict_native_rejects_offset_aware_general_fallback(self):
        class SparseMorphBPE:
            vocab = {"<unk>": 0, LETTER_A: 1}

            def encode_with_offsets(self, text):
                from Tokenizer.morphbpe.offsets import MorphToken

                return [
                    MorphToken(
                        ch,
                        self.vocab.get(ch, self.vocab["<unk>"]),
                        index,
                        index + 1,
                    )
                    for index, ch in enumerate(text)
                ]

        general = GeneralBPEModel.minimal()
        vocab = build_unified_vocab(
            morphbpe_vocab=SparseMorphBPE.vocab,
            general_vocab=general.get_vocab(),
        )
        tokenizer = DualTrackTokenizer(vocab, SparseMorphBPE(), general)
        encode_native = make_ocr_target_encoder(tokenizer, mode="native")
        with self.assertRaisesRegex(ValueError, "silently route"):
            encode_native(UNCOVERED_LETTER)
        self.assertEqual(
            encode_native.stats["rejected_mn_general_fallback"],
            1,
        )

    def test_strict_native_mode_does_not_require_byte_tokens(self):
        tokenizer = _make_tokenizer(
            {"<unk>": 0, MULTI_CHAR_WORD: 1, LETTER_A: 2}
        )
        encode_native = make_ocr_target_encoder(tokenizer, mode="native")
        self.assertEqual(
            encode_native(MULTI_CHAR_WORD),
            [tokenizer.vocab[MULTI_CHAR_WORD]],
        )

    def test_strict_native_mode_applies_only_pretraining_space_folding(self):
        encode_native = make_ocr_target_encoder(self.tokenizer, mode="native")
        ids = encode_native("\u00a0")
        self.assertEqual(ids, [self.tokenizer.vocab["▁"]])
        self.assertEqual(self.tokenizer.decode(ids), " ")
        self.assertEqual(encode_native.stats["canonicalized"], 1)

    def test_contextual_mongolian_nnbsp_is_not_folded(self):
        text = f"{LETTER_A}{NNBSP}{LETTER_A}"
        vocab = dict(FakeMorphBPE.vocab)
        vocab[text] = 501
        tokenizer = _make_tokenizer(vocab)
        encode_native = make_ocr_target_encoder(tokenizer, mode="native")
        ids = encode_native(text)
        self.assertEqual(tokenizer.decode(ids), text)
        self.assertEqual(encode_native.stats["canonicalized"], 0)

    def test_legacy_byte_mode_is_explicit(self):
        encode_legacy = make_ocr_target_encoder(
            self.tokenizer,
            mode="byte_fallback",
        )
        ids = encode_legacy(MULTI_CHAR_WORD)
        self.assertNotEqual(ids, [self.tokenizer.vocab[MULTI_CHAR_WORD]])
        self.assertEqual(self.tokenizer.decode(ids), MULTI_CHAR_WORD)
        self.assertEqual(encode_legacy.stats["native"], 0)
        self.assertEqual(encode_legacy.stats["byte_fallback"], 1)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown OCR target encoding mode"):
            make_ocr_target_encoder(self.tokenizer, mode="unsafe")

    def test_missing_byte_tokens_raises_with_remedy(self):
        sparse_vocab = {"<unk>": 0, LETTER_A: 300}
        tokenizer = _make_tokenizer(sparse_vocab)
        with self.assertRaisesRegex(ValueError, "byte-fallback tokens"):
            make_ocr_target_encoder(tokenizer, mode="native_fallback")

    def test_lone_surrogate_is_rejected_by_roundtrip_check(self):
        text = "a\ud800b"
        with self.assertRaisesRegex(ValueError, "round-trip mismatch"):
            self.encode_target(text)

    def test_manifest_hash_is_canonical_json_not_file_formatting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = {"schema_version": 2, "nested": {"b": 2, "a": 1}}
            (root / "manifest.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            first = tokenizer_manifest_canonical_sha256(root)
            (root / "manifest.json").write_text(
                '{"nested":{"a":1,"b":2},"schema_version":2}\n',
                encoding="utf-8",
            )
            self.assertEqual(first, tokenizer_manifest_canonical_sha256(root))

    def test_vocab_hash_detects_same_size_with_permuted_token_ids(self):
        class _Tokenizer:
            vocab = {"a": 1, "b": 2}

        class _Permuted:
            vocab = {"a": 2, "b": 1}

        self.assertNotEqual(
            tokenizer_vocab_sha256(_Tokenizer()),
            tokenizer_vocab_sha256(_Permuted()),
        )

    def test_morphology_track_table_cache_is_zero_copy_and_immutable(self):
        first = tokenizer_morphology_track_table(self.tokenizer)
        second = tokenizer_morphology_track_table(self.tokenizer)
        self.assertIs(first, second)
        self.assertIsInstance(first, tuple)

    def test_ocr_adapter_algorithm_binds_its_source(self):
        contract = ocr_adapter_algorithm_contract()
        self.assertEqual(contract["source_file"], "Model/ocr/tokenization.py")
        self.assertEqual(len(contract["source_sha256"]), 64)
        self.assertEqual(contract["tokenization_contract_version"], 3)


if __name__ == "__main__":
    unittest.main()
