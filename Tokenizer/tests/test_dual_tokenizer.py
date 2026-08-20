# -*- coding: utf-8 -*-

import unittest

from Tokenizer.generic_bpe import GeneralBPEModel
from Tokenizer.multimodal import (
    IMAGE_END,
    IMAGE_PATCH,
    IMAGE_PLACEHOLDER,
    IMAGE_START,
    expand_image_placeholders,
)
from Tokenizer.traditional_mongolian import NNBSP
from Tokenizer.unified.dual_tokenizer import (
    DualTrackTokenizer,
    SEGMENT,
    SPECIAL_TOKENS,
    build_unified_vocab,
    segment_by_language,
)


class FakeMorphBPE:
    vocab = {"ᠮᠣᠩᠭᠣᠯ": 0, "ᠪᠢᠴᠢᠭ": 1}

    def encode(self, text):
        return [self.vocab[text]]


def build_fake_tokenizer():
    general = GeneralBPEModel.minimal()
    vocab = build_unified_vocab(
        morphbpe_vocab=FakeMorphBPE.vocab,
        general_vocab=general.get_vocab(),
    )
    return DualTrackTokenizer(vocab, FakeMorphBPE(), general)


class DualTokenizerTest(unittest.TestCase):
    def test_segments_special_tokens_before_language_routing(self):
        spans = segment_by_language("这张图 " + IMAGE_PLACEHOLDER + " test")
        self.assertEqual(
            [(span.lang, span.text) for span in spans],
            [
                ("general", "这张图"),
                ("space", " "),
                ("special", IMAGE_PLACEHOLDER),
                ("space", " "),
                ("general", "test"),
            ],
        )

    def test_plain_text_encoding_escapes_literal_control_surfaces(self):
        tokenizer = build_fake_tokenizer()
        text = "▁◈<bos><image_patch>"
        ids = tokenizer.encode_plain_text(text)
        self.assertNotIn(SPECIAL_TOKENS["▁"], ids)
        self.assertNotIn(SPECIAL_TOKENS["◈"], ids)
        self.assertNotIn(SPECIAL_TOKENS["<bos>"], ids)
        self.assertNotIn(SPECIAL_TOKENS["<image_patch>"], ids)
        self.assertEqual(tokenizer.decode(ids), text)

    def test_fullwidth_latin_and_digits_route_to_general(self):
        spans = segment_by_language("Ａ３！test")
        self.assertEqual(
            [(span.lang, span.text) for span in spans],
            [("general", "Ａ３！test")],
        )

    def test_mongolian_nnbsp_stays_inside_mongolian_span(self):
        text = "ᠨᠡᠷ" + NNBSP + "ᠦᠦ"
        spans = segment_by_language(text)
        self.assertEqual([(span.lang, span.text) for span in spans], [("mn", text)])

        latin = "hello" + NNBSP + "world"
        self.assertEqual(
            [(span.lang, span.text) for span in segment_by_language(latin)],
            [("general", "hello"), ("space", NNBSP), ("general", "world")],
        )

    def test_digits_inside_mongolian_run_round_trip_segmentation(self):
        # Mixed-script boundary inside one word-like run: segmentation must
        # reconstruct the original text exactly (lossless routing), with the
        # digits on the general track.
        text = "ᠮᠣᠩᠭᠣᠯ123ᠪᠢᠴᠢᠭ"
        spans = segment_by_language(text)
        self.assertEqual("".join(span.text for span in spans), text)
        self.assertEqual([span.lang for span in spans], ["mn", "general", "mn"])

    def test_cyrillic_between_mongolian_round_trips(self):
        text = "ᠮᠣᠩᠭᠣᠯ бичиг ᠪᠢᠴᠢᠭ"
        spans = segment_by_language(text)
        self.assertEqual("".join(span.text for span in spans), text)
        self.assertEqual(
            [span.lang for span in spans],
            ["mn", "space", "general", "space", "mn"],
        )

    def test_nnbsp_adjacent_to_plain_space_round_trips(self):
        # NNBSP straddling a Mongolian word and a plain space: whatever track
        # the router picks, the concatenation of spans must stay lossless.
        text = "ᠨᠡᠷ" + NNBSP + " ᠦᠦ"
        spans = segment_by_language(text)
        self.assertEqual("".join(span.text for span in spans), text)

    def test_cjk_punctuation_routes_to_general(self):
        spans = segment_by_language("这。图")
        self.assertEqual(
            [(span.lang, span.text) for span in spans],
            [("general", "这。图")],
        )

    def test_mongolian_punctuation_routes_to_general(self):
        spans = segment_by_language("ᠰᠠᠢᠨ᠃")
        self.assertEqual(
            [(span.lang, span.text) for span in spans],
            [
                ("mn", "ᠰᠠᠢᠨ"),
                ("general", "᠃"),
            ],
        )

    def test_encode_routes_tracks_to_global_id_ranges(self):
        tokenizer = build_fake_tokenizer()
        result = tokenizer.encode_with_spans(
            "ᠮᠣᠩᠭᠣᠯ 这 test!", add_bos=True, add_eos=True
        )
        self.assertEqual(result.ids[0], SPECIAL_TOKENS["<bos>"])
        self.assertEqual(result.ids[-1], SPECIAL_TOKENS["<eos>"])
        mn_lo, mn_hi = SEGMENT["mongolian"]
        gen_lo, gen_hi = SEGMENT["general"]
        self.assertTrue(any(mn_lo <= i < mn_hi for i in result.ids))
        self.assertTrue(any(gen_lo <= i < gen_hi for i in result.ids))
        self.assertEqual(result.input_ids, result.ids)
        self.assertEqual(len(result.tokens), len(result.ids))
        self.assertEqual(result.tokens[0].start, -1)
        self.assertEqual(result.tokens[-1].end, -1)

    def test_multimodal_placeholder_and_patch_tokens_are_special(self):
        tokenizer = build_fake_tokenizer()
        expanded = expand_image_placeholders(IMAGE_PLACEHOLDER, patches_per_image=2)
        self.assertEqual(expanded, IMAGE_START + IMAGE_PATCH + IMAGE_PATCH + IMAGE_END)
        ids = tokenizer.encode(expanded)
        self.assertEqual(
            ids,
            [
                SPECIAL_TOKENS[IMAGE_START],
                SPECIAL_TOKENS[IMAGE_PATCH],
                SPECIAL_TOKENS[IMAGE_PATCH],
                SPECIAL_TOKENS[IMAGE_END],
            ],
        )

    def test_emoji_round_trip(self):
        tokenizer = build_fake_tokenizer()
        ids = tokenizer.encode("🙂")
        self.assertGreater(len(ids), 0)
        self.assertEqual(tokenizer.decode(ids), "🙂")

    def test_mixed_script_round_trip(self):
        tokenizer = build_fake_tokenizer()
        text = "mixed 中 ᠮᠣᠩᠭᠣᠯ 🙂 日本語"
        ids = tokenizer.encode(text)
        self.assertEqual(tokenizer.decode(ids), text)

    def test_token_level_offsets_cover_tracks(self):
        tokenizer = build_fake_tokenizer()
        result = tokenizer.encode_with_spans("这 test 🙂")
        tracks = {tok.track for tok in result.tokens}
        self.assertIn("general", tracks)
        self.assertIn("space", tracks)
        # offsets are monotonic and within bounds
        for tok in result.tokens:
            self.assertLessEqual(tok.start, tok.end)

    def test_mongolian_punctuation_round_trips_without_unk(self):
        tokenizer = build_fake_tokenizer()
        ids = tokenizer.encode("᠃")
        self.assertNotIn(tokenizer.unk_id, ids)
        self.assertEqual(tokenizer.decode(ids), "᠃")

    def test_newline_and_tab_are_preserved_distinctly(self):
        # Newline/tab/CR must NOT collapse into the space token "▁". They route
        # to the general byte-level track so document structure survives.
        tokenizer = build_fake_tokenizer()
        for text in ("hello\nhello", "test\ttest", "a\r\nb", "x\n\ny"):
            result = tokenizer.encode_with_spans(text)
            self.assertEqual(tokenizer.decode(result.input_ids), text)
            self.assertFalse(
                any(tok.track == "space" for tok in result.tokens),
                f"structural whitespace wrongly folded to ▁ for {text!r}",
            )

    def test_plain_space_still_folds_to_space_token(self):
        tokenizer = build_fake_tokenizer()
        result = tokenizer.encode_with_spans("hello  hello")
        space_toks = [t for t in result.tokens if t.track == "space"]
        self.assertEqual(len(space_toks), 2)
        self.assertEqual(tokenizer.decode(result.input_ids), "hello  hello")


class MongolianFallbackOffsetTest(unittest.TestCase):
    """The no-offset MorphBPE fallback must yield monotonic per-piece spans."""

    def test_multi_piece_fallback_offsets_are_monotonic(self):
        class MultiPieceMorphBPE:
            vocab = {"ᠮᠣᠩ": 0, "ᠭᠣᠯ": 1}

            def encode(self, text):
                return [0, 1]

        word = "ᠮᠣᠩᠭᠣᠯ"
        general = GeneralBPEModel.minimal()
        vocab = build_unified_vocab(
            morphbpe_vocab=MultiPieceMorphBPE.vocab,
            general_vocab=general.get_vocab(),
        )
        tokenizer = DualTrackTokenizer(vocab, MultiPieceMorphBPE(), general)

        result = tokenizer.encode_with_spans(word)
        mn = [t for t in result.tokens if t.track == "mn"]
        self.assertEqual(len(mn), 2)
        self.assertEqual((mn[0].start, mn[0].end), (0, 3))
        self.assertEqual((mn[1].start, mn[1].end), (3, 6))

    def test_uncovered_mongolian_scalar_uses_lossless_general_fallback(self):
        class SparseMorphBPE:
            vocab = {"<unk>": 0, "ᠮ": 1}

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
        text = "ᠮ᠐ᠮ"
        result = tokenizer.encode_with_spans(text)
        self.assertNotIn(tokenizer.unk_id, result.input_ids)
        self.assertEqual(tokenizer.decode(result.input_ids), text)
        self.assertEqual(
            [token.track for token in result.tokens],
            [
                "mn",
                "mn_general_fallback",
                "mn_general_fallback",
                "mn_general_fallback",
                "mn",
            ],
        )


class SharedSurfaceCollisionTest(unittest.TestCase):
    """Surfaces present in BOTH track vocabs share one MorphBPE-segment id.

    Regression guard: the general track's forward map used to be gated on the
    general segment range, so every shared surface (ASCII letters/digits that
    MorphBPE picked up from Latin fragments in the Mongolian corpus) encoded
    to <unk> at runtime — ~27% of English and ~29% of code tokens in the v2
    bundle. The fixture vocabs here are deliberately overlapping; the default
    ``FakeMorphBPE`` is disjoint from the general vocab and cannot catch this.
    """

    class CollidingMorphBPE:
        # "a" and "1" stand in for Latin/digit fragments absorbed into the
        # MorphBPE vocab at build time; build_unified_vocab assigns them
        # MorphBPE-segment ids, which the general track must then share.
        vocab = {"ᠮᠣᠩᠭᠣᠯ": 0, "a": 1, "1": 2, "±": 3}

        def encode(self, text):
            return [self.vocab[text]]

    def _build(self):
        general = GeneralBPEModel.minimal()
        vocab = build_unified_vocab(
            morphbpe_vocab=self.CollidingMorphBPE.vocab,
            general_vocab=general.get_vocab(),
        )
        return DualTrackTokenizer(vocab, self.CollidingMorphBPE(), general), vocab

    def test_shared_surface_encodes_to_shared_id_not_unk(self):
        tokenizer, vocab = self._build()
        mn_lo, mn_hi = SEGMENT["mongolian"]
        for surface in ("a", "1"):
            ids = tokenizer.encode(surface)  # routes to the general track
            self.assertEqual(ids, [vocab[surface]])
            self.assertNotEqual(ids[0], tokenizer.unk_id)
            # The shared id lives in the MorphBPE segment, not the general one.
            self.assertTrue(mn_lo <= ids[0] < mn_hi)

    def test_general_text_with_shared_pieces_round_trips_without_unk(self):
        tokenizer, _ = self._build()
        for text in ("a1", "var a = 1;", "中a中", "a 中 1"):
            ids = tokenizer.encode(text)
            self.assertNotIn(tokenizer.unk_id, ids)
            self.assertEqual(tokenizer.decode(ids), text)

    def test_non_ascii_shared_byte_atom_stays_in_general_decode_buffer(self):
        tokenizer, vocab = self._build()
        # U+FE31 is UTF-8 EF B8 B1. ByteLevel represents the final B1 byte as
        # the visible atom "±", which deliberately owns a MorphBPE-segment id.
        text = "︱"
        ids = tokenizer.encode(text)
        self.assertIn(vocab["±"], ids)
        self.assertEqual(tokenizer.decode(ids), text)

    def test_mongolian_ids_unaffected_by_collisions(self):
        tokenizer, _ = self._build()
        baseline = build_fake_tokenizer()
        word = "ᠮᠣᠩᠭᠣᠯ"
        self.assertEqual(tokenizer.encode(word), baseline.encode(word))


if __name__ == "__main__":
    unittest.main()
