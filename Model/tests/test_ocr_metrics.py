# -*- coding: utf-8 -*-

"""Unit tests for OCR accuracy metrics (torch-free)."""

import unittest

from Model.ocr.metrics import (
    cer,
    edit_distance,
    grapheme_clusters,
    nominal_normalize,
    ocr_report,
    script_bucket_cer,
    script_of,
    wer,
)

# Free variation selectors and Mongolian vowel separator — pure encoding
# variation that normalized CER must ignore.
FVS1 = "\u180b"
FVS4 = "\u180f"
MVS = "\u180e"
NNBSP = "\u202f"
# A short traditional-Mongolian letter run (nominal code points).
MONG = "\u182d\u1820\u1837"  # GA A RA
# A short CJK run (Han characters) for script-bucket tests.
HAN = "\u6c49\u5b57"  # \u6c49\u5b57 ("Chinese characters")


class TestEditDistance(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(edit_distance("abc", "abc"), 0)

    def test_empty(self):
        self.assertEqual(edit_distance("", "abc"), 3)
        self.assertEqual(edit_distance("abc", ""), 3)

    def test_substitution_insertion_deletion(self):
        self.assertEqual(edit_distance("kitten", "sitting"), 3)

    def test_symmetric(self):
        self.assertEqual(edit_distance("abcd", "abx"), edit_distance("abx", "abcd"))


class TestNominalNormalize(unittest.TestCase):
    def test_python_fallback_strips_fvs_mvs(self):
        folded = nominal_normalize([MONG + FVS1 + MONG + MVS], backend="python")
        self.assertEqual(folded, [MONG + MONG])

    def test_python_fallback_maps_nnbsp_to_space(self):
        folded = nominal_normalize([MONG + NNBSP + MONG], backend="python")
        self.assertEqual(folded, [MONG + " " + MONG])

    def test_auto_backend_returns_same_length(self):
        # auto must never change the number of items, regardless of backend.
        items = [MONG, MONG + FVS1, "", "abc"]
        self.assertEqual(len(nominal_normalize(items, backend="auto")), len(items))


class TestCER(unittest.TestCase):
    def test_fvs_difference_is_free_under_normalized_cer(self):
        # Prediction differs from reference only by an FVS -> normalized CER 0,
        # raw CER non-zero. This is the core Mongolian-OCR measurement point.
        pred = [MONG + FVS1]
        ref = [MONG]
        self.assertAlmostEqual(cer(pred, ref, backend="python"), 0.0)
        self.assertGreater(cer(pred, ref, normalize=False), 0.0)

    def test_corpus_micro_average(self):
        # One substitution over 3+3 reference chars -> 1/6.
        preds = ["abc", "xyz"]
        refs = ["abc", "xyq"]
        self.assertAlmostEqual(cer(preds, refs, normalize=False), 1 / 6)

    def test_perfect(self):
        self.assertEqual(cer(["abc"], ["abc"], normalize=False), 0.0)

    def test_empty_ref_counts_insertions(self):
        # Pure insertion against an empty reference must be penalized, not 0/0.
        self.assertGreater(cer(["abc"], [""], normalize=False), 0.0)

    def test_single_backend_decision_on_auto(self):
        # The bug: under auto, folding preds/refs separately could pick different
        # backends (one side has a newline -> Python fallback, the other -> Rust).
        # _fold_pair folds the concatenated batch once, so a newline on *either*
        # side forces the same backend for both.
        from Model.ocr.metrics import _fold_pair

        _, _, used = _fold_pair(["a\nb"], ["ab"], backend="auto")
        self.assertEqual(used, "python")
        # And an FVS-only difference stays free regardless.
        pred = ["a\nb" + FVS1]
        ref = ["a\nb"]
        self.assertAlmostEqual(cer(pred, ref, backend="auto"), 0.0)

    def test_grapheme_unit_differs_from_codepoint_unit(self):
        # pred is missing the FVS1 that ref carries. Under codepoint units
        # that is 1 substitution-equivalent edit over a 2-codepoint ref ->
        # 0.5. Under grapheme units the ref clusters to a single grapheme
        # (base + FVS1), so the whole thing is 1 edit over a 1-grapheme ref
        # -> 1.0. This is the core grapheme-vs-codepoint distinction.
        pred = [MONG[0]]
        ref = [MONG[0] + FVS1]
        self.assertAlmostEqual(cer(pred, ref, normalize=False), 0.5)
        self.assertAlmostEqual(
            cer(pred, ref, normalize=False, unit="grapheme"), 1.0
        )

    def test_unknown_unit_raises(self):
        with self.assertRaises(ValueError):
            cer(["a"], ["a"], unit="bogus")


class TestGraphemeClusters(unittest.TestCase):
    def test_fvs1_attaches_to_base(self):
        self.assertEqual(grapheme_clusters(MONG[0] + FVS1), [MONG[0] + FVS1])

    def test_fvs4_attaches_regardless_of_unicodedata_category(self):
        # FVS4 (U+180F) must attach whether or not the running interpreter's
        # unicodedata reports it as category Mn (see the _FVS comment in
        # Model/ocr/metrics.py): the explicit code-point set is load-bearing.
        self.assertEqual(grapheme_clusters(MONG[0] + FVS4), [MONG[0] + FVS4])

    def test_mvs_and_nnbsp_are_standalone_clusters(self):
        self.assertEqual(
            grapheme_clusters(MONG[0] + MVS + MONG[0]),
            [MONG[0], MVS, MONG[0]],
        )
        self.assertEqual(
            grapheme_clusters(MONG[0] + NNBSP + MONG[0]),
            [MONG[0], NNBSP, MONG[0]],
        )

    def test_combining_acute_forms_one_cluster(self):
        # "e" + COMBINING ACUTE ACCENT (U+0301) is the decomposed form of "é".
        self.assertEqual(grapheme_clusters("é"), ["é"])

    def test_leading_combining_mark_is_its_own_cluster(self):
        # No preceding base character to attach to.
        self.assertEqual(grapheme_clusters("́a"), ["́", "a"])

    def test_empty_text_yields_no_clusters(self):
        self.assertEqual(grapheme_clusters(""), [])


class TestWER(unittest.TestCase):
    def test_word_error(self):
        preds = ["a b c d"]
        refs = ["a b x d"]
        self.assertAlmostEqual(wer(preds, refs, normalize=False), 1 / 4)


class TestOCRReport(unittest.TestCase):
    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            ocr_report(["a"], ["a", "b"], backend="python")

    def test_rejection_excludes_samples(self):
        preds = ["good", "WRONG"]
        refs = ["good", "right"]
        # Reject the second (wrong) sample -> scored set is perfect.
        rep = ocr_report(
            preds, refs, backend="python", rejected=[False, True]
        )
        self.assertEqual(rep.n, 1)
        self.assertAlmostEqual(rep.rejection_rate, 0.5)
        self.assertEqual(rep.norm_cer, 0.0)
        self.assertEqual(rep.line_exact, 1.0)

    def test_report_fields_consistent(self):
        preds = [MONG + FVS1, "abc"]
        refs = [MONG, "abd"]
        rep = ocr_report(preds, refs, backend="python")
        self.assertEqual(rep.n, 2)
        self.assertEqual(rep.backend, "python")
        # FVS-only diff on line 1 is free; line 2 has 1/3 char error -> over
        # ref length 3+3=6, total dist 1 -> 1/6.
        self.assertAlmostEqual(rep.norm_cer, 1 / 6)
        self.assertGreaterEqual(rep.raw_cer, rep.norm_cer)

    def test_grapheme_cer_matches_cer_grapheme_unit_on_raw_text(self):
        # grapheme_cer is computed on the RAW (unfolded) kept texts, so it
        # must equal cer(..., normalize=False, unit="grapheme") on the same
        # pairs -- not the normalized/folded pairs.
        preds = [MONG[0], "abc"]
        refs = [MONG[0] + FVS1, "abd"]
        rep = ocr_report(preds, refs, backend="python")
        self.assertAlmostEqual(
            rep.grapheme_cer,
            cer(preds, refs, normalize=False, unit="grapheme"),
        )


class TestScriptOf(unittest.TestCase):
    def test_mongolian_letter_is_mn(self):
        self.assertEqual(script_of(MONG[0]), "mn")

    def test_fvs_is_mn(self):
        self.assertEqual(script_of(FVS1), "mn")
        self.assertEqual(script_of(FVS4), "mn")

    def test_nnbsp_is_mn(self):
        self.assertEqual(script_of(NNBSP), "mn")

    def test_mvs_is_mn(self):
        self.assertEqual(script_of(MVS), "mn")

    def test_han_character_is_cjk(self):
        self.assertEqual(script_of(HAN[0]), "cjk")

    def test_cjk_punctuation_is_cjk(self):
        self.assertEqual(script_of("、"), "cjk")  # IDEOGRAPHIC COMMA

    def test_ascii_letter_and_digit_are_latin(self):
        self.assertEqual(script_of("A"), "latin")
        self.assertEqual(script_of("7"), "latin")

    def test_ascii_punctuation_and_space_are_other(self):
        self.assertEqual(script_of(","), "other")
        self.assertEqual(script_of(" "), "other")

    def test_rejects_multi_character_input(self):
        with self.assertRaises(ValueError):
            script_of("ab")


class TestScriptBucketCER(unittest.TestCase):
    def test_mn_perfect_cjk_one_error(self):
        # Mongolian run identical on both sides; one Han substitution (子 ->
        # 字) plus untouched Latin/digit/punctuation tails.
        pred = [MONG + "汉子A1,"]
        ref = [MONG + "汉字A1,"]
        buckets = script_bucket_cer(pred, ref, backend="python")
        self.assertAlmostEqual(buckets["mn"]["cer"], 0.0)
        self.assertEqual(buckets["mn"]["n_ref"], len(MONG))
        self.assertGreater(buckets["cjk"]["cer"], 0.0)
        self.assertEqual(buckets["cjk"]["n_ref"], 2)
        self.assertAlmostEqual(buckets["latin"]["cer"], 0.0)
        self.assertAlmostEqual(buckets["other"]["cer"], 0.0)

    def test_empty_bucket_omitted(self):
        # Pure-Mongolian pair: cjk/latin/other buckets have n_ref == 0 and
        # must not appear in the result at all.
        buckets = script_bucket_cer([MONG], [MONG], backend="python")
        self.assertEqual(set(buckets), {"mn"})
        self.assertEqual(buckets["mn"]["n_ref"], len(MONG))

    def test_unknown_unit_raises(self):
        with self.assertRaises(ValueError):
            script_bucket_cer(["a"], ["a"], unit="bogus")

    def test_codepoint_unit_does_not_cluster(self):
        # Under unit="codepoint", a base+FVS pair is two separate code points,
        # both bucketed "mn" -- unlike unit="grapheme" where they merge into
        # one cluster. normalize=False so the fold doesn't strip the FVS
        # before bucketing. n_ref for "mn" should reflect 2 code points, not 1.
        pred = [MONG[0] + FVS1]
        ref = [MONG[0] + FVS1]
        buckets = script_bucket_cer(pred, ref, normalize=False, unit="codepoint")
        self.assertEqual(buckets["mn"]["n_ref"], 2)


class TestOCRReportScriptCER(unittest.TestCase):
    def test_script_cer_matches_direct_call(self):
        preds = [MONG + "汉子A1,", "abc"]
        refs = [MONG + "汉字A1,", "abd"]
        rep = ocr_report(preds, refs, backend="python")
        # script_cer is computed on the same RAW (unfolded) kept texts as
        # grapheme_cer, so it must equal a direct script_bucket_cer call with
        # normalize=False on the same pairs.
        direct = script_bucket_cer(preds, refs, normalize=False, unit="grapheme")
        self.assertEqual(rep.script_cer, direct)

    def test_script_cer_is_populated_by_default(self):
        rep = ocr_report([MONG], [MONG], backend="python")
        self.assertIsNotNone(rep.script_cer)
        self.assertEqual(set(rep.script_cer), {"mn"})


if __name__ == "__main__":
    unittest.main()
