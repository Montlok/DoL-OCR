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
    symbol_metrics,
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
    def test_python_fallback_strips_fvs_but_preserves_mvs(self):
        folded = nominal_normalize([MONG + FVS1 + MONG + MVS], backend="python")
        self.assertEqual(folded, [MONG + MONG + MVS])

    def test_python_fallback_maps_nnbsp_to_mvs_like_rust(self):
        folded = nominal_normalize([MONG + NNBSP + MONG], backend="python")
        self.assertEqual(folded, [MONG + MVS + MONG])

    def test_python_fallback_removes_same_zero_width_noise_as_rust(self):
        noise = "\u200b\u200c\u200d\u2060\ufeff"
        self.assertEqual(
            nominal_normalize(["a" + noise + "b"], backend="python"),
            ["ab"],
        )

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

    def test_raw_and_normalized_line_exact_are_reported_separately(self):
        pred = [MONG]
        ref = [MONG + FVS1]
        rep = ocr_report(pred, ref, backend="python")
        self.assertEqual(rep.raw_line_exact, 0.0)
        self.assertEqual(rep.normalized_line_exact, 1.0)
        # Backward compatibility: line_exact remains the normalized rate.
        self.assertEqual(rep.line_exact, rep.normalized_line_exact)

    def test_rejected_rows_are_excluded_from_symbol_metrics(self):
        rep = ocr_report(
            ["7", "9"],
            ["7", "8"],
            backend="python",
            rejected=[False, True],
        )
        digits = rep.symbol_metrics["digit"]
        self.assertEqual(digits["n_ref"], 1)
        self.assertEqual(digits["n_pred"], 1)
        self.assertEqual(digits["correct"], 1)
        self.assertEqual(digits["line_support"], 1)


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


class TestSymbolMetrics(unittest.TestCase):
    def test_full_alignment_attributes_all_required_symbol_classes(self):
        preds = [
            "2",  # in-class digit substitution
            "",  # digit deletion
            "9",  # prediction-only digit insertion
            ",",  # punctuation match
            ".",  # punctuation substitution
            FVS4,  # FVS1 -> FVS4 substitution
            "",  # MVS deletion
            "",  # NNBSP deletion
        ]
        refs = [
            "1",
            "7",
            "",
            ",",
            "!",
            FVS1,
            MVS,
            NNBSP,
        ]
        metrics = symbol_metrics(preds, refs)

        self.assertEqual(
            metrics["digit"],
            {
                "n_ref": 2,
                "n_pred": 2,
                "correct": 0,
                "substitutions": 1,
                "deletions": 1,
                "insertions": 1,
                "line_support": 2,
                "error_rate": 1.5,
            },
        )
        self.assertEqual(
            metrics["punctuation"],
            {
                "n_ref": 2,
                "n_pred": 2,
                "correct": 1,
                "substitutions": 1,
                "deletions": 0,
                "insertions": 0,
                "line_support": 2,
                "error_rate": 0.5,
            },
        )
        self.assertEqual(metrics["fvs"]["n_ref"], 1)
        self.assertEqual(metrics["fvs"]["n_pred"], 1)
        self.assertEqual(metrics["fvs"]["substitutions"], 1)
        self.assertEqual(metrics["fvs"]["insertions"], 0)
        self.assertEqual(metrics["fvs"]["error_rate"], 1.0)
        self.assertEqual(metrics["mvs"]["deletions"], 1)
        self.assertEqual(metrics["mvs"]["error_rate"], 1.0)
        self.assertEqual(metrics["nnbsp"]["deletions"], 1)
        self.assertEqual(metrics["nnbsp"]["error_rate"], 1.0)

    def test_fvs_variants_have_independent_support_and_errors(self):
        variants = symbol_metrics([FVS4], [FVS1])["fvs"]["variants"]
        self.assertEqual(variants["fvs1"]["n_ref"], 1)
        self.assertEqual(variants["fvs1"]["substitutions"], 1)
        self.assertEqual(variants["fvs1"]["error_rate"], 1.0)
        self.assertEqual(variants["fvs4"]["n_ref"], 0)
        self.assertEqual(variants["fvs4"]["n_pred"], 1)
        self.assertEqual(variants["fvs4"]["insertions"], 1)
        self.assertIsNone(variants["fvs4"]["error_rate"])
        self.assertEqual(variants["fvs2"]["n_ref"], 0)
        self.assertIsNone(variants["fvs2"]["error_rate"])

    def test_zero_reference_support_is_explicitly_unsupported(self):
        metrics = symbol_metrics(["!"], [""])
        self.assertEqual(metrics["punctuation"]["n_ref"], 0)
        self.assertEqual(metrics["punctuation"]["n_pred"], 1)
        self.assertEqual(metrics["punctuation"]["insertions"], 1)
        self.assertEqual(metrics["punctuation"]["line_support"], 0)
        self.assertIsNone(metrics["punctuation"]["error_rate"])
        self.assertIsNone(metrics["digit"]["error_rate"])

    def test_mongolian_and_ascii_decimal_digits_share_digit_class(self):
        mongolian_digit_one = "\u1811"
        metrics = symbol_metrics(
            ["7" + mongolian_digit_one],
            ["7" + mongolian_digit_one],
        )
        self.assertEqual(metrics["digit"]["n_ref"], 2)
        self.assertEqual(metrics["digit"]["correct"], 2)
        self.assertEqual(metrics["digit"]["error_rate"], 0.0)

    def test_length_mismatch_raises(self):
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            symbol_metrics(["1"], ["1", "2"])


if __name__ == "__main__":
    unittest.main()
