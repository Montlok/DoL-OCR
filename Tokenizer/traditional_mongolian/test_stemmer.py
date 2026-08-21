# -*- coding: utf-8 -*-

import unittest

from .stemmer import MongolStemmer
from .unicode_norm import FVS1, MVS, NNBSP, control_boundaries, strip_all, strip_all_with_map


class TraditionalMongolianStemmerTest(unittest.TestCase):
    def setUp(self):
        self.stemmer = MongolStemmer()

    def test_genitive_suffix_with_mvs_boundary(self):
        word = "ᠪᠢᠴᠢᠭ" + MVS + "ᠦᠨ"
        result = self.stemmer.analyze(word)
        self.assertEqual(result.root, "ᠪᠢᠴᠢᠭ")
        self.assertEqual(result.suffixes, [MVS + "ᠦᠨ"])
        self.assertEqual(result.suffix_ids, ["GEN"])
        self.assertEqual(result.boundaries, [0, 5, 8])
        self.assertEqual(result.skeleton_boundaries, [0, 5, 7])
        self.assertGreaterEqual(result.confidence, 0.70)

    def test_plural_case_stack_with_original_boundaries(self):
        word = "ᠪᠢᠴᠢᠭ" + MVS + "ᠦᠳ" + MVS + "ᠦᠨ"
        result = self.stemmer.analyze(word)
        self.assertEqual(result.root, "ᠪᠢᠴᠢᠭ")
        self.assertEqual(result.suffix_ids, ["PL_UD", "GEN"])
        self.assertEqual(result.suffix_types, ["plural", "case"])
        self.assertEqual(result.suffixes, [MVS + "ᠦᠳ", MVS + "ᠦᠨ"])
        self.assertEqual(result.skeleton_boundaries, [0, 5, 7, 9])

    def test_nnbsp_question_particle(self):
        word = "ᠨᠡᠷ" + NNBSP + "ᠦᠦ"
        result = self.stemmer.analyze(word)
        self.assertEqual(result.root, "ᠨᠡᠷ")
        self.assertEqual(result.suffix_ids, ["Q_UU"])
        self.assertEqual(result.suffix_types, ["particle"])
        self.assertEqual(result.suffixes, [NNBSP + "ᠦᠦ"])

    def test_encoding_mapping_output_shape_is_accepted(self):
        # The accepted nominal Unicode form may map NNBSP suffix spacing to MVS.
        word = "ᠨᠡᠷ" + MVS + "ᠡ"
        result = self.stemmer.analyze(word)
        self.assertEqual(result.root, "ᠨᠡᠷ")
        self.assertEqual(result.suffix_ids, ["DAT_LOC"])
        self.assertGreaterEqual(result.confidence, 0.60)

    def test_skeleton_map_ignores_controls(self):
        word = "ᠨᠡᠷ" + NNBSP + "ᠦᠦ"
        skeleton, boundary_map = strip_all_with_map(word)
        self.assertEqual(skeleton, strip_all(word))
        self.assertEqual(skeleton, "ᠨᠡᠷᠦᠦ")
        self.assertEqual(boundary_map, [0, 1, 2, 3, 5, 6])
        self.assertEqual(control_boundaries(word), {3})

    def test_trailing_controls_are_covered_by_offset_map(self):
        word = "ᠨᠡᠷ" + FVS1 + MVS
        skeleton, boundary_map = strip_all_with_map(word)
        self.assertEqual(skeleton, "ᠨᠡᠷ")
        self.assertEqual(boundary_map[-1], len(word))

    def test_duplicate_surface_disambiguation_does_not_crash(self):
        result = self.stemmer.analyze("ᠪᠠᠷᠢ" + MVS + "ᠳᠠ")
        self.assertIsInstance(result.suffix_ids, list)

    def test_short_word_over_stemming_prevention(self):
        result = self.stemmer.analyze("ᠨᠡᠷ")
        self.assertEqual(result.root, "ᠨᠡᠷ")
        self.assertEqual(result.suffixes, [])

    def test_low_confidence_inner_suffix_does_not_split_lexical_word(self):
        result = self.stemmer.analyze("ᠮᠣᠩᠭᠣᠯ")
        self.assertEqual(result.root, "ᠮᠣᠩᠭᠣᠯ")
        self.assertEqual(result.suffixes, [])
        self.assertEqual(result.confidence, 0.52)

    def test_modern_derived_verb_preserves_stable_root(self):
        result = self.stemmer.analyze("ᠪᠠᠶᠢᠭᠤᠯᠤᠭᠳᠠᠬᠤ")
        root_skeleton = strip_all(result.root)
        self.assertNotEqual(root_skeleton, "ᠪᠠᠶᠢ")
        self.assertGreaterEqual(len(root_skeleton), len("ᠪᠠᠶᠢᠭᠤᠯ"))
        self.assertNotEqual(result.suffix_ids, ["CAUS_GUL", "PASS", "PTCP_FUT_QU"])


if __name__ == "__main__":
    unittest.main()
