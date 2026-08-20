# -*- coding: utf-8 -*-

import unittest

from Tokenizer.pretraining import derive_morph_info_from_boundary_ids
from Tokenizer.unified.vocab import SPECIAL_TOKENS


WB = SPECIAL_TOKENS["▁"]
MB = SPECIAL_TOKENS["◈"]
BOS = SPECIAL_TOKENS["<bos>"]
EOS = SPECIAL_TOKENS["<eos>"]


class BoundaryDerivationTest(unittest.TestCase):
    def test_word_boundary_advances_word_pos(self):
        ids = [BOS, WB, 300, 301, WB, 302, EOS]
        wp, md = derive_morph_info_from_boundary_ids(ids, WB, MB)
        self.assertEqual(wp, [0, 0, 0, 0, 1, 1, 1])
        self.assertEqual(md, [0, 0, 0, 0, 0, 0, 0])

    def test_morpheme_boundary_bumps_depth(self):
        ids = [WB, 300, MB, 301, MB, 302]
        wp, md = derive_morph_info_from_boundary_ids(ids, WB, MB)
        self.assertEqual(wp, [0, 0, 0, 0, 0, 0])
        self.assertEqual(md, [0, 0, 1, 1, 2, 2])

    def test_other_special_resets_depth(self):
        ids = [WB, 300, MB, 301, EOS, WB, 400]
        wp, md = derive_morph_info_from_boundary_ids(ids, WB, MB)
        # EOS keeps word_pos at the current word; next WB advances to word 1.
        self.assertEqual(wp, [0, 0, 0, 0, 0, 1, 1])
        self.assertEqual(md, [0, 0, 1, 1, 0, 0, 0])

    def test_max_depth_clamps(self):
        ids = [WB, MB, MB, MB, MB, 300]
        _wp, md = derive_morph_info_from_boundary_ids(ids, WB, MB, max_depth=3)
        # depth would be 0,1,2,3,4,4; max_depth is the inclusive upper bound.
        self.assertEqual(md, [0, 1, 2, 3, 3, 3])

    def test_max_depth_zero_clamps_to_zero(self):
        # The optional helper uses the same inclusive convention as
        # MorphologicalRoPE; production configs require a positive bound.
        ids = [WB, MB, MB, 300]
        _wp, md = derive_morph_info_from_boundary_ids(ids, WB, MB, max_depth=0)
        self.assertEqual(md, [0, 0, 0, 0])
        self.assertTrue(all(d >= 0 for d in md))


if __name__ == "__main__":
    unittest.main()
