# -*- coding: utf-8 -*-

import unittest

from .suffixes import (
    ALL_SUFFIXES,
    LEGACY_MONGOL_CODE_SUFFIX_SURFACES,
    allows_nnbsp,
    duplicate_surfaces,
    validate_suffix_inventory,
)


def suffix_item(item_id):
    for item in ALL_SUFFIXES:
        if item["id"] == item_id:
            return item
    raise AssertionError(f"missing suffix item {item_id}")


class TraditionalMongolianSuffixInventoryTest(unittest.TestCase):
    def test_inventory_is_valid(self):
        self.assertTrue(validate_suffix_inventory())

    def test_legacy_mongol_code_suffix_surfaces_are_covered(self):
        surfaces = {surface for item in ALL_SUFFIXES for surface in item["surface"]}
        self.assertFalse(LEGACY_MONGOL_CODE_SUFFIX_SURFACES - surfaces)

    def test_reviewed_legacy_gaps_are_present(self):
        surfaces = {surface for item in ALL_SUFFIXES for surface in item["surface"]}
        expected = {
            "ᠤ",
            "ᠦ",
            "ᠳᠠᠬᠢ",
            "ᠳᠡᠬᠢ",
            "ᠲᠠᠬᠢ",
            "ᠲᠡᠬᠢ",
            "ᠶᠤᠭᠠᠨ",
            "ᠶᠦᠭᠡᠨ",
            "ᠳᠠᠭᠠᠨ",
            "ᠳᠡᠭᠡᠨ",
            "ᠲᠠᠭᠠᠨ",
            "ᠲᠡᠭᠡᠨ",
            "ᠠᠴᠠᠭᠠᠨ",
            "ᠡᠴᠡᠭᠡᠨ",
            "ᠲᠠᠢᠭᠠᠨ",
            "ᠲᠡᠢᠭᠡᠨ",
            "ᠤᠤ",
            "ᠦᠦ",
        }
        self.assertFalse(expected - surfaces)

    def test_nnbsp_support_is_item_specific(self):
        self.assertTrue(allows_nnbsp(suffix_item("DAT_LOC"), "ᠳᠠ"))
        self.assertFalse(allows_nnbsp(suffix_item("PASS"), "ᠳᠠ"))
        self.assertTrue(allows_nnbsp(suffix_item("Q_UU"), "ᠤᠤ"))
        self.assertFalse(allows_nnbsp(suffix_item("PROG_JU"), "ᠴᠤ"))
        self.assertTrue(allows_nnbsp(suffix_item("CVB_COORD_JU"), "ᠴᠤ"))

    def test_duplicate_surfaces_are_reported(self):
        duplicates = duplicate_surfaces()
        self.assertEqual(duplicates["ᠳᠠ"], ["DAT_LOC", "PASS"])
        self.assertEqual(duplicates["ᠴᠤ"], ["PROG_JU", "CVB_COORD_JU"])

    def test_inventory_has_no_empty_surface(self):
        self.assertFalse(
            [
                item["id"]
                for item in ALL_SUFFIXES
                if any(surface == "" for surface in item["surface"])
            ]
        )


if __name__ == "__main__":
    unittest.main()
