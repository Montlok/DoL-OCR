# -*- coding: utf-8 -*-

"""The unsafe SSL-token re-wrapper must stay retired."""

from __future__ import annotations

import unittest

from scripts.build_vlm_ocr_data import main


class RetiredVLMOCRBuilderTest(unittest.TestCase):
    def test_entrypoint_fails_before_writing_legacy_rows(self) -> None:
        self.assertEqual(main([]), 2)


if __name__ == "__main__":
    unittest.main()
