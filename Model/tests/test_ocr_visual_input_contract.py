# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest

from Model.ocr.visual_input_contract import (
    DOL_OCR_ANYRES_V2,
    DOL_OCR_LINE_LETTERBOX_224_V1,
    resolve_checkpoint_ocr_visual_input_contract,
)


class OCRVisualInputContractTests(unittest.TestCase):
    def test_resolves_only_exact_name_and_version_pairs(self) -> None:
        cases = (
            (DOL_OCR_LINE_LETTERBOX_224_V1, 1),
            (DOL_OCR_ANYRES_V2, 2),
        )
        for contract, version in cases:
            with self.subTest(contract=contract):
                metadata = {
                    "ocr_visual_input_contract": contract,
                    "ocr_visual_input_contract_version": version,
                }
                self.assertEqual(
                    resolve_checkpoint_ocr_visual_input_contract(metadata),
                    contract,
                )
                self.assertEqual(
                    resolve_checkpoint_ocr_visual_input_contract(
                        metadata,
                        requested=contract,
                    ),
                    contract,
                )

    def test_missing_or_partial_metadata_never_guesses(self) -> None:
        invalid = (
            {},
            {"ocr_visual_input_contract": DOL_OCR_LINE_LETTERBOX_224_V1},
            {"ocr_visual_input_contract_version": 1},
        )
        for metadata in invalid:
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                resolve_checkpoint_ocr_visual_input_contract(metadata)

    def test_anyres_requires_v2_and_line_v1_cannot_request_anyres(self) -> None:
        with self.assertRaisesRegex(ValueError, "version mismatch"):
            resolve_checkpoint_ocr_visual_input_contract(
                {
                    "ocr_visual_input_contract": DOL_OCR_ANYRES_V2,
                    "ocr_visual_input_contract_version": 1,
                }
            )

        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            resolve_checkpoint_ocr_visual_input_contract(
                {
                    "ocr_visual_input_contract": DOL_OCR_LINE_LETTERBOX_224_V1,
                    "ocr_visual_input_contract_version": 1,
                },
                requested=DOL_OCR_ANYRES_V2,
            )

    def test_unknown_contract_and_boolean_version_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported"):
            resolve_checkpoint_ocr_visual_input_contract(
                {
                    "ocr_visual_input_contract": "anyres",
                    "ocr_visual_input_contract_version": 2,
                }
            )
        with self.assertRaisesRegex(ValueError, "version mismatch"):
            resolve_checkpoint_ocr_visual_input_contract(
                {
                    "ocr_visual_input_contract": DOL_OCR_LINE_LETTERBOX_224_V1,
                    "ocr_visual_input_contract_version": True,
                }
            )


if __name__ == "__main__":
    unittest.main()
