# -*- coding: utf-8 -*-

from __future__ import annotations

from copy import deepcopy
import inspect
import unittest
from types import SimpleNamespace
from unittest import mock

from Model.ocr.anyres_preprocess_contract import (
    ANYRES_GLOBAL_VISUAL_PREFIX_TOKENS,
    anyres_preprocess_contract_sha256,
    build_anyres_preprocess_contract,
    canonical_anyres_preprocess_contract_json,
    validate_anyres_preprocess_contract,
)
from Model.ocr.visual_input_contract import DOL_OCR_ANYRES_V2
from Model.omvt.native_planner import COVERAGE_PROOF_METHOD, PLAN_CONTRACT


def _cfg(**overrides):
    values = {
        "in_channels": 3,
        "compress_to": 256,
        "vertical_patch": (32, 8),
        "horizontal_patch": (8, 32),
        "square_patch": (16, 16),
        "layout_patch": (56, 56),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _build(**overrides):
    values = {
        "omvt_cfg": _cfg(),
        "max_decode_pixels": 12_000_000,
        "max_raw_patch_tokens_per_view": 8192,
        "max_windows": 8,
        "halo_px": 32,
        "max_detail_tokens": 512,
        "source_tokens_per_detail_token": 8,
        "max_seq_len": 1024,
        "recommended_max_new_tokens": 256,
    }
    values.update(overrides)
    return build_anyres_preprocess_contract(**values)


class AnyresPreprocessContractTests(unittest.TestCase):
    def test_builds_deterministic_self_verifying_contract(self) -> None:
        first = _build()
        second = _build()
        self.assertEqual(first, second)
        self.assertEqual(
            first["visual_input_contract"],
            {"name": DOL_OCR_ANYRES_V2, "version": 2},
        )
        self.assertEqual(
            first["planner_contract"],
            {
                "name": PLAN_CONTRACT,
                "coverage_proof_method": COVERAGE_PROOF_METHOD,
            },
        )
        self.assertEqual(
            first["patch_contract"]["shapes_hw"],
            {
                "vertical": [32, 8],
                "horizontal": [8, 32],
                "square": [16, 16],
                "layout": [56, 56],
            },
        )
        self.assertEqual(
            set(first["implementation_sources"]),
            {
                "native_processor",
                "native_patcher",
                "native_planner",
                "native_tower",
                "detail_compressor",
                "detail_bridge",
            },
        )
        self.assertEqual(
            first["contract_canonical_sha256"],
            anyres_preprocess_contract_sha256(first),
        )
        self.assertEqual(validate_anyres_preprocess_contract(first), first)
        self.assertEqual(
            canonical_anyres_preprocess_contract_json(first),
            canonical_anyres_preprocess_contract_json(second),
        )

    def test_binds_exif_rgb_imagenet_and_normalized_white_padding(self) -> None:
        pixel = _build()["pixel_contract"]
        self.assertEqual(pixel["orientation"], "pil_imageops_exif_transpose_v1")
        self.assertEqual(pixel["color_mode"], "RGB")
        self.assertEqual(pixel["normalization"]["name"], "imagenet_rgb_v1")
        self.assertEqual(pixel["padding"]["rgb_u8"], [255, 255, 255])
        expected_white = [
            (1.0 - mean) / std
            for mean, std in zip(
                pixel["normalization"]["mean"],
                pixel["normalization"]["std"],
                strict=True,
            )
        ]
        self.assertEqual(pixel["padding"]["normalized_rgb"], expected_white)
        self.assertFalse(pixel["padding"]["pixel_valid_mask"])

    def test_all_budget_arguments_are_required_positive_integers(self) -> None:
        signature = inspect.signature(build_anyres_preprocess_contract)
        for parameter in signature.parameters.values():
            self.assertIs(parameter.default, inspect.Parameter.empty)

        fields = (
            "max_decode_pixels",
            "max_raw_patch_tokens_per_view",
            "max_windows",
            "halo_px",
            "max_detail_tokens",
            "source_tokens_per_detail_token",
            "max_seq_len",
            "recommended_max_new_tokens",
        )
        for field in fields:
            for invalid in (0, -1, True, 1.5):
                with self.subTest(field=field, invalid=invalid):
                    with self.assertRaisesRegex(ValueError, "positive integer"):
                        _build(**{field: invalid})

    def test_requires_rgb_and_exact_256_token_global_prefix(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires RGB"):
            _build(omvt_cfg=_cfg(in_channels=1))
        with self.assertRaisesRegex(ValueError, "exactly 256"):
            _build(omvt_cfg=_cfg(compress_to=255))
        self.assertEqual(
            _build()["budgets"]["context"]["global_visual_prefix_tokens"],
            ANYRES_GLOBAL_VISUAL_PREFIX_TOKENS,
        )

    def test_context_and_derived_resource_budgets_are_consistent(self) -> None:
        payload = _build(
            max_raw_patch_tokens_per_view=1000,
            max_windows=3,
            max_detail_tokens=80,
            source_tokens_per_detail_token=16,
            max_seq_len=515,
            recommended_max_new_tokens=256,
        )
        budgets = payload["budgets"]
        self.assertEqual(budgets["patch"]["max_raw_tokens_per_asset"], 3000)
        self.assertEqual(budgets["detail"]["max_effective_tokens_per_view"], 63)
        self.assertEqual(budgets["detail"]["max_effective_tokens_per_asset"], 189)
        self.assertEqual(
            budgets["context"][
                "remaining_nonvisual_prompt_tokens_at_recommended_output"
            ],
            0,
        )
        self.assertEqual(validate_anyres_preprocess_contract(payload), payload)
        with self.assertRaisesRegex(ValueError, "exceeds max_seq_len"):
            _build(max_seq_len=514, recommended_max_new_tokens=256)

    def test_exact_schema_hash_and_current_source_bytes_are_enforced(self) -> None:
        payload = _build()

        extra = deepcopy(payload)
        extra["unreviewed"] = True
        with self.assertRaisesRegex(ValueError, "keys differ"):
            validate_anyres_preprocess_contract(extra)

        tampered = deepcopy(payload)
        tampered["budgets"]["output"]["recommended_max_new_tokens"] -= 1
        with self.assertRaisesRegex(ValueError, "derived causal context"):
            validate_anyres_preprocess_contract(tampered)

        tampered_source = deepcopy(payload)
        tampered_source["implementation_sources"]["native_processor"][
            "sha256"
        ] = "0" * 64
        with mock.patch(
            "Model.ocr.anyres_preprocess_contract._current_implementation_sources",
            return_value=tampered_source["implementation_sources"],
        ):
            with self.assertRaisesRegex(ValueError, "canonical SHA256"):
                validate_anyres_preprocess_contract(tampered_source)

        drifted_current = deepcopy(payload["implementation_sources"])
        drifted_current["native_processor"]["sha256"] = "f" * 64
        with mock.patch(
            "Model.ocr.anyres_preprocess_contract._current_implementation_sources",
            return_value=drifted_current,
        ):
            with self.assertRaisesRegex(ValueError, "source drift"):
                validate_anyres_preprocess_contract(payload)


if __name__ == "__main__":
    unittest.main()
