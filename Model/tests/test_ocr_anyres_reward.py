# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import unittest

import torch

from Model.posttrain import ocr_anyres_reward
from Model.posttrain.ocr_anyres_reward import (
    AnyresOCRRewardAdapter,
    AnyresOCRRewardConfig,
    score_anyres_ocr_group,
)


EOS = 3


def _encode(text: str) -> list[int]:
    return [300 + ord(character) for character in text]


def _decode(ids) -> str:
    return "".join(chr(int(token) - 300) for token in ids)


def _completion(text: str, *, eos: bool = True) -> list[int]:
    return [*_encode(text), *([EOS] if eos else [])]


def _valid(*texts: str) -> set[int]:
    result = {EOS, 17}
    for text in texts:
        result.update(_encode(text))
    return result


def _score(responses, references, completions, **kwargs):
    return score_anyres_ocr_group(
        responses,
        references,
        completions,
        EOS,
        _valid(*responses, *references),
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        **kwargs,
    )


class AnyresOCRRewardTests(unittest.TestCase):
    def test_perfect_beats_single_symbol_error_then_repeat_and_empty(self) -> None:
        reference = "ᠠ1!"
        responses = [reference, "ᠠ2!", reference * 3, ""]
        result = _score(
            responses,
            [reference] * len(responses),
            [_completion(response) for response in responses],
        )
        rewards = result["group_rewards"].tolist()
        self.assertGreater(rewards[0], rewards[1])
        self.assertGreater(rewards[1], rewards[2])
        self.assertGreater(rewards[1], rewards[3])
        self.assertLessEqual(rewards[3], 0.0)
        self.assertEqual(result["samples"][0]["breakdown"]["exact_match"], 0.10)
        self.assertEqual(result["samples"][0]["breakdown"]["valid_eos"], 0.05)

    def test_raw_supported_control_deletion_is_not_normalized_away(self) -> None:
        reference = "ᠠ\u180b\u180e\u202f"
        perfect = _score(
            [reference], [reference], [_completion(reference)]
        )["samples"][0]
        deleted = _score(
            ["ᠠ"], [reference], [_completion("ᠠ")]
        )["samples"][0]
        self.assertGreater(perfect["total"], deleted["total"])
        self.assertGreater(
            deleted["diagnostics"]["supported_symbol_reference_count"], 0
        )
        self.assertGreater(
            deleted["diagnostics"]["supported_symbol_micro_error"], 0.0
        )
        self.assertLess(deleted["breakdown"]["supported_symbol_error"], 0.0)
        self.assertGreater(deleted["diagnostics"]["raw_grapheme_cer"], 0.0)

    def test_eos_hitcap_reserved_unassigned_and_roundtrip_are_independent(self) -> None:
        text = "ᠠ"
        no_eos = _score([text], [text], [_completion(text, eos=False)])["samples"][0]
        self.assertTrue(no_eos["diagnostics"]["hit_cap"])
        self.assertEqual(no_eos["breakdown"]["missing_eos_or_hit_cap"], -0.10)

        valid_ids = _valid(text)
        reserved = score_anyres_ocr_group(
            [text],
            [text],
            [[2, *_encode(text), EOS]],
            EOS,
            valid_ids | {2},
            tokenizer_encode=_encode,
            tokenizer_decode=_decode,
        )["samples"][0]
        self.assertTrue(reserved["diagnostics"]["invalid"])
        self.assertTrue(
            any(
                reason.startswith("reserved_token")
                for reason in reserved["diagnostics"]["invalid_reasons"]
            )
        )
        self.assertEqual(reserved["breakdown"]["exact_match"], 0.0)
        self.assertEqual(reserved["breakdown"]["valid_eos"], 0.0)
        self.assertEqual(reserved["breakdown"]["grounding_margin"], 0.0)

        unassigned = score_anyres_ocr_group(
            [text],
            [text],
            [[999999, EOS]],
            EOS,
            valid_ids,
            tokenizer_encode=_encode,
            tokenizer_decode=_decode,
        )["samples"][0]
        self.assertTrue(unassigned["diagnostics"]["invalid"])
        self.assertEqual(unassigned["breakdown"]["invalid_output"], -0.20)

        mismatch = score_anyres_ocr_group(
            [text],
            [text],
            [_completion(text)],
            EOS,
            valid_ids,
            tokenizer_encode=lambda _text: [777],
            tokenizer_decode=lambda _ids: text,
        )["samples"][0]
        self.assertIn(
            "tokenizer_completion_roundtrip",
            mismatch["diagnostics"]["invalid_reasons"],
        )

        with self.assertRaisesRegex(RuntimeError, "tokenizer infrastructure"):
            score_anyres_ocr_group(
                [text],
                [text],
                [_completion(text)],
                EOS,
                valid_ids,
                tokenizer_encode=lambda _text: (_ for _ in ()).throw(
                    RuntimeError("tokenizer infrastructure")
                ),
                tokenizer_decode=_decode,
            )

    def test_grounding_margin_is_external_clipped_and_breakdown_is_finite(self) -> None:
        text = "ᠠ"
        result = _score(
            [text, text],
            [text, text],
            [_completion(text), _completion(text)],
            grounding_margin=[9.0, -9.0],
            config=AnyresOCRRewardConfig(grounding_margin_weight=0.05),
        )
        first, second = result["samples"]
        self.assertEqual(first["diagnostics"]["grounding_margin_clipped"], 1.0)
        self.assertEqual(second["diagnostics"]["grounding_margin_clipped"], -1.0)
        self.assertAlmostEqual(first["breakdown"]["grounding_margin"], 0.05)
        self.assertAlmostEqual(second["breakdown"]["grounding_margin"], -0.05)
        self.assertTrue(torch.isfinite(result["group_rewards"]).all())
        self.assertTrue(
            all(
                math.isfinite(value)
                for sample in result["samples"]
                for value in sample["breakdown"].values()
            )
        )

    def test_config_and_production_roundtrip_gate_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            AnyresOCRRewardConfig(exact_match_bonus=float("nan"))
        with self.assertRaises(ValueError):
            AnyresOCRRewardConfig(
                grounding_margin_clip_min=1.0,
                grounding_margin_clip_max=-1.0,
            )
        with self.assertRaisesRegex(ValueError, "requires tokenizer"):
            score_anyres_ocr_group(
                ["ᠠ"], ["ᠠ"], [_completion("ᠠ")], EOS, _valid("ᠠ")
            )
        debug = score_anyres_ocr_group(
            ["ᠠ"],
            ["ᠠ"],
            [_completion("ᠠ")],
            EOS,
            _valid("ᠠ"),
            config=AnyresOCRRewardConfig(require_tokenizer_roundtrip=False),
        )
        self.assertTrue(torch.isfinite(debug["group_rewards"]).all())

    def test_production_adapter_decodes_ids_and_exposes_bound_contract(self) -> None:
        text = "ᠠ"
        valid = sorted(_valid(text))
        with self.assertRaisesRegex(ValueError, "reviewed TokenizerBundle"):
            AnyresOCRRewardAdapter(
                eos_id=EOS,
                valid_token_ids=valid,
                tokenizer_encode=_encode,
                tokenizer_decode=_decode,
                tokenizer_contract_sha256="a" * 64,
            )
        adapter = AnyresOCRRewardAdapter(
            eos_id=EOS,
            valid_token_ids=valid,
            tokenizer_encode=_encode,
            tokenizer_decode=_decode,
            tokenizer_contract_sha256="a" * 64,
            _factory_token=ocr_anyres_reward._REVIEWED_FACTORY_TOKEN,
        )
        clean_ids = torch.tensor([_completion(text)], dtype=torch.long)
        clean = adapter.score_group(
            [text],
            clean_ids,
            clean_ids == EOS,
        )
        invalid_ids = torch.tensor([[2, *_encode(text), EOS]], dtype=torch.long)
        invalid = adapter.score_group(
            [text],
            invalid_ids,
            invalid_ids == EOS,
        )
        self.assertEqual(clean["responses"], [text])
        self.assertIn("\ufffd", invalid["responses"][0])
        self.assertGreater(
            float(clean["group_rewards"][0]),
            float(invalid["group_rewards"][0]),
        )
        self.assertEqual(
            adapter.contract["kind"],
            "dol_ocr_anyres_reward_adapter_v1",
        )
        with self.assertRaisesRegex(ValueError, "eos_mask differs"):
            adapter.score_group(
                [text],
                clean_ids,
                torch.zeros_like(clean_ids, dtype=torch.bool),
            )


if __name__ == "__main__":
    unittest.main()
