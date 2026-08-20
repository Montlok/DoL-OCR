# -*- coding: utf-8 -*-

from __future__ import annotations

import copy
import unittest

from Model.ocr.tokenization import canonical_json_sha256
from Model.posttrain.ocr_anyres_grpo_formal_protocol import (
    build_formal_grpo_protocol,
    formal_grpo_attempt,
    rebuild_and_match_formal_protocol,
    validate_formal_grpo_protocol,
    validate_formal_sampler_boundary,
)
from Model.posttrain.ocr_quota_sampler import OCRQuotaSampler


def _buckets() -> dict[str, str]:
    return {
        f"{bucket}-{index:03d}": bucket
        for bucket in (
            "print",
            "handwritten_good",
            "handwritten_medium",
            "handwritten_poor",
        )
        for index in range(20)
    }


def _sampler() -> OCRQuotaSampler:
    return OCRQuotaSampler(
        _buckets(),
        global_batch_size=20,
        seed=99,
        world_size=1,
    )


def _build(sampler: OCRQuotaSampler | None = None) -> dict:
    return build_formal_grpo_protocol(
        sampler=sampler or _sampler(),
        ocr_dataset_contract_sha256="a" * 64,
        text_sample_ids=tuple(f"text-{index}" for index in range(13)),
        text_dataset_contract_sha256="b" * 64,
        text_batch_size=4,
        base_seed=20260821,
        max_rollout_attempts=17,
        max_optimizer_steps=13,
        eval_every=3,
        save_every=5,
        early_stop_patience=2,
    )


class FormalGRPOProtocolTest(unittest.TestCase):
    def test_protocol_is_dynamic_deterministic_and_rebuildable(self):
        first = _build()
        second = _build()
        self.assertEqual(first, second)
        payload = validate_formal_grpo_protocol(first)
        self.assertEqual(payload["max_rollout_attempts"], 17)
        self.assertEqual(payload["max_optimizer_steps"], 13)
        rebuilt = rebuild_and_match_formal_protocol(
            first,
            sampler=_sampler(),
            ocr_dataset_contract_sha256="a" * 64,
            text_sample_ids=tuple(f"text-{index}" for index in range(13)),
            text_dataset_contract_sha256="b" * 64,
        )
        self.assertEqual(rebuilt, first)

    def test_consumed_sampler_cannot_define_a_formal_schedule(self):
        sampler = _sampler()
        sampler.prepare_global_batch()
        sampler.commit_global_batch()
        with self.assertRaisesRegex(ValueError, "fresh sampler"):
            _build(sampler)

    def test_resume_boundary_replays_the_exact_next_ids(self):
        protocol = _build()
        restored = _sampler()
        for _ in range(7):
            restored.prepare_global_batch()
            restored.commit_global_batch()
        validate_formal_sampler_boundary(protocol, restored, 7)
        pending = list(restored.prepare_global_batch())
        self.assertEqual(pending, formal_grpo_attempt(protocol, 7)["ocr_prompt_ids"])

    def test_registered_ids_and_seed_tamper_fail_closed(self):
        protocol = _build()
        bad = copy.deepcopy(protocol)
        bad["payload"]["text_sample_ids"][0] = ""
        bad["canonical_sha256"] = canonical_json_sha256(bad["payload"])
        with self.assertRaisesRegex(ValueError, "registered text IDs"):
            validate_formal_grpo_protocol(bad)

        bad = copy.deepcopy(protocol)
        bad["payload"]["attempts"][0]["cpu_seed"] = -1
        bad["canonical_sha256"] = canonical_json_sha256(bad["payload"])
        with self.assertRaisesRegex(ValueError, "seed range"):
            validate_formal_grpo_protocol(bad)


if __name__ == "__main__":
    unittest.main()
