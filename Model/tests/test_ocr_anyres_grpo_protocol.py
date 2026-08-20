# -*- coding: utf-8 -*-

"""Focused tests for the fixed, KL-independent anyres GRPO pilot protocol."""

from __future__ import annotations

import random
import unittest
from copy import deepcopy

import torch

from Model.ocr.tokenization import canonical_json_sha256
from Model.posttrain.ocr_anyres_grpo_protocol import (
    FIXED_KL_PILOT_ROLLOUT_ATTEMPTS,
    advance_fixed_kl_pilot_attempt,
    apply_attempt_seed,
    build_fixed_kl_pilot_protocol,
    fixed_kl_pilot_attempt,
    next_fixed_kl_pilot_attempt,
    rebuild_and_match_fixed_kl_pilot_protocol,
    validate_fixed_kl_pilot_protocol,
    validate_sampler_resume_state,
)
from Model.posttrain.ocr_quota_sampler import OCR_QUOTA_BUCKETS, OCRQuotaSampler


OCR_DATASET_SHA256 = "a" * 64
TEXT_DATASET_SHA256 = "b" * 64


def _sample_buckets() -> dict[str, str]:
    sizes = {
        "print": 17,
        "handwritten_good": 7,
        "handwritten_medium": 11,
        "handwritten_poor": 7,
    }
    return {
        f"{bucket}-{index:03d}": bucket
        for bucket in OCR_QUOTA_BUCKETS
        for index in range(sizes[bucket])
    }


def _text_sample_ids() -> tuple[str, ...]:
    return tuple(f"text-{index:03d}" for index in range(23))


def _sampler(
    *,
    seed: int = 811,
    samples: dict[str, str] | None = None,
    global_batch_size: int = 20,
) -> OCRQuotaSampler:
    return OCRQuotaSampler(
        samples or _sample_buckets(),
        global_batch_size=global_batch_size,
        seed=seed,
        world_size=1,
    )


def _protocol(sampler: OCRQuotaSampler | None = None) -> dict[str, object]:
    return build_fixed_kl_pilot_protocol(
        sampler=sampler or _sampler(),
        ocr_dataset_contract_sha256=OCR_DATASET_SHA256,
        text_sample_ids=_text_sample_ids(),
        text_dataset_contract_sha256=TEXT_DATASET_SHA256,
        text_batch_size=5,
        base_seed=20260820,
    )


class FixedKLPilotProtocolTest(unittest.TestCase):
    def test_three_independent_samplers_build_identical_200_attempt_schedules(self):
        protocols = [_protocol(_sampler()) for _ in range(3)]

        self.assertEqual(protocols[0], protocols[1])
        self.assertEqual(protocols[1], protocols[2])
        payload = validate_fixed_kl_pilot_protocol(protocols[0])
        self.assertEqual(
            payload["rollout_attempts"],
            FIXED_KL_PILOT_ROLLOUT_ATTEMPTS,
        )
        self.assertEqual(len(payload["attempts"]), 200)
        self.assertEqual(payload["global_batch_size"], 20)
        self.assertEqual(payload["text_batch_size"], 5)
        self.assertEqual(payload["world_size"], 1)
        self.assertEqual(len(payload["ocr_prompt_schedule_sha256"]), 64)
        self.assertEqual(len(payload["text_batch_schedule_sha256"]), 64)
        self.assertEqual(len(payload["rollout_seed_schedule_sha256"]), 64)

    def test_planning_deepcopy_never_mutates_live_sampler(self):
        sampler = _sampler()
        before = deepcopy(sampler.state_dict())

        _protocol(sampler)

        self.assertEqual(sampler.state_dict(), before)
        self.assertIsNone(sampler.pending_global_batch)
        self.assertEqual(sampler.draw_counter, 0)

    def test_full_rebuild_authenticates_saved_protocol(self):
        saved = _protocol()

        payload = rebuild_and_match_fixed_kl_pilot_protocol(
            saved,
            fresh_sampler=_sampler(),
            ocr_dataset_contract_sha256=OCR_DATASET_SHA256,
            text_sample_ids=_text_sample_ids(),
            text_dataset_contract_sha256=TEXT_DATASET_SHA256,
            text_batch_size=5,
            base_seed=20260820,
        )

        self.assertEqual(payload, saved["payload"])

    def test_manual_id_edit_with_all_digests_recomputed_fails_rebuild(self):
        saved = deepcopy(_protocol())
        payload = saved["payload"]
        payload["attempts"][0]["ocr_prompt_ids"][0] = "forged-prompt-id"
        payload["ocr_prompt_schedule_sha256"] = canonical_json_sha256(
            [
                {
                    "attempt_index": row["attempt_index"],
                    "ocr_prompt_ids": row["ocr_prompt_ids"],
                    "sampler_state_before_sha256": row[
                        "sampler_state_before_sha256"
                    ],
                    "sampler_state_after_sha256": row[
                        "sampler_state_after_sha256"
                    ],
                }
                for row in payload["attempts"]
            ]
        )
        payload["text_batch_schedule_sha256"] = canonical_json_sha256(
            [
                {
                    "attempt_index": row["attempt_index"],
                    "text_sample_ids": row["text_sample_ids"],
                }
                for row in payload["attempts"]
            ]
        )
        payload["rollout_seed_schedule_sha256"] = canonical_json_sha256(
            [
                {
                    "attempt_index": row["attempt_index"],
                    "cpu_seed": row["cpu_seed"],
                    "cuda_seed": row["cuda_seed"],
                }
                for row in payload["attempts"]
            ]
        )
        saved["canonical_sha256"] = canonical_json_sha256(payload)

        # The forged wrapper is internally self-consistent; provenance must be
        # supplied by the deterministic reconstruction, not by trusting hashes.
        validate_fixed_kl_pilot_protocol(
            saved,
            sampler=_sampler(),
            ocr_dataset_contract_sha256=OCR_DATASET_SHA256,
            text_sample_ids=_text_sample_ids(),
            text_dataset_contract_sha256=TEXT_DATASET_SHA256,
            sampler_attempt_index=0,
        )
        with self.assertRaisesRegex(ValueError, "deterministic rebuild"):
            rebuild_and_match_fixed_kl_pilot_protocol(
                saved,
                fresh_sampler=_sampler(),
                ocr_dataset_contract_sha256=OCR_DATASET_SHA256,
                text_sample_ids=_text_sample_ids(),
                text_dataset_contract_sha256=TEXT_DATASET_SHA256,
                text_batch_size=5,
                base_seed=20260820,
            )

    def test_build_and_rebuild_reject_nonzero_sampler(self):
        advanced = _sampler()
        advanced.prepare_global_batch()
        advanced.commit_global_batch()

        with self.assertRaisesRegex(ValueError, "fresh initial sampler"):
            _protocol(advanced)
        with self.assertRaisesRegex(ValueError, "fresh initial sampler"):
            rebuild_and_match_fixed_kl_pilot_protocol(
                _protocol(),
                fresh_sampler=advanced,
                ocr_dataset_contract_sha256=OCR_DATASET_SHA256,
                text_sample_ids=_text_sample_ids(),
                text_dataset_contract_sha256=TEXT_DATASET_SHA256,
                text_batch_size=5,
                base_seed=20260820,
            )

    def test_attempt_index_replays_exact_next_ocr_and_text_batches(self):
        protocol = _protocol()
        restored = _sampler()
        for _ in range(73):
            restored.prepare_global_batch()
            restored.commit_global_batch()

        validate_sampler_resume_state(
            protocol,
            sampler=restored,
            attempt_index=73,
        )
        expected = next_fixed_kl_pilot_attempt(
            protocol,
            completed_rollout_attempts=73,
        )
        self.assertEqual(expected, fixed_kl_pilot_attempt(protocol, 73))
        self.assertEqual(
            list(restored.prepare_global_batch()),
            expected["ocr_prompt_ids"],
        )
        self.assertEqual(
            expected["text_sample_ids"],
            ["text-020", "text-021", "text-022", "text-000", "text-001"],
        )

    def test_zero_active_attempt_still_consumes_slot_without_shifting_ids(self):
        protocol = _protocol()
        attempt_zero = fixed_kl_pilot_attempt(protocol, 0)
        attempt_one_before = fixed_kl_pilot_attempt(protocol, 1)

        next_index = advance_fixed_kl_pilot_attempt(
            protocol,
            attempt_index=0,
            active_group_count=0,
        )
        attempt_one_after = next_fixed_kl_pilot_attempt(
            protocol,
            completed_rollout_attempts=next_index,
        )

        self.assertEqual(next_index, 1)
        self.assertEqual(attempt_one_after, attempt_one_before)
        self.assertNotEqual(
            attempt_zero["ocr_prompt_ids"],
            attempt_one_after["ocr_prompt_ids"],
        )
        self.assertEqual(
            advance_fixed_kl_pilot_attempt(
                protocol,
                attempt_index=0,
                active_group_count=4,
            ),
            next_index,
        )

    def test_hash_seeds_are_reproducible_and_reset_python_torch_and_numpy(self):
        protocol = _protocol()
        attempt = fixed_kl_pilot_attempt(protocol, 37)

        applied_first = apply_attempt_seed(protocol, 37)
        first_python = random.random()
        first_torch = torch.rand(4)
        try:
            import numpy as np
        except ImportError:
            first_numpy = None
        else:
            first_numpy = np.random.random(4)

        random.seed(1)
        torch.manual_seed(2)
        if first_numpy is not None:
            np.random.seed(3)

        applied_second = apply_attempt_seed(protocol, 37)
        second_python = random.random()
        second_torch = torch.rand(4)

        self.assertEqual(applied_first, applied_second)
        self.assertEqual(applied_first["cpu_seed"], attempt["cpu_seed"])
        self.assertEqual(applied_first["cuda_seed"], attempt["cuda_seed"])
        self.assertEqual(first_python, second_python)
        self.assertTrue(torch.equal(first_torch, second_torch))
        if first_numpy is not None:
            self.assertTrue((first_numpy == np.random.random(4)).all())

        independent_protocol = _protocol(_sampler())
        self.assertEqual(
            fixed_kl_pilot_attempt(independent_protocol, 37)["cpu_seed"],
            attempt["cpu_seed"],
        )
        self.assertEqual(
            fixed_kl_pilot_attempt(independent_protocol, 37)["cuda_seed"],
            attempt["cuda_seed"],
        )

    def test_sampler_config_sample_and_dataset_drift_fail_closed(self):
        protocol = _protocol()

        with self.assertRaisesRegex(ValueError, "config drift"):
            validate_fixed_kl_pilot_protocol(
                protocol,
                sampler=_sampler(seed=812),
            )

        changed_samples = _sample_buckets()
        changed_samples["print-new"] = "print"
        with self.assertRaisesRegex(ValueError, "sample-list drift"):
            validate_fixed_kl_pilot_protocol(
                protocol,
                sampler=_sampler(samples=changed_samples),
            )

        changed_text_ids = list(_text_sample_ids())
        changed_text_ids[-1] = "text-drift"
        with self.assertRaisesRegex(ValueError, "text sample-list drift"):
            validate_fixed_kl_pilot_protocol(
                protocol,
                text_sample_ids=changed_text_ids,
            )

        with self.assertRaisesRegex(ValueError, "OCR dataset contract drift"):
            validate_fixed_kl_pilot_protocol(
                protocol,
                ocr_dataset_contract_sha256="c" * 64,
            )
        with self.assertRaisesRegex(ValueError, "text dataset contract drift"):
            validate_fixed_kl_pilot_protocol(
                protocol,
                text_dataset_contract_sha256="d" * 64,
            )

    def test_invalid_batch_contracts_and_resume_boundaries_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "divisible by 20"):
            _protocol(_sampler(global_batch_size=10))

        protocol = _protocol()
        advanced = _sampler()
        advanced.prepare_global_batch()
        advanced.commit_global_batch()
        with self.assertRaisesRegex(ValueError, "requested attempt boundary"):
            validate_sampler_resume_state(
                protocol,
                sampler=advanced,
                attempt_index=0,
            )

        pending = _sampler()
        pending.prepare_global_batch()
        with self.assertRaisesRegex(ValueError, "committed attempt boundary"):
            validate_sampler_resume_state(
                protocol,
                sampler=pending,
                attempt_index=1,
            )

        tampered = deepcopy(protocol)
        tampered["payload"]["text_batch_size"] = 6
        with self.assertRaisesRegex(ValueError, "canonical_sha256"):
            validate_fixed_kl_pilot_protocol(tampered)


if __name__ == "__main__":
    unittest.main()
