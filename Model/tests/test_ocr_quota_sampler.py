# -*- coding: utf-8 -*-

"""Focused replay and quota tests for the anyres OCR rank-0 sampler."""

from __future__ import annotations

import json
import unittest
from collections import Counter

from Model.posttrain.ocr_quota_sampler import (
    OCR_QUOTA_BUCKETS,
    OCR_QUOTA_PER_20,
    OCRQuotaSampler,
    slice_global_batch,
)


def _sample_buckets() -> dict[str, str]:
    sizes = {
        "print": 5,
        "handwritten_good": 3,
        "handwritten_medium": 4,
        "handwritten_poor": 3,
    }
    return {
        f"{bucket}-{index:02d}": bucket
        for bucket in OCR_QUOTA_BUCKETS
        for index in range(sizes[bucket])
    }


def _draw_and_commit(sampler: OCRQuotaSampler) -> tuple[str, ...]:
    batch = sampler.prepare_global_batch()
    sampler.commit_global_batch()
    return batch


class OCRQuotaSamplerTest(unittest.TestCase):
    def test_short_batches_close_exact_quota_and_do_not_replace_within_epoch(self):
        sample_buckets = _sample_buckets()
        sampler = OCRQuotaSampler(
            sample_buckets,
            global_batch_size=4,
            seed=90210,
            world_size=2,
        )
        drawn = [
            sample_id
            for _ in range(10)
            for sample_id in _draw_and_commit(sampler)
        ]

        for start in range(21):
            counts = Counter(sample_buckets[sample_id] for sample_id in drawn[start:start + 20])
            self.assertEqual(dict(counts), OCR_QUOTA_PER_20)
        self.assertEqual(sampler.quota_deficit, {bucket: 0 for bucket in OCR_QUOTA_BUCKETS})

        first_print = [
            sample_id
            for sample_id in drawn[:20]
            if sample_buckets[sample_id] == "print"
        ]
        print_ids = {
            sample_id for sample_id, bucket in sample_buckets.items() if bucket == "print"
        }
        self.assertEqual(set(first_print[:5]), print_ids)
        self.assertEqual(set(first_print[5:10]), print_ids)
        self.assertEqual(len(set(first_print[10:12])), 2)

        state = sampler.state_dict()
        self.assertEqual(state["buckets"]["print"]["epoch"], 4)
        self.assertEqual(state["buckets"]["print"]["cursor"], 4)
        json.dumps(state, ensure_ascii=False, allow_nan=False)

    def test_pending_global_batch_is_idempotent_and_rank_slice_is_lossless(self):
        sampler = OCRQuotaSampler(
            _sample_buckets(),
            global_batch_size=8,
            seed=7,
            world_size=4,
        )
        global_batch = sampler.prepare_global_batch()
        self.assertEqual(sampler.prepare_global_batch(), global_batch)
        self.assertEqual(sampler.draw_counter, 8)

        rank_batches = [sampler.pending_batch_for_rank(rank) for rank in range(4)]
        self.assertTrue(all(len(batch) == 2 for batch in rank_batches))
        reconstructed = [None] * len(global_batch)
        for rank, rank_batch in enumerate(rank_batches):
            reconstructed[rank::4] = rank_batch
            self.assertEqual(
                rank_batch,
                slice_global_batch(global_batch, rank=rank, world_size=4),
            )
        self.assertEqual(tuple(reconstructed), global_batch)

        state = sampler.state_dict()
        self.assertEqual(state["pending_global_batch"], list(global_batch))
        self.assertEqual(state["pending_world_size"], 4)

    def test_json_restore_replays_pending_and_next_batches_exactly(self):
        kwargs = {
            "global_batch_size": 4,
            "seed": 42,
            "world_size": 2,
        }
        original = OCRQuotaSampler(_sample_buckets(), **kwargs)
        for _ in range(3):
            _draw_and_commit(original)

        committed_state = json.loads(json.dumps(original.state_dict()))
        committed_restore = OCRQuotaSampler(_sample_buckets(), **kwargs)
        committed_restore.load_state_dict(committed_state)
        self.assertEqual(
            original.prepare_global_batch(),
            committed_restore.prepare_global_batch(),
        )

        pending_state = json.loads(json.dumps(original.state_dict()))
        pending_restore = OCRQuotaSampler(_sample_buckets(), **kwargs)
        pending_restore.load_state_dict(pending_state)
        self.assertEqual(
            pending_restore.prepare_global_batch(),
            original.pending_global_batch,
        )
        for rank in range(2):
            self.assertEqual(
                pending_restore.pending_batch_for_rank(rank),
                original.pending_batch_for_rank(rank),
            )

        original.commit_global_batch()
        pending_restore.commit_global_batch()
        for _ in range(8):
            self.assertEqual(
                _draw_and_commit(original),
                _draw_and_commit(pending_restore),
            )

    def test_world_config_and_sample_list_drift_fail_closed(self):
        samples = _sample_buckets()
        source = OCRQuotaSampler(
            samples,
            global_batch_size=8,
            seed=9,
            world_size=2,
        )
        _draw_and_commit(source)
        state = json.loads(json.dumps(source.state_dict()))

        with self.assertRaisesRegex(ValueError, "world_size drift"):
            OCRQuotaSampler(
                samples,
                global_batch_size=8,
                seed=9,
                world_size=4,
            ).load_state_dict(state)

        with self.assertRaisesRegex(ValueError, "config drift"):
            OCRQuotaSampler(
                samples,
                global_batch_size=8,
                seed=10,
                world_size=2,
            ).load_state_dict(state)

        expanded = dict(samples)
        expanded["print-new"] = "print"
        with self.assertRaisesRegex(ValueError, "sample-list hash drift"):
            OCRQuotaSampler(
                expanded,
                global_batch_size=8,
                seed=9,
                world_size=2,
            ).load_state_dict(state)

        reassigned = dict(samples)
        reassigned["print-00"] = "handwritten_good"
        with self.assertRaisesRegex(ValueError, "sample-list hash drift"):
            OCRQuotaSampler(
                reassigned,
                global_batch_size=8,
                seed=9,
                world_size=2,
            ).load_state_dict(state)


if __name__ == "__main__":
    unittest.main()
