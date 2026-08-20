# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import threading
import time
import traceback
import unittest
from unittest import mock

from Model.posttrain.verified_prefetch import VerifiedBatchPrefetcher


class VerifiedBatchPrefetcherTest(unittest.TestCase):
    def test_slow_loader_overlaps_in_order_without_exceeding_one_slot(self) -> None:
        started: list[int] = []
        started_events = [threading.Event() for _ in range(3)]
        release_events = [threading.Event() for _ in range(3)]
        release_events[0].set()

        def load(batch_id: int) -> str:
            started.append(batch_id)
            started_events[batch_id].set()
            if not release_events[batch_id].wait(timeout=2.0):
                raise TimeoutError(f"test did not release {batch_id}")
            return f"verified-{batch_id}"

        prefetcher = VerifiedBatchPrefetcher(range(3), load, max_prefetch=1)
        with prefetcher:
            self.assertTrue(started_events[0].wait(timeout=1.0))
            time.sleep(0.05)
            self.assertEqual(started, [0])
            self.assertFalse(prefetcher._thread.daemon)

            self.assertEqual(next(prefetcher), (0, "verified-0"))
            self.assertTrue(started_events[1].wait(timeout=1.0))
            self.assertEqual(started, [0, 1])
            self.assertFalse(started_events[2].is_set())

            release_events[1].set()
            self.assertEqual(next(prefetcher), (1, "verified-1"))
            self.assertTrue(started_events[2].wait(timeout=1.0))
            self.assertEqual(started, [0, 1, 2])
            release_events[2].set()
            self.assertEqual(next(prefetcher), (2, "verified-2"))
            with self.assertRaises(StopIteration):
                next(prefetcher)
        self.assertFalse(prefetcher.worker_alive)

    def test_loader_exception_keeps_trace_context_and_stops_before_next_id(self) -> None:
        consumed: list[str] = []

        def load(batch_id: str) -> str:
            consumed.append(batch_id)
            if batch_id == "bad":
                try:
                    raise KeyError("verified-byte-cause")
                except KeyError as cause:
                    raise RuntimeError(f"load failed for {batch_id}") from cause
            return batch_id.upper()

        prefetcher = VerifiedBatchPrefetcher(
            ["ok", "bad", "must-not-load"],
            load,
            max_prefetch=1,
        )
        self.assertEqual(next(prefetcher), ("ok", "OK"))
        try:
            next(prefetcher)
        except RuntimeError as exception:
            caught = exception
            frames = traceback.extract_tb(exception.__traceback__)
        else:  # pragma: no cover - assertion path.
            self.fail("loader exception was not propagated")
        self.assertRegex(str(caught), "load failed for bad")
        self.assertIsInstance(caught.__cause__, KeyError)
        self.assertTrue(any(frame.name == "load" for frame in frames))
        self.assertEqual(consumed, ["ok", "bad"])
        self.assertFalse(prefetcher.worker_alive)
        prefetcher.close()
        prefetcher.close()

    def test_early_close_cancels_future_work_joins_and_is_idempotent(self) -> None:
        started: list[int] = []
        second_started = threading.Event()

        def load(batch_id: int) -> int:
            started.append(batch_id)
            if batch_id == 1:
                second_started.set()
                time.sleep(0.1)
            return batch_id

        prefetcher = VerifiedBatchPrefetcher(range(100), load, max_prefetch=1)
        self.assertEqual(next(prefetcher), (0, 0))
        self.assertTrue(second_started.wait(timeout=1.0))
        before = time.monotonic()
        prefetcher.close()
        elapsed = time.monotonic() - before
        self.assertLess(elapsed, 1.0)
        self.assertFalse(prefetcher.worker_alive)
        self.assertEqual(started, [0, 1])
        prefetcher.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            next(prefetcher)

    def test_instance_cannot_be_reused_after_fork(self) -> None:
        prefetcher = VerifiedBatchPrefetcher([], lambda value: value)
        with mock.patch.object(os, "getpid", return_value=os.getpid() + 1):
            with self.assertRaisesRegex(RuntimeError, "after fork"):
                iter(prefetcher)
            with self.assertRaisesRegex(RuntimeError, "after fork"):
                prefetcher.close()
        prefetcher.close()


if __name__ == "__main__":
    unittest.main()
