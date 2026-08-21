# -*- coding: utf-8 -*-

"""Per-pass reseed of the streaming shuffle.

A pass-stable seed shuffles every epoch into the same order, and a resumed
run (fresh iterator, ``state.step`` intact) re-trains on the head of the
stream — the OPT-class replay failure. Pin both halves of the contract:
(a) successive passes of one iterator emit different orders, and (b) a
rebuilt iterator reproduces the original stream exactly, which is the
property that makes the resume fast-forward in ``scripts/train_rdt`` valid.
"""

import itertools
import json
import tempfile
import unittest
from pathlib import Path

from Model.training.data import StreamingJsonlDataset


def _write_shard(dirpath: Path, n: int) -> Path:
    path = dirpath / "shard.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"i": i}) + "\n")
    return path


class StreamingReseedTest(unittest.TestCase):
    def test_passes_differ_and_rebuild_reproduces(self):
        with tempfile.TemporaryDirectory() as td:
            path = _write_shard(Path(td), 64)
            ds = StreamingJsonlDataset(
                [str(path)],
                world_size=1,
                rank=0,
                shuffle_buffer=8,
                seed=3,
                infinite=True,
            )
            stream = (row["i"] for row in iter(ds))
            pass0 = list(itertools.islice(stream, 64))
            pass1 = list(itertools.islice(stream, 64))
            # Each pass is a clean permutation (buffer drains at pass end) …
            self.assertEqual(sorted(pass0), list(range(64)))
            self.assertEqual(sorted(pass1), list(range(64)))
            # … but the orders must differ across passes.
            self.assertNotEqual(pass0, pass1)

            rebuilt = list(itertools.islice((r["i"] for r in iter(ds)), 64))
            self.assertEqual(pass0, rebuilt)

    def test_no_shuffle_passes_are_identity(self):
        with tempfile.TemporaryDirectory() as td:
            path = _write_shard(Path(td), 16)
            ds = StreamingJsonlDataset(
                [str(path)],
                world_size=1,
                rank=0,
                shuffle_buffer=0,
                seed=3,
                infinite=True,
            )
            stream = (row["i"] for row in iter(ds))
            pass0 = list(itertools.islice(stream, 16))
            pass1 = list(itertools.islice(stream, 16))
            self.assertEqual(pass0, list(range(16)))
            self.assertEqual(pass1, list(range(16)))


if __name__ == "__main__":
    unittest.main()
