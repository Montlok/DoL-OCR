# -*- coding: utf-8 -*-

"""Verify ``StreamingJsonlDataset`` does not silently starve ranks when the
shard count is smaller than the DDP world size — in that regime every rank
must still receive the full shard list (replicated) so collective ops do
not deadlock on idle workers.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from Model.training.data import StreamingJsonlDataset


def _write_shards(root: Path, n: int) -> list[Path]:
    paths = []
    for i in range(n):
        p = root / f"shard_{i}.jsonl"
        p.write_text(
            json.dumps({"input_ids": [i], "attention_mask": [1], "labels": [i]}) + "\n",
            encoding="utf-8",
        )
        paths.append(p)
    return paths


class UnevenShardsTest(unittest.TestCase):
    def test_replicates_when_paths_lt_world_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _write_shards(root, 2)
            # world_size=4 strictly partitioned would leave ranks 2,3 empty.
            for rank in range(4):
                ds = StreamingJsonlDataset(
                    paths, world_size=4, rank=rank, infinite=False
                )
                files = ds._files_for_worker()
                self.assertEqual(
                    sorted(files), sorted(paths),
                    f"rank {rank} should see all shards (replicated fallback)",
                )

    def test_strict_partition_when_paths_ge_world_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _write_shards(root, 4)
            seen: list[Path] = []
            for rank in range(2):
                ds = StreamingJsonlDataset(
                    paths, world_size=2, rank=rank, infinite=False
                )
                seen.extend(ds._files_for_worker())
            self.assertEqual(sorted(seen), sorted(paths))
            self.assertEqual(len(seen), len(set(seen)), "ranks must not overlap")

    def test_small_shard_flushes_before_shuffle_buffer_is_full(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_shards(Path(tmp), 1)
            ds = StreamingJsonlDataset(paths, shuffle_buffer=1024, infinite=True)
            row = next(iter(ds))
            self.assertEqual(row["input_ids"], [0])

    def test_empty_shard_raises_instead_of_spinning_forever(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.jsonl"
            path.write_text("", encoding="utf-8")
            ds = StreamingJsonlDataset([path], infinite=True)
            with self.assertRaisesRegex(ValueError, "no JSONL rows"):
                next(iter(ds))

    def test_malformed_json_raises_with_location(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            path.write_text("{bad json}\n", encoding="utf-8")
            ds = StreamingJsonlDataset([path], infinite=False)
            with self.assertRaisesRegex(ValueError, "bad.jsonl:1"):
                next(iter(ds))


if __name__ == "__main__":
    unittest.main()
