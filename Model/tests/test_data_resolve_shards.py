# -*- coding: utf-8 -*-

"""Unit tests for ``Model.training.data._resolve_shards``."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from Model.training.data import _resolve_shards
from Model.training.data import build_dataloader
from Model.config import TrainingConfig


class ResolveShardsTest(unittest.TestCase):
    def test_directory_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.jsonl").write_text("{}\n", encoding="utf-8")
            (root / "b.jsonl").write_text("{}\n", encoding="utf-8")
            (root / "c.txt").write_text("nope\n", encoding="utf-8")

            shards = _resolve_shards(str(root))
            self.assertEqual([p.name for p in shards], ["a.jsonl", "b.jsonl"])

    def test_absolute_glob_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "shard_0.jsonl").write_text("{}\n", encoding="utf-8")
            (root / "shard_1.jsonl").write_text("{}\n", encoding="utf-8")
            (root / "other.txt").write_text("nope\n", encoding="utf-8")

            spec = str(root / "*.jsonl")  # absolute path with wildcard
            shards = _resolve_shards(spec)
            self.assertEqual(
                [p.name for p in shards],
                ["shard_0.jsonl", "shard_1.jsonl"],
            )

    def test_recursive_glob(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sub").mkdir()
            (root / "sub" / "deep.jsonl").write_text("{}\n", encoding="utf-8")
            (root / "top.jsonl").write_text("{}\n", encoding="utf-8")

            spec = str(root / "**" / "*.jsonl")
            shards = _resolve_shards(spec)
            names = sorted(p.name for p in shards)
            self.assertIn("deep.jsonl", names)
            self.assertIn("top.jsonl", names)

    def test_single_file_passthrough(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            f = root / "only.jsonl"
            f.write_text("{}\n", encoding="utf-8")
            shards = _resolve_shards(str(f))
            self.assertEqual(shards, [f])

    def test_missing_single_file_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shards = _resolve_shards(str(Path(tmp) / "missing.jsonl"))
            self.assertEqual(shards, [])

    def test_sequence_passthrough(self) -> None:
        shards = _resolve_shards(["/a.jsonl", Path("/b.jsonl")])
        self.assertEqual([str(p) for p in shards], ["/a.jsonl", "/b.jsonl"])

    def test_build_dataloader_rejects_missing_sequence_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.jsonl"
            cfg = TrainingConfig(train_data="", seq_len=8, num_workers=0)
            with self.assertRaisesRegex(FileNotFoundError, "do not exist"):
                build_dataloader([missing], cfg)


if __name__ == "__main__":
    unittest.main()
