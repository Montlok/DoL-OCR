# -*- coding: utf-8 -*-

"""Regression tests for the generic ``parquet`` and ``jsonl`` corpus sources.

These sources let the pretraining pipeline ingest arbitrary local HF dumps
(FineMath/OpenWebMath parquet, MC2 Mongolian jsonl) by text column, so the
data-mixing tool can weight every source uniformly.
"""

import json
import os
import tempfile
import unittest

from Tokenizer.tools import prepare_corpus

try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    _HAS_PYARROW = True
except ImportError:  # pragma: no cover - exercised only without pyarrow
    _HAS_PYARROW = False


class JsonlSourceTest(unittest.TestCase):
    def test_iter_jsonl_text_reads_field_and_skips_blank(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "data.jsonl")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"text": "hello"}) + "\n")
                fh.write("\n")  # blank line ignored
                fh.write(json.dumps({"text": ""}) + "\n")  # empty text skipped
                fh.write("not json\n")  # undecodable ignored
                fh.write(json.dumps({"text": "world"}) + "\n")
            got = list(prepare_corpus._iter_jsonl_text(path, "text", 0))
        self.assertEqual(got, ["hello", "world"])

    def test_iter_jsonl_text_custom_column_and_limit(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "data.jsonl")
            with open(path, "w", encoding="utf-8") as fh:
                for i in range(5):
                    fh.write(json.dumps({"body": f"line{i}"}) + "\n")
            got = list(prepare_corpus._iter_jsonl_text(path, "body", 3))
        self.assertEqual(got, ["line0", "line1", "line2"])


@unittest.skipUnless(_HAS_PYARROW, "pyarrow not installed")
class ParquetSourceTest(unittest.TestCase):
    def _write_parquet(self, path, texts, column="text"):
        table = pa.table({column: texts})
        pq.write_table(table, path)

    def test_iter_parquet_single_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "shard.parquet")
            self._write_parquet(path, ["alpha", "", "beta"])
            got = list(prepare_corpus._iter_parquet(path, "text", 0))
        self.assertEqual(got, ["alpha", "beta"])

    def test_iter_parquet_directory_recursive_and_sorted(self):
        with tempfile.TemporaryDirectory() as d:
            sub = os.path.join(d, "nested")
            os.makedirs(sub)
            self._write_parquet(os.path.join(d, "a.parquet"), ["a1", "a2"])
            self._write_parquet(os.path.join(sub, "b.parquet"), ["b1"])
            got = list(prepare_corpus._iter_parquet(d, "text", 0))
        self.assertEqual(sorted(got), ["a1", "a2", "b1"])

    def test_iter_parquet_limit(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "shard.parquet")
            self._write_parquet(path, [f"t{i}" for i in range(10)])
            got = list(prepare_corpus._iter_parquet(path, "text", 4))
        self.assertEqual(got, ["t0", "t1", "t2", "t3"])

    def test_iter_parquet_missing_column_raises(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "shard.parquet")
            self._write_parquet(path, ["x"], column="body")
            with self.assertRaises(SystemExit):
                list(prepare_corpus._iter_parquet(path, "text", 0))

    def test_iter_parquet_no_shards_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                list(prepare_corpus._iter_parquet(d, "text", 0))


if __name__ == "__main__":
    unittest.main()
