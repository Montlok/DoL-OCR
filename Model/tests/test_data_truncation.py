# -*- coding: utf-8 -*-

"""Tests for span-aware truncation guard in ``_normalize_row`` (M2)."""

from __future__ import annotations

import unittest

from Model.training.data import _normalize_row


def _row(n: int, *, images: list | None = None) -> dict:
    row = {
        "input_ids": list(range(n)),
        "attention_mask": [1] * n,
        "labels": list(range(n)),
    }
    if images is not None:
        row["images"] = images
    return row


class NormalizeRowTruncationTest(unittest.TestCase):
    def test_text_row_truncates(self) -> None:
        out = _normalize_row(_row(10), max_seq_len=4)
        self.assertEqual(len(out["input_ids"]), 4)
        self.assertEqual(len(out["labels"]), 4)
        self.assertEqual(len(out["word_pos"]), 4)

    def test_text_row_not_truncated_when_within_limit(self) -> None:
        out = _normalize_row(_row(4), max_seq_len=8)
        self.assertEqual(len(out["input_ids"]), 4)

    def test_multimodal_row_refuses_truncation(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _normalize_row(_row(10, images=["img0.png"]), max_seq_len=4)
        self.assertIn("seq_len", str(ctx.exception))

    def test_multimodal_row_within_limit_ok(self) -> None:
        out = _normalize_row(_row(4, images=["img0.png"]), max_seq_len=8)
        self.assertEqual(out["images"], ["img0.png"])


if __name__ == "__main__":
    unittest.main()
