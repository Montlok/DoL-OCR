# -*- coding: utf-8 -*-

"""Tests for tool-result span masking (Model/posttrain/masking.py)."""

import unittest

import torch

from Model.posttrain.masking import (
    completion_mask_excluding_tool_results,
    span_mask,
    tool_result_span_mask,
)

OPEN = [90, 91]   # pretend-encoded "<tool_result>"
CLOSE = [92, 93]  # pretend-encoded "</tool_result>"


class ToolResultMaskTest(unittest.TestCase):
    def test_masks_inclusive_span(self):
        # [a, <tr>, X, Y, </tr>, b]
        ids = torch.tensor([1, 90, 91, 7, 8, 92, 93, 2])
        m = tool_result_span_mask(ids, OPEN, CLOSE)
        self.assertEqual(m.tolist(), [1, 0, 0, 0, 0, 0, 0, 1])

    def test_no_markers_all_ones(self):
        ids = torch.tensor([1, 2, 3, 4])
        m = tool_result_span_mask(ids, OPEN, CLOSE)
        self.assertEqual(m.tolist(), [1, 1, 1, 1])

    def test_dangling_open_masks_to_end(self):
        ids = torch.tensor([1, 90, 91, 7, 8])
        m = tool_result_span_mask(ids, OPEN, CLOSE)
        self.assertEqual(m.tolist(), [1, 0, 0, 0, 0])

    def test_multiple_spans(self):
        ids = torch.tensor([90, 91, 5, 92, 93, 6, 90, 91, 7, 92, 93])
        m = tool_result_span_mask(ids, OPEN, CLOSE)
        self.assertEqual(m.tolist(), [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])

    def test_batched(self):
        ids = torch.tensor([[1, 90, 91, 92, 93, 2], [3, 4, 5, 6, 7, 8]])
        m = tool_result_span_mask(ids, OPEN, CLOSE)
        self.assertEqual(m[0].tolist(), [1, 0, 0, 0, 0, 1])
        self.assertEqual(m[1].tolist(), [1, 1, 1, 1, 1, 1])

    def test_think_and_tool_call_not_masked(self):
        # think (80,81..82,83) and tool_call (84,85..86,87) stay supervised.
        ids = torch.tensor([80, 81, 9, 82, 83, 84, 85, 1, 86, 87])
        m = tool_result_span_mask(ids, OPEN, CLOSE)
        self.assertTrue(torch.all(m == 1.0))

    def test_combines_with_base_completion_mask(self):
        # First two tokens are the prompt (base mask 0).
        ids = torch.tensor([1, 2, 5, 90, 91, 92, 93, 8])
        base = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        out = completion_mask_excluding_tool_results(base, ids, OPEN, CLOSE)
        # prompt masked AND tool-result span masked; only positions 2 and 7 stay.
        self.assertEqual(out.tolist(), [0, 0, 1, 0, 0, 0, 0, 1])

    def test_generic_span_mask_multiple_marker_types(self):
        # Two different injected span types masked in one pass (extensible API).
        ids = torch.tensor([1, 90, 91, 5, 92, 93, 2, 70, 9, 71, 3])
        m = span_mask(ids, [(OPEN, CLOSE), ([70], [71])])
        self.assertEqual(m.tolist(), [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1])


if __name__ == "__main__":
    unittest.main()
