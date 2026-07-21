# -*- coding: utf-8 -*-

"""Unit tests for OCR training-row construction (torch/render-free)."""

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from Model.ocr.data import build_ocr_row, split_ocr_row
from scripts.build_ocr_data import _iter_input_text

BOS, ISTART, IPATCH, IEND, EOS = 2, 6, 7, 8, 3
IGNORE = -100


class TestBuildOCRRow(unittest.TestCase):
    def _row(self, target, n=4, instr=(), add_eos=True):
        return build_ocr_row(
            target,
            n,
            "images/0.png",
            bos_id=BOS,
            image_start_id=ISTART,
            image_patch_id=IPATCH,
            image_end_id=IEND,
            eos_id=EOS,
            instruction_ids=instr,
            add_eos=add_eos,
        )

    def test_slot_count_matches_n_image_tokens(self):
        row = self._row([10, 11, 12], n=5)
        self.assertEqual(row["input_ids"].count(IPATCH), 5)

    def test_layout_order(self):
        row = self._row([10, 11], n=3)
        # [BOS] <image_start> <patch>*3 <image_end> 10 11 [EOS]
        self.assertEqual(
            row["input_ids"], [BOS, ISTART, IPATCH, IPATCH, IPATCH, IEND, 10, 11, EOS]
        )

    def test_labels_mask_prompt_only(self):
        row = self._row([10, 11], n=3)
        # prompt = BOS + ISTART + 3 patches + IEND = 6 positions masked
        labels = row["labels"]
        self.assertEqual(labels[:6], [IGNORE] * 6)
        self.assertEqual(labels[6:], [10, 11, EOS])

    def test_instruction_is_masked(self):
        row = self._row([10], n=2, instr=[99, 98])
        # BOS, ISTART, patch, patch, IEND, 99, 98 -> 7 masked, then target 10, EOS
        self.assertEqual(row["labels"], [IGNORE] * 7 + [10, EOS])
        self.assertEqual(
            row["input_ids"], [BOS, ISTART, IPATCH, IPATCH, IEND, 99, 98, 10, EOS]
        )

    def test_lengths_align(self):
        row = self._row([10, 11, 12], n=4, instr=[5])
        n = len(row["input_ids"])
        self.assertEqual(len(row["attention_mask"]), n)
        self.assertEqual(len(row["labels"]), n)
        self.assertTrue(all(m == 1 for m in row["attention_mask"]))

    def test_single_image_passthrough(self):
        row = self._row([10], n=2)
        self.assertEqual(row["images"], ["images/0.png"])

    def test_no_eos(self):
        row = self._row([10, 11], n=2, add_eos=False)
        self.assertEqual(row["input_ids"][-1], 11)
        self.assertEqual(row["labels"][-2:], [10, 11])

    def test_invalid_n_image_tokens(self):
        with self.assertRaises(ValueError):
            self._row([10], n=0)

    def test_empty_target_raises(self):
        with self.assertRaises(ValueError):
            self._row([], n=2)

    def test_target_cannot_add_extra_image_patch_slots(self):
        with self.assertRaisesRegex(ValueError, "exactly n_image_tokens"):
            self._row([10, IPATCH, 11], n=2)

    def test_instruction_cannot_add_extra_image_patch_slots(self):
        with self.assertRaisesRegex(ValueError, "exactly n_image_tokens"):
            self._row([10], n=2, instr=[99, IPATCH])

    def test_row_length_matches_prompt_overhead_arithmetic(self):
        # 4 = BOS + <image_start> + <image_end> + EOS (build_ocr_row's
        # add_eos=True default); total row length = 4 + n_image_tokens +
        # len(instruction_ids) + len(target_ids). This is the formula
        # scripts/build_ocr_data.py's --max-seq-len guard relies on to
        # skip over-length rows before rendering.
        cases = [
            {"target": [10], "n": 4, "instr": ()},
            {"target": [10, 11, 12], "n": 5, "instr": ()},
            {"target": [10, 11], "n": 3, "instr": [99, 98]},
            {"target": list(range(100, 120)), "n": 64, "instr": [1, 2, 3, 4, 5]},
        ]
        for case in cases:
            with self.subTest(case=case):
                row = self._row(case["target"], n=case["n"], instr=case["instr"])
                expected = 4 + case["n"] + len(case["instr"]) + len(case["target"])
                self.assertEqual(len(row["input_ids"]), expected)


class TestSplitOCRRow(unittest.TestCase):
    def _row(self, target, instr=(), add_eos=True):
        return build_ocr_row(
            target,
            4,
            "img.png",
            bos_id=BOS,
            image_start_id=ISTART,
            image_patch_id=IPATCH,
            image_end_id=IEND,
            eos_id=EOS,
            instruction_ids=instr,
            add_eos=add_eos,
        )

    def test_round_trips_build_ocr_row(self):
        row = self._row([100, 101, 102], instr=[50, 51])
        prompt, target, image = split_ocr_row(row, eos_id=EOS)
        self.assertEqual(prompt, [BOS, ISTART] + [IPATCH] * 4 + [IEND, 50, 51])
        self.assertEqual(target, [100, 101, 102])
        self.assertEqual(image, "img.png")

    def test_keeps_eos_when_id_not_given(self):
        row = self._row([100, 101])
        _, target, _ = split_ocr_row(row)
        self.assertEqual(target, [100, 101, EOS])

    def test_no_eos_row_is_not_truncated(self):
        row = self._row([100, 101], add_eos=False)
        _, target, _ = split_ocr_row(row, eos_id=EOS)
        self.assertEqual(target, [100, 101])

    def test_rejects_fully_masked_and_unmasked_rows(self):
        row = self._row([100])
        masked = dict(row, labels=[IGNORE] * len(row["labels"]))
        with self.assertRaises(ValueError):
            split_ocr_row(masked)
        unmasked = dict(row, labels=list(row["input_ids"]))
        with self.assertRaises(ValueError):
            split_ocr_row(unmasked)

    def test_rejects_non_contiguous_target(self):
        row = self._row([100, 101, 102])
        labels = list(row["labels"])
        labels[-2] = IGNORE
        with self.assertRaises(ValueError):
            split_ocr_row(dict(row, labels=labels))

    def test_rejects_misaligned_lengths(self):
        row = self._row([100])
        with self.assertRaises(ValueError):
            split_ocr_row(dict(row, labels=row["labels"][:-1]))

    def test_missing_images_field_yields_none(self):
        row = self._row([100, 101])
        row.pop("images")
        _, _, image = split_ocr_row(row, eos_id=EOS)
        self.assertIsNone(image)


class TestBuildOCRInputText(unittest.TestCase):
    def _write_input(self, text: str) -> str:
        td = TemporaryDirectory()
        self.addCleanup(td.cleanup)
        path = Path(td.name) / "input.txt"
        path.write_text(text, encoding="utf-8")
        return str(path)

    def test_iter_input_text_accepts_plain_and_jsonl(self):
        path = self._write_input('{"text": "abc"}\n\nplain\n')
        self.assertEqual(list(_iter_input_text(path)), [(1, "abc"), (3, "plain")])

    def test_iter_input_text_rejects_invalid_json_with_line_number(self):
        path = self._write_input('{"text":\n')
        with self.assertRaisesRegex(ValueError, r"input\.txt:1: invalid JSONL"):
            list(_iter_input_text(path))

    def test_iter_input_text_rejects_json_arrays_with_line_number(self):
        path = self._write_input('["abc"]\n')
        with self.assertRaisesRegex(ValueError, r"input\.txt:1: JSONL row must be"):
            list(_iter_input_text(path))

    def test_iter_input_text_requires_non_empty_text_field(self):
        path = self._write_input('{"text": ""}\n')
        with self.assertRaisesRegex(ValueError, r"input\.txt:1: JSONL row missing"):
            list(_iter_input_text(path))


if __name__ == "__main__":
    unittest.main()
