# -*- coding: utf-8 -*-

import json
import os
import tempfile
import unittest

from Tokenizer.evals.pretraining_gate import run_gate
from Tokenizer.pretraining import PretrainingDataBuilder, encoded_sample_to_dict
from Tokenizer.tests.test_pretraining_builder import build_smoke_bundle


class PretrainingGateTest(unittest.TestCase):
    def test_smoke_bundle_and_jsonl_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = build_smoke_bundle(tmp)
            bundle_dir = os.path.join(tmp, "bundle")
            bundle.save_dir(bundle_dir)
            inp = os.path.join(tmp, "input.jsonl")
            with open(inp, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "type": "image_text",
                            "text": "文字 <image> test",
                            "images": ["x.jpg"],
                            "image_sizes": [[14, 14]],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            result = run_gate(bundle_dir, inp, max_length=128)

        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(result["num_samples"], 1)
        self.assertIn("unk_rate", result["metrics"])
        self.assertIn("max_morph_depth", result["metrics"])
        self.assertGreater(result["metrics"]["supervised_tokens"], 0)

    def test_bad_sample_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = build_smoke_bundle(tmp)
            bundle_dir = os.path.join(tmp, "bundle")
            bundle.save_dir(bundle_dir)
            inp = os.path.join(tmp, "bad.jsonl")
            with open(inp, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "type": "image_text",
                            "text": "文字 <image> test",
                            "images": [],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            result = run_gate(bundle_dir, inp, max_length=128)

        self.assertFalse(result["passed"])
        self.assertEqual(result["num_samples"], 0)
        self.assertTrue(
            any("encode failed" in item["message"] for item in result["failures"])
        )
        self.assertTrue(
            any("no valid samples" in item["message"] for item in result["failures"])
        )

    def test_encoded_pretraining_rows_are_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = build_smoke_bundle(tmp)
            bundle_dir = os.path.join(tmp, "bundle")
            bundle.save_dir(bundle_dir)
            builder = PretrainingDataBuilder(bundle, max_length=128)
            sample = builder.encode_json_obj(
                {
                    "type": "image_text",
                    "text": "文字 <image> test",
                    "images": ["x.jpg"],
                    "image_sizes": [[14, 14]],
                }
            )
            inp = os.path.join(tmp, "encoded.jsonl")
            with open(inp, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(encoded_sample_to_dict(sample), ensure_ascii=False)
                    + "\n"
                )

            result = run_gate(bundle_dir, inp, max_length=128)

        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(result["num_samples"], 1)

    def test_bad_morph_feature_length_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = build_smoke_bundle(tmp)
            bundle_dir = os.path.join(tmp, "bundle")
            bundle.save_dir(bundle_dir)
            builder = PretrainingDataBuilder(bundle, max_length=128)
            sample = builder.encode_text("ᠮᠣᠩᠭᠣᠯ text")
            row = encoded_sample_to_dict(sample)
            row["word_pos"] = row["word_pos"][:-1]
            inp = os.path.join(tmp, "bad_morph.jsonl")
            with open(inp, "w", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            result = run_gate(bundle_dir, inp, max_length=128)

        self.assertFalse(result["passed"])
        self.assertTrue(
            any(
                "word_pos length mismatch" in item["message"]
                for item in result["failures"]
            ),
            result["failures"],
        )

    def test_encoded_row_media_payload_count_must_match_spans(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = build_smoke_bundle(tmp)
            bundle_dir = os.path.join(tmp, "bundle")
            bundle.save_dir(bundle_dir)
            builder = PretrainingDataBuilder(bundle, max_length=128)
            sample = builder.encode_json_obj(
                {
                    "type": "image_text",
                    "text": "文字 <image> test",
                    "images": ["x.jpg"],
                    "image_sizes": [[14, 14]],
                }
            )
            row = encoded_sample_to_dict(sample)
            row["images"].append("extra.jpg")
            inp = os.path.join(tmp, "bad_media.jsonl")
            with open(inp, "w", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            result = run_gate(bundle_dir, inp, max_length=128)

        self.assertFalse(result["passed"])
        self.assertTrue(
            any("images count" in item["message"] for item in result["failures"]),
            result["failures"],
        )

    def test_malformed_encoded_row_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = build_smoke_bundle(tmp)
            bundle_dir = os.path.join(tmp, "bundle")
            bundle.save_dir(bundle_dir)
            inp = os.path.join(tmp, "malformed.jsonl")
            with open(inp, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "input_ids": ["not", "an", "int"],
                            "attention_mask": [1, 1, 1],
                            "labels": [0, 0, 0],
                            "token_offsets": [[0, 1], [1, 2], [2, 3]],
                            "modality_spans": {
                                "image_token_spans": [],
                                "video_token_spans": [],
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            result = run_gate(bundle_dir, inp, max_length=128)

        self.assertFalse(result["passed"])
        self.assertEqual(result["num_samples"], 0)
        self.assertTrue(
            any(
                "encoded row parse failed" in item["message"]
                for item in result["failures"]
            ),
            result["failures"],
        )

    def test_short_labels_with_modality_span_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = build_smoke_bundle(tmp)
            bundle_dir = os.path.join(tmp, "bundle")
            bundle.save_dir(bundle_dir)
            builder = PretrainingDataBuilder(bundle, max_length=128)
            sample = builder.encode_json_obj(
                {
                    "type": "image_text",
                    "text": "文字 <image> test",
                    "images": ["x.jpg"],
                    "image_sizes": [[14, 14]],
                }
            )
            row = encoded_sample_to_dict(sample)
            # Truncate labels so they are shorter than input_ids while keeping
            # a modality span that extends past the labels length. The gate
            # should report structured failures (length mismatch) without
            # raising an IndexError from _validate_spans.
            row["labels"] = row["labels"][: max(1, len(row["labels"]) // 2)]
            inp = os.path.join(tmp, "short_labels.jsonl")
            with open(inp, "w", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            result = run_gate(bundle_dir, inp, max_length=128)

        self.assertFalse(result["passed"])
        self.assertTrue(
            any("length mismatch" in item["message"] for item in result["failures"]),
            result["failures"],
        )


if __name__ == "__main__":
    unittest.main()
