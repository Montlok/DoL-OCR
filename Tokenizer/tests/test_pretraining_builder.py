# -*- coding: utf-8 -*-

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from Tokenizer.morphbpe import MorphBPETrainer
from Tokenizer.pretraining import (
    IGNORE_INDEX,
    PretrainingDataBuilder,
    iter_pack_samples,
    pack_samples,
)
from Tokenizer.pretraining.data_contract import (
    PRETRAINING_PRODUCER_GENERIC_BUILDER,
    build_pretraining_data_contract,
    canonical_json_sha256,
    load_and_validate_pretraining_data_contract,
    pretraining_producer_algorithm_contract,
    write_pretraining_data_contract,
)
from Tokenizer.unified.bundle import TokenizerBundle
from Tokenizer.unified.contract import (
    tokenizer_algorithm_contract,
    tokenizer_bundle_contract,
)


def build_smoke_bundle(tmp: str) -> TokenizerBundle:
    trainer = MorphBPETrainer(vocab_size=200, min_pair_freq=1)
    morphbpe = trainer.train(["ᠮᠣᠩᠭᠣᠯ ᠪᠢᠴᠢᠭ", "ᠮᠣᠩᠭᠣᠯ text"])
    morph_path = os.path.join(tmp, "morphbpe.json")
    morphbpe.save(morph_path)
    bundle = TokenizerBundle.from_files(morph_path)
    bundle_dir = os.path.join(tmp, "bundle")
    bundle.save_dir(bundle_dir)
    return TokenizerBundle.from_dir(bundle_dir)


class PretrainingBuilderTest(unittest.TestCase):
    def test_encode_pure_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = PretrainingDataBuilder(build_smoke_bundle(tmp), max_length=128)
            sample = builder.encode_text("ᠮᠣᠩᠭᠣᠯ 文字 test")
        self.assertEqual(len(sample.input_ids), len(sample.attention_mask))
        self.assertEqual(sample.labels[0], IGNORE_INDEX)
        self.assertEqual(sample.labels[1:], sample.input_ids[1:])
        self.assertEqual(len(sample.token_offsets), len(sample.input_ids))
        self.assertEqual(len(sample.word_pos), len(sample.input_ids))
        self.assertEqual(len(sample.morph_depth), len(sample.input_ids))
        self.assertEqual(sample.modality_spans["image_token_spans"], [])

    def test_morph_features_reset_at_spaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = build_smoke_bundle(tmp)
            builder = PretrainingDataBuilder(
                bundle, max_length=128, add_bos=False, add_eos=False
            )
            sample = builder.encode_text("ᠮᠣᠩᠭᠣᠯ test")

        space_idx = sample.input_ids.index(bundle.tokenizer.vocab["▁"])
        self.assertEqual(sample.morph_depth[space_idx], 0)
        self.assertLess(sample.word_pos[space_idx - 1], sample.word_pos[space_idx + 1])

    def test_encode_image_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = PretrainingDataBuilder(build_smoke_bundle(tmp), max_length=128)
            sample = builder.encode_json_obj(
                {
                    "type": "image_text",
                    "text": "文字 <image> test",
                    "images": ["x.jpg"],
                    "image_sizes": [[14, 14]],
                }
            )
        self.assertEqual(len(sample.modality_spans["image_token_spans"]), 1)
        self.assertEqual(sample.metadata["images"], ["x.jpg"])
        start, end = sample.modality_spans["image_token_spans"][0]
        self.assertTrue(
            all(label == IGNORE_INDEX for label in sample.labels[start:end])
        )
        self.assertEqual(sample.labels[0], IGNORE_INDEX)
        self.assertGreater(
            sum(1 for label in sample.labels if label != IGNORE_INDEX),
            0,
        )

    def test_ocr_metadata_is_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = PretrainingDataBuilder(build_smoke_bundle(tmp), max_length=128)
            sample = builder.encode_json_obj(
                {
                    "type": "ocr",
                    "text": "<image> text",
                    "images": ["x.jpg"],
                    "image_sizes": [[14, 14]],
                    "ocr": [{"text": "hello", "bbox": [0, 0, 1, 1]}],
                }
            )
        self.assertEqual(sample.metadata["type"], "ocr")
        self.assertEqual(sample.metadata["ocr"][0]["text"], "hello")

    def test_flat_ocr_labels_are_normalized_for_single_image_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = PretrainingDataBuilder(build_smoke_bundle(tmp), max_length=128)
            sample = builder.encode_json_obj(
                {
                    "type": "ocr",
                    "text": "<image> text",
                    "images": ["x.jpg"],
                    "image_sizes": [[14, 14]],
                    "ocr_labels": [1, 2, 3],
                    "reading_order": [0, 1, 2],
                }
            )
        self.assertEqual(sample.ocr_labels, [[1, 2, 3]])
        self.assertEqual(sample.reading_order, [[0, 1, 2]])

    def test_structural_special_labels_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = PretrainingDataBuilder(build_smoke_bundle(tmp), max_length=128)
            sample = builder.encode_text("<ocr> hello")
        self.assertEqual(sample.labels[0], IGNORE_INDEX)

    def test_empty_label_ignore_tokens_disables_masking(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = build_smoke_bundle(tmp)
            default_builder = PretrainingDataBuilder(bundle, max_length=128)
            default_sample = default_builder.encode_text("<ocr> hello")
            builder = PretrainingDataBuilder(
                bundle, max_length=128, label_ignore_tokens=set()
            )
            sample = builder.encode_text("<ocr> hello")
        # Default behaviour masks at least one structural token (e.g. <bos>,
        # <ocr>). With an empty ignore set, no token should be masked.
        self.assertIn(IGNORE_INDEX, default_sample.labels)
        self.assertNotIn(IGNORE_INDEX, sample.labels)

    def test_truncation_does_not_cut_image_span(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = PretrainingDataBuilder(build_smoke_bundle(tmp), max_length=3)
            sample = builder.encode_json_obj(
                {
                    "type": "image_text",
                    "text": "<image> test",
                    "images": ["x.jpg"],
                    "image_sizes": [[29, 29]],
                }
            )
        self.assertLessEqual(len(sample.input_ids), 3)
        self.assertEqual(sample.modality_spans["image_token_spans"], [])
        self.assertEqual(sample.images, [])
        self.assertEqual(sample.image_sizes, [])
        self.assertTrue(sample.metadata["truncated"])

    def test_pack_samples_for_text_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = build_smoke_bundle(tmp)
            builder = PretrainingDataBuilder(
                bundle, max_length=64, add_bos=False, add_eos=False
            )
            first = builder.encode_text("hello")
            second = builder.encode_text("test")
            packed = pack_samples(
                [first, second],
                max_length=64,
                pad_id=bundle.tokenizer.vocab["<pad>"],
                eos_id=bundle.tokenizer.vocab["<eos>"],
            )
        self.assertEqual(len(packed), 1)
        self.assertEqual(packed[0].metadata["num_samples"], 2)
        self.assertLessEqual(len(packed[0].input_ids), 64)
        self.assertEqual(len(packed[0].labels), len(packed[0].attention_mask))
        self.assertEqual(len(packed[0].word_pos), len(packed[0].input_ids))
        self.assertGreater(max(packed[0].word_pos), max(first.word_pos))

    def test_iter_pack_samples_matches_list_pack(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = build_smoke_bundle(tmp)
            builder = PretrainingDataBuilder(
                bundle, max_length=64, add_bos=False, add_eos=False
            )
            samples = [
                builder.encode_text("hello"),
                builder.encode_text("test"),
                builder.encode_text("ᠮᠣᠩᠭᠣᠯ"),
            ]
            kwargs = {
                "max_length": 64,
                "pad_id": bundle.tokenizer.vocab["<pad>"],
                "eos_id": bundle.tokenizer.vocab["<eos>"],
            }
            eager = pack_samples(samples, **kwargs)
            streamed = list(iter_pack_samples(iter(samples), **kwargs))
        self.assertEqual(eager, streamed)

    def test_pack_samples_can_pad_to_max_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = build_smoke_bundle(tmp)
            builder = PretrainingDataBuilder(
                bundle, max_length=64, add_bos=False, add_eos=False
            )
            sample = builder.encode_text("hello")
            packed = pack_samples(
                [sample],
                max_length=8,
                pad_id=bundle.tokenizer.vocab["<pad>"],
                eos_id=bundle.tokenizer.vocab["<eos>"],
                pad_to_max_length=True,
            )
        self.assertEqual(len(packed), 1)
        self.assertEqual(len(packed[0].input_ids), 8)
        self.assertEqual(packed[0].attention_mask[-1], 0)
        self.assertEqual(packed[0].labels[-1], IGNORE_INDEX)
        self.assertEqual(packed[0].word_pos[-1], 0)
        self.assertEqual(packed[0].morph_depth[-1], 0)

    def test_pack_samples_trims_modality_spans_to_sequence_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = build_smoke_bundle(tmp)
            builder = PretrainingDataBuilder(
                bundle, max_length=128, add_bos=False, add_eos=False
            )
            image_sample = builder.encode_json_obj(
                {
                    "type": "image_text",
                    "text": "<image> test",
                    "images": ["x.jpg"],
                    "image_sizes": [[29, 29]],
                }
            )
            packed = pack_samples(
                [image_sample],
                max_length=3,
                pad_id=bundle.tokenizer.vocab["<pad>"],
                eos_id=bundle.tokenizer.vocab["<eos>"],
            )

        self.assertEqual(packed, [])

    def test_pack_trim_does_not_leave_partial_image_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = build_smoke_bundle(tmp)
            builder = PretrainingDataBuilder(
                bundle, max_length=128, add_bos=False, add_eos=False
            )
            image_sample = builder.encode_json_obj(
                {
                    "type": "image_text",
                    "text": "a <image> b",
                    "images": ["x.jpg"],
                    "image_sizes": [[29, 29]],
                }
            )
            packed = pack_samples(
                [image_sample],
                max_length=3,
                pad_id=bundle.tokenizer.vocab["<pad>"],
                eos_id=bundle.tokenizer.vocab["<eos>"],
            )
        forbidden = {
            bundle.tokenizer.vocab["<image_start>"],
            bundle.tokenizer.vocab["<image_patch>"],
            bundle.tokenizer.vocab["<image_end>"],
        }
        self.assertEqual(len(packed), 1)
        self.assertTrue(forbidden.isdisjoint(packed[0].input_ids))
        self.assertEqual(packed[0].modality_spans["image_token_spans"], [])

    def test_build_pretraining_data_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = build_smoke_bundle(tmp)
            bundle_dir = os.path.join(tmp, "bundle")
            bundle.save_dir(bundle_dir)
            inp = os.path.join(tmp, "input.jsonl")
            out = os.path.join(tmp, "out.jsonl")
            with open(inp, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "type": "image_text",
                            "text": "文字 <image>",
                            "images": ["x.jpg"],
                            "image_sizes": [[14, 14]],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "Tokenizer.tools.build_pretraining_data",
                    "--tokenizer-bundle",
                    bundle_dir,
                    "--input",
                    inp,
                    "--output",
                    out,
                    "--max-length",
                    "128",
                    "--pack",
                    "--pack-max-length",
                    "16",
                    "--pad-to-max-length",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(proc.stdout)
            with open(out, "r", encoding="utf-8") as f:
                row = json.loads(f.readline())
            receipt = load_and_validate_pretraining_data_contract(
                summary["receipt"],
                out,
                tokenizer_bundle=tokenizer_bundle_contract(bundle_dir),
                tokenizer_algorithm=tokenizer_algorithm_contract(),
            )
        self.assertEqual(summary["num_samples"], 1)
        self.assertGreater(summary["supervised_tokens"], 0)
        self.assertIn("input_ids", row)
        self.assertEqual(len(row["input_ids"]), len(row["labels"]))
        self.assertEqual(len(row["input_ids"]), len(row["word_pos"]))
        self.assertEqual(len(row["input_ids"]), len(row["morph_depth"]))
        self.assertEqual(len(row["input_ids"]), 16)
        self.assertEqual(row["labels"][-1], IGNORE_INDEX)
        self.assertEqual(receipt["data_file_count"], 1)
        self.assertEqual(receipt["data_sha256"], summary["data_sha256"])
        self.assertEqual(
            receipt["producer_kind"],
            PRETRAINING_PRODUCER_GENERIC_BUILDER,
        )
        self.assertEqual(
            receipt["producer_algorithm"],
            pretraining_producer_algorithm_contract(
                PRETRAINING_PRODUCER_GENERIC_BUILDER
            ),
        )

    def test_build_pretraining_data_cli_can_rotate_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = build_smoke_bundle(tmp)
            bundle_dir = os.path.join(tmp, "bundle")
            bundle.save_dir(bundle_dir)
            inp = os.path.join(tmp, "input.jsonl")
            out = os.path.join(tmp, "out.jsonl")
            with open(inp, "w", encoding="utf-8") as f:
                for text in ("ᠮᠣᠩᠭᠣᠯ", "hello"):
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "Tokenizer.tools.build_pretraining_data",
                    "--tokenizer-bundle",
                    bundle_dir,
                    "--input",
                    inp,
                    "--output",
                    out,
                    "--max-length",
                    "64",
                    "--shard-sample-budget",
                    "1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(proc.stdout)
            shards = summary["shards"]
            receipt = json.loads(
                Path(summary["receipt"]).read_text(encoding="utf-8")
            )
        self.assertEqual(summary["num_samples"], 2)
        self.assertEqual(len(shards), 2)
        self.assertTrue(shards[0].endswith("out-00000.jsonl"))
        self.assertTrue(shards[1].endswith("out-00001.jsonl"))
        self.assertEqual(
            [entry["name"] for entry in receipt["data_files"]],
            ["out-00000.jsonl", "out-00001.jsonl"],
        )

    def test_pretraining_receipt_detects_shard_failure_injections(self):
        scenarios = ("mutate", "add", "remove", "reorder")
        for scenario in scenarios:
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                build_smoke_bundle(tmp)
                bundle_dir = root / "bundle"
                shard_dir = root / "shards"
                shard_dir.mkdir()
                first = shard_dir / "shard-00000.jsonl"
                second = shard_dir / "shard-00001.jsonl"
                first.write_text('{"input_ids":[2,3],"labels":[-100,3]}\n')
                second.write_text('{"input_ids":[2,4],"labels":[-100,4]}\n')
                bundle_identity = tokenizer_bundle_contract(bundle_dir)
                algorithm_identity = tokenizer_algorithm_contract()
                payload = build_pretraining_data_contract(
                    [first, second],
                    producer_kind=PRETRAINING_PRODUCER_GENERIC_BUILDER,
                    tokenizer_bundle=bundle_identity,
                    tokenizer_algorithm=algorithm_identity,
                )
                receipt = root / "receipt.json"
                write_pretraining_data_contract(receipt, payload)

                if scenario == "mutate":
                    first.write_text(
                        '{"input_ids":[2,9],"labels":[-100,9]}\n',
                        encoding="utf-8",
                    )
                    data_spec = shard_dir
                elif scenario == "add":
                    (shard_dir / "shard-00002.jsonl").write_text(
                        '{"input_ids":[2,5],"labels":[-100,5]}\n',
                        encoding="utf-8",
                    )
                    data_spec = shard_dir
                elif scenario == "remove":
                    second.unlink()
                    data_spec = shard_dir
                else:
                    data_spec = [second, first]

                with self.assertRaisesRegex(
                    ValueError,
                    "resolved data shards differ",
                ):
                    load_and_validate_pretraining_data_contract(
                        receipt,
                        data_spec,
                        tokenizer_bundle=bundle_identity,
                        tokenizer_algorithm=algorithm_identity,
                    )

    def test_pretraining_receipt_rejects_tampered_or_unknown_producer(self):
        for scenario in ("algorithm", "unknown-kind"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                build_smoke_bundle(tmp)
                shard = root / "train.jsonl"
                shard.write_text(
                    '{"input_ids":[2,3],"labels":[-100,3],'
                    '"word_pos":[0,0],"morph_depth":[0,0]}\n',
                    encoding="utf-8",
                )
                bundle_identity = tokenizer_bundle_contract(root / "bundle")
                algorithm_identity = tokenizer_algorithm_contract()
                payload = build_pretraining_data_contract(
                    shard,
                    producer_kind=PRETRAINING_PRODUCER_GENERIC_BUILDER,
                    tokenizer_bundle=bundle_identity,
                    tokenizer_algorithm=algorithm_identity,
                )
                if scenario == "algorithm":
                    payload["producer_algorithm"]["source_sha256"] = "0" * 64
                    expected = "producer_algorithm differs"
                else:
                    payload["producer_kind"] = "unregistered_builder"
                    expected = "unknown pretraining producer_kind"
                core = dict(payload)
                core.pop("contract_canonical_sha256")
                payload["contract_canonical_sha256"] = canonical_json_sha256(
                    core
                )
                receipt = root / "receipt.json"
                write_pretraining_data_contract(receipt, payload)

                with self.assertRaisesRegex(ValueError, expected):
                    load_and_validate_pretraining_data_contract(
                        receipt,
                        shard,
                        tokenizer_bundle=bundle_identity,
                        tokenizer_algorithm=algorithm_identity,
                    )

    def test_pretraining_receipt_allows_mount_relocation_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_smoke_bundle(tmp)
            bundle_dir = root / "bundle"
            original = root / "original"
            original.mkdir()
            (original / "shard-00000.jsonl").write_text(
                '{"input_ids":[2,3],"labels":[-100,3]}\n',
                encoding="utf-8",
            )
            bundle_identity = tokenizer_bundle_contract(bundle_dir)
            algorithm_identity = tokenizer_algorithm_contract()
            receipt = root / "receipt.json"
            write_pretraining_data_contract(
                receipt,
                build_pretraining_data_contract(
                    original,
                    producer_kind=PRETRAINING_PRODUCER_GENERIC_BUILDER,
                    tokenizer_bundle=bundle_identity,
                    tokenizer_algorithm=algorithm_identity,
                ),
            )
            relocated_root = root / "different-mount"
            relocated_bundle = relocated_root / "bundle"
            shutil.copytree(bundle_dir, relocated_bundle)
            relocated_identity = tokenizer_bundle_contract(relocated_bundle)
            self.assertEqual(relocated_identity, bundle_identity)
            relocated = relocated_root / "rows"
            shutil.copytree(original, relocated)
            validated = load_and_validate_pretraining_data_contract(
                receipt,
                relocated,
                tokenizer_bundle=relocated_identity,
                tokenizer_algorithm=algorithm_identity,
            )
        self.assertEqual(validated["data_file_count"], 1)

    def test_build_pretraining_data_cli_streams_without_pack(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = build_smoke_bundle(tmp)
            bundle_dir = os.path.join(tmp, "bundle")
            bundle.save_dir(bundle_dir)
            inp = os.path.join(tmp, "input.txt")
            out = os.path.join(tmp, "out.jsonl")
            with open(inp, "w", encoding="utf-8") as f:
                f.write("ᠮᠣᠩᠭᠣᠯ 文字 test\n")
                f.write("hello\n")
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "Tokenizer.tools.build_pretraining_data",
                    "--tokenizer-bundle",
                    bundle_dir,
                    "--input",
                    inp,
                    "--output",
                    out,
                    "--max-length",
                    "128",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(proc.stdout)
            with open(out, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]

        self.assertEqual(summary["num_samples"], 2)
        self.assertEqual(len(rows), 2)
        self.assertIn("input_ids", rows[0])
        self.assertGreater(summary["supervised_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
