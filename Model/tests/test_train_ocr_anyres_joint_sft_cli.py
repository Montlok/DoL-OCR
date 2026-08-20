# -*- coding: utf-8 -*-

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from Model.posttrain.ocr_quota_sampler import OCRQuotaSampler
from scripts import train_ocr_anyres_joint_sft as joint_cli


class _OCRDataset:
    def __init__(self) -> None:
        buckets = (
            "print",
            "handwritten_good",
            "handwritten_medium",
            "handwritten_poor",
        )
        self.sample_ids = tuple(
            f"{bucket}-{index}" for bucket in buckets for index in range(5)
        )
        self.quota_buckets = tuple(bucket for bucket in buckets for _ in range(5))
        self.dataset_contract = {
            "kind": "ocr",
            "contract_sha256": "a" * 64,
        }

    def get_by_sample_id(self, sample_id):
        return sample_id


class _TextDataset:
    def __init__(self, split: str = "train", digest: str = "b") -> None:
        self.contract_sha256 = digest * 64
        self.dataset_contract = {
            "kind": "text",
            "split": split,
            "contract_sha256": self.contract_sha256,
            "exclusion_contract": {"sha": "c" * 64},
        }

    def __len__(self):
        return 5

    def __getitem__(self, index):
        return {"index": index}


class _TextPartition:
    def __init__(self) -> None:
        self._datasets = {
            split: _TextDataset(split, digest)
            for split, digest in zip(
                (
                    "train",
                    "sft_validation",
                    "kl_selection",
                    "formal_monitor",
                ),
                "bcde",
                strict=True,
            )
        }
        self.partition_contract = {
            "kind": "dol_text_ce_replay_partition",
            "contract_sha256": "1" * 64,
        }

    def dataset(self, split: str) -> _TextDataset:
        return self._datasets[split]

    @property
    def datasets(self) -> dict[str, _TextDataset]:
        return dict(self._datasets)


def _args(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        dist="single",
        resume="",
        output=str(root / "output"),
        root=str(root),
        assets=str(root / "assets.jsonl"),
        views=str(root / "views.jsonl"),
        train_samples=str(root / "train.jsonl"),
        sft_validation_samples=str(root / "sft-validation.jsonl"),
        kl_selection_samples=str(root / "kl-selection.jsonl"),
        formal_monitor_samples=str(root / "formal-monitor.jsonl"),
        preprocess_contract=str(root / "preprocess.json"),
        tokenizer=str(root / "tokenizer"),
        text_replay=str(root / "text.jsonl"),
        reviewed_exclusions=str(root / "exclude.json"),
        cycles=2,
        global_batch_size=20,
        text_batch_size=3,
        save_every=1,
        eval_every=1,
        early_stop_patience=2,
        warmup_cycles=0,
        seed=42,
        tower_lr=1e-5,
        bridge_lr=2e-5,
        projector_lr=3e-5,
        lm_lr=4e-6,
        grad_clip=1.0,
        weight_decay=0.01,
        precision="bf16",
    )


class TrainOCRAnyresJointSFTCLITest(unittest.TestCase):
    def test_validator_preflight_binds_all_four_image_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = _args(Path(tmp))
            validator_args = joint_cli._validator_args(args)
            self.assertEqual(validator_args.train_samples, args.train_samples)
            self.assertEqual(
                validator_args.sft_validation_samples,
                args.sft_validation_samples,
            )
            self.assertEqual(
                validator_args.kl_selection_samples,
                args.kl_selection_samples,
            )
            self.assertEqual(
                validator_args.formal_monitor_samples,
                args.formal_monitor_samples,
            )

    def test_scope_and_output_guards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = _args(Path(tmp))
            joint_cli._validate_args(args)
            for field, value, pattern in (
                ("dist", "ddp", "single"),
                ("resume", "step", "resume"),
                ("global_batch_size", 21, "divisible by 20"),
            ):
                changed = SimpleNamespace(**vars(args))
                setattr(changed, field, value)
                with self.subTest(field=field), self.assertRaisesRegex(
                    ValueError, pattern
                ):
                    joint_cli._validate_args(changed)

    def test_four_ocr_batches_and_text_cursor_commit_are_transactional(self) -> None:
        dataset = _OCRDataset()
        sampler = OCRQuotaSampler(
            dict(zip(dataset.sample_ids, dataset.quota_buckets, strict=True)),
            global_batch_size=20,
            seed=9,
        )
        batches, staged_state, ids = joint_cli._prepare_ocr_microbatches(
            sampler,
            dataset,
            lambda rows: tuple(rows),
        )
        self.assertEqual(len(batches), 4)
        self.assertTrue(all(len(batch) == 20 for batch in batches))
        self.assertEqual(len(ids), 4)
        self.assertEqual(sampler.draw_counter, 0)
        self.assertEqual(staged_state["draw_counter"], 80)
        sampler.load_state_dict(staged_state)
        self.assertEqual(sampler.draw_counter, 80)

        text = _TextDataset()
        batch, next_cursor, indices = joint_cli._text_batch_at_cursor(
            text,
            lambda rows: tuple(row["index"] for row in rows),
            cursor=1,
            batch_size=3,
        )
        self.assertEqual(batch, (3, 4, 0))
        self.assertEqual(indices, [3, 4, 0])
        self.assertEqual(next_cursor, 2)

    def test_metadata_binds_sampler_text_cursor_runtime_and_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = _args(Path(tmp))
            ocr = _OCRDataset()
            text_partition = _TextPartition()
            text_train = text_partition.dataset("train")
            text_validation = text_partition.dataset("sft_validation")
            sampler = OCRQuotaSampler(
                dict(zip(ocr.sample_ids, ocr.quota_buckets, strict=True)),
                global_batch_size=20,
                seed=9,
            )
            prepared = SimpleNamespace(
                metadata_template={"stage": "joint", "parent_anyres_checkpoint": {}},
            )
            metadata = joint_cli._metadata(
                prepared,
                cycle=7,
                args=args,
                train_dataset=ocr,
                val_dataset=ocr,
                text_train_dataset=text_train,
                text_validation_dataset=text_validation,
                text_partition=text_partition,
                sampler=sampler,
                text_cursor=5,
                dataset_report={
                    "datasets": {
                        "train": {
                            "contract": ocr.dataset_contract,
                            "contract_sha256": "a" * 64,
                        },
                        "sft_validation": {
                            "contract": ocr.dataset_contract,
                            "contract_sha256": "a" * 64,
                        },
                        "kl_selection": {
                            "contract": {
                                "kind": "kl_selection",
                                "contract_sha256": "f" * 64,
                            },
                            "contract_sha256": "f" * 64,
                        },
                        "formal_monitor": {
                            "contract": {
                                "kind": "formal_monitor",
                                "contract_sha256": "9" * 64,
                            },
                            "contract_sha256": "9" * 64,
                        },
                    },
                    "text_replay": {
                        "partition_contract": text_partition.partition_contract,
                        "partition_contract_sha256": (
                            text_partition.partition_contract["contract_sha256"]
                        ),
                        "splits": {
                            split: {
                                "contract": text_partition.dataset(
                                    split
                                ).dataset_contract,
                                "contract_sha256": text_partition.dataset(
                                    split
                                ).contract_sha256,
                            }
                            for split in (
                                "train",
                                "sft_validation",
                                "kl_selection",
                                "formal_monitor",
                            )
                        }
                    },
                },
                historical_baseline={"historical": True},
                baseline={"baseline": True},
                last_validation={"eligibility": {"eligible": True}},
                optimizer_contract={"canonical_sha256": "d" * 64},
                runtime_source_receipt={"canonical_sha256": "e" * 64},
                runtime_environment={"device": "test"},
                final=False,
            )
            self.assertEqual(metadata["training_stage"], "joint")
            self.assertEqual(
                metadata["training_config"]["cadence"],
                "4_ocr_microbatches_then_1_text_microbatch",
            )
            self.assertEqual(
                metadata["training_config"]["objective"],
                "mean_ocr_plus_0.2_text_ce",
            )
            self.assertEqual(
                metadata["historical_visual_best_validation"],
                {"historical": True},
            )
            self.assertEqual(
                metadata["joint_runtime_baseline"],
                {"baseline": True},
            )
            self.assertEqual(metadata["training_config"]["ocr_batches_per_cycle"], 4)
            self.assertEqual(metadata["training_config"]["text_weight"], 0.2)
            self.assertEqual(metadata["text_replay_cursor"]["cursor"], 5)
            self.assertEqual(
                metadata["quota_sampler_state"]["draw_counter"],
                sampler.draw_counter,
            )
            self.assertIn("runtime_source_receipt", metadata)
            self.assertIn("optimizer_contract", metadata)
            self.assertEqual(
                metadata["sft_validation_dataset_contract"],
                ocr.dataset_contract,
            )
            self.assertEqual(
                metadata["kl_selection_dataset_contract"],
                {
                    "kind": "kl_selection",
                    "contract_sha256": "f" * 64,
                },
            )
            self.assertEqual(
                metadata["formal_monitor_dataset_contract"],
                {
                    "kind": "formal_monitor",
                    "contract_sha256": "9" * 64,
                },
            )
            self.assertEqual(
                metadata["text_replay_validation_contract"],
                metadata["text_replay_sft_validation_contract"],
            )
            self.assertEqual(
                {
                    metadata["text_replay_train_contract"]["contract_sha256"],
                    metadata["text_replay_sft_validation_contract"][
                        "contract_sha256"
                    ],
                    metadata["text_replay_kl_selection_contract"][
                        "contract_sha256"
                    ],
                    metadata["text_replay_formal_monitor_contract"][
                        "contract_sha256"
                    ],
                },
                {character * 64 for character in "bcde"},
            )

    def test_result_identity_and_preflight_failure_do_not_create_output(self) -> None:
        identity = {
            "path": "/tmp/best",
            "model_sha256": "a" * 64,
            "metadata_sha256": "b" * 64,
        }
        self.assertEqual(
            joint_cli._best_identity(
                {
                    "promotion_allowed": True,
                    "best_eligible_checkpoint": identity,
                }
            ),
            identity,
        )
        with self.assertRaisesRegex(ValueError, "not promotion eligible"):
            joint_cli._best_identity(
                {
                    "promotion_allowed": False,
                    "best_eligible_checkpoint": identity,
                }
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output"
            argv = [
                "--visual-stage-result", str(root / "result.json"),
                "--tokenizer", str(root / "tokenizer"),
                "--root", str(root),
                "--assets", str(root / "assets.jsonl"),
                "--views", str(root / "views.jsonl"),
                "--train-samples", str(root / "train.jsonl"),
                "--sft-validation-samples", str(root / "sft-validation.jsonl"),
                "--kl-selection-samples", str(root / "kl-selection.jsonl"),
                "--formal-monitor-samples", str(root / "formal-monitor.jsonl"),
                "--preprocess-contract", str(root / "preprocess.json"),
                "--text-replay", str(root / "text.jsonl"),
                "--reviewed-exclusions", str(root / "exclusions.json"),
                "--output", str(output),
                "--cycles", "2",
                "--global-batch-size", "20",
                "--tower-lr", "1e-5",
                "--bridge-lr", "2e-5",
                "--projector-lr", "3e-5",
                "--lm-lr", "4e-6",
            ]
            with patch.object(
                joint_cli,
                "build_dataset_report",
                side_effect=ValueError("preflight rejected"),
            ):
                with self.assertRaisesRegex(ValueError, "preflight rejected"):
                    joint_cli.main(argv)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
