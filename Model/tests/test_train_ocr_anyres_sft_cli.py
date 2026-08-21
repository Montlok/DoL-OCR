# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import train_ocr_anyres_sft


def _namespace(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        stage="visual",
        dist="single",
        resume="",
        source_checkpoint=str(root / "source"),
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
        warmup_cycles=0,
        save_every=1,
        eval_every=1,
        early_stop_patience=2,
        keep_last=1,
        seed=42,
        precision="bf16",
        tower_lr=1e-5,
        bridge_lr=3e-5,
        projector_lr=1e-5,
        lm_lr=3e-7,
        grad_clip=1.0,
        weight_decay=0.01,
    )


class _Dataset:
    def __init__(self) -> None:
        buckets = {
            "print": 4,
            "handwritten_good": 4,
            "handwritten_medium": 4,
            "handwritten_poor": 4,
        }
        self.sample_ids = tuple(
            f"{bucket}-{index}"
            for bucket, count in buckets.items()
            for index in range(count)
        )
        self.quota_buckets = tuple(
            bucket for bucket, count in buckets.items() for _ in range(count)
        )

    def get_by_sample_id(self, sample_id: str):
        return sample_id


class _ContractDataset:
    def __init__(self, kind: str) -> None:
        digest = {
            "train": "1" * 64,
            "sft_validation": "2" * 64,
            "text_validation": "3" * 64,
        }[kind]
        self.dataset_contract = {
            "kind": kind,
            "contract_sha256": digest,
        }


class _TextPartition:
    def __init__(self) -> None:
        self._datasets = {
            split: SimpleNamespace(
                dataset_contract={
                    "kind": f"text_{split}",
                    "contract_sha256": digest * 64,
                }
            )
            for split, digest in zip(
                (
                    "train",
                    "sft_validation",
                    "kl_selection",
                    "formal_monitor",
                ),
                "6789",
                strict=True,
            )
        }
        self.partition_contract = {
            "kind": "dol_text_ce_replay_partition",
            "contract_sha256": "a" * 64,
        }

    def dataset(self, split: str):
        return self._datasets[split]


class _Sampler:
    @staticmethod
    def state_dict():
        return {"draw_counter": 0}


class TrainOCRAnyresSFTCLITest(unittest.TestCase):
    def test_all_four_manifest_hashes_are_rechecked_after_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = _namespace(root)
            paths = {
                "assets_sha256": Path(args.assets),
                "views_sha256": Path(args.views),
                "train_samples_sha256": Path(args.train_samples),
                "sft_validation_samples_sha256": Path(
                    args.sft_validation_samples
                ),
                "kl_selection_samples_sha256": Path(args.kl_selection_samples),
                "formal_monitor_samples_sha256": Path(args.formal_monitor_samples),
            }
            for index, path in enumerate(paths.values()):
                path.write_bytes(f"manifest-{index}\n".encode())
            Path(args.text_replay).write_text("text manifest\n", encoding="utf-8")
            report = {
                "manifests": {
                    field: hashlib.sha256(path.read_bytes()).hexdigest()
                    for field, path in paths.items()
                },
                "text_replay": {
                    "manifest_sha256": hashlib.sha256(
                        Path(args.text_replay).read_bytes()
                    ).hexdigest()
                },
            }
            train_ocr_anyres_sft._require_image_manifests_unchanged(
                args,
                report,
            )
            Path(args.formal_monitor_samples).write_text(
                "changed\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "changed after validation"):
                train_ocr_anyres_sft._require_image_manifests_unchanged(
                    args,
                    report,
                )

    def test_validator_preflight_binds_all_four_image_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = _namespace(Path(temporary))
            validator_args = train_ocr_anyres_sft._validator_args(args)
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

    def test_visual_metadata_records_distinct_public_split_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = _namespace(Path(temporary))
            train = _ContractDataset("train")
            sft_validation = _ContractDataset("sft_validation")
            text_partition = _TextPartition()
            split_contracts = {
                "train": train.dataset_contract,
                "sft_validation": sft_validation.dataset_contract,
                "kl_selection": {
                    "kind": "kl_selection",
                    "contract_sha256": "4" * 64,
                },
                "formal_monitor": {
                    "kind": "formal_monitor",
                    "contract_sha256": "5" * 64,
                },
            }
            report = {
                "datasets": {
                    split: {
                        "contract": contract,
                        "contract_sha256": contract["contract_sha256"],
                    }
                    for split, contract in split_contracts.items()
                },
                "text_replay": {
                    "partition_contract": text_partition.partition_contract,
                    "partition_contract_sha256": text_partition.partition_contract[
                        "contract_sha256"
                    ],
                    "splits": {
                        split: {
                            "contract": text_partition.dataset(
                                split
                            ).dataset_contract,
                            "contract_sha256": text_partition.dataset(
                                split
                            ).dataset_contract["contract_sha256"],
                        }
                        for split in (
                            "train",
                            "sft_validation",
                            "kl_selection",
                            "formal_monitor",
                        )
                    }
                },
            }
            metadata = train_ocr_anyres_sft._metadata(
                SimpleNamespace(metadata_template={"stage": "visual"}),
                cycle=1,
                args=args,
                train_dataset=train,
                val_dataset=sft_validation,
                text_partition=text_partition,
                sampler=_Sampler(),
                dataset_report=report,
                baseline_report={"baseline": True},
                last_eval={"current": True},
                final=False,
                language_parameter_sha256="a" * 64,
                legacy_omvt_parameter_sha256="b" * 64,
                runtime_source_receipt={"canonical_sha256": "c" * 64},
                runtime_environment={"device": "test"},
            )
            self.assertEqual(
                metadata["validation_dataset_contract"],
                sft_validation.dataset_contract,
            )
            self.assertEqual(
                metadata["sft_validation_dataset_contract"],
                sft_validation.dataset_contract,
            )
            self.assertEqual(
                metadata["kl_selection_dataset_contract"],
                split_contracts["kl_selection"],
            )
            self.assertEqual(
                metadata["formal_monitor_dataset_contract"],
                split_contracts["formal_monitor"],
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
                {character * 64 for character in "6789"},
            )

    def test_scope_and_resume_guards_fail_before_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = _namespace(Path(temporary))
            train_ocr_anyres_sft._validate_args(args)
            for field, value, expected in (
                ("stage", "joint", "stage=visual"),
                ("dist", "fsdp", "distributed"),
                ("resume", "checkpoint", "resume is disabled"),
            ):
                changed = SimpleNamespace(**vars(args))
                setattr(changed, field, value)
                with self.subTest(field=field), self.assertRaisesRegex(
                    ValueError, expected
                ):
                    train_ocr_anyres_sft._validate_args(changed)

    def test_validation_batches_are_same_bucket_pairs(self) -> None:
        dataset = _Dataset()
        batches = train_ocr_anyres_sft._validation_batches(
            dataset,
            lambda rows: tuple(rows),
        )
        self.assertEqual(len(batches), 8)
        for batch in batches:
            counts = [
                sum(value.startswith(bucket) for value in batch)
                for bucket in (
                    "print",
                    "handwritten_good",
                    "handwritten_medium",
                    "handwritten_poor",
                )
            ]
            self.assertEqual(sorted(counts), [0, 0, 0, 2])

        dataset = _Dataset()
        dataset.sample_ids = tuple(
            [*dataset.sample_ids, *(f"{bucket}-extra" for bucket in (
                "print",
                "handwritten_good",
                "handwritten_medium",
                "handwritten_poor",
            ))]
        )
        dataset.quota_buckets = tuple(
            [*dataset.quota_buckets, *(
                "print",
                "handwritten_good",
                "handwritten_medium",
                "handwritten_poor",
            )]
        )
        odd_batches = train_ocr_anyres_sft._validation_batches(
            dataset,
            lambda rows: tuple(rows),
        )
        self.assertEqual(sorted(len(batch) for batch in odd_batches), [2] * 4 + [3] * 4)

    def test_dataset_preflight_failure_does_not_create_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            argv = [
                "--source-checkpoint", str(root / "source"),
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
                "--reviewed-exclusions", str(root / "exclude.json"),
                "--output", str(output),
                "--cycles", "2",
                "--global-batch-size", "20",
                "--tower-lr", "1e-5",
                "--bridge-lr", "3e-5",
                "--projector-lr", "1e-5",
                "--lm-lr", "3e-7",
            ]
            with patch.object(
                train_ocr_anyres_sft,
                "build_dataset_report",
                side_effect=ValueError("dataset rejected"),
            ):
                with self.assertRaisesRegex(ValueError, "dataset rejected"):
                    train_ocr_anyres_sft.main(argv)
            self.assertFalse(output.exists())

    def test_material_improvement_is_strict_at_zero_and_relative_elsewhere(self):
        helper = train_ocr_anyres_sft._strict_relative_improvement
        self.assertTrue(helper(0.994, 1.0))
        self.assertFalse(helper(0.995, 1.0))
        self.assertFalse(helper(0.0, 0.0))
        self.assertFalse(helper(0.9999999999995, 1.0))

    def test_production_source_receipt_binds_full_runtime_closure(self):
        receipt = train_ocr_anyres_sft._runtime_source_receipt()
        validated = train_ocr_anyres_sft._validate_runtime_source_receipt(
            receipt
        )
        paths = {row["path"] for row in validated["files"]}
        self.assertIn("Model/omvt/native_mixers.py", paths)
        self.assertIn("Model/omvt/injector.py", paths)
        self.assertIn("Tokenizer/multimodal/image_io.py", paths)
        self.assertIn("Model/config.py", paths)
        self.assertFalse(any("/tests/" in f"/{path}" for path in paths))
        tampered = {**receipt, "canonical_sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "canonical SHA256"):
            train_ocr_anyres_sft._validate_runtime_source_receipt(tampered)
        with patch.object(
            train_ocr_anyres_sft,
            "_runtime_source_receipt",
            return_value=tampered,
        ), self.assertRaisesRegex(RuntimeError, "changed during training"):
            train_ocr_anyres_sft._require_runtime_source_unchanged(receipt)


if __name__ == "__main__":
    unittest.main()
