# -*- coding: utf-8 -*-

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import scripts.validate_ocr_anyres_dataset as validator


_BUCKETS = (
    "print",
    "handwritten_good",
    "handwritten_medium",
    "handwritten_poor",
)


class _Encoder:
    mode = "native"

    def __init__(self) -> None:
        self.stats = {"native": 0, "byte_fallback": 0, "canonicalized": 0}

    def __call__(self, text: str) -> list[int]:
        self.stats["native"] += 1
        return [300 + index for index, _ in enumerate(text)]


class _Dataset:
    def __init__(self, split: str, samples: list[dict]) -> None:
        selected = [sample for sample in samples if sample["split"] == split]
        self.sample_ids = tuple(sample["sample_id"] for sample in selected)
        self.quota_buckets = tuple(sample["quota_bucket"] for sample in selected)
        self.quota_counts = {
            bucket: self.quota_buckets.count(bucket) for bucket in _BUCKETS
        }
        self.dataset_contract = {
            "kind": "fake-anyres-selection",
            "split": split,
            "sample_ids": list(self.sample_ids),
        }
        self.contract_sha256 = {
            "train": "1" * 64,
            "sft_validation": "2" * 64,
            "kl_selection": "6" * 64,
            "formal_monitor": "7" * 64,
        }[split]


class _TextDataset:
    def __init__(self, split: str) -> None:
        self.dataset_contract = {
            "kind": "fake-text-replay",
            "split": split,
            "contract_sha256": {
                "train": "3" * 64,
                "sft_validation": "4" * 64,
                "kl_selection": "5" * 64,
                "formal_monitor": "6" * 64,
            }[split],
        }
        self.contract_sha256 = self.dataset_contract["contract_sha256"]
        self.document_sha256 = {f"doc-{split}": "7" * 64}
        self.text_sha256 = {f"text-{split}": "8" * 64}
        self.token_stats = {"samples": 1, "total_content_tokens": 3}


class _TextPartition:
    def __init__(self, path, *_args, **_kwargs) -> None:
        self._datasets = {
            split: _TextDataset(split)
            for split in (
                "train",
                "sft_validation",
                "kl_selection",
                "formal_monitor",
            )
        }
        self.contract_sha256 = "9" * 64
        self.partition_contract = {
            "kind": "dol_text_ce_replay_partition",
            "manifest_file_sha256": validator._file_sha256(path),
            "contract_sha256": self.contract_sha256,
        }

    def dataset(self, split: str) -> _TextDataset:
        return self._datasets[split]


def _samples() -> dict[str, list[dict]]:
    result = {
        "train": [],
        "sft_validation": [],
        "kl_selection": [],
        "formal_monitor": [],
    }
    for index, bucket in enumerate(_BUCKETS):
        result["train"].append(
            {
                "sample_id": f"train-{index}",
                "split": "train",
                "quota_bucket": bucket,
                "reference_model": {"text": "abc"},
                "reference_token_count": 3,
            }
        )
        for split in ("sft_validation", "kl_selection", "formal_monitor"):
            for copy_index in range(2):
                result[split].append(
                    {
                        "sample_id": f"{split}-{index}-{copy_index}",
                        "split": split,
                        "quota_bucket": bucket,
                        "reference_model": {"text": "abc"},
                        "reference_token_count": 3,
                    }
                )
    return result


def _preprocess(output_budget: int = 4) -> dict:
    return {
        "contract_canonical_sha256": "a" * 64,
        "planner_contract": {"name": "planner", "coverage_proof_method": "proof"},
        "budgets": {
            "decode": {"max_pixels_per_asset": 40_000_000},
            "window": {"max_windows_per_asset": 32},
            "output": {"recommended_max_new_tokens": output_budget},
            "context": {"max_sequence_tokens": 512},
        },
        "implementation_sources": {"native_planner": {"sha256": "b" * 64}},
    }


class AnyresValidatorCLITest(unittest.TestCase):
    def _run_patches(
        self,
        paths: dict[str, Path],
        *,
        output_budget: int = 4,
        duplicate_contracts: bool = False,
    ):
        samples = _samples()
        encoder = _Encoder()
        lookup = {
            str(paths["assets"]): [{"asset_id": "asset"}],
            str(paths["views"]): [{"view_id": "view"}],
            **{
                str(paths[split]): samples[split]
                for split in samples
            },
        }
        normalized = {
            "assets": [],
            "views": [
                {
                    "qa": {
                        "review_status": "accepted",
                        "coverage": 1.0,
                        "cc_crossing_count": 0,
                        "cc_uncovered_count": 0,
                    }
                }
            ],
            "samples": [
                row for split_rows in samples.values() for row in split_rows
            ],
        }
        quarantine = [
            {"entity_type": "view", "entity_id": "bad-view", "reasons": ["cut"]}
        ]

        def dataset_factory(**kwargs):
            dataset = _Dataset(
                kwargs["split"],
                [row for split_rows in samples.values() for row in split_rows],
            )
            if duplicate_contracts and kwargs["split"] == "formal_monitor":
                dataset.contract_sha256 = "6" * 64
            return dataset

        return (
            mock.patch.object(
                validator,
                "validate_anyres_preprocess_contract",
                return_value=_preprocess(output_budget),
            ),
            mock.patch.object(
                validator.TokenizerBundle,
                "from_dir",
                return_value=SimpleNamespace(
                    tokenizer=object(),
                    validate=lambda: [],
                ),
            ),
            mock.patch.object(
                validator,
                "make_ocr_target_encoder",
                return_value=encoder,
            ),
            mock.patch.object(
                validator,
                "load_anyres_jsonl",
                side_effect=lambda path: lookup[str(path)],
            ),
            mock.patch.object(
                validator,
                "validate_anyres_assets_views_samples",
                return_value=(normalized, quarantine),
            ),
            mock.patch.object(
                validator,
                "_resource_budget_summary",
                return_value={
                    "planner_contract_verified": True,
                    "assets_checked": 12,
                    "views_checked": 12,
                    "max_observed_views_per_asset": 1,
                    "max_observed_raw_tokens_per_view": 4,
                    "max_observed_raw_tokens_per_asset": 4,
                },
            ),
            mock.patch.object(
                validator,
                "validate_anyres_ready_dataset",
                return_value={
                    "kind": "dol_ocr_anyres_ready_admission_v1",
                    "canonical_sha256": "9" * 64,
                },
            ),
            mock.patch.object(
                validator,
                "AnyresOCRDataset",
                side_effect=dataset_factory,
            ),
            mock.patch.object(
                validator,
                "tokenizer_manifest_canonical_sha256",
                return_value="c" * 64,
            ),
            mock.patch.object(
                validator,
                "tokenizer_vocab_sha256",
                return_value="d" * 64,
            ),
            mock.patch.object(validator, "TextReplayPartition", _TextPartition),
        )

    @staticmethod
    def _arguments(paths: dict[str, Path], output: Path) -> list[str]:
        return [
            "--root",
            str(paths["root"]),
            "--assets",
            str(paths["assets"]),
            "--views",
            str(paths["views"]),
            "--train-samples",
            str(paths["train"]),
            "--sft-validation-samples",
            str(paths["sft_validation"]),
            "--kl-selection-samples",
            str(paths["kl_selection"]),
            "--formal-monitor-samples",
            str(paths["formal_monitor"]),
            "--preprocess-contract",
            str(paths["preprocess"]),
            "--tokenizer",
            str(paths["tokenizer"]),
            "--text-replay",
            str(paths["text"]),
            "--reviewed-exclusions",
            str(paths["exclusions"]),
            "--out",
            str(output),
        ]

    def test_success_prints_one_json_and_atomically_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {
                name: root / name
                for name in (
                    "assets",
                    "views",
                    "train",
                    "sft_validation",
                    "kl_selection",
                    "formal_monitor",
                    "preprocess",
                    "tokenizer",
                    "text",
                    "exclusions",
                )
            }
            paths["root"] = root
            for name in (
                "assets",
                "views",
                "train",
                "sft_validation",
                "kl_selection",
                "formal_monitor",
                "text",
            ):
                paths[name].write_text("fixture\n", encoding="utf-8")
            paths["tokenizer"].mkdir()
            paths["preprocess"].write_text("{}\n", encoding="utf-8")
            paths["exclusions"].write_text(
                json.dumps(
                    {"schema_version": 1, "document_ids": [], "ngram_sha256": []}
                ),
                encoding="utf-8",
            )
            output = root / "report.json"
            stdout = io.StringIO()
            patches = self._run_patches(paths)
            with contextlib.ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                with contextlib.redirect_stdout(stdout):
                    self.assertEqual(
                        validator.main(self._arguments(paths, output)),
                        0,
                    )

            rendered = stdout.getvalue()
            report = json.loads(rendered)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)
            self.assertEqual(rendered.count("\n"), 1)
            self.assertTrue(report["read_only"])
            self.assertFalse(report["model_loaded"])
            self.assertFalse(report["gpu_used"])
            self.assertFalse(report["golden_opened"])
            self.assertTrue(
                report["manifests"]["global_cross_split_leakage_validated"]
            )
            self.assertEqual(report["datasets"]["train"]["quota_counts"]["print"], 1)
            self.assertEqual(
                report["datasets"]["sft_validation"]["quota_counts"]["print"],
                2,
            )
            self.assertEqual(
                report["datasets"]["kl_selection"]["quota_counts"]["print"],
                2,
            )
            self.assertEqual(
                report["datasets"]["formal_monitor"]["quota_counts"]["print"],
                2,
            )
            self.assertEqual(
                set(report["datasets"]),
                {
                    "train",
                    "sft_validation",
                    "kl_selection",
                    "formal_monitor",
                },
            )
            self.assertEqual(report["recommended_max_new_tokens"], 4)
            self.assertEqual(report["preprocess_output_budget"], 4)
            self.assertEqual(report["quarantine"]["count"], 1)
            self.assertEqual(
                report["text_replay"]["partition_contract_sha256"],
                "9" * 64,
            )
            self.assertEqual(
                set(report["text_replay"]["splits"]),
                {
                    "train",
                    "sft_validation",
                    "kl_selection",
                    "formal_monitor",
                },
            )
            self.assertTrue(
                report["text_replay"]["global_cross_split_leakage_validated"]
            )

    def test_failure_does_not_create_out_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {name: root / name for name in (
                "assets", "views", "train", "sft_validation",
                "kl_selection", "formal_monitor", "preprocess", "tokenizer",
                "text", "exclusions",
            )}
            paths["root"] = root
            for name in (
                "assets", "views", "train", "sft_validation",
                "kl_selection", "formal_monitor", "text",
            ):
                paths[name].write_text("fixture\n", encoding="utf-8")
            paths["tokenizer"].mkdir()
            paths["preprocess"].write_text("{}\n", encoding="utf-8")
            paths["exclusions"].write_text(
                '{"schema_version":1,"document_ids":[],"ngram_sha256":[]}',
                encoding="utf-8",
            )
            output = root / "must-not-exist.json"
            patches = self._run_patches(paths, output_budget=5)
            with contextlib.ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                with self.assertRaisesRegex(ValueError, "must equal"):
                    validator.main(self._arguments(paths, output))
            self.assertFalse(output.exists())

    def test_duplicate_public_split_contracts_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {
                name: root / name
                for name in (
                    "assets", "views", "train", "sft_validation",
                    "kl_selection", "formal_monitor", "preprocess",
                    "tokenizer", "text", "exclusions",
                )
            }
            paths["root"] = root
            for name in (
                "assets", "views", "train", "sft_validation",
                "kl_selection", "formal_monitor", "text",
            ):
                paths[name].write_text("fixture\n", encoding="utf-8")
            paths["tokenizer"].mkdir()
            paths["preprocess"].write_text("{}\n", encoding="utf-8")
            paths["exclusions"].write_text(
                '{"schema_version":1,"document_ids":[],"ngram_sha256":[]}',
                encoding="utf-8",
            )
            output = root / "must-not-exist.json"
            patches = self._run_patches(paths, duplicate_contracts=True)
            with contextlib.ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                with self.assertRaisesRegex(ValueError, "distinct"):
                    validator.main(self._arguments(paths, output))
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
