# -*- coding: utf-8 -*-

"""Final-selection tests for formal anyres GRPO checkpoints."""

from __future__ import annotations

import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from Model.ocr.tokenization import canonical_json_sha256
from Model.posttrain.ocr_anyres_grpo_run import (
    build_anyres_grpo_progress,
    build_text_schedule_cursor_state,
    validate_anyres_grpo_run_contract,
)
from Model.posttrain.ocr_joint_eval import (
    DEPLOYMENT_BUCKET_WEIGHTS,
    joint_eval_eligibility,
)
from Model.posttrain.ocr_selection import (
    OCR_ANYRES_GRPO_SELECTION_RECEIPT_FILENAME,
    OCR_ANYRES_GRPO_SELECTION_RECEIPT_KIND,
    build_anyres_grpo_selection_receipt,
    validate_anyres_grpo_selection_receipt,
    write_anyres_grpo_selection_receipt,
)
from Model.tests.test_ocr_anyres_grpo_run import _run


class AnyresGRPOSelectionReceiptTest(unittest.TestCase):
    _FORMAL_IMAGE_CONTRACT = "f" * 64
    _FORMAL_TEXT_CONTRACT = "3" * 64
    _FORMAL_SAMPLE_IDS = tuple(
        sorted(f"{bucket}-0" for bucket in DEPLOYMENT_BUCKET_WEIGHTS)
    )
    _FORMAL_TEXT_TARGETS = 8

    @staticmethod
    def _runtime_baseline() -> dict:
        baseline = {
            "image_dataset_contract_sha256": (
                AnyresGRPOSelectionReceiptTest._FORMAL_IMAGE_CONTRACT
            ),
            "real": {
                "overall": {
                    "eos_rate": 1.0,
                    "invalid_count": 0,
                    "hit_cap_rate": 0.0,
                },
                "buckets": {
                    bucket: {"raw_grapheme_cer": 0.2}
                    for bucket in DEPLOYMENT_BUCKET_WEIGHTS
                }
            },
            "grounding": {
                "blank_cer_gap": 0.1,
                "shuffled_cer_gap": 0.1,
                "blank_first_token_nll_gap": 0.1,
                "shuffled_first_token_nll_gap": 0.1,
                "paired_bootstrap_95ci": {
                    name: {"low": 0.05}
                    for name in (
                        "blank_cer_gap",
                        "shuffled_cer_gap",
                        "blank_first_token_nll_gap",
                        "shuffled_first_token_nll_gap",
                    )
                },
            },
            "deployment_weighted_cer": 0.2,
            "text_replay": {
                "dataset_contract_sha256": (
                    AnyresGRPOSelectionReceiptTest._FORMAL_TEXT_CONTRACT
                ),
                "token_nll": 1.0,
                "target_tokens": (
                    AnyresGRPOSelectionReceiptTest._FORMAL_TEXT_TARGETS
                ),
            },
            "selection_records": [
                {"sample_id": sample_id}
                for sample_id in (
                    AnyresGRPOSelectionReceiptTest._FORMAL_SAMPLE_IDS
                )
            ],
        }
        baseline["eligibility"] = joint_eval_eligibility(
            baseline,
            baseline_bucket_cer={
                bucket: 0.2 for bucket in DEPLOYMENT_BUCKET_WEIGHTS
            },
            baseline_text_token_nll=1.0,
            min_relative_cer_improvement=0.0,
        )
        return baseline

    @staticmethod
    def _formal_monitor_report(baseline: dict) -> dict:
        report = {
            "image_dataset_contract_sha256": (
                AnyresGRPOSelectionReceiptTest._FORMAL_IMAGE_CONTRACT
            ),
            "real": {
                "overall": {
                    "eos_rate": 1.0,
                    "invalid_count": 0,
                    "hit_cap_rate": 0.0,
                },
                "buckets": {
                    bucket: {"raw_grapheme_cer": 0.1}
                    for bucket in DEPLOYMENT_BUCKET_WEIGHTS
                },
            },
            "grounding": {
                "blank_cer_gap": 0.1,
                "shuffled_cer_gap": 0.1,
                "blank_first_token_nll_gap": 0.1,
                "shuffled_first_token_nll_gap": 0.1,
                "paired_bootstrap_95ci": {
                    name: {"low": 0.05}
                    for name in (
                        "blank_cer_gap",
                        "shuffled_cer_gap",
                        "blank_first_token_nll_gap",
                        "shuffled_first_token_nll_gap",
                    )
                },
            },
            "deployment_weighted_cer": 0.1,
            "text_replay": {
                "dataset_contract_sha256": (
                    AnyresGRPOSelectionReceiptTest._FORMAL_TEXT_CONTRACT
                ),
                "token_nll": 1.0,
                "target_tokens": (
                    AnyresGRPOSelectionReceiptTest._FORMAL_TEXT_TARGETS
                ),
            },
            "selection_records": [
                {"sample_id": sample_id}
                for sample_id in (
                    AnyresGRPOSelectionReceiptTest._FORMAL_SAMPLE_IDS
                )
            ],
        }
        baseline_buckets = {
            bucket: float(
                baseline["real"]["buckets"][bucket]["raw_grapheme_cer"]
            )
            for bucket in DEPLOYMENT_BUCKET_WEIGHTS
        }
        report["eligibility"] = joint_eval_eligibility(
            report,
            baseline_bucket_cer=baseline_buckets,
            baseline_text_token_nll=1.0,
            min_relative_cer_improvement=0.0,
        )
        return report

    @staticmethod
    def _rebind_formal_lineage(
        run: dict,
        *,
        dataset_admission_report: dict,
        runtime_source_receipt: dict,
    ) -> dict:
        result = copy.deepcopy(run)
        dataset_sha = canonical_json_sha256(dataset_admission_report)
        source_sha = runtime_source_receipt["canonical_sha256"]
        result["dataset_ready_admission_sha256"] = dataset_sha
        result["source_closure_sha256"] = source_sha
        result["admission"]["dataset_admission_report_sha256"] = dataset_sha
        result["admission_sha256"] = canonical_json_sha256(result["admission"])
        selection = result["formal_selection_receipt"]
        selection["admission_sha256"] = result["admission_sha256"]
        selection["dataset_ready_admission_sha256"] = dataset_sha
        selection["source_closure_sha256"] = source_sha
        selection["canonical_sha256"] = canonical_json_sha256(
            {
                key: value
                for key, value in selection.items()
                if key != "canonical_sha256"
            }
        )
        result["formal_selection_receipt_sha256"] = selection[
            "canonical_sha256"
        ]
        result["canonical_sha256"] = canonical_json_sha256(
            {
                key: value
                for key, value in result.items()
                if key != "canonical_sha256"
            }
        )
        return validate_anyres_grpo_run_contract(result)

    @staticmethod
    def _progress(
        run: dict,
        *,
        optimizer_steps: int,
        selected: Path,
        validation_sha256: str,
        terminal: bool,
    ) -> dict:
        return build_anyres_grpo_progress(
            run,
            optimizer_steps=optimizer_steps,
            rollout_attempts=optimizer_steps,
            no_update_count=0,
            consecutive_no_update=0,
            optimizer_skip_count=0,
            consecutive_optimizer_skips=0,
            sampler_state={
                "pending_global_batch": None,
                "draw_counter": optimizer_steps
                * int(run["schedule"]["global_batch_size"]),
            },
            text_cursor_state=build_text_schedule_cursor_state(
                dataset_contract_sha256=run[
                    "text_replay_train_contract_sha256"
                ],
                batch_size=int(run["schedule"]["text_batch_size"]),
                schedule_cursor=optimizer_steps,
                ce_steps=optimizer_steps,
            ),
            text_ce_steps=optimizer_steps,
            eval_count=optimizer_steps,
            bad_eval_count=0,
            best_validation_eligible=True,
            best_validation_step=1,
            best_checkpoint=str(selected),
            best_validation_sha256=validation_sha256,
            last_validation_sha256=validation_sha256,
            stop_reason="validation_plateau" if terminal else "running",
            terminal=terminal,
        )

    @staticmethod
    def _checkpoint(
        path: Path,
        *,
        run: dict,
        progress: dict,
        locked_golden_anchor_sha256: str,
        dataset_admission_report: dict,
        runtime_source_receipt: dict,
        runtime_baseline: dict,
        formal_monitor_best_validation: dict,
    ) -> None:
        path.mkdir(parents=True)
        (path / "model.pt").write_bytes(
            f"model-{progress['optimizer_steps']}".encode("ascii")
        )
        torch.save(
            {
                "step": progress["optimizer_steps"],
                "metadata": {
                    "phase": "grpo",
                    "task": "ocr",
                    "training_stage": "grpo_formal",
                    "final": progress["terminal"],
                    "stop_reason": progress["stop_reason"],
                    "ocr_anyres_grpo_run_contract": copy.deepcopy(run),
                    "ocr_anyres_grpo_run_contract_sha256": run[
                        "canonical_sha256"
                    ],
                    "ocr_anyres_grpo_progress": copy.deepcopy(progress),
                    "ocr_anyres_grpo_progress_sha256": progress[
                        "canonical_sha256"
                    ],
                    "locked_golden_opened": False,
                    "locked_golden_anchor_sha256": (
                        locked_golden_anchor_sha256
                    ),
                    "dataset_admission_report": copy.deepcopy(
                        dataset_admission_report
                    ),
                    "runtime_source_receipt": copy.deepcopy(
                        runtime_source_receipt
                    ),
                    "formal_monitor_runtime_baseline": copy.deepcopy(
                        runtime_baseline
                    ),
                    "formal_monitor_best_validation": copy.deepcopy(
                        formal_monitor_best_validation
                    ),
                },
            },
            path / "meta.pt",
        )
        (path / "COMPLETE").write_text(
            f"step={progress['optimizer_steps']}\n",
            encoding="ascii",
        )
        for name in ("optimizer.pt", "scheduler.pt", "rng.pt"):
            torch.save({}, path / name)

    def _fixture(self, root: Path) -> tuple[dict, dict, Path, Path, str, str]:
        runtime_baseline = self._runtime_baseline()
        formal_monitor_best = self._formal_monitor_report(runtime_baseline)
        image_contract = {
            "contract_sha256": self._FORMAL_IMAGE_CONTRACT,
            "sample_ids": list(self._FORMAL_SAMPLE_IDS),
        }
        text_contract = {
            "contract_sha256": self._FORMAL_TEXT_CONTRACT,
            "token_stats": {
                "samples": 2,
                "total_sequence_tokens": 10,
            },
        }
        dataset_admission = {
            "schema_version": 1,
            "kind": "test_anyres_dataset_admission",
            "datasets": {
                "formal_monitor": {
                    "contract": image_contract,
                    "contract_sha256": self._FORMAL_IMAGE_CONTRACT,
                }
            },
            "text_replay": {
                "splits": {
                    "formal_monitor": {
                        "contract": text_contract,
                        "contract_sha256": self._FORMAL_TEXT_CONTRACT,
                    }
                }
            },
        }
        source_base = {
            "schema_version": 1,
            "kind": "dol_ocr_anyres_production_source_closure_v1",
            "files": [
                {
                    "path": "Model/model.py",
                    "sha256": "c" * 64,
                }
            ],
        }
        runtime_source = {
            **source_base,
            "canonical_sha256": canonical_json_sha256(source_base),
        }
        run = self._rebind_formal_lineage(
            _run(
                1,
                mode="formal",
                baseline_validation=runtime_baseline,
            ),
            dataset_admission_report=dataset_admission,
            runtime_source_receipt=runtime_source,
        )
        root = root.resolve()
        selected = root / "best" / "step_00000001"
        terminal = root / "step_00000002"
        validation_sha = canonical_json_sha256(formal_monitor_best)
        golden_anchor_sha = "b" * 64
        selected_progress = self._progress(
            run,
            optimizer_steps=1,
            selected=selected,
            validation_sha256=validation_sha,
            terminal=False,
        )
        terminal_progress = self._progress(
            run,
            optimizer_steps=2,
            selected=selected,
            validation_sha256=validation_sha,
            terminal=True,
        )
        self._checkpoint(
            selected,
            run=run,
            progress=selected_progress,
            locked_golden_anchor_sha256=golden_anchor_sha,
            dataset_admission_report=dataset_admission,
            runtime_source_receipt=runtime_source,
            runtime_baseline=runtime_baseline,
            formal_monitor_best_validation=formal_monitor_best,
        )
        self._checkpoint(
            terminal,
            run=run,
            progress=terminal_progress,
            locked_golden_anchor_sha256=golden_anchor_sha,
            dataset_admission_report=dataset_admission,
            runtime_source_receipt=runtime_source,
            runtime_baseline=runtime_baseline,
            formal_monitor_best_validation=formal_monitor_best,
        )
        return (
            run,
            terminal_progress,
            selected,
            terminal,
            validation_sha,
            golden_anchor_sha,
        )

    def test_build_validate_and_durable_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (
                run,
                progress,
                selected,
                terminal,
                validation_sha,
                golden_anchor_sha,
            ) = self._fixture(Path(tmp))
            receipt = build_anyres_grpo_selection_receipt(
                selected_checkpoint=selected,
                terminal_checkpoint=terminal,
                formal_run_contract=run,
                terminal_progress=progress,
                formal_monitor_best_validation_sha256=validation_sha,
                locked_golden_anchor_sha256=golden_anchor_sha,
            )
            self.assertEqual(receipt["kind"], OCR_ANYRES_GRPO_SELECTION_RECEIPT_KIND)
            self.assertEqual(
                receipt["selected_checkpoint"]["outer_optimizer_step"], 1
            )
            self.assertEqual(
                receipt["terminal_checkpoint"]["outer_optimizer_step"], 2
            )
            self.assertEqual(
                receipt["kl_selection_receipt_sha256"],
                run["formal_selection_receipt"]["canonical_sha256"],
            )
            self.assertEqual(receipt["selected_kl_coef"], 0.01)
            self.assertEqual(
                [key for key in receipt if "golden" in key],
                ["locked_golden_anchor_sha256"],
            )
            self.assertEqual(
                validate_anyres_grpo_selection_receipt(
                    receipt,
                    formal_run_contract=run,
                    terminal_progress=progress,
                ),
                receipt,
            )

            destination = (
                selected.parent / OCR_ANYRES_GRPO_SELECTION_RECEIPT_FILENAME
            )
            with mock.patch(
                "Model.posttrain.ocr_selection.os.fsync",
                wraps=os.fsync,
            ) as fsync:
                write_anyres_grpo_selection_receipt(
                    destination,
                    receipt,
                    formal_run_contract=run,
                    terminal_progress=progress,
                )
            self.assertGreaterEqual(fsync.call_count, 2)
            write_anyres_grpo_selection_receipt(
                destination,
                receipt,
                formal_run_contract=run,
                terminal_progress=progress,
            )

    def test_tampered_checkpoint_bytes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (
                run,
                progress,
                selected,
                terminal,
                validation_sha,
                golden_anchor_sha,
            ) = self._fixture(Path(tmp))
            receipt = build_anyres_grpo_selection_receipt(
                selected_checkpoint=selected,
                terminal_checkpoint=terminal,
                formal_run_contract=run,
                terminal_progress=progress,
                formal_monitor_best_validation_sha256=validation_sha,
                locked_golden_anchor_sha256=golden_anchor_sha,
            )
            (selected / "model.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "identity changed"):
                validate_anyres_grpo_selection_receipt(
                    receipt,
                    formal_run_contract=run,
                    terminal_progress=progress,
                )

    def test_terminal_progress_must_point_exactly_to_eligible_best(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (
                run,
                progress,
                selected,
                terminal,
                validation_sha,
                golden_anchor_sha,
            ) = self._fixture(Path(tmp))
            bad = copy.deepcopy(progress)
            bad["best_checkpoint"] = str(terminal)
            unhashed = {
                key: value
                for key, value in bad.items()
                if key != "canonical_sha256"
            }
            bad["canonical_sha256"] = canonical_json_sha256(unhashed)
            with self.assertRaisesRegex(
                ValueError,
                "progress state|best checkpoint",
            ):
                build_anyres_grpo_selection_receipt(
                    selected_checkpoint=selected,
                    terminal_checkpoint=terminal,
                    formal_run_contract=run,
                    terminal_progress=bad,
                    formal_monitor_best_validation_sha256=validation_sha,
                    locked_golden_anchor_sha256=golden_anchor_sha,
                )

    def test_failure_stop_reason_cannot_publish_an_old_best(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (
                run,
                progress,
                selected,
                terminal,
                validation_sha,
                golden_anchor_sha,
            ) = self._fixture(Path(tmp))
            failed = copy.deepcopy(progress)
            failed["stop_reason"] = "integrity_failure"
            failed["canonical_sha256"] = canonical_json_sha256(
                {
                    key: value
                    for key, value in failed.items()
                    if key != "canonical_sha256"
                }
            )
            with self.assertRaisesRegex(ValueError, "successful terminal reason"):
                build_anyres_grpo_selection_receipt(
                    selected_checkpoint=selected,
                    terminal_checkpoint=terminal,
                    formal_run_contract=run,
                    terminal_progress=failed,
                    formal_monitor_best_validation_sha256=validation_sha,
                    locked_golden_anchor_sha256=golden_anchor_sha,
                )

    def test_alias_path_and_incomplete_terminal_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (
                run,
                progress,
                selected,
                terminal,
                validation_sha,
                golden_anchor_sha,
            ) = self._fixture(Path(tmp))
            alias = selected.parent / "selected-alias"
            alias.symlink_to(selected.name)
            with self.assertRaisesRegex(ValueError, "canonical absolute"):
                build_anyres_grpo_selection_receipt(
                    selected_checkpoint=alias,
                    terminal_checkpoint=terminal,
                    formal_run_contract=run,
                    terminal_progress=progress,
                    formal_monitor_best_validation_sha256=validation_sha,
                    locked_golden_anchor_sha256=golden_anchor_sha,
                )

            (terminal / "optimizer.pt").unlink()
            with self.assertRaisesRegex(ValueError, "fully resumable"):
                build_anyres_grpo_selection_receipt(
                    selected_checkpoint=selected,
                    terminal_checkpoint=terminal,
                    formal_run_contract=run,
                    terminal_progress=progress,
                    formal_monitor_best_validation_sha256=validation_sha,
                    locked_golden_anchor_sha256=golden_anchor_sha,
                )


if __name__ == "__main__":
    unittest.main()
