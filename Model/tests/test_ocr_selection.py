# -*- coding: utf-8 -*-

"""Terminal-state tests for locked OCR checkpoint selection."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from Model.posttrain.ocr_selection import (
    build_selection_receipt,
    load_and_validate_selection_receipt,
    write_selection_receipt,
)


class OCRSelectionReceiptTest(unittest.TestCase):
    @staticmethod
    def _checkpoint(
        path: Path,
        *,
        step: int,
        data_contract: dict,
        final: bool,
        best_checkpoint: Path | None = None,
        best_step: int | None = None,
    ) -> Path:
        if final and (best_checkpoint is None or best_step is None):
            raise ValueError("terminal fixture requires the selected best")
        health_state = {
            "stop_reason": "validation_plateau",
        }
        if best_checkpoint is not None and best_step is not None:
            health_state.update(
                {
                    "best_val_eligible": True,
                    "best_val_step": best_step,
                    "best_checkpoint": str(best_checkpoint),
                }
            )
        path.mkdir(parents=True)
        (path / "model.pt").write_bytes(f"model-{step}".encode("ascii"))
        torch.save(
            {
                "step": step,
                "metadata": {
                    "phase": "grpo",
                    "task": "ocr",
                    "final": final,
                    "stop_reason": "validation_plateau",
                    "health_state": health_state,
                    "data_contract": data_contract,
                },
            },
            path / "meta.pt",
        )
        (path / "COMPLETE").write_text(
            f"step={step}\n",
            encoding="ascii",
        )
        return path

    def test_terminal_metadata_is_revalidated_at_evaluation_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_contract = {"schema_version": 1, "dataset": "fixed"}
            selected = self._checkpoint(
                root / "best" / "step_00000001",
                step=1,
                data_contract=data_contract,
                final=False,
            )
            terminal = self._checkpoint(
                root / "step_00000002",
                step=2,
                data_contract=data_contract,
                final=True,
                best_checkpoint=selected,
                best_step=1,
            )
            receipt_path = selected.parent / "SELECTION_FINALIZED.json"
            write_selection_receipt(
                receipt_path,
                build_selection_receipt(
                    selected_checkpoint=selected,
                    selected_step=1,
                    terminal_checkpoint=terminal,
                    terminal_step=2,
                    stop_reason="validation_plateau",
                    data_contract=data_contract,
                ),
            )
            load_and_validate_selection_receipt(
                receipt_path,
                selected_checkpoint=selected,
                data_contract=data_contract,
            )

            payload = torch.load(
                terminal / "meta.pt",
                map_location="cpu",
                weights_only=False,
            )
            payload["metadata"]["final"] = False
            torch.save(payload, terminal / "meta.pt")
            with self.assertRaisesRegex(ValueError, "final flag"):
                load_and_validate_selection_receipt(
                    receipt_path,
                    selected_checkpoint=selected,
                    data_contract=data_contract,
                )

    def test_receipt_must_be_beside_selected_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_contract = {"dataset": "fixed"}
            selected = self._checkpoint(
                root / "best" / "step_00000001",
                step=1,
                data_contract=data_contract,
                final=False,
            )
            terminal = self._checkpoint(
                root / "step_00000002",
                step=2,
                data_contract=data_contract,
                final=True,
                best_checkpoint=selected,
                best_step=1,
            )
            copied = root / "copied-selection.json"
            write_selection_receipt(
                copied,
                build_selection_receipt(
                    selected_checkpoint=selected,
                    selected_step=1,
                    terminal_checkpoint=terminal,
                    terminal_step=2,
                    stop_reason="validation_plateau",
                    data_contract=data_contract,
                ),
            )
            with self.assertRaisesRegex(ValueError, "canonical"):
                load_and_validate_selection_receipt(
                    copied,
                    selected_checkpoint=selected,
                    data_contract=data_contract,
                )


if __name__ == "__main__":
    unittest.main()
