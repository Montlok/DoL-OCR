# -*- coding: utf-8 -*-

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts import eval_ocr_anyres_locked as cli


def _args(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        device="cpu",
        precision="bf16",
        text_batch_size=4,
        sealed_root=str(root / "sealed"),
        ledger_dir=str(root / "ledger"),
        out=str(root / "report.json"),
    )


class LockedEvalCLITest(unittest.TestCase):
    def test_preclaim_failure_never_claims_or_opens_locked_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = _args(Path(temporary))
            with mock.patch.object(
                cli,
                "_preclaim",
                side_effect=ValueError("public receipt mismatch"),
            ), mock.patch.object(cli, "claim_locked_benchmark_once") as claim, mock.patch.object(
                cli, "load_locked_benchmark_after_claim"
            ) as locked_load:
                with self.assertRaisesRegex(ValueError, "receipt mismatch"):
                    cli._run(args)
            claim.assert_not_called()
            locked_load.assert_not_called()
            self.assertFalse(Path(args.out).exists())

    def test_claim_and_incomplete_stub_precede_first_locked_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = _args(root)
            events: list[str] = []
            preclaim = SimpleNamespace(
                anchor_canonical_sha256="1" * 64,
                build_receipt_sha256="2" * 64,
            )
            state = {
                "output": Path(args.out),
                "preclaim": preclaim,
                "selected_identity": {
                    "model_sha256": "3" * 64,
                    "metadata_sha256": "4" * 64,
                },
                "reference_identity": {
                    "model_sha256": "5" * 64,
                    "metadata_sha256": "6" * 64,
                },
                "selection_file_sha256": "7" * 64,
                "run": {
                    "canonical_sha256": "8" * 64,
                    "source_closure_sha256": "9" * 64,
                },
                "kl_file_sha256": "a" * 64,
                "joint_result": {"canonical_sha256": "b" * 64},
                "evaluator_source": {"canonical_sha256": "c" * 64},
                "bundle": SimpleNamespace(tokenizer=object()),
            }
            claim = SimpleNamespace(marker_sha256="d" * 64)

            def do_claim(*_args, **_kwargs):
                events.append("claim")
                return claim

            def do_stub(*_args, **_kwargs):
                events.append("stub")

            def do_load(*_args, **_kwargs):
                events.append("locked-read")
                raise RuntimeError("post-claim injected failure")

            with mock.patch.object(cli, "_preclaim", return_value=state), mock.patch.object(
                cli, "claim_locked_benchmark_once", side_effect=do_claim
            ), mock.patch.object(
                cli, "write_claimed_evaluation_incomplete_stub", side_effect=do_stub
            ), mock.patch.object(
                cli, "make_ocr_target_encoder", return_value=lambda text: [1]
            ), mock.patch.object(
                cli,
                "build_anyres_ocr_reward_adapter",
                return_value=SimpleNamespace(decode_completion=lambda values: str(values)),
            ), mock.patch.object(
                cli, "load_locked_benchmark_after_claim", side_effect=do_load
            ):
                with self.assertRaisesRegex(RuntimeError, "post-claim"):
                    cli._run(args)
            self.assertEqual(events, ["claim", "stub", "locked-read"])


if __name__ == "__main__":
    unittest.main()
