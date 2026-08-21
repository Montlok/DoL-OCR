# -*- coding: utf-8 -*-

"""CLI guard tests for ``scripts/train_vlm_align``.

Ensures bad CLI input fails cleanly (exit code 2 + stderr message) rather than
raising a raw traceback — in particular ``--image-size`` is validated before
``image_patch_count`` derives the default ``--n-image-tokens``.
"""

from __future__ import annotations

import copy
import contextlib
import io
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from Model.config import OMVTConfig, RDTConfig, TrainingConfig
from Model.model import RDTForCausalLM
from Model.ocr.position_contract import (
    BOUNDARY_V1,
    OCR_POSITION_CONTRACT_METADATA_VERSION,
)
from Model.omvt import OMVTInjector, collate_omvt_batch
from Model.training import build_optimizer
from scripts import train_vlm_align


class TrainVlmAlignCliGuardsTest(unittest.TestCase):
    def test_steps_alias_resolves_to_max_steps(self) -> None:
        args = train_vlm_align.parse_args(["--steps", "9"])
        self.assertEqual(args.max_steps, 9)
        self.assertEqual(args.steps, 9)
        self.assertEqual(args.ocr_target_encoding, "native")
        self.assertEqual(args.ocr_position_contract, BOUNDARY_V1)

    def test_early_stop_cli_bounds_are_validated(self) -> None:
        invalid = (
            ["--max-steps", "0"],
            ["--max-steps", "5", "--min-steps", "6"],
            ["--early-stop-patience", "-1"],
            ["--early-stop-min-delta", "nan"],
            ["--early-stop-ema-alpha", "0"],
            ["--early-stop-window", "0"],
        )
        for argv in invalid:
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                train_vlm_align.parse_args(argv)

    def test_full_corpus_stream_requires_frozen_language(self) -> None:
        stderr = io.StringIO()
        with (
            mock.patch.object(train_vlm_align, "RDTForCausalLM") as model_ctor,
            contextlib.redirect_stderr(stderr),
        ):
            rc = train_vlm_align.main(
                [
                    "--stream-wds-dir",
                    "wds",
                    "--stream-hanshi-meta",
                    "meta.jsonl",
                    "--stream-hanshi-pages",
                    "pages",
                    "--stream-tokenizer-bundle",
                    "tokenizer",
                ]
            )
        self.assertEqual(rc, 2)
        model_ctor.assert_not_called()
        self.assertIn("--freeze-rdt", stderr.getvalue())

    def test_full_corpus_stream_rejects_non_native_targets(self) -> None:
        stderr = io.StringIO()
        with (
            mock.patch.object(train_vlm_align, "RDTForCausalLM") as model_ctor,
            contextlib.redirect_stderr(stderr),
        ):
            rc = train_vlm_align.main(
                [
                    "--stream-wds-dir",
                    "wds",
                    "--stream-hanshi-meta",
                    "meta.jsonl",
                    "--stream-hanshi-pages",
                    "pages",
                    "--stream-tokenizer-bundle",
                    "tokenizer",
                    "--freeze-rdt",
                    "--ocr-target-encoding",
                    "native_fallback",
                ]
            )
        self.assertEqual(rc, 2)
        model_ctor.assert_not_called()
        self.assertIn("--ocr-target-encoding native", stderr.getvalue())

    def test_partial_wds_cap_is_smoke_only(self) -> None:
        stderr = io.StringIO()
        with (
            mock.patch.object(train_vlm_align, "RDTForCausalLM") as model_ctor,
            contextlib.redirect_stderr(stderr),
        ):
            rc = train_vlm_align.main(
                [
                    "--stream-wds-dir",
                    "wds",
                    "--stream-hanshi-meta",
                    "meta.jsonl",
                    "--stream-hanshi-pages",
                    "pages",
                    "--stream-tokenizer-bundle",
                    "tokenizer",
                    "--stream-max-wds-shards",
                    "1",
                    "--freeze-rdt",
                ]
            )
        self.assertEqual(rc, 2)
        model_ctor.assert_not_called()
        self.assertIn("restricted to --smoke", stderr.getvalue())

    def test_non_positive_image_size_fails_cleanly(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            rc = train_vlm_align.main(["--image-size", "0", "--seq-len", "16"])
        self.assertEqual(rc, 2)
        self.assertIn("--image-size", stderr.getvalue())

    def test_non_multiple_of_four_image_size_fails_cleanly(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            rc = train_vlm_align.main(["--image-size", "6", "--seq-len", "16"])
        self.assertEqual(rc, 2)
        self.assertIn("--image-size", stderr.getvalue())

    def test_official_mamba_failure_returns_exit_code_two(self) -> None:
        stderr = io.StringIO()
        with (
            mock.patch.object(
                train_vlm_align.torch.cuda,
                "is_available",
                return_value=False,
            ),
            contextlib.redirect_stderr(stderr),
        ):
            rc = train_vlm_align.main(
                [
                    "--mamba",
                    "official",
                    "--image-size",
                    "4",
                    "--n-image-tokens",
                    "1",
                    "--seq-len",
                    "8",
                ]
            )
        self.assertEqual(rc, 2)
        msg = stderr.getvalue()
        self.assertIn("--mamba official", msg)
        self.assertIn("target device is cpu", msg)

    def test_non_positive_recurrent_steps_fails_cleanly(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            rc = train_vlm_align.main(
                [
                    "--image-size",
                    "4",
                    "--n-image-tokens",
                    "1",
                    "--seq-len",
                    "8",
                    "--recurrent-steps",
                    "0",
                ]
            )
        self.assertEqual(rc, 2)
        self.assertIn("--recurrent-steps", stderr.getvalue())


class LossPlateauEarlyStopTest(unittest.TestCase):
    def _tracker(self, **overrides) -> train_vlm_align.LossPlateauEarlyStop:
        values = {
            "mode": "ema",
            "min_steps": 0,
            "patience": 2,
            "min_delta": 0.0,
            "ema_alpha": 1.0,
            "window_size": 3,
        }
        values.update(overrides)
        return train_vlm_align.LossPlateauEarlyStop(**values)

    def test_burn_in_does_not_anchor_to_an_early_low_outlier(self) -> None:
        tracker = self._tracker(min_steps=3)
        self.assertFalse(tracker.observe(1.0, 1))
        self.assertFalse(tracker.observe(5.0, 2))
        self.assertEqual(tracker.best, 5.0)
        self.assertFalse(tracker.observe(5.0, 3))
        self.assertTrue(tracker.observe(5.0, 4))

    def test_min_delta_requires_a_material_loss_improvement(self) -> None:
        tracker = self._tracker(patience=3, min_delta=0.1)
        self.assertFalse(tracker.observe(5.0, 1))
        self.assertFalse(tracker.observe(4.95, 2))
        self.assertEqual(tracker.bad_steps, 1)
        self.assertFalse(tracker.observe(4.8, 3))
        self.assertEqual(tracker.best, 4.8)
        self.assertEqual(tracker.bad_steps, 0)

    def test_rolling_window_state_resumes_exactly(self) -> None:
        tracker = self._tracker(mode="window", patience=2)
        tracker.observe(5.0, 1)
        tracker.observe(4.0, 2)
        args = SimpleNamespace(
            early_stop_mode="window",
            min_steps=0,
            early_stop_patience=2,
            early_stop_min_delta=0.0,
            early_stop_ema_alpha=1.0,
            early_stop_window=3,
        )
        restored = train_vlm_align.LossPlateauEarlyStop.from_args(
            args, {"early_stop": tracker.metadata_dict()}
        )
        self.assertEqual(restored.state_dict(), tracker.state_dict())
        for step, loss in enumerate((3.0, 3.0, 3.0), start=3):
            self.assertEqual(
                restored.observe(loss, step),
                tracker.observe(loss, step),
            )
            self.assertEqual(restored.state_dict(), tracker.state_dict())

    def test_enabled_resume_requires_saved_plateau_state(self) -> None:
        args = train_vlm_align.parse_args(["--early-stop-patience", "10"])
        conflicts = train_vlm_align._early_stop_resume_conflicts(args, {})
        self.assertTrue(conflicts)
        self.assertIn("no early_stop metadata", conflicts[0])

    def test_cursor_advances_only_after_optimizer_commit(self) -> None:
        initial = {"counts": {"total": 0}}
        iterator = train_vlm_align.CursorTrackingIterator(
            iter(
                [
                    {"corpus_cursor": {"counts": {"total": 2}}},
                    {"corpus_cursor": {"counts": {"total": 4}}},
                ]
            ),
            initial_cursor=initial,
        )
        next(iterator)
        self.assertEqual(iterator.committed_cursor, initial)
        iterator.commit()
        self.assertEqual(iterator.committed_cursor["counts"]["total"], 2)
        next(iterator)
        self.assertEqual(iterator.committed_cursor["counts"]["total"], 2)

    def test_stream_loop_stops_on_plateau_and_checkpoints_committed_cursor(
        self,
    ) -> None:
        cursors = [{"version": 1, "counts": {"total": total}} for total in range(4)]
        batch_iter = train_vlm_align.CursorTrackingIterator(
            iter({"corpus_cursor": cursor} for cursor in cursors[1:]),
            initial_cursor=cursors[0],
        )
        spec = train_vlm_align.StreamingCorpusSpec(
            paths=(Path("shard-00000.tar"),),
            manifest={"version": 1},
            manifest_sha256="corpus-sha",
            tokenizer_manifest_sha256="tokenizer-sha",
            tokenizer_bundle=None,
            resume_cursor=None,
        )
        losses = iter((1.0, 1.0, 1.0))

        def fake_train_one_step(
            _model,
            iterator,
            _optimizer,
            _scheduler,
            _train_cfg,
            state,
            *,
            device,
        ):
            self.assertEqual(device.type, "cpu")
            next(iterator)
            state.step += 1
            return {"loss": next(losses), "grad_norm": 1.0, "lr": 1e-4}

        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch.object(
                    train_vlm_align, "_checkpoint_metadata", return_value={}
                ),
                mock.patch.object(
                    train_vlm_align, "_prepare_streaming_corpus", return_value=spec
                ),
                mock.patch.object(
                    train_vlm_align, "_build_omvt_cfg", return_value=_tiny_omvt()
                ),
                mock.patch.object(
                    train_vlm_align, "_build_rdt_cfg", return_value=_tiny_rdt()
                ),
                mock.patch.object(
                    train_vlm_align,
                    "_stream_vlm_batches",
                    return_value=batch_iter,
                ),
                mock.patch.object(
                    train_vlm_align,
                    "train_one_step",
                    side_effect=fake_train_one_step,
                ),
                mock.patch.object(train_vlm_align, "save_checkpoint") as save_mock,
            ):
                rc = train_vlm_align.main(
                    [
                        "--stream-wds-dir",
                        "wds",
                        "--stream-hanshi-meta",
                        "meta.jsonl",
                        "--stream-hanshi-pages",
                        "pages",
                        "--stream-tokenizer-bundle",
                        "tokenizer",
                        "--freeze-rdt",
                        "--max-steps",
                        "5",
                        "--early-stop-patience",
                        "2",
                        "--early-stop-ema-alpha",
                        "1",
                        "--device",
                        "cpu",
                        "--mamba",
                        "naive",
                        "--output",
                        tmp,
                    ]
                )
        self.assertEqual(rc, 0)
        save_mock.assert_called_once()
        metadata = save_mock.call_args.kwargs["metadata"]
        self.assertEqual(metadata["stop_reason"], "loss_plateau")
        self.assertTrue(metadata["final"])
        self.assertEqual(metadata["early_stop"]["state"]["bad_steps"], 2)
        self.assertEqual(metadata["streaming"]["corpus_cursor"]["counts"]["total"], 3)
        self.assertEqual(metadata["ocr_position_contract"], BOUNDARY_V1)
        self.assertEqual(
            metadata["ocr_position_contract_version"],
            OCR_POSITION_CONTRACT_METADATA_VERSION,
        )


def _tiny_rdt() -> RDTConfig:
    return RDTConfig(
        d_model=32,
        n_heads=4,
        head_dim=8,
        kv_lora_rank=8,
        rope_head_dim=4,
        nope_head_dim=4,
        ffn_hidden=64,
        ffn_multiple=32,
        n_prelude=1,
        n_coda=1,
        mamba_per_block=1,
        attn_per_block=1,
        recurrent_steps=2,
        mamba_d_state=8,
        mamba_expand=2,
        mamba_headdim=16,
        use_official_mamba=False,
        max_seq_len=32,
    )


def _tiny_omvt() -> OMVTConfig:
    return OMVTConfig(
        image_size=16,
        vertical_patch=(8, 4),
        horizontal_patch=(4, 8),
        square_patch=(4, 4),
        layout_patch=(16, 16),
        d_vision=32,
        vision_n_heads=4,
        vision_ffn_hidden=64,
        compress_to=2,
        compressor_layers=1,
        compressor_heads=4,
        n_vertical_layers=1,
        n_horizontal_layers=1,
        n_local_attn_layers=1,
        n_layout_layers=1,
    )


class TrainVlmAlignFreezeContractTest(unittest.TestCase):
    def _model(self) -> RDTForCausalLM:
        rdt_cfg, omvt_cfg = _tiny_rdt(), _tiny_omvt()
        model = RDTForCausalLM(rdt_cfg)
        model.vision._omvt_cfg = omvt_cfg
        model.vision.omvt = OMVTInjector(rdt_cfg, omvt_cfg)
        return model

    def test_text_checkpoint_requires_every_language_key(self) -> None:
        text_model = RDTForCausalLM(_tiny_rdt())
        state = text_model.state_dict()
        target = self._model()
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "good.pt"
            torch.save(state, good)
            train_vlm_align._load_rdt_init(target, str(good))

            bad_state = dict(state)
            missing_key = next(k for k in bad_state if k.startswith("recurrent."))
            bad_state.pop(missing_key)
            bad = Path(tmp) / "bad.pt"
            torch.save(bad_state, bad)
            with self.assertRaisesRegex(RuntimeError, "partially loaded"):
                train_vlm_align._load_rdt_init(self._model(), str(bad))

    def test_checkpoint_metadata_is_the_model_geometry_source(self) -> None:
        rdt_cfg, omvt_cfg = _tiny_rdt(), _tiny_omvt()
        args = SimpleNamespace(
            config="tiny",
            seq_len=8,
            recurrent_steps=None,
            grad_ckpt=False,
            mamba="naive",
            image_size=56,
            d_vision=64,
            n_image_tokens=None,
            patch_preset="derived",
            init_omvt_checkpoint="",
        )
        restored_rdt = train_vlm_align._build_rdt_cfg(
            args,
            {"rdt_config": asdict(rdt_cfg)},
            torch.device("cpu"),
        )
        restored_omvt = train_vlm_align._build_omvt_cfg(
            args, {"omvt_config": asdict(omvt_cfg)}
        )
        self.assertEqual(restored_rdt.d_model, rdt_cfg.d_model)
        self.assertEqual(restored_rdt.recurrent_steps, rdt_cfg.recurrent_steps)
        self.assertEqual(restored_omvt.image_size, omvt_cfg.image_size)
        self.assertEqual(restored_omvt.compress_to, omvt_cfg.compress_to)

    def test_native_ocr_producer_persists_consumer_lineage(self) -> None:
        args = SimpleNamespace(
            config="tiny",
            resume="",
            freeze_rdt=True,
            frozen_vision=False,
            init_rdt_checkpoint="/checkpoints/lm",
            init_omvt_checkpoint="",
            use_ema_tower=False,
        )
        source_bundle = {
            "schema_version": 1,
            "files": [
                {
                    "role": "config",
                    "name": "config.json",
                    "size_bytes": 1,
                    "sha256": "a" * 64,
                }
            ],
            "files_canonical_sha256": "b" * 64,
        }
        ocr_data_contract = {
            "ocr_tokenization_contract": {
                "target_encoding": "native",
                "tokenization_contract_version": 3,
                "pretraining_tokenizer_algorithm": {
                    "contract_version": 1,
                    "files": {"dual.py": "f" * 64},
                },
            }
        }
        metadata = train_vlm_align._alignment_metadata(
            args,
            _tiny_rdt(),
            _tiny_omvt(),
            TrainingConfig(),
            inherited={
                "tokenizer_bundle": source_bundle,
                "tokenizer_algorithm": ocr_data_contract[
                    "ocr_tokenization_contract"
                ]["pretraining_tokenizer_algorithm"],
            },
            stop_reason="loss_plateau",
            final=True,
            ocr_data_contract=ocr_data_contract,
        )
        self.assertEqual(metadata["ocr_target_encoding"], "native")
        self.assertEqual(metadata["ocr_tokenization_contract_version"], 3)
        self.assertEqual(
            metadata["source_rdt_tokenizer_bundle"],
            source_bundle,
        )
        self.assertEqual(
            metadata["source_rdt_tokenizer_algorithm"],
            ocr_data_contract["ocr_tokenization_contract"][
                "pretraining_tokenizer_algorithm"
            ],
        )
        self.assertEqual(metadata["ocr_data_contract"], ocr_data_contract)

    def test_rdt_to_vlm_requires_one_exact_bundle_contract(self) -> None:
        bundle = {
            "schema_version": 1,
            "files": [
                {
                    "role": "vocab",
                    "name": "vocab.json",
                    "size_bytes": 123,
                    "sha256": "a" * 64,
                }
            ],
            "files_canonical_sha256": "b" * 64,
        }
        algorithm = {
            "contract_version": 1,
            "files": {"Tokenizer/unified/dual_tokenizer.py": "c" * 64},
            "files_canonical_sha256": "d" * 64,
        }
        token_contract = {
            "tokenizer_bundle": bundle,
            "pretraining_tokenizer_algorithm": algorithm,
        }
        self.assertEqual(
            train_vlm_align._validate_frozen_rdt_tokenizer_lineage(
                {
                    "tokenizer_bundle": bundle,
                    "tokenizer_algorithm": algorithm,
                },
                token_contract,
            ),
            (bundle, algorithm),
        )

        drifted = copy.deepcopy(bundle)
        drifted["files"][0]["sha256"] = "0" * 64
        for source_bundle in (
            drifted,
            {"files": {"vocab.json": "a" * 64}},
        ):
            with self.subTest(source_bundle=source_bundle), self.assertRaisesRegex(
                ValueError,
                "bundle contract differs",
            ):
                train_vlm_align._validate_frozen_rdt_tokenizer_lineage(
                    {
                        "tokenizer_bundle": source_bundle,
                        "tokenizer_algorithm": algorithm,
                    },
                    token_contract,
                )

    def test_resume_rejects_batch_or_schedule_drift(self) -> None:
        saved = TrainingConfig(
            train_data="align.jsonl",
            seq_len=512,
            micro_batch_size=32,
            learning_rate=3e-4,
            weight_decay=0.05,
            max_steps=6000,
            warmup_steps=500,
            precision="bf16",
            output_dir="old-output",
            save_every=1000,
        )
        same_run_new_logging = TrainingConfig(
            train_data="align.jsonl",
            seq_len=512,
            micro_batch_size=32,
            learning_rate=3e-4,
            weight_decay=0.05,
            max_steps=6000,
            warmup_steps=500,
            precision="bf16",
            output_dir="new-output",
            save_every=250,
            resume="latest",
        )
        self.assertEqual(
            train_vlm_align._resume_training_conflicts(
                same_run_new_logging, {"training_config": asdict(saved)}
            ),
            [],
        )

        drifted = TrainingConfig(
            train_data="align.jsonl",
            seq_len=512,
            micro_batch_size=16,
            learning_rate=3e-4,
            weight_decay=0.05,
            max_steps=8000,
            warmup_steps=1000,
            precision="bf16",
        )
        conflicts = train_vlm_align._resume_training_conflicts(
            drifted, {"training_config": asdict(saved)}
        )
        self.assertTrue(any(item.startswith("micro_batch_size:") for item in conflicts))
        self.assertTrue(any(item.startswith("max_steps:") for item in conflicts))
        self.assertTrue(any(item.startswith("warmup_steps:") for item in conflicts))

    def test_legacy_resume_without_training_metadata_is_rejected(self) -> None:
        conflicts = train_vlm_align._resume_training_conflicts(TrainingConfig(), {})
        self.assertEqual(len(conflicts), 1)
        self.assertIn("--init-rdt-checkpoint", conflicts[0])

    def test_resume_drift_fails_before_model_allocation(self) -> None:
        saved_train = TrainingConfig(
            train_data="align.jsonl",
            seq_len=512,
            micro_batch_size=32,
            learning_rate=3e-4,
            weight_decay=0.05,
            max_steps=6000,
            warmup_steps=500,
            precision="bf16",
        )
        metadata = {
            "rdt_config": asdict(_tiny_rdt()),
            "omvt_config": asdict(_tiny_omvt()),
            "training_config": asdict(saved_train),
            "freeze_rdt": True,
            "frozen_vision": False,
            "ocr_position_contract": BOUNDARY_V1,
            "ocr_position_contract_version": (
                OCR_POSITION_CONTRACT_METADATA_VERSION
            ),
        }
        with tempfile.TemporaryDirectory() as tmp:
            step = Path(tmp) / "step_00000001"
            step.mkdir()
            torch.save({}, step / "model.pt")
            torch.save({"step": 1, "metadata": metadata}, step / "meta.pt")
            for name in ("optimizer.pt", "scheduler.pt", "rng.pt"):
                (step / name).write_bytes(name.encode("ascii"))
            (step / "COMPLETE").write_text("step=1\n", encoding="ascii")
            stderr = io.StringIO()
            with (
                mock.patch.object(train_vlm_align, "RDTForCausalLM") as model_ctor,
                contextlib.redirect_stderr(stderr),
            ):
                rc = train_vlm_align.main(
                    [
                        "--resume",
                        str(step),
                        "--freeze-rdt",
                        "--data",
                        "align.jsonl",
                        "--image-size",
                        "16",
                        "--n-image-tokens",
                        "2",
                        "--seq-len",
                        "512",
                        "--steps",
                        "6000",
                        "--batch-size",
                        "16",
                        "--warmup-steps",
                        "500",
                        "--precision",
                        "bf16",
                        "--device",
                        "cpu",
                        "--mamba",
                        "naive",
                    ]
                )
            self.assertEqual(rc, 2)
            model_ctor.assert_not_called()
            self.assertIn("micro_batch_size", stderr.getvalue())

    def test_resume_rejects_incomplete_checkpoint_before_model_allocation(
        self,
    ) -> None:
        metadata = {
            "rdt_config": asdict(_tiny_rdt()),
            "omvt_config": asdict(_tiny_omvt()),
            "training_config": asdict(TrainingConfig()),
        }
        with tempfile.TemporaryDirectory() as tmp:
            step = Path(tmp) / "step_00000001"
            step.mkdir()
            torch.save({}, step / "model.pt")
            torch.save({"step": 1, "metadata": metadata}, step / "meta.pt")
            (step / "COMPLETE").write_text("step=1\n", encoding="ascii")
            stderr = io.StringIO()
            with (
                mock.patch.object(train_vlm_align, "RDTForCausalLM") as model_ctor,
                contextlib.redirect_stderr(stderr),
            ):
                rc = train_vlm_align.main(
                    [
                        "--resume",
                        str(step),
                        "--image-size",
                        "16",
                        "--n-image-tokens",
                        "2",
                        "--seq-len",
                        "32",
                        "--device",
                        "cpu",
                        "--mamba",
                        "naive",
                    ]
                )
            self.assertEqual(rc, 2)
            model_ctor.assert_not_called()
            self.assertIn("optimizer.pt", stderr.getvalue())

    def test_freeze_updates_only_omvt(self) -> None:
        torch.manual_seed(0)
        model = self._model()
        args = SimpleNamespace(freeze_rdt=True, frozen_vision=False)
        names = train_vlm_align._configure_trainable_modules(model, args)
        self.assertTrue(names)
        self.assertTrue(all(name.startswith("vision.omvt.") for name in names))

        train_cfg = TrainingConfig(
            train_data="",
            seq_len=8,
            micro_batch_size=1,
            learning_rate=1e-3,
            max_steps=1,
            warmup_steps=1,
            precision="fp32",
        )
        optimizer = build_optimizer(model, train_cfg)
        optimizer_ids = {
            id(param) for group in optimizer.param_groups for param in group["params"]
        }
        trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
        self.assertEqual(optimizer_ids, trainable_ids)

        language_before = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if not name.startswith("vision.omvt.")
        }
        visual_before = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if name.startswith("vision.omvt.")
        }
        image = torch.randn(1, 3, 16, 16)
        pixels = dict(collate_omvt_batch(image, _tiny_omvt()))
        cfg = _tiny_rdt()
        ids = torch.tensor(
            [
                [
                    cfg.bos_id,
                    cfg.image_patch_id,
                    cfg.image_patch_id,
                    300,
                    301,
                    302,
                    303,
                    cfg.eos_id,
                ]
            ]
        )
        out = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            labels=ids,
            pixel_values=pixels,
            steps=2,
        )
        out["loss"].backward()
        self.assertTrue(
            any(
                p.grad is not None
                and bool(torch.isfinite(p.grad).all())
                and float(p.grad.abs().sum()) > 0
                for name, p in model.named_parameters()
                if name.startswith("vision.omvt.")
            )
        )
        self.assertTrue(
            all(
                p.grad is None
                for name, p in model.named_parameters()
                if not name.startswith("vision.omvt.")
            )
        )
        optimizer.step()
        self.assertTrue(
            all(
                torch.equal(language_before[name], p.detach())
                for name, p in model.named_parameters()
                if name in language_before
            )
        )
        self.assertTrue(
            any(
                not torch.equal(visual_before[name], p.detach())
                for name, p in model.named_parameters()
                if name in visual_before
            )
        )


if __name__ == "__main__":
    unittest.main()
