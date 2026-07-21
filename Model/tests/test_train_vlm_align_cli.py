# -*- coding: utf-8 -*-

"""CLI guard tests for ``scripts/train_vlm_align``.

Ensures bad CLI input fails cleanly (exit code 2 + stderr message) rather than
raising a raw traceback — in particular ``--image-size`` is validated before
``image_patch_count`` derives the default ``--n-image-tokens``.
"""

from __future__ import annotations

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
from Model.omvt import OMVTInjector, collate_omvt_batch
from Model.training import build_optimizer
from scripts import train_vlm_align


class TrainVlmAlignCliGuardsTest(unittest.TestCase):
    def test_invalid_early_stop_fails_before_model_allocation(self) -> None:
        cases = (
            (["--early-stop-patience", "-1"], "patience"),
            (["--early-stop-min-delta", "nan"], "min_delta"),
            (["--early-stop-ema-alpha", "0"], "ema_alpha"),
            (["--early-stop-window-size", "0"], "window_size"),
            (
                ["--steps", "5", "--early-stop-min-steps", "6"],
                "cannot exceed",
            ),
        )
        for argv, expected in cases:
            stderr = io.StringIO()
            with (
                self.subTest(argv=argv),
                mock.patch.object(train_vlm_align, "RDTForCausalLM") as model_ctor,
                contextlib.redirect_stderr(stderr),
            ):
                rc = train_vlm_align.main(argv)
            self.assertEqual(rc, 2)
            model_ctor.assert_not_called()
            self.assertIn(expected, stderr.getvalue())

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
            rc = train_vlm_align.main([
                "--mamba", "official",
                "--image-size", "4",
                "--n-image-tokens", "1",
                "--seq-len", "8",
            ])
        self.assertEqual(rc, 2)
        msg = stderr.getvalue()
        self.assertIn("--mamba official", msg)
        self.assertIn("target device is cpu", msg)

    def test_non_positive_recurrent_steps_fails_cleanly(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            rc = train_vlm_align.main([
                "--image-size", "4",
                "--n-image-tokens", "1",
                "--seq-len", "8",
                "--recurrent-steps", "0",
            ])
        self.assertEqual(rc, 2)
        self.assertIn("--recurrent-steps", stderr.getvalue())


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


class _ConstantLossInjector(torch.nn.Module):
    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))


class _ConstantLossVlm(torch.nn.Module):
    def __init__(self, _cfg) -> None:
        super().__init__()
        self.language_weight = torch.nn.Parameter(torch.zeros(()))
        self.vision = torch.nn.Module()
        self.vision.omvt = None

    def forward(self, **_kwargs):
        loss = self.language_weight * 0.0 + self.vision.omvt.weight * 0.0 + 1.0
        return {"loss": loss}


class TrainVlmAlignEarlyStoppingIntegrationTest(unittest.TestCase):
    def test_plateau_writes_one_final_resumable_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            with (
                mock.patch.object(
                    train_vlm_align,
                    "_build_omvt_cfg",
                    return_value=_tiny_omvt(),
                ),
                mock.patch.object(
                    train_vlm_align,
                    "_build_rdt_cfg",
                    return_value=_tiny_rdt(),
                ),
                mock.patch.object(
                    train_vlm_align,
                    "RDTForCausalLM",
                    _ConstantLossVlm,
                ),
                mock.patch.object(
                    train_vlm_align,
                    "OMVTInjector",
                    _ConstantLossInjector,
                ),
                contextlib.redirect_stdout(stdout),
            ):
                rc = train_vlm_align.main(
                    [
                        "--steps",
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
            self.assertIn("stop_reason=loss_plateau", stdout.getvalue())
            step_dir = Path(tmp) / "step_00000003"
            self.assertTrue((step_dir / "model.pt").is_file())
            metadata = torch.load(
                step_dir / "meta.pt",
                map_location="cpu",
                weights_only=False,
            )["metadata"]
            self.assertTrue(metadata["final"])
            self.assertEqual(metadata["stop_reason"], "loss_plateau")
            self.assertEqual(
                metadata["early_stopping"]["state"]["last_step"],
                3,
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
        conflicts = train_vlm_align._resume_training_conflicts(
            TrainingConfig(), {}
        )
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
        }
        with tempfile.TemporaryDirectory() as tmp:
            step = Path(tmp) / "step_00000001"
            step.mkdir()
            torch.save({}, step / "model.pt")
            torch.save({"step": 1, "metadata": metadata}, step / "meta.pt")
            stderr = io.StringIO()
            with (
                mock.patch.object(train_vlm_align, "RDTForCausalLM") as model_ctor,
                contextlib.redirect_stderr(stderr),
            ):
                rc = train_vlm_align.main([
                    "--resume", str(step),
                    "--freeze-rdt",
                    "--data", "align.jsonl",
                    "--image-size", "16",
                    "--n-image-tokens", "2",
                    "--seq-len", "512",
                    "--steps", "6000",
                    "--batch-size", "16",
                    "--warmup-steps", "500",
                    "--precision", "bf16",
                    "--device", "cpu",
                    "--mamba", "naive",
                ])
            self.assertEqual(rc, 2)
            model_ctor.assert_not_called()
            self.assertIn("micro_batch_size", stderr.getvalue())

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
        ids = torch.tensor([[
            cfg.bos_id,
            cfg.image_patch_id,
            cfg.image_patch_id,
            300,
            301,
            302,
            303,
            cfg.eos_id,
        ]])
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
