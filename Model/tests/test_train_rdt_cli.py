# -*- coding: utf-8 -*-

"""CLI guard tests for ``scripts/train_rdt``.

These guards prevent the most dangerous foot-gun: kicking off what looks
like a real pretraining run but silently feeding the model random tokens
and emitting checkpoints with no useful signal.

Tests are hermetic: ``--output`` is redirected to a ``TemporaryDirectory``
so they never touch ``outputs/`` in the working tree, and we assert on
exit codes / stderr messages so a regression that exits for a different
reason (e.g. missing module) would not silently pass.
"""

from __future__ import annotations

import contextlib
import io
import copy
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

from Model.config import _REMOVED_INACTIVE_RDT_CONFIG_FIELDS
from Model.training.checkpoint import validate_resumable_checkpoint
from scripts import train_rdt


class TrainRdtCliGuardsTest(unittest.TestCase):
    def test_resume_normalizes_reviewed_inactive_legacy_model_fields(self) -> None:
        current = asdict(train_rdt.CONFIG_CHOICES["two_stage_tiny"]())
        legacy = {**current, **_REMOVED_INACTIVE_RDT_CONFIG_FIELDS}
        self.assertEqual(train_rdt._resume_rdt_config(legacy), current)

    def test_resume_rejects_model_only_and_interrupted_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_only = root / "step_00000001"
            model_only.mkdir()
            (model_only / "model.pt").write_bytes(b"model")
            (model_only / "COMPLETE").write_text(
                "step=1\n",
                encoding="ascii",
            )
            train_rdt.torch.save(
                {"step": 1, "metadata": {}},
                model_only / "meta.pt",
            )
            with self.assertRaisesRegex(
                ValueError,
                "optimizer.pt",
            ):
                validate_resumable_checkpoint(
                    model_only,
                    require_scaler=False,
                    context="RDT --resume",
                )

            interrupted = root / "step_00000002"
            interrupted.mkdir()
            for name in (
                "model.pt",
                "optimizer.pt",
                "scheduler.pt",
                "rng.pt",
            ):
                (interrupted / name).write_bytes(name.encode("ascii"))
            train_rdt.torch.save(
                {"step": 2, "metadata": {}},
                interrupted / "meta.pt",
            )
            with self.assertRaisesRegex(ValueError, "COMPLETE"):
                validate_resumable_checkpoint(
                    interrupted,
                    require_scaler=False,
                    context="RDT --resume",
                )
            (interrupted / "COMPLETE").write_text(
                "step=2\n",
                encoding="ascii",
            )
            self.assertEqual(
                validate_resumable_checkpoint(
                    interrupted,
                    require_scaler=False,
                    context="RDT --resume",
                ),
                interrupted,
            )
            with self.assertRaisesRegex(ValueError, "scaler.pt"):
                validate_resumable_checkpoint(
                    interrupted,
                    require_scaler=True,
                    context="RDT --resume",
                )

    def test_resume_binds_tokenizer_algorithm_and_data_lineage(self) -> None:
        current = {
            "config_name": "tiny",
            "rdt_config": {"d_model": 64, "recurrent_steps": 2},
            "omvt_config": None,
            "tokenizer_bundle": {
                "schema_version": 1,
                "files": [
                    {
                        "role": "vocab",
                        "name": "vocab.json",
                        "size_bytes": 123,
                        "sha256": "a" * 64,
                    }
                ],
                "files_canonical_sha256": "d" * 64,
            },
            "tokenizer_algorithm": {
                "contract_version": 1,
                "files_canonical_sha256": "b" * 64,
            },
            "training_config": {
                "train_data": "/new/mount/train/*.jsonl",
                "eval_data": "/new/mount/eval/*.jsonl",
                "output_dir": "/new/output",
                "resume": "/new/checkpoint",
                "learning_rate": 3e-4,
                "grad_accum_steps": 8,
            },
            "mix": {"mix_every": 8},
            "data_lineage": {
                "train": {
                    "data_sha256": "e" * 64,
                    "data_files": [
                        {
                            "name": "train-00000.jsonl",
                            "size_bytes": 10,
                            "sha256": "f" * 64,
                        }
                    ],
                },
                "eval": {
                    "data_sha256": "1" * 64,
                    "data_files": [
                        {
                            "name": "eval-00000.jsonl",
                            "size_bytes": 8,
                            "sha256": "2" * 64,
                        }
                    ],
                },
            },
        }
        saved = copy.deepcopy(current)
        saved["training_config"].update(
            {
                "train_data": "/old/mount/train/*.jsonl",
                "eval_data": "/old/mount/eval/*.jsonl",
                "output_dir": "/old/output",
                "resume": "",
            }
        )
        with mock.patch.object(
            train_rdt,
            "load_checkpoint_metadata",
            return_value=saved,
        ):
            train_rdt._validate_resume_tokenizer_lineage(
                "/checkpoint",
                current,
            )
            drifted = {
                **current,
                "tokenizer_algorithm": {
                    **current["tokenizer_algorithm"],
                    "files_canonical_sha256": "c" * 64,
                },
            }
            with self.assertRaisesRegex(
                ValueError,
                "tokenizer_algorithm differs",
            ):
                train_rdt._validate_resume_tokenizer_lineage(
                    "/checkpoint",
                    drifted,
                )

    def test_resume_rejects_model_training_and_shard_byte_drift(self) -> None:
        current = {
            "config_name": "tiny",
            "rdt_config": {"d_model": 64},
            "omvt_config": None,
            "tokenizer_bundle": {"files_canonical_sha256": "a" * 64},
            "tokenizer_algorithm": {"files_canonical_sha256": "b" * 64},
            "training_config": {
                "train_data": "/new/data",
                "eval_data": "",
                "output_dir": "/new/output",
                "resume": "/checkpoint",
                "learning_rate": 3e-4,
                "optimizer": "adamw",
                "seed": 42,
            },
            "mix": None,
            "data_lineage": {
                "train": {
                    "data_sha256": "c" * 64,
                    "data_files": [
                        {
                            "name": "train.jsonl",
                            "size_bytes": 99,
                            "sha256": "d" * 64,
                        }
                    ],
                }
            },
        }
        mutations = (
            ("rdt_config", "rdt_config", {"d_model": 128}),
            (
                "training_config",
                "training_config",
                {
                    **current["training_config"],
                    "learning_rate": 1e-4,
                },
            ),
            (
                "data_lineage",
                "data_lineage",
                {
                    "train": {
                        "data_sha256": "e" * 64,
                        "data_files": current["data_lineage"]["train"][
                            "data_files"
                        ],
                    }
                },
            ),
        )
        for expected, key, value in mutations:
            with self.subTest(expected=expected):
                saved = copy.deepcopy(current)
                saved[key] = value
                with (
                    mock.patch.object(
                        train_rdt,
                        "load_checkpoint_metadata",
                        return_value=saved,
                    ),
                    self.assertRaisesRegex(ValueError, expected),
                ):
                    train_rdt._validate_resume_tokenizer_lineage(
                        "/checkpoint",
                        current,
                    )

    def test_resume_rejects_legacy_checkpoint_without_data_lineage(self) -> None:
        current = {
            "config_name": "tiny",
            "rdt_config": {"d_model": 64},
            "omvt_config": None,
            "tokenizer_bundle": {"files_canonical_sha256": "a" * 64},
            "tokenizer_algorithm": {"files_canonical_sha256": "b" * 64},
            "training_config": {
                "train_data": "/data",
                "eval_data": "",
                "output_dir": "/output",
                "resume": "/checkpoint",
                "learning_rate": 3e-4,
            },
            "mix": None,
            "data_lineage": {"train": {"data_sha256": "c" * 64}},
        }
        saved = {key: value for key, value in current.items() if key != "data_lineage"}
        with (
            mock.patch.object(
                train_rdt,
                "load_checkpoint_metadata",
                return_value=saved,
            ),
            self.assertRaisesRegex(ValueError, "data_lineage differs"),
        ):
            train_rdt._validate_resume_tokenizer_lineage(
                "/checkpoint",
                current,
            )

    def test_learning_framework_flags_reach_training_config(self) -> None:
        args = train_rdt.parse_args([
            "--config", "two_stage_tiny",
            "--smoke",
            "--optimizer", "muon",
            "--adam-use-atan2",
            "--muon-momentum", "0.9",
            "--muon-ns-steps", "3",
            "--lr-schedule", "wsd",
            "--wsd-stable-ratio", "0.7",
            "--wsd-decay-shape", "linear",
            "--min-lr-ratio", "0.05",
            "--lr-decay-steps", "123",
            "--recurrent-steps-start", "2",
            "--recurrent-steps-ramp", "99",
        ])
        model_cfg = train_rdt._build_model_cfg(args)
        train_cfg = train_rdt._build_train_cfg(args, model_cfg)
        self.assertEqual(train_cfg.optimizer, "muon")
        self.assertTrue(train_cfg.adam_use_atan2)
        self.assertEqual(train_cfg.muon_momentum, 0.9)
        self.assertEqual(train_cfg.muon_ns_steps, 3)
        self.assertEqual(train_cfg.lr_schedule, "wsd")
        self.assertEqual(train_cfg.wsd_stable_ratio, 0.7)
        self.assertEqual(train_cfg.wsd_decay_shape, "linear")
        self.assertEqual(train_cfg.min_lr_ratio, 0.05)
        self.assertEqual(train_cfg.lr_decay_steps, 123)
        self.assertEqual(train_cfg.recurrent_steps_start, 2)
        self.assertEqual(train_cfg.recurrent_steps_ramp, 99)

    def test_auto_mamba_prefers_official_on_cuda_linux(self) -> None:
        cfg = train_rdt.CONFIG_CHOICES["two_stage_tiny"]()
        self.assertFalse(cfg.use_official_mamba)
        with (
            mock.patch.object(train_rdt.platform, "system", return_value="Linux"),
            mock.patch.object(train_rdt.torch.cuda, "is_available", return_value=True),
            mock.patch.object(train_rdt, "official_available", return_value=True),
        ):
            resolved = train_rdt._resolve_mamba_backend(
                cfg,
                "auto",
                device="cuda",
            )
        self.assertTrue(resolved.use_official_mamba)

    def test_auto_mamba_uses_naive_on_macos(self) -> None:
        cfg = train_rdt.CONFIG_CHOICES["two_stage_pretrain"]()
        self.assertTrue(cfg.use_official_mamba)
        with (
            mock.patch.object(train_rdt.platform, "system", return_value="Darwin"),
            mock.patch.object(train_rdt.torch.cuda, "is_available", return_value=True),
            mock.patch.object(train_rdt, "official_available", return_value=True),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            resolved = train_rdt._resolve_mamba_backend(
                cfg,
                "auto",
                device="cuda",
            )
        self.assertFalse(resolved.use_official_mamba)

    def test_auto_mamba_uses_naive_on_non_linux_cuda_host(self) -> None:
        cfg = train_rdt.CONFIG_CHOICES["two_stage_pretrain"]()
        with (
            mock.patch.object(train_rdt.platform, "system", return_value="Windows"),
            mock.patch.object(train_rdt.torch.cuda, "is_available", return_value=True),
            mock.patch.object(train_rdt, "official_available", return_value=True),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            resolved = train_rdt._resolve_mamba_backend(
                cfg,
                "auto",
                device="cuda",
            )
        self.assertFalse(resolved.use_official_mamba)

    def test_auto_mamba_uses_naive_for_cpu_target(self) -> None:
        cfg = train_rdt.CONFIG_CHOICES["two_stage_pretrain"]()
        stderr = io.StringIO()
        with (
            mock.patch.object(train_rdt.platform, "system", return_value="Linux"),
            mock.patch.object(train_rdt.torch.cuda, "is_available", return_value=True),
            mock.patch.object(train_rdt, "official_available", return_value=True),
            contextlib.redirect_stderr(stderr),
        ):
            resolved = train_rdt._resolve_mamba_backend(
                cfg,
                "auto",
                device="cpu",
            )
        self.assertFalse(resolved.use_official_mamba)
        self.assertIn("target device is cpu", stderr.getvalue())
        self.assertIn("CPU-only/CPU-target", stderr.getvalue())

    def test_official_mamba_fails_fast_without_cuda(self) -> None:
        cfg = train_rdt.CONFIG_CHOICES["two_stage_pretrain"]()
        with (
            mock.patch.object(train_rdt.platform, "system", return_value="Linux"),
            mock.patch.object(train_rdt.torch.cuda, "is_available", return_value=False),
            mock.patch.object(train_rdt, "official_available", return_value=True),
        ):
            with self.assertRaisesRegex(RuntimeError, "CUDA is not available"):
                train_rdt._resolve_mamba_backend(cfg, "official", device="cuda")

    def test_cached_decode_requires_naive_mamba(self) -> None:
        cfg = train_rdt.CONFIG_CHOICES["two_stage_pretrain"]()
        with self.assertRaisesRegex(ValueError, "--use-cache requires --mamba naive"):
            train_rdt._resolve_mamba_backend(
                cfg,
                "auto",
                use_cache=True,
                context="scripts.generate",
            )
        resolved = train_rdt._resolve_mamba_backend(
            cfg,
            "naive",
            use_cache=True,
            context="scripts.generate",
        )
        self.assertFalse(resolved.use_official_mamba)

    def test_requires_data_or_smoke(self) -> None:
        # No --data, no --smoke ⇒ must return non-zero exit code with a
        # diagnostic message before any expensive setup (model alloc,
        # output-dir creation, distributed init).
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rc = train_rdt.main(["--config", "tiny", "--output", tmp])
            self.assertEqual(rc, 2)
            msg = stderr.getvalue()
            self.assertIn("--data is required", msg)
            # And no checkpoint dirs should have been created.
            import os
            self.assertEqual(os.listdir(tmp), [])

    def test_resume_path_must_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rc = train_rdt.main([
                    "--config", "tiny",
                    "--output", tmp,
                    "--smoke",
                    "--resume", "/definitely/does/not/exist",
                ])
            self.assertEqual(rc, 2)
            self.assertIn("--resume", stderr.getvalue())

    def test_tokenizer_bundle_path_must_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rc = train_rdt.main([
                    "--config", "tiny",
                    "--output", tmp,
                    "--smoke",
                    "--tokenizer-bundle", "/definitely/not/a/bundle",
                ])
            self.assertEqual(rc, 2)
            self.assertIn("--tokenizer-bundle", stderr.getvalue())

    def test_non_smoke_requires_bundle_and_producer_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = f"{tmp}/train.jsonl"
            with open(data, "w", encoding="utf-8") as fh:
                fh.write(
                    '{"input_ids":[2,3],"attention_mask":[1,1],'
                    '"labels":[-100,3]}\n'
                )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rc = train_rdt.main([
                    "--config", "tiny",
                    "--output", tmp,
                    "--data", data,
                ])
            self.assertEqual(rc, 2)
            self.assertIn("--tokenizer-bundle is required", stderr.getvalue())

    def test_production_cannot_disable_resume_stream_fast_forward(self) -> None:
        args = train_rdt.parse_args([
            "--data", "/not/read/because/flag/fails/first.jsonl",
            "--no-resume-skip-data",
        ])
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            rc = train_rdt._validate_args(args)
        self.assertEqual(rc, 2)
        self.assertIn("smoke-only", stderr.getvalue())

    def test_mix_accepts_native_ocr_alignment_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            receipt = f"{tmp}/ocr_data_contract.json"
            with open(receipt, "w", encoding="utf-8") as fh:
                fh.write('{"kind":"pretokenized_ocr_alignment"}')
            args = train_rdt.parse_args([
                "--data", f"{tmp}/train.jsonl",
                "--data-receipt", f"{tmp}/train.receipt.json",
                "--tokenizer-bundle", f"{tmp}/bundle",
                "--mix-data", f"{tmp}/ocr.jsonl",
                "--mix-data-receipt", receipt,
                "--mix-every", "4",
                "--multimodal",
            ])
            bundle_identity = {"files_canonical_sha256": "a" * 64}
            algorithm_identity = {"files_canonical_sha256": "b" * 64}
            train_lineage = {
                "tokenizer_bundle": bundle_identity,
                "tokenizer_algorithm": algorithm_identity,
                "data_sha256": "c" * 64,
            }
            ocr_lineage = {
                "kind": "pretokenized_ocr_alignment",
                "data_sha256": "d" * 64,
            }
            with (
                mock.patch.object(
                    train_rdt,
                    "_tokenizer_bundle_metadata",
                    return_value=bundle_identity,
                ),
                mock.patch.object(
                    train_rdt,
                    "tokenizer_algorithm_contract",
                    return_value=algorithm_identity,
                ),
                mock.patch.object(
                    train_rdt,
                    "load_and_validate_pretraining_data_contract",
                    return_value=train_lineage,
                ),
                mock.patch(
                    "Tokenizer.unified.bundle.TokenizerBundle.from_dir",
                    return_value=mock.Mock(tokenizer=object()),
                ),
                mock.patch(
                    "Model.ocr.tokenization.native_tokenization_contract",
                    return_value={"target_encoding": "native"},
                ),
                mock.patch(
                    "Model.ocr.alignment_contract."
                    "load_and_validate_ocr_alignment_data_contract",
                    return_value=ocr_lineage,
                ) as validate_ocr,
            ):
                lineage = train_rdt._validated_data_lineage(args)
            self.assertEqual(lineage["mix"], ocr_lineage)
            validate_ocr.assert_called_once_with(
                receipt,
                f"{tmp}/ocr.jsonl",
                {"target_encoding": "native"},
            )

    def test_receipt_backed_dataloaders_require_persisted_representation(self) -> None:
        lineage = {
            "train": {"kind": "pretokenized_rdt_jsonl"},
            "eval": {"kind": "pretokenized_rdt_jsonl"},
            "mix": {"kind": "pretokenized_ocr_alignment"},
        }
        self.assertEqual(
            train_rdt._dataloader_contract_flags(lineage, "train"),
            {
                "require_precomputed_morphology": True,
                "require_verified_images": False,
            },
        )
        self.assertEqual(
            train_rdt._dataloader_contract_flags(lineage, "eval"),
            {
                "require_precomputed_morphology": True,
                "require_verified_images": False,
            },
        )
        self.assertEqual(
            train_rdt._dataloader_contract_flags(lineage, "mix"),
            {
                "require_precomputed_morphology": True,
                "require_verified_images": True,
            },
        )
        row_without_morphology = {
            "input_ids": [2, 3],
            "attention_mask": [1, 1],
            "labels": [-100, 3],
        }
        collator = train_rdt.PretrainingCollator(
            require_precomputed_morphology=True
        )
        with self.assertRaisesRegex(
            ValueError,
            "persist word_pos and morph_depth",
        ):
            collator([row_without_morphology])

    def test_rank_zero_preflight_broadcasts_lineage_without_worker_io(self) -> None:
        lineage = {"train": {"data_sha256": "a" * 64}}
        root_callback = mock.Mock(return_value=lineage)
        with mock.patch.object(
            train_rdt.torch.distributed,
            "broadcast_object_list",
        ) as broadcast:
            root_result = train_rdt._rank_zero_broadcast_result(
                root_callback,
                rank=0,
                world_size=2,
            )
        self.assertEqual(root_result, lineage)
        root_callback.assert_called_once_with()
        broadcast.assert_called_once()

        worker_callback = mock.Mock(
            side_effect=AssertionError("worker must not scan data")
        )

        def receive_success(box, *, src):
            self.assertEqual(src, 0)
            box[0] = {"ok": True, "value": lineage}

        with mock.patch.object(
            train_rdt.torch.distributed,
            "broadcast_object_list",
            side_effect=receive_success,
        ):
            worker_result = train_rdt._rank_zero_broadcast_result(
                worker_callback,
                rank=1,
                world_size=2,
            )
        self.assertEqual(worker_result, lineage)
        worker_callback.assert_not_called()

    def test_rank_zero_preflight_broadcasts_one_failure_without_barrier(self) -> None:
        root_callback = mock.Mock(side_effect=OSError("NAS read failed"))
        with mock.patch.object(
            train_rdt.torch.distributed,
            "broadcast_object_list",
        ) as broadcast:
            with self.assertRaisesRegex(ValueError, "NAS read failed") as root_error:
                train_rdt._rank_zero_broadcast_result(
                    root_callback,
                    rank=0,
                    world_size=2,
                )
        broadcast.assert_called_once()

        def receive_failure(box, *, src):
            self.assertEqual(src, 0)
            box[0] = {
                "ok": False,
                "error_type": "OSError",
                "error": "NAS read failed",
            }

        with mock.patch.object(
            train_rdt.torch.distributed,
            "broadcast_object_list",
            side_effect=receive_failure,
        ):
            with self.assertRaisesRegex(ValueError, "NAS read failed") as worker_error:
                train_rdt._rank_zero_broadcast_result(
                    mock.Mock(
                        side_effect=AssertionError("worker must not scan data")
                    ),
                    rank=1,
                    world_size=2,
                )
        self.assertEqual(str(root_error.exception), str(worker_error.exception))

    def test_empty_shard_glob_fails_fast(self) -> None:
        # An empty glob (typo'd shard pattern) must abort with exit 2
        # *before* the model is allocated.
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rc = train_rdt.main([
                    "--config", "tiny",
                    "--output", tmp,
                    "--data", f"{tmp}/no_such_shards_*.jsonl",
                ])
            self.assertEqual(rc, 2)
            self.assertIn("zero shards", stderr.getvalue())

    def test_missing_exact_data_file_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rc = train_rdt.main([
                    "--config", "tiny",
                    "--output", tmp,
                    "--data", f"{tmp}/missing.jsonl",
                ])
            self.assertEqual(rc, 2)
            self.assertIn("zero shards", stderr.getvalue())

    def test_empty_eval_shard_glob_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            import os

            data = os.path.join(tmp, "train.jsonl")
            with open(data, "w", encoding="utf-8") as fh:
                fh.write(
                    '{"input_ids":[2,3],"attention_mask":[1,1],'
                    '"labels":[-100,3]}\n'
                )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rc = train_rdt.main([
                    "--config", "tiny",
                    "--output", tmp,
                    "--data", data,
                    "--eval-data", f"{tmp}/no_eval_shards_*.jsonl",
                ])
            self.assertEqual(rc, 2)
            self.assertIn("--eval-data", stderr.getvalue())

    def test_low_supervised_rate_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            import os

            data = os.path.join(tmp, "train.jsonl")
            with open(data, "w", encoding="utf-8") as fh:
                fh.write(
                    '{"input_ids":[2,3],"attention_mask":[1,1],'
                    '"labels":[-100,-100]}\n'
                )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rc = train_rdt.main([
                    "--config", "tiny",
                    "--output", tmp,
                    "--data", data,
                ])
            self.assertEqual(rc, 2)
            self.assertIn("supervised_rate", stderr.getvalue())

    def test_min_supervised_rate_zero_disables_data_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            import os

            data = os.path.join(tmp, "train.jsonl")
            with open(data, "w", encoding="utf-8") as fh:
                fh.write(
                    '{"input_ids":[2,3],"attention_mask":[1,1],'
                    '"labels":[-100,-100]}\n'
                )
            args = train_rdt.parse_args([
                "--config", "tiny",
                "--output", tmp,
                "--data", data,
                "--min-supervised-rate", "0",
            ])
            self.assertEqual(train_rdt._validate_args(args), 0)

    def test_smoke_runs_without_data(self) -> None:
        # --smoke explicitly opts into the synthetic batch generator.
        with tempfile.TemporaryDirectory() as tmp:
            rc = train_rdt.main([
                "--config", "tiny", "--smoke", "--output", tmp,
            ])
            self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
