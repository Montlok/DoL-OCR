# -*- coding: utf-8 -*-

"""Focused contract tests for :mod:`scripts.train_sft`.

These tests guard the SFT-specific P1: every SFT batch must omit synthetic
sequential positions and make the model derive ``boundary_v1`` positions, and
a resume may not silently cross that semantic boundary.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from Model.config import IGNORE_INDEX, RDTConfig, TrainingConfig
from Model.model import RDTForCausalLM
from Model.ocr.position_contract import (
    BOUNDARY_V1,
    LEGACY_SEQUENTIAL_V0,
    OCR_POSITION_CONTRACT_METADATA_VERSION,
)
from scripts import train_sft


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
        max_seq_len=16,
        bidirectional=False,
    )


class SFTPositionContractTest(unittest.TestCase):
    def test_smoke_and_real_loaders_emit_only_boundary_contract(self) -> None:
        train_cfg = TrainingConfig(seq_len=128, micro_batch_size=1)
        smoke_batch = next(iter(train_sft._smoke_loader(train_cfg)))
        self.assertEqual(smoke_batch["position_contract"], BOUNDARY_V1)
        self.assertNotIn("word_pos", smoke_batch)
        self.assertNotIn("morph_depth", smoke_batch)

        row = {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "sft.jsonl"
            data_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            fake_bundle = SimpleNamespace(
                encode=lambda text: [300 + (ord(ch) % 97) for ch in text]
            )
            args = SimpleNamespace(
                tokenizer="unused-tokenizer",
                data=str(data_path),
                seed=42,
                num_workers=0,
            )
            with mock.patch(
                "Tokenizer.unified.bundle.TokenizerBundle.from_dir",
                return_value=fake_bundle,
            ):
                real_batch = next(iter(train_sft._real_loader(args, train_cfg)))

        self.assertEqual(real_batch["position_contract"], BOUNDARY_V1)
        self.assertNotIn("word_pos", real_batch)
        self.assertNotIn("morph_depth", real_batch)

    def test_training_batch_uses_model_boundary_position_semantics(self) -> None:
        cfg = _tiny_rdt()
        ids = [
            cfg.bos_id,
            cfg.word_boundary_id,
            300,
            cfg.morpheme_boundary_id,
            301,
            cfg.word_boundary_id,
            302,
            cfg.eos_id,
        ]
        row = {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "labels": [IGNORE_INDEX] * (len(ids) - 2) + ids[-2:],
        }
        batch = train_sft._sft_collator(
            TrainingConfig(seq_len=16, micro_batch_size=1)
        )([row])
        self.assertNotIn("word_pos", batch)
        self.assertNotIn("morph_depth", batch)

        model = RDTForCausalLM(cfg).eval()
        original = model._morph_info_for_position_contract
        observed: dict[str, object] = {}

        def capture(input_ids, attention_mask, position_contract):
            word_pos, morph_depth = original(
                input_ids,
                attention_mask,
                position_contract,
            )
            observed["contract"] = position_contract
            observed["word_pos"] = word_pos.detach().clone()
            observed["morph_depth"] = morph_depth.detach().clone()
            return word_pos, morph_depth

        with (
            mock.patch.object(
                model,
                "_morph_info_for_position_contract",
                side_effect=capture,
            ) as derive_mock,
            torch.no_grad(),
        ):
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                word_pos=batch.get("word_pos"),
                morph_depth=batch.get("morph_depth"),
                position_contract=batch["position_contract"],
            )

        derive_mock.assert_called_once()
        self.assertEqual(observed["contract"], BOUNDARY_V1)
        self.assertEqual(
            observed["word_pos"].tolist(),
            [[0, 0, 0, 0, 0, 1, 1, 1]],
        )
        self.assertEqual(
            observed["morph_depth"].tolist(),
            [[0, 0, 0, 1, 1, 0, 0, 0]],
        )
        self.assertNotEqual(
            observed["word_pos"].tolist(),
            [list(range(len(ids)))],
        )
        self.assertTrue(bool(torch.isfinite(output["loss"])))


class SFTCheckpointContractTest(unittest.TestCase):
    def test_checkpoint_metadata_pins_rdt_and_boundary_contract(self) -> None:
        args = train_sft.parse_args(["--smoke", "--config", "tiny"])
        cfg = _tiny_rdt()
        metadata = train_sft._sft_checkpoint_metadata(args, cfg, final=True)

        self.assertEqual(metadata["phase"], "sft")
        self.assertEqual(metadata["rdt_config"], asdict(cfg))
        self.assertEqual(metadata["ocr_position_contract"], BOUNDARY_V1)
        self.assertEqual(
            metadata["ocr_position_contract_version"],
            OCR_POSITION_CONTRACT_METADATA_VERSION,
        )
        self.assertIs(metadata["final"], True)

    def test_resume_rejects_position_contract_before_model(self) -> None:
        current_cfg = train_sft._build_model_cfg(
            train_sft.parse_args(["--smoke", "--config", "tiny"])
        )
        cases = {
            "missing": {"rdt_config": asdict(current_cfg)},
            "different": {
                "rdt_config": asdict(current_cfg),
                "ocr_position_contract": LEGACY_SEQUENTIAL_V0,
                "ocr_position_contract_version": (
                    OCR_POSITION_CONTRACT_METADATA_VERSION
                ),
            },
            "wrong_version": {
                "rdt_config": asdict(current_cfg),
                "ocr_position_contract": BOUNDARY_V1,
                "ocr_position_contract_version": 999,
            },
        }

        for name, metadata in cases.items():
            stderr = io.StringIO()
            with (
                self.subTest(case=name),
                mock.patch.object(
                    train_sft,
                    "load_checkpoint_metadata",
                    return_value=metadata,
                ),
                mock.patch.object(train_sft, "RDTForCausalLM") as model_ctor,
                contextlib.redirect_stderr(stderr),
            ):
                rc = train_sft.main(
                    [
                        "--smoke",
                        "--config",
                        "tiny",
                        "--resume",
                        "/untrusted/checkpoint",
                    ]
                )

            self.assertEqual(rc, 2)
            model_ctor.assert_not_called()
            self.assertIn("unsafe SFT resume", stderr.getvalue())

    def test_resume_rejects_rdt_config_drift(self) -> None:
        args = train_sft.parse_args(["--smoke", "--config", "tiny"])
        cfg = train_sft._build_model_cfg(args)
        metadata = train_sft._sft_checkpoint_metadata(args, cfg)
        metadata["rdt_config"] = dict(metadata["rdt_config"])
        metadata["rdt_config"]["recurrent_steps"] += 1

        with (
            mock.patch.object(
                train_sft,
                "load_checkpoint_metadata",
                return_value=metadata,
            ),
            self.assertRaisesRegex(ValueError, "rdt_config differs"),
        ):
            train_sft._validate_sft_resume_metadata("/checkpoint", cfg)


if __name__ == "__main__":
    unittest.main()
