# -*- coding: utf-8 -*-

from __future__ import annotations

import copy
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from Model.config import OMVTConfig, RDTConfig
from Model.model import RDTForCausalLM
from Model.ocr.visual_input_contract import (
    DOL_OCR_ANYRES_V2,
    DOL_OCR_LINE_LETTERBOX_224_V1,
)
from Model.omvt import OMVTInjector
from Model.posttrain.checkpointing import reconstruct_policy_from_checkpoint


def _rdt_config() -> RDTConfig:
    return RDTConfig(
        d_model=8,
        n_heads=1,
        head_dim=8,
        kv_lora_rank=4,
        rope_head_dim=4,
        nope_head_dim=4,
        ffn_hidden=16,
        ffn_multiple=8,
        n_prelude=1,
        n_coda=0,
        mamba_per_block=0,
        attn_per_block=1,
        recurrent_steps=1,
        mamba_d_state=4,
        mamba_expand=2,
        mamba_headdim=8,
        use_official_mamba=False,
        max_seq_len=16,
        bidirectional=False,
    )


def _omvt_config() -> OMVTConfig:
    return OMVTConfig(
        image_size=16,
        vertical_patch=(8, 4),
        horizontal_patch=(4, 8),
        square_patch=(4, 4),
        layout_patch=(16, 16),
        d_vision=8,
        n_vertical_layers=1,
        n_horizontal_layers=1,
        n_local_attn_layers=1,
        n_layout_layers=1,
        vision_n_heads=2,
        vision_ffn_hidden=16,
        vision_dropout=0.0,
        compress_to=3,
        compressor_layers=1,
        compressor_heads=1,
    )


def _write_checkpoint(
    root: Path,
    *,
    anyres: bool,
) -> tuple[Path, dict, dict[str, torch.Tensor]]:
    rdt_cfg = _rdt_config()
    omvt_cfg = _omvt_config()
    model = RDTForCausalLM(rdt_cfg)
    model.vision._omvt_cfg = omvt_cfg
    model.vision.omvt = OMVTInjector(rdt_cfg, omvt_cfg)
    metadata = {
        "rdt_config": asdict(rdt_cfg),
        "omvt_config": asdict(omvt_cfg),
    }
    if anyres:
        receipt = model.vision.install_native_detail_tower(
            omvt_cfg,
            max_detail_tokens=4,
            ratio=5,
            initialize_from_legacy=True,
        )
        model.install_vision_cross_attention(
            memory_dim=omvt_cfg.d_vision,
            n_heads=1,
            dropout=0.0,
        )
        metadata.update(
            {
                "ocr_visual_input_contract": DOL_OCR_ANYRES_V2,
                "ocr_visual_input_contract_version": 2,
                "native_detail_config": {
                    "max_detail_tokens": 4,
                    "source_tokens_per_detail_token": 5,
                },
                "vision_cross_attention_config": {
                    "memory_dim": omvt_cfg.d_vision,
                    "n_heads": 1,
                    "dropout": 0.0,
                },
                "native_migration_receipt": {
                    "payload": receipt.canonical_payload(),
                    "canonical_sha256": receipt.canonical_sha256,
                },
            }
        )
    else:
        metadata.update(
            {
                "ocr_visual_input_contract": DOL_OCR_LINE_LETTERBOX_224_V1,
                "ocr_visual_input_contract_version": 1,
            }
        )
    step = root / ("v2" if anyres else "v1")
    step.mkdir()
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    torch.save(state, step / "model.pt")
    torch.save({"metadata": metadata}, step / "meta.pt")
    return step, metadata, state


class OMVTV2CheckpointingTest(unittest.TestCase):
    def test_anyres_v2_roundtrip_strictly_reconstructs_every_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            step, metadata, expected = _write_checkpoint(Path(tmp), anyres=True)
            self.assertTrue(
                metadata["native_migration_receipt"]["payload"][
                    "initialized_from_legacy"
                ]
            )
            restored = reconstruct_policy_from_checkpoint(step, require_vision=True)
            self.assertIsNotNone(restored.model.vision.native_detail_tower)
            self.assertIsNotNone(restored.model.vision_cross_attention)
            self.assertEqual(
                restored.native_detail_config,
                metadata["native_detail_config"],
            )
            self.assertEqual(
                restored.vision_cross_attention_config,
                metadata["vision_cross_attention_config"],
            )
            actual = restored.model.state_dict()
            self.assertEqual(set(actual), set(expected))
            for name, tensor in expected.items():
                self.assertTrue(torch.equal(actual[name], tensor), name)

    def test_v1_reconstruction_never_creates_anyres_modules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            step, _metadata, expected = _write_checkpoint(Path(tmp), anyres=False)
            self.assertFalse(
                any("native_detail_tower" in name for name in expected)
            )
            restored = reconstruct_policy_from_checkpoint(step, require_vision=True)
            self.assertIsNone(restored.model.vision.native_detail_tower)
            self.assertIsNone(restored.model.vision_cross_attention)
            self.assertIsNone(restored.native_detail_config)
            self.assertFalse(
                any(
                    "native_detail_tower" in name
                    or name.startswith("vision_cross_attention.")
                    for name in restored.model.state_dict()
                )
            )

    def test_missing_tampered_and_conflicting_v2_metadata_fail_closed(self) -> None:
        mutations = {
            "missing": lambda metadata: metadata.pop("native_detail_config"),
            "tampered": lambda metadata: metadata[
                "native_migration_receipt"
            ].update(canonical_sha256="0" * 64),
            "conflict": lambda metadata: metadata.update(
                ocr_visual_input_contract=DOL_OCR_LINE_LETTERBOX_224_V1,
                ocr_visual_input_contract_version=1,
            ),
            "shape": lambda metadata: metadata[
                "vision_cross_attention_config"
            ].update(memory_dim=9),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                step, metadata, _state = _write_checkpoint(Path(tmp), anyres=True)
                changed = copy.deepcopy(metadata)
                mutate(changed)
                torch.save({"metadata": changed}, step / "meta.pt")
                with self.assertRaises(ValueError):
                    reconstruct_policy_from_checkpoint(step, require_vision=True)


if __name__ == "__main__":
    unittest.main()
