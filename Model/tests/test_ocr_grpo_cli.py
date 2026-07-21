# -*- coding: utf-8 -*-

"""CLI contract tests for OCR RL validation and locked-golden evaluation."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch

from Model.config import OMVTConfig, RDTConfig
from Model.model import RDTForCausalLM
from Model.omvt import OMVTInjector
from Model.posttrain.checkpointing import OCR_GRPO_CONTRACT_VERSION


def _rdt() -> RDTConfig:
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
        recurrent_steps=1,
        mamba_d_state=8,
        mamba_expand=2,
        mamba_headdim=16,
        use_official_mamba=False,
        max_seq_len=24,
    )


def _omvt() -> OMVTConfig:
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


def _checkpoint(root: Path, metadata: dict) -> Path:
    rdt_cfg, omvt_cfg = _rdt(), _omvt()
    model = RDTForCausalLM(rdt_cfg)
    model.vision._omvt_cfg = omvt_cfg
    model.vision.omvt = OMVTInjector(rdt_cfg, omvt_cfg)
    step = root / "step_00000000"
    step.mkdir(parents=True)
    torch.save(model.state_dict(), step / "model.pt")
    payload = {
        "rdt_config": asdict(rdt_cfg),
        "omvt_config": asdict(omvt_cfg),
        **metadata,
    }
    torch.save({"step": 0, "metadata": payload}, step / "meta.pt")
    (step / "COMPLETE").write_text("step=0\n", encoding="ascii")
    return step


def _image_and_row(root: Path, name: str, split: str) -> dict:
    from PIL import Image

    image = root / f"{name}.png"
    color = tuple(hashlib.sha256(name.encode("utf-8")).digest()[:3])
    Image.new("RGB", (8, 12), color).save(image)
    return {
        "id": name,
        "group_id": f"group-{name}",
        "split": split,
        "image": image.name,
        "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "reference": "a",
        "domain": "photo",
    }


class _FakeTokenizer:
    vocab = {"a": 300, "<0x61>": 301}
    id_to_token = {300: "a", 301: "<0x61>"}

    @staticmethod
    def decode(ids):
        return "".join("a" for token in ids if token in {300, 301})


class _FakeBundle:
    tokenizer = _FakeTokenizer()

    @staticmethod
    def validate():
        return []

    @staticmethod
    def encode(_text, add_bos=False, add_eos=False):
        del add_bos, add_eos
        return [300]


class OCRGRPOCLITest(unittest.TestCase):
    def test_manifest_validator_reports_lossless_completion_budget(self):
        from scripts.validate_ocr_rl_manifests import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = _checkpoint(root / "checkpoint", {})
            paths = {}
            for name, split in (
                ("train", "rl_train"),
                ("validation", "rl_val"),
                ("golden", "golden"),
            ):
                manifest = root / f"{name}.jsonl"
                manifest.write_text(
                    json.dumps(_image_and_row(root, name, split)) + "\n",
                    encoding="utf-8",
                )
                paths[name] = manifest
            output = io.StringIO()
            with (
                patch(
                    "Tokenizer.unified.bundle.TokenizerBundle.from_dir",
                    return_value=_FakeBundle(),
                ),
                patch(
                    "scripts.build_ocr_data.make_ocr_target_encoder",
                    return_value=lambda _text: [300],
                ),
                contextlib.redirect_stdout(output),
            ):
                rc = main(
                    [
                        "--checkpoint", str(checkpoint),
                        "--tokenizer", str(root / "tokenizer"),
                        "--train", str(paths["train"]),
                        "--validation", str(paths["validation"]),
                        "--golden", str(paths["golden"]),
                        "--image-root", str(root),
                    ]
                )
            self.assertEqual(rc, 0)
            report = json.loads(output.getvalue())
            self.assertEqual(report["recommended_max_new_tokens"], 2)
            self.assertEqual(report["splits"]["golden"]["samples"], 1)
            self.assertTrue(report["splits"]["golden"]["locked"])
            self.assertNotIn(
                "completion_tokens_including_eos", report["splits"]["golden"]
            )

    def test_locked_golden_cli_accepts_only_registered_selected_artifacts(self):
        from scripts.eval_ocr_grpo import _vocab_sha256, main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            golden = root / "golden.jsonl"
            golden.write_text(
                json.dumps(_image_and_row(root, "golden", "golden")) + "\n",
                encoding="utf-8",
            )
            tokenizer_dir = root / "tokenizer"
            tokenizer_dir.mkdir()
            (tokenizer_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
            checkpoint_root = root / "best"
            step_path = checkpoint_root / "step_00000000"
            source_checkpoint = "visual-source"
            reference = _checkpoint(
                root / "reference",
                {
                    "phase": "grpo_reference",
                    "contract_version": OCR_GRPO_CONTRACT_VERSION,
                    "source_checkpoint": source_checkpoint,
                    "immutable": True,
                },
            )
            metadata = {
                "phase": "grpo",
                "contract_version": OCR_GRPO_CONTRACT_VERSION,
                "task": "ocr",
                "source_checkpoint": source_checkpoint,
                "reference_checkpoint": str(reference),
                "golden_manifest": str(golden),
                "golden_split": "golden",
                "grpo_config": {"max_new_tokens": 2, "recurrent_steps": 1},
                "health_state": {
                    "best_checkpoint": str(step_path),
                    "best_val_grapheme_cer": 0.2,
                    "best_val_step": 1,
                    "best_val_eligible": True,
                },
                "data_contract": {
                    "golden_manifest_sha256": hashlib.sha256(
                        golden.read_bytes()
                    ).hexdigest(),
                    "tokenizer_vocab_sha256": _vocab_sha256(_FakeTokenizer()),
                    "tokenizer_manifest_sha256": hashlib.sha256(
                        (tokenizer_dir / "manifest.json").read_bytes()
                    ).hexdigest(),
                    "reference_model_sha256": hashlib.sha256(
                        (reference / "model.pt").read_bytes()
                    ).hexdigest(),
                    "image_root": str(root),
                },
            }
            checkpoint = _checkpoint(checkpoint_root, metadata)
            real = {
                "grapheme_cer": 0.1,
                "norm_cer": 0.1,
                "raw_cer": 0.1,
                "wer": 0.1,
                "line_exact": 0.9,
                "eos_rate": 1.0,
                "invalid_output_rate": 0.0,
                "samples": 1.0,
            }
            blank = {**real, "grapheme_cer": 0.8}
            reference_metrics = {**real, "grapheme_cer": 0.3}
            rendered = io.StringIO()
            with (
                patch(
                    "Tokenizer.unified.bundle.TokenizerBundle.from_dir",
                    return_value=_FakeBundle(),
                ),
                patch(
                    "scripts.eval_ocr_grpo.evaluate_ocr_manifest",
                    side_effect=[real, blank, reference_metrics],
                ),
                contextlib.redirect_stdout(rendered),
            ):
                rc = main(
                    [
                        "--checkpoint", str(checkpoint),
                        "--tokenizer", str(tokenizer_dir),
                        "--golden-manifest", str(golden),
                        "--device", "cpu",
                    ]
                )
            self.assertEqual(rc, 0)
            report = json.loads(rendered.getvalue())
            self.assertAlmostEqual(
                report["reference_grapheme_cer_improvement"], 0.2
            )
            self.assertTrue(report["gates"]["reference_improvement_pass"])


if __name__ == "__main__":
    unittest.main()
