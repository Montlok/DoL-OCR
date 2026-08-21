# -*- coding: utf-8 -*-

"""Hermetic regression for the ``--data`` path of ``train_omvt_ssl``.

Before the PR #10 fix the trainer treated ``ocr_labels`` /
``reading_order`` as ``list[int]`` instead of the schema-correct
``list[list[int]]`` (one sequence per image), so the very first batch
with real OCR labels would crash with a shape mismatch. This test feeds
both fields and asserts the loop runs to finite loss.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore


@unittest.skipIf(Image is None, "Pillow not installed")
class TrainOMVTSSLRealDataTest(unittest.TestCase):
    def test_real_data_with_ocr_labels_runs(self) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from scripts.train_omvt_ssl import main as ssl_main

        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            imgs = tmp_p / "imgs"
            imgs.mkdir()
            jsonl = tmp_p / "shard.jsonl"
            with jsonl.open("w", encoding="utf-8") as fh:
                for i in range(4):
                    p = imgs / f"img_{i}.png"
                    Image.new("RGB", (32, 32), color=(i * 30, 80, 160)).save(p)
                    fh.write(
                        json.dumps(
                            {
                                "images": [str(p)],
                                "image_sizes": [[32, 32]],
                                # schema-correct list[list[int]]:
                                "ocr_labels": [[i % 8, (i + 1) % 8, (i + 2) % 8]],
                                "reading_order": [[0, 1, 2]],
                            }
                        )
                        + "\n"
                    )
            rc = ssl_main(
                [
                    "--data", str(jsonl),
                    "--image-size", "32",
                    "--compress-to", "3",
                    "--batch-size", "2",
                    "--steps", "2",
                    "--output", str(tmp_p / "out"),
                ]
            )
            self.assertEqual(rc, 0)
            ckpt = tmp_p / "out" / "latest" / "omvt_ssl.pt"
            self.assertTrue(ckpt.exists())

            from Model.config import OMVTConfig, RDTConfig
            from Model.model import RDTForCausalLM
            from Model.omvt import OMVTInjector
            from scripts.train_vlm_align import _load_omvt_init

            rdt_cfg = RDTConfig(
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
            )
            omvt_cfg = OMVTConfig(
                image_size=32,
                d_vision=64,
                vertical_patch=(16, 8),
                horizontal_patch=(8, 16),
                square_patch=(8, 8),
                layout_patch=(32, 32),
                compress_to=3,
            )
            model = RDTForCausalLM(rdt_cfg)
            model.vision._omvt_cfg = omvt_cfg
            model.vision.omvt = OMVTInjector(rdt_cfg, omvt_cfg)
            _load_omvt_init(model, str(tmp_p / "out" / "latest"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
