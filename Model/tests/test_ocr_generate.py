# -*- coding: utf-8 -*-

"""Generate-time image conditioning (OCR prefill) tests.

Covers threading ``pixel_values`` through :meth:`RDTForCausalLM.generate`:
the image must be injected at prefill, must affect the output, and the cached
and cache-free decode paths must agree (the same bit-exact invariant the decode
gate enforces for text).
"""

import unittest

import torch

from Model.config import RDTConfig
from Model.model import RDTForCausalLM


def _two_stage_cfg() -> RDTConfig:
    return RDTConfig(
        d_model=32, n_heads=4, head_dim=8, kv_lora_rank=8, rope_head_dim=4,
        nope_head_dim=4, ffn_hidden=64, ffn_multiple=32, n_prelude=2, n_coda=2,
        recurrent_steps=3, mamba_d_state=8, mamba_expand=2, mamba_headdim=8,
        use_official_mamba=False, max_seq_len=32, core_type="two_stage",
        stage1_mamba_layers=2, stage2_attn_layers=2, recurrent_drift_mode="mhc",
    )


def _image_prompt(cfg: RDTConfig) -> torch.Tensor:
    return torch.tensor(
        [[
            cfg.bos_id,
            cfg.image_start_id,
            cfg.image_patch_id,
            cfg.image_end_id,
            256,
        ]]
    )


class TestGeneratePixelValues(unittest.TestCase):
    def test_cachefree_accepts_pixel_values(self):
        torch.manual_seed(0)
        cfg = _two_stage_cfg()
        model = RDTForCausalLM(cfg, patch_pixels=4).eval()
        prompt = _image_prompt(cfg)
        out = model.generate(
            prompt,
            max_new_tokens=3,
            greedy=True,
            use_cache=False,
            pixel_values=torch.randn(1, 4),
        )
        self.assertEqual(out.shape, (1, prompt.shape[1] + 3))
        self.assertTrue(torch.all(out[:, : prompt.shape[1]] == prompt))

    def test_image_changes_output(self):
        cfg = _two_stage_cfg()
        model = RDTForCausalLM(cfg, patch_pixels=4).eval()
        prompt = _image_prompt(cfg)

        def fake_forward(
            input_ids,
            *,
            steps=None,
            return_logits=True,
            pixel_values=None,
            **kwargs,
        ):
            self.assertTrue(return_logits)
            self.assertIsNotNone(pixel_values)
            self.assertIsNone(steps)
            logits = torch.zeros(
                input_ids.shape[0],
                input_ids.shape[1],
                cfg.vocab_size,
                dtype=torch.float32,
                device=input_ids.device,
            )
            token = 300 if float(pixel_values.sum()) > 0 else 301
            logits[:, -1, token] = 100.0
            return {"logits": logits}

        model.forward = fake_forward  # type: ignore[method-assign]
        a = model.generate(
            prompt, max_new_tokens=1, greedy=True, use_cache=False,
            pixel_values=torch.full((1, 4), 5.0),
        )
        b = model.generate(
            prompt, max_new_tokens=1, greedy=True, use_cache=False,
            pixel_values=torch.full((1, 4), -5.0),
        )
        self.assertEqual(int(a[0, -1]), 300)
        self.assertEqual(int(b[0, -1]), 301)

    def test_cached_matches_cachefree_with_image(self):
        torch.manual_seed(0)
        cfg = _two_stage_cfg()
        model = RDTForCausalLM(cfg, patch_pixels=4).eval()
        prompt = _image_prompt(cfg)
        pixels = torch.randn(1, 4)
        track_table = torch.zeros(cfg.vocab_size, dtype=torch.long)
        track_table[256:512] = 1

        free = model.generate(
            prompt, max_new_tokens=5, greedy=True, use_cache=False,
            pixel_values=pixels,
            morphology_track_table=track_table,
        )
        cached = model.generate(
            prompt, max_new_tokens=5, greedy=True, use_cache=True,
            pixel_values=pixels,
            morphology_track_table=track_table,
        )
        self.assertTrue(
            torch.equal(free, cached),
            f"cached/cache-free diverged: {free.tolist()} vs {cached.tolist()}",
        )

    def test_cachefree_truncation_with_image_raises(self):
        torch.manual_seed(0)
        cfg = _two_stage_cfg()
        cfg.max_seq_len = 6
        model = RDTForCausalLM(cfg, patch_pixels=4).eval()
        prompt = _image_prompt(cfg)  # length 5
        with self.assertRaises(ValueError):
            # 5 + 4 generated > max_seq_len=6 -> window truncates -> refuse.
            model.generate(
                prompt, max_new_tokens=4, greedy=True, use_cache=False,
                pixel_values=torch.randn(1, 4),
            )


if __name__ == "__main__":
    unittest.main()
