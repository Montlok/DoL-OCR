# -*- coding: utf-8 -*-

"""Zero-shot KL early-exit for ``generate()`` (Huginn arXiv:2502.05171 §6.1).

The early-exit adaptively shortens the recurrent-depth loop per token when the
output distribution has converged. It must be (a) *disabled by default* so the
existing fixed-depth decode is bit-exact, (b) equivalent to ``steps=1`` when the
threshold is so large every token exits immediately, and (c) equivalent to the
full fixed-depth decode when the threshold is ~0 (never converges early).
"""

import unittest

import torch

from Model.config import RDTConfig
from Model.model import RDTForCausalLM


def _two_stage_cfg() -> RDTConfig:
    return RDTConfig(
        d_model=32,
        n_heads=4,
        head_dim=8,
        kv_lora_rank=8,
        rope_head_dim=4,
        nope_head_dim=4,
        ffn_hidden=64,
        ffn_multiple=32,
        n_prelude=2,
        n_coda=2,
        recurrent_steps=3,
        mamba_d_state=8,
        mamba_expand=2,
        mamba_headdim=8,
        use_official_mamba=False,
        max_seq_len=64,
        core_type="two_stage",
        stage1_mamba_layers=3,
        stage2_attn_layers=2,
        recurrent_drift_mode="none",
    )


def _model(threshold: float):
    torch.manual_seed(0)
    cfg = _two_stage_cfg()
    cfg.kl_exit_threshold = threshold
    return RDTForCausalLM(cfg).eval()


class EarlyExitTest(unittest.TestCase):
    def test_disabled_by_default_is_fixed_depth(self):
        model = _model(0.0)
        self.assertFalse(model._kl_exit_active(None))
        ids = torch.randint(300, 320, (2, 6))
        with torch.no_grad():
            a = model.generate(ids, greedy=True, max_new_tokens=8)
        # A second run must be deterministic / identical (no hidden state leak).
        with torch.no_grad():
            b = model.generate(ids, greedy=True, max_new_tokens=8)
        self.assertTrue(torch.equal(a, b))

    def test_explicit_steps_override_disables_exit(self):
        model = _model(1.0)
        self.assertFalse(model._kl_exit_active(recurrent_steps=2))
        self.assertTrue(model._kl_exit_active(recurrent_steps=None))

    def test_huge_threshold_matches_min_depth(self):
        # Threshold so large the loop exits at the first comparison (depth 2,
        # the shallowest depth where convergence can be measured), so the logits
        # equal a plain forward at steps=2.
        model = _model(threshold=1e9)
        ids = torch.randint(300, 320, (2, 7))
        window = ids
        with torch.no_grad():
            adaptive = model._adaptive_depth_logits(window, None)
            fixed2 = model.forward(window, steps=2, return_logits=True)[
                "logits"
            ][:, -1, :].float()
        self.assertTrue(torch.allclose(adaptive, fixed2, atol=1e-5))

    def test_tiny_threshold_matches_full_depth(self):
        # Threshold ~0 never triggers early exit -> deepest distribution, equal
        # to a fixed forward at cfg.recurrent_steps.
        model = _model(threshold=1e-12)
        ids = torch.randint(300, 320, (2, 7))
        window = ids
        with torch.no_grad():
            adaptive = model._adaptive_depth_logits(window, None)
            full = model.forward(
                window, steps=model.cfg.recurrent_steps, return_logits=True
            )["logits"][:, -1, :].float()
        self.assertTrue(torch.allclose(adaptive, full, atol=1e-5))

    def test_generate_runs_with_early_exit(self):
        model = _model(threshold=0.05)
        ids = torch.randint(300, 320, (2, 5))
        with torch.no_grad():
            out = model.generate(ids, greedy=True, max_new_tokens=10)
        self.assertEqual(out.shape, (2, 15))


if __name__ == "__main__":
    unittest.main()
