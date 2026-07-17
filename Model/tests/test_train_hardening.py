# -*- coding: utf-8 -*-

"""Depth-scaled init + recurrent step sampling + ACT halt norm."""

from __future__ import annotations

import unittest

import torch

from Model.config import RDTConfig, TrainingConfig, tiny_config
from Model.model import RDTForCausalLM
from Model.recurrent import RecurrentCore
from Model.training.optim import recurrent_steps_for_step


def _tiny() -> RDTConfig:
    return tiny_config()


class TestResidualInitScaling(unittest.TestCase):
    def test_core_aware_depth_counts(self) -> None:
        interleaved = RDTConfig(
            n_prelude=2,
            n_coda=3,
            mamba_per_block=2,
            attn_per_block=1,
            recurrent_steps=4,
        )
        self.assertEqual(interleaved.actual_layers, 8)
        self.assertEqual(interleaved.effective_depth, 17)

        two_stage = RDTConfig(
            core_type="two_stage",
            n_prelude=2,
            n_coda=3,
            stage1_mamba_layers=4,
            stage2_attn_layers=2,
            recurrent_steps=5,
            recurrent_drift_mode="mhc",
        )
        self.assertEqual(two_stage.actual_layers, 11)
        self.assertEqual(two_stage.effective_depth, 19)

        segmented = RDTConfig(
            core_type="segmented",
            n_prelude=2,
            n_coda=3,
            stage1_mamba_layers=4,
            stage2_attn_layers=2,
            segmented_local_layers=3,
            recurrent_steps=5,
        )
        self.assertEqual(segmented.actual_layers, 14)
        self.assertEqual(segmented.effective_depth, 22)

    def test_output_projections_are_depth_scaled(self) -> None:
        torch.manual_seed(0)
        cfg = _tiny()
        model = RDTForCausalLM(cfg)

        scale = (2.0 * cfg.effective_depth) ** -0.5
        expected = cfg.init_std * scale

        block = model.recurrent.block
        attn_layer = next(layer for layer in block.layers if hasattr(layer, "attn"))
        mamba_layer = next(layer for layer in block.layers if hasattr(layer, "mamba"))

        for tensor in (
            model.prelude[0].attn.o_proj.weight,
            model.prelude[0].ffn.w_down.weight,
            model.coda[-1].attn.o_proj.weight,
            attn_layer.attn.o_proj.weight,
            mamba_layer.ffn.w_down.weight,
            mamba_layer.mamba.mamba.out_proj.weight,
        ):
            self.assertAlmostEqual(
                float(tensor.detach().std()), expected, delta=expected * 0.15
            )

    def test_non_residual_projections_keep_base_std(self) -> None:
        torch.manual_seed(0)
        cfg = _tiny()
        model = RDTForCausalLM(cfg)

        for tensor in (
            model.prelude[0].attn.q_proj.weight,
            model.prelude[0].ffn.w_in.weight,
            model.embed.weight,
        ):
            self.assertAlmostEqual(
                float(tensor.detach().std()), cfg.init_std, delta=cfg.init_std * 0.1
            )

    def test_hidden_norm_growth_is_tamed(self) -> None:
        torch.manual_seed(0)
        cfg = _tiny()
        model = RDTForCausalLM(cfg).eval()

        B, L = 2, 32
        ids = torch.randint(300, 24000, (B, L))
        am = torch.ones(B, L, dtype=torch.long)
        wp, md = model._default_morph_info(input_ids=ids, attention_mask=am)

        with torch.no_grad():
            h = model.embed(ids)
            for blk in model.prelude:
                h = blk(h, word_pos=wp, morph_depth=md, attn_mask=am, causal=True)
            e0 = h
            base = h.norm().item()
            for _ in range(4 * cfg.recurrent_steps):
                h = h + e0
                h = model.recurrent.block(
                    h, word_pos=wp, morph_depth=md, attn_mask=am, causal=True
                )

        # Pre-fix this ratio was >50x at 4x trained depth; the depth-scaled
        # init keeps drift dominated by the (intentional) e0 injection.
        self.assertLess(h.norm().item() / base, 25.0)


class TestRecurrentStepSampling(unittest.TestCase):
    def _cfg(self, **kw) -> TrainingConfig:
        return TrainingConfig(recurrent_steps_sampling="poisson", **kw)

    def test_fixed_mode_returns_target(self) -> None:
        cfg = TrainingConfig()
        for step in range(5):
            self.assertEqual(recurrent_steps_for_step(step, cfg, 8), 8)

    def test_deterministic_across_ranks(self) -> None:
        cfg_a = self._cfg(seed=7)
        cfg_b = self._cfg(seed=7)
        draws_a = [recurrent_steps_for_step(s, cfg_a, 8) for s in range(64)]
        draws_b = [recurrent_steps_for_step(s, cfg_b, 8) for s in range(64)]
        self.assertEqual(draws_a, draws_b)

    def test_varies_across_steps_and_seeds(self) -> None:
        cfg = self._cfg(seed=7)
        draws = {recurrent_steps_for_step(s, cfg, 8) for s in range(64)}
        self.assertGreater(len(draws), 3)

        other = self._cfg(seed=8)
        self.assertNotEqual(
            [recurrent_steps_for_step(s, cfg, 8) for s in range(64)],
            [recurrent_steps_for_step(s, other, 8) for s in range(64)],
        )

    def test_mean_tracks_target_and_bounds_hold(self) -> None:
        cfg = self._cfg(seed=3)
        target = 8
        draws = [recurrent_steps_for_step(s, cfg, target) for s in range(2000)]

        self.assertGreaterEqual(min(draws), cfg.recurrent_steps_min)
        self.assertLessEqual(max(draws), 2 * target)

        mean = sum(draws) / len(draws)
        self.assertGreater(mean, target * 0.75)
        self.assertLess(mean, target * 1.25)

    def test_explicit_max_is_respected(self) -> None:
        cfg = self._cfg(seed=3, recurrent_steps_min=2, recurrent_steps_max=6)
        draws = [recurrent_steps_for_step(s, cfg, 8) for s in range(500)]
        self.assertGreaterEqual(min(draws), 2)
        self.assertLessEqual(max(draws), 6)

    def test_ramp_then_sampling(self) -> None:
        cfg = self._cfg(seed=3, recurrent_steps_start=2, recurrent_steps_ramp=100)
        early = [recurrent_steps_for_step(0, cfg, 8) for _ in range(1)]
        self.assertLessEqual(early[0], 2 * 8)

    def test_config_validation(self) -> None:
        with self.assertRaises(ValueError):
            TrainingConfig(recurrent_steps_sampling="bogus")
        with self.assertRaises(ValueError):
            TrainingConfig(recurrent_steps_min=0)
        with self.assertRaises(ValueError):
            TrainingConfig(recurrent_steps_min=4, recurrent_steps_max=2)
        with self.assertRaises(ValueError):
            TrainingConfig(recurrent_steps_sigma=0.0)

    def test_model_runs_at_sampled_depths(self) -> None:
        torch.manual_seed(0)
        cfg = _tiny()
        model = RDTForCausalLM(cfg).eval()
        ids = torch.randint(300, 320, (1, 16))
        with torch.no_grad():
            for steps in (1, 3, 8):
                out = model(ids, labels=ids, steps=steps)
                self.assertEqual(out["rec_info"]["steps_used"], steps)
                self.assertTrue(torch.isfinite(out["loss"]))


class TestActHaltNorm(unittest.TestCase):
    def test_halt_norm_exists_and_act_runs(self) -> None:
        torch.manual_seed(0)
        cfg = tiny_config()
        object.__setattr__(cfg, "use_act", True)
        object.__setattr__(cfg, "act_max_steps", 4)

        core = RecurrentCore(cfg)
        self.assertTrue(hasattr(core, "halt_norm"))

        e0 = torch.randn(2, 8, cfg.d_model)
        out, info = core(e0)
        self.assertEqual(out.shape, e0.shape)
        self.assertGreater(info["steps_used"], 0.0)

    def test_halt_probability_is_scale_invariant(self) -> None:
        torch.manual_seed(0)
        cfg = tiny_config()
        object.__setattr__(cfg, "use_act", True)

        core = RecurrentCore(cfg)
        h = torch.randn(2, 8, cfg.d_model)
        p_small = torch.sigmoid(core.halt_proj(core.halt_norm(h)))
        p_large = torch.sigmoid(core.halt_proj(core.halt_norm(h * 1000.0)))
        self.assertTrue(torch.allclose(p_small, p_large, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
