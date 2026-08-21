# -*- coding: utf-8 -*-

import unittest

import torch

from Model.config import two_stage_tiny_config
from Model.layers.mhc import ManifoldHyperConnection, sinkhorn_knopp
from Model.two_stage import MHCAttnSubLayer


class SinkhornTest(unittest.TestCase):
    def test_projection_is_doubly_stochastic(self):
        torch.manual_seed(0)
        mat = sinkhorn_knopp(torch.randn(4, 4), n_iters=20)

        self.assertTrue(torch.all(mat >= 0))
        self.assertTrue(torch.allclose(mat.sum(dim=-1), torch.ones(4), atol=1e-3))
        self.assertTrue(torch.allclose(mat.sum(dim=-2), torch.ones(4), atol=1e-3))

    def test_batched_projection(self):
        torch.manual_seed(0)
        mat = sinkhorn_knopp(torch.randn(3, 5, 5), n_iters=20)

        self.assertEqual(mat.shape, (3, 5, 5))
        self.assertTrue(torch.allclose(mat.sum(dim=-1), torch.ones(3, 5), atol=1e-3))
        self.assertTrue(torch.allclose(mat.sum(dim=-2), torch.ones(3, 5), atol=1e-3))


class ManifoldHyperConnectionTest(unittest.TestCase):
    def test_residual_matrix_is_doubly_stochastic(self):
        torch.manual_seed(0)
        hc = ManifoldHyperConnection(8, n_streams=4, sinkhorn_iters=20)

        with torch.no_grad():
            hc.res_bias.copy_(torch.randn(4, 4))

        mat = hc.residual_matrix()
        self.assertTrue(torch.allclose(mat.sum(dim=-1), torch.ones(4), atol=1e-3))
        self.assertTrue(torch.allclose(mat.sum(dim=-2), torch.ones(4), atol=1e-3))

    def test_composite_gain_is_bounded_over_deep_stack(self):
        torch.manual_seed(0)
        d_model = 16
        hc = ManifoldHyperConnection(d_model, n_streams=4, sinkhorn_iters=20)
        hc.eval()

        with torch.no_grad():
            hc.res_bias.copy_(torch.randn(4, 4))

        x = torch.randn(2, 6, d_model)
        streams = hc.expand(x)

        # Pure cross-stream mixing (no write): a doubly-stochastic matrix has
        # spectral norm <= 1, so the composite gain over 60 layers stays bounded.
        def zero_fn(inp):
            return torch.zeros_like(inp)

        s = streams
        for _ in range(60):
            s = hc(s, zero_fn)

        gain = hc.collapse(s).norm() / hc.collapse(streams).norm().clamp(min=1e-6)
        self.assertTrue(torch.isfinite(gain))
        self.assertLessEqual(gain.item(), 1.1)

    def test_unconstrained_control_blows_up(self):
        torch.manual_seed(0)
        d_model = 16
        hc = ManifoldHyperConnection(
            d_model, n_streams=4, sinkhorn_iters=20, constrain=False
        )
        hc.eval()

        with torch.no_grad():
            # Deterministic all-ones logits: softplus(1) ~ 1.31 per entry, so each
            # row sums to ~5.25 >> 1 and the unconstrained mixing is guaranteed to
            # blow up over repeated application (no reliance on a random draw).
            hc.res_bias.copy_(torch.ones(4, 4))

        x = torch.randn(2, 6, d_model)
        streams = hc.expand(x)

        def zero_fn(inp):
            return torch.zeros_like(inp)

        s = streams
        for _ in range(60):
            s = hc(s, zero_fn)

        gain = hc.collapse(s).norm() / hc.collapse(streams).norm().clamp(min=1e-6)
        # Without the doubly-stochastic constraint the repeated mixing explodes.
        self.assertTrue((not torch.isfinite(gain)) or gain.item() > 1e3)

    def test_n_equals_one_degenerates_to_plain_residual(self):
        torch.manual_seed(0)
        d_model = 8
        hc = ManifoldHyperConnection(d_model, n_streams=1, sinkhorn_iters=20)
        hc.eval()

        x = torch.randn(2, 4, d_model)
        streams = hc.expand(x)
        self.assertEqual(streams.shape, (2, 4, 1, d_model))

        # n=1: residual matrix is [[1]] and post init gives weight 1, so the
        # update is exactly x + fn(x).
        out = hc(streams, lambda inp: inp)
        collapsed = hc.collapse(out)
        self.assertTrue(torch.allclose(collapsed, 2.0 * x, atol=1e-5))

    def test_backprop_flows(self):
        torch.manual_seed(0)
        d_model = 8
        hc = ManifoldHyperConnection(d_model, n_streams=4, sinkhorn_iters=20)

        x = torch.randn(2, 4, d_model, requires_grad=True)
        loss = hc.collapse(hc(hc.expand(x), lambda inp: inp)).sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.norm().item(), 0.0)
        self.assertIsNotNone(hc.res_bias.grad)

    def test_maps_are_data_dependent_when_gated(self):
        """With a non-zero dynamic gate the maps vary with the input (Eq. 5/7)."""

        torch.manual_seed(0)
        d_model = 8
        hc = ManifoldHyperConnection(d_model, n_streams=4, sinkhorn_iters=20)
        hc.eval()

        # At init the gates are zero, so the residual matrix is identical for
        # every token regardless of the input (data-independent).
        x = torch.randn(2, 4, d_model)
        ref = hc._dyn_ref(hc.expand(x))
        mat_init = hc.residual_matrix(ref)
        self.assertTrue(
            torch.allclose(mat_init, mat_init[:, :1], atol=1e-5),
            "maps must be data-independent at init (alpha == 0)",
        )

        # Turn the dynamic gates on; now the per-token maps must differ across
        # positions with different inputs.
        with torch.no_grad():
            hc.pre_alpha.fill_(1.0)
            hc.post_alpha.fill_(1.0)
            hc.res_alpha.fill_(1.0)
            hc.res_proj.weight.normal_()
            hc.pre_proj.weight.normal_()
            hc.post_proj.weight.normal_()

        ref = hc._dyn_ref(hc.expand(x))
        mat = hc.residual_matrix(ref)
        self.assertFalse(
            torch.allclose(mat[:, 0], mat[:, 1], atol=1e-4),
            "maps must vary with the input once gated",
        )
        # Doubly-stochastic constraint must still hold per token.
        self.assertTrue(torch.allclose(mat.sum(dim=-1), torch.ones_like(mat.sum(dim=-1)), atol=1e-2))
        self.assertTrue(torch.allclose(mat.sum(dim=-2), torch.ones_like(mat.sum(dim=-2)), atol=1e-2))


class MHCSubLayerStabilityTest(unittest.TestCase):
    def test_mhc_sublayer_stability(self):
        """Stacking MHCAttnSubLayer over many recurrent steps stays bounded.

        With mHC sunk into each residual, the write term ``H_post * F`` is
        redistributed by a doubly-stochastic matrix every layer instead of
        accumulating, so running 16 steps does not blow up relative to 1 step.
        """

        torch.manual_seed(0)
        cfg = two_stage_tiny_config()
        layer = MHCAttnSubLayer(cfg)
        layer.eval()

        bsz, seq_len = 2, 12
        x = torch.randn(bsz, seq_len, cfg.d_model)
        word_pos = torch.arange(seq_len).unsqueeze(0).expand(bsz, seq_len)
        morph_depth = torch.zeros(bsz, seq_len, dtype=torch.long)

        def run(steps: int) -> float:
            streams = x.unsqueeze(-2).expand(-1, -1, cfg.mhc_n_streams, -1).contiguous()
            with torch.no_grad():
                for _ in range(steps):
                    streams = layer(
                        streams,
                        word_pos=word_pos,
                        morph_depth=morph_depth,
                    )
            return streams.mean(dim=-2).norm().item()

        norm_1 = run(1)
        norm_8 = run(8)
        norm_16 = run(16)

        for value in (norm_1, norm_8, norm_16):
            self.assertTrue(torch.isfinite(torch.tensor(value)))

        self.assertLessEqual(norm_16, 10.0 * norm_1)


if __name__ == "__main__":
    unittest.main()
