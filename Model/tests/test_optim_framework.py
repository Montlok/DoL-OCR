# -*- coding: utf-8 -*-

"""Guards for the learning-framework upgrades (Adam-atan2, Muon, WSD)."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from Model.config import TrainingConfig
from Model.training.optim import (
    AdamAtan2,
    CombinedOptimizer,
    Muon,
    build_optimizer,
    build_scheduler,
    zeropower_via_newtonschulz5,
)


def _toy_model() -> nn.Module:
    model = nn.Module()
    model.embed = nn.Embedding(16, 8)
    model.w = nn.Linear(8, 8, bias=True)
    model.lm_head = nn.Linear(8, 16, bias=False)
    return model


class _MuonRoutingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.hidden = nn.Linear(8, 8, bias=True)
        self.conv = nn.Conv1d(8, 8, kernel_size=3, bias=False)
        self.no_wd_matrix = nn.Parameter(torch.randn(8, 8))
        self.no_wd_matrix._no_weight_decay = True
        self.lm_head = nn.Linear(8, 16, bias=False)


def _step_once(model, optim):
    x = torch.randint(0, 16, (2, 4))
    h = model.embed(x)
    logits = model.lm_head(model.w(h))
    loss = logits.float().pow(2).mean()
    optim.zero_grad()
    loss.backward()
    optim.step()
    return float(loss.detach())


class AdamAtan2Test(unittest.TestCase):
    def test_decreases_quadratic_loss(self) -> None:
        torch.manual_seed(0)
        model = _toy_model()
        optim = AdamAtan2(model.parameters(), lr=1e-2)
        first = _step_once(model, optim)
        for _ in range(40):
            last = _step_once(model, optim)
        self.assertLess(last, first)

    def test_no_eps_no_nan_with_zero_grad_dims(self) -> None:
        torch.manual_seed(0)
        p = nn.Parameter(torch.randn(4, 4))
        optim = AdamAtan2([p], lr=1e-2, weight_decay=0.1)
        loss = (p[:, :2]).pow(2).sum()  # half the columns get zero grad
        loss.backward()
        optim.step()
        self.assertFalse(torch.isnan(p).any())

    def test_moments_stay_fp32_for_low_precision_params(self) -> None:
        p = nn.Parameter(torch.randn(4, 4, dtype=torch.bfloat16))
        optim = AdamAtan2([p], lr=1e-2)
        p.grad = torch.randn_like(p)
        optim.step()
        state = optim.state[p]
        self.assertEqual(state["exp_avg"].dtype, torch.float32)
        self.assertEqual(state["exp_avg_sq"].dtype, torch.float32)


class MuonTest(unittest.TestCase):
    def test_newton_schulz_orthogonalizes(self) -> None:
        torch.manual_seed(0)
        g = torch.randn(6, 4)
        q = zeropower_via_newtonschulz5(g, steps=5)
        # The quintic iteration drives the singular values toward 1; check
        # they are all close to 1 (orthogonalized), unlike the raw matrix.
        sv = torch.linalg.svdvals(q)
        self.assertTrue(torch.all(sv > 0.7) and torch.all(sv < 1.3))

    def test_muon_decreases_loss(self) -> None:
        torch.manual_seed(0)
        p = nn.Parameter(torch.randn(8, 8))
        target = torch.randn(8, 8)
        optim = Muon([p], lr=1e-2, momentum=0.9, ns_steps=5)
        first = None
        for _ in range(50):
            loss = (p - target).pow(2).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            if first is None:
                first = float(loss.detach())
        self.assertLess(float(loss.detach()), first)

    def test_muon_rejects_non_2d(self) -> None:
        p = nn.Parameter(torch.randn(8))
        optim = Muon([p], lr=1e-2)
        p.grad = torch.randn(8)
        with self.assertRaises(ValueError):
            optim.step()

    def test_momentum_stays_fp32_for_low_precision_params(self) -> None:
        p = nn.Parameter(torch.randn(8, 8, dtype=torch.bfloat16))
        optim = Muon([p], lr=1e-2)
        p.grad = torch.randn_like(p)
        optim.step()
        self.assertEqual(optim.state[p]["momentum_buffer"].dtype, torch.float32)


class BuildOptimizerTest(unittest.TestCase):
    def test_adamw_atan2_path(self) -> None:
        model = _toy_model()
        cfg = TrainingConfig(train_data="x", adam_use_atan2=True, max_steps=10)
        optim = build_optimizer(model, cfg)
        self.assertIsInstance(optim, AdamAtan2)

    def test_muon_hybrid_routes_params(self) -> None:
        model = _toy_model()
        cfg = TrainingConfig(train_data="x", optimizer="muon", max_steps=10)
        optim = build_optimizer(model, cfg)
        self.assertIsInstance(optim, CombinedOptimizer)
        # embeddings + lm_head must NOT be on the Muon sub-optimizer.
        muon = optim.optimizers[0]
        muon_param_ids = {id(p) for g in muon.param_groups for p in g["params"]}
        self.assertNotIn(id(model.embed.weight), muon_param_ids)
        self.assertNotIn(id(model.lm_head.weight), muon_param_ids)
        self.assertIn(id(model.w.weight), muon_param_ids)

    def test_muon_preserves_weight_decay_policy(self) -> None:
        model = _MuonRoutingModel()
        cfg = TrainingConfig(
            train_data="x",
            optimizer="muon",
            weight_decay=0.2,
            max_steps=10,
        )
        optim = build_optimizer(model, cfg)
        self.assertIsInstance(optim, CombinedOptimizer)

        muon = optim.optimizers[0]
        adam = optim.optimizers[1]
        muon_ids = {id(p) for group in muon.param_groups for p in group["params"]}
        self.assertIn(id(model.hidden.weight), muon_ids)
        self.assertNotIn(id(model.no_wd_matrix), muon_ids)
        self.assertNotIn(id(model.conv.weight), muon_ids)
        self.assertNotIn(id(model.lm_head.weight), muon_ids)

        wd_by_param: dict[int, float] = {}
        for group in adam.param_groups:
            for param in group["params"]:
                wd_by_param[id(param)] = group["weight_decay"]
        self.assertEqual(wd_by_param[id(model.no_wd_matrix)], 0.0)
        self.assertEqual(wd_by_param[id(model.embed.weight)], 0.0)
        self.assertEqual(wd_by_param[id(model.hidden.bias)], 0.0)
        self.assertEqual(wd_by_param[id(model.conv.weight)], cfg.weight_decay)
        self.assertEqual(wd_by_param[id(model.lm_head.weight)], cfg.weight_decay)

    def test_muon_train_step_runs(self) -> None:
        torch.manual_seed(0)
        model = _toy_model()
        cfg = TrainingConfig(train_data="x", optimizer="muon", max_steps=10)
        optim = build_optimizer(model, cfg)
        first = _step_once(model, optim)
        for _ in range(30):
            last = _step_once(model, optim)
        self.assertLessEqual(last, first + 1e-3)

    def test_combined_optimizer_exposes_mutable_base_state(self) -> None:
        p1 = nn.Parameter(torch.randn(4, 4))
        p2 = nn.Parameter(torch.randn(4))
        opt1 = Muon([p1], lr=1e-2)
        opt2 = AdamAtan2([p2], lr=1e-2)
        optim = CombinedOptimizer([opt1, opt2])
        self.assertIsInstance(optim.state, dict)

        p1.grad = torch.randn_like(p1)
        p2.grad = torch.randn_like(p2)
        optim.step()

        self.assertIn(p1, optim.state)
        self.assertIn(p2, optim.state)
        self.assertIs(optim.param_groups[0], opt1.param_groups[0])
        self.assertIs(optim.param_groups[1], opt2.param_groups[0])


class WSDScheduleTest(unittest.TestCase):
    def _factors(self, cfg, optim):
        base = optim.param_groups[0]["lr"]  # capture before LambdaLR scales it
        sched = build_scheduler(optim, cfg)
        factors = []
        for _ in range(cfg.max_steps):
            factors.append(optim.param_groups[0]["lr"] / base)
            optim.step()
            sched.step()
        return factors

    def test_wsd_has_warmup_plateau_decay(self) -> None:
        model = _toy_model()
        cfg = TrainingConfig(
            train_data="x",
            lr_schedule="wsd",
            warmup_steps=5,
            max_steps=100,
            wsd_stable_ratio=0.8,
            min_lr_ratio=0.0,
            wsd_decay_shape="1-sqrt",
        )
        optim = torch.optim.SGD(model.parameters(), lr=1.0)
        f = self._factors(cfg, optim)
        self.assertLess(f[0], 1.0)              # warmup ramps up
        self.assertAlmostEqual(f[40], 1.0, places=6)  # stable plateau at full lr
        self.assertLess(f[99], 0.2)             # decay tail near zero

    def test_cosine_still_default(self) -> None:
        cfg = TrainingConfig(train_data="x", max_steps=10)
        self.assertEqual(cfg.lr_schedule, "cosine")


class TrainingConfigValidationTest(unittest.TestCase):
    def test_rejects_bad_knobs(self) -> None:
        for kw in (
            dict(lr_schedule="bogus"),
            dict(wsd_stable_ratio=1.0),
            dict(wsd_decay_shape="zzz"),
            dict(optimizer="adam"),
            dict(muon_ns_steps=0),
            dict(muon_momentum=1.0),
            dict(lr_decay_steps=0),
            dict(lr_decay_steps=-1),
        ):
            with self.assertRaises(ValueError):
                TrainingConfig(train_data="x", max_steps=10, **kw)


if __name__ == "__main__":
    unittest.main()
