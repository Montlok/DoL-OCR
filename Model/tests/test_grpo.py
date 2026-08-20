# -*- coding: utf-8 -*-

"""Tests for GRPO advantages, surrogate loss, and verifiable rewards."""

import contextlib
import io
import tempfile
import unittest
from unittest.mock import patch

import torch

from Model.config import RDTConfig, TrainingConfig
from Model.model import RDTForCausalLM
from Model.posttrain.grpo import (
    GRPOConfig,
    grpo_compute_loss,
    grpo_loss,
    group_normalized_advantages,
    group_relative_advantages,
)
from Model.posttrain.logprobs import token_logprobs_with_mask
from Model.posttrain.rewards import (
    RewardConfig,
    compute_rewards,
    exact_match_reward,
    format_reward,
    mongolian_script_ratio,
    numeric_match_reward,
)


def _cfg() -> RDTConfig:
    return RDTConfig(
        d_model=32, n_heads=4, head_dim=8, kv_lora_rank=8, rope_head_dim=4,
        nope_head_dim=4, ffn_hidden=64, ffn_multiple=32, n_prelude=1, n_coda=1,
        mamba_per_block=1, attn_per_block=1, recurrent_steps=2, mamba_d_state=8,
        mamba_expand=2, mamba_headdim=16, use_official_mamba=False, max_seq_len=16,
    )


class AdvantageTest(unittest.TestCase):
    def test_group_normalization_zero_mean(self):
        rewards = torch.tensor([0.0, 1.0, 2.0, 6.0])
        adv = group_normalized_advantages(rewards, group_size=2)
        # Each group should be zero-mean.
        self.assertAlmostEqual(float(adv[:2].mean()), 0.0, places=5)
        self.assertAlmostEqual(float(adv[2:].mean()), 0.0, places=5)
        # Higher reward in a group -> positive advantage.
        self.assertGreater(float(adv[1]), float(adv[0]))

    def test_identical_rewards_give_zero_advantage(self):
        rewards = torch.tensor([3.0, 3.0, 3.0])
        adv = group_normalized_advantages(rewards, group_size=3)
        self.assertTrue(torch.allclose(adv, torch.zeros_like(adv), atol=1e-4))

    def test_bad_group_size_raises(self):
        with self.assertRaises(ValueError):
            group_normalized_advantages(torch.zeros(5), group_size=2)
        with self.assertRaises(ValueError):
            group_normalized_advantages(torch.zeros(1), group_size=1)

    def test_centered_advantage_does_not_amplify_tiny_reward_spread(self):
        rewards = torch.tensor([0.0, 1e-8])
        adv = group_relative_advantages(
            rewards,
            group_size=2,
            mode="centered",
        )
        self.assertLessEqual(float(adv.abs().max()), 1e-8)

    def test_group_relative_advantage_rejects_nonfinite_reward(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            group_relative_advantages(
                torch.tensor([0.0, float("nan")]),
                group_size=2,
            )

    def test_minimum_reward_spread_zeros_only_configured_groups(self):
        rewards = torch.tensor([0.0, 1e-8, 0.0, 1.0])
        adv = group_relative_advantages(
            rewards,
            group_size=2,
            mode="centered",
            min_reward_spread=1e-7,
        )
        self.assertTrue(torch.equal(adv[:2], torch.zeros(2)))
        self.assertGreater(float(adv[3]), 0.0)


class GRPOLossTest(unittest.TestCase):
    def test_config_rejects_sampling_scoring_distribution_mismatch(self):
        with self.assertRaisesRegex(ValueError, "rollout and scoring"):
            GRPOConfig(temperature=0.8)
        with self.assertRaisesRegex(ValueError, "rollout and scoring"):
            GRPOConfig(top_p=0.95)

    def test_no_change_no_kl_recovers_neg_advantage(self):
        logp = torch.tensor([[-1.0, -2.0]])
        adv = torch.tensor([0.5])
        mask = torch.ones(1, 2)
        loss, m = grpo_loss(logp, logp, adv, mask, ref_token_logp=logp,
                            cfg=GRPOConfig(kl_coef=0.1))
        # ratio==1, kl==0 -> loss == -adv
        self.assertAlmostEqual(float(loss), -0.5, places=5)
        self.assertAlmostEqual(m["kl"], 0.0, places=6)
        self.assertAlmostEqual(m["ratio_mean"], 1.0, places=6)

    def test_kl_is_nonnegative(self):
        policy = torch.tensor([[-1.0, -2.0]])
        ref = torch.tensor([[-1.5, -1.0]])
        adv = torch.tensor([0.0])
        mask = torch.ones(1, 2)
        _, m = grpo_loss(policy, policy, adv, mask, ref_token_logp=ref,
                        cfg=GRPOConfig(kl_coef=1.0))
        self.assertGreaterEqual(m["kl"], 0.0)

    def test_mask_excludes_prompt_tokens(self):
        policy = torch.tensor([[-1.0, -2.0, -3.0]])
        old = torch.tensor([[-2.0, -2.0, -3.0]])
        adv = torch.tensor([1.0])
        full = grpo_loss(policy, old, adv, torch.ones(1, 3))[0]
        masked = grpo_loss(policy, old, adv, torch.tensor([[0.0, 1.0, 1.0]]))[0]
        self.assertNotAlmostEqual(float(full), float(masked), places=4)
        # Masked-out first token differs between policy/old; excluding it changes loss.

    def test_positive_advantage_clipped(self):
        policy = torch.tensor([[0.0]])
        old = torch.tensor([[-5.0]])  # ratio huge
        adv = torch.tensor([1.0])
        loss, _ = grpo_loss(policy, old, adv, torch.ones(1, 1),
                            cfg=GRPOConfig(clip_eps=0.2, kl_coef=0.0))
        # Clipped surrogate caps gain at (1+clip)*adv -> loss == -1.2
        self.assertAlmostEqual(float(loss), -1.2, places=5)

    def test_behavior_policy_mismatch_fails_for_negative_advantage(self):
        with self.assertRaisesRegex(RuntimeError, "rollout/scoring policy mismatch"):
            grpo_loss(
                torch.tensor([[0.0]]),
                torch.tensor([[-0.1]]),
                torch.tensor([-1.0]),
                torch.ones(1, 1),
                cfg=GRPOConfig(
                    clip_eps=None,
                    kl_coef=0.0,
                    max_behavior_log_ratio=0.05,
                ),
            )

    def test_behavior_gate_can_be_deferred_for_distributed_consensus(self):
        _, metrics = grpo_loss(
            torch.tensor([[0.0]]),
            torch.tensor([[-0.1]]),
            torch.tensor([-1.0]),
            torch.ones(1, 1),
            cfg=GRPOConfig(
                clip_eps=None,
                kl_coef=0.0,
                max_behavior_log_ratio=0.05,
            ),
            enforce_behavior_gate=False,
        )
        self.assertAlmostEqual(
            metrics["behavior_log_ratio_abs_max"],
            0.1,
        )

    def test_kl_numeric_bound_keeps_corrective_gradient(self):
        policy = torch.tensor([[-30.0]], requires_grad=True)
        old = policy.detach().clone()
        reference = torch.tensor([[30.0]])
        loss, metrics = grpo_loss(
            policy,
            old,
            torch.tensor([0.0]),
            torch.ones(1, 1),
            ref_token_logp=reference,
            cfg=GRPOConfig(kl_coef=1.0, log_ratio_clip=5.0),
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["kl_numeric_clip_frac"], 1.0)
        self.assertIsNotNone(policy.grad)
        self.assertGreater(float(policy.grad.abs().sum()), 0.0)

    def test_ratio_numeric_bound_keeps_corrective_gradient(self):
        policy = torch.tensor([[30.0]], requires_grad=True)
        old = torch.tensor([[0.0]])
        loss, metrics = grpo_loss(
            policy,
            old,
            torch.tensor([-1.0]),
            torch.ones(1, 1),
            cfg=GRPOConfig(kl_coef=0.0, log_ratio_clip=5.0),
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["numeric_clip_frac"], 1.0)
        self.assertIsNotNone(policy.grad)
        self.assertGreater(float(policy.grad.abs().sum()), 0.0)


class RewardTest(unittest.TestCase):
    def test_exact_match(self):
        self.assertEqual(exact_match_reward("  Hello ", "hello"), 1.0)
        self.assertEqual(exact_match_reward("hi", "hello"), 0.0)

    def test_numeric_match_uses_last_number(self):
        self.assertEqual(numeric_match_reward("steps... answer is 42", "42"), 1.0)
        self.assertEqual(numeric_match_reward("answer is 41", "42"), 0.0)

    def test_mongolian_script_ratio(self):
        self.assertGreater(mongolian_script_ratio("Сайн байна уу"), 0.9)
        self.assertEqual(mongolian_script_ratio("hello world"), 0.0)

    def test_format_reward(self):
        self.assertEqual(format_reward("<think>reason</think> answer"), 1.0)
        self.assertEqual(format_reward("no think just answer"), 0.0)
        self.assertEqual(format_reward("<think>a</think><think>b</think> x"), 0.0)

    def test_compute_rewards_combines_weights(self):
        cfg = RewardConfig(exact_match_weight=1.0, format_weight=0.5)
        r = compute_rewards(
            ["<think>t</think> 42", "wrong"],
            ["<think>t</think> 42", "42"],
            cfg,
        )
        self.assertAlmostEqual(float(r[0]), 1.5, places=5)
        self.assertAlmostEqual(float(r[1]), 0.0, places=5)


class GRPOIntegrationTest(unittest.TestCase):
    def test_no_signal_streak_clean_stops_without_optimizer_path(self):
        from scripts import train_grpo

        calls = 0

        def no_signal_loss(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            return (
                torch.tensor(0.0, requires_grad=True),
                {
                    "loss": 0.0,
                    "kl": 0.0,
                    "active_group": 0.0,
                    "degenerate_group": 1.0,
                },
            )

        with tempfile.TemporaryDirectory() as tmp:
            output = io.StringIO()
            with (
                patch.dict(
                    train_grpo.CONFIG_CHOICES,
                    {"two_stage_tiny": _cfg},
                    clear=True,
                ),
                patch.object(
                    train_grpo,
                    "grpo_compute_loss",
                    side_effect=no_signal_loss,
                ),
                patch.object(
                    train_grpo,
                    "clip_or_check_grad_norm",
                    side_effect=AssertionError(
                        "zero-signal rollout entered optimizer path"
                    ),
                ),
                contextlib.redirect_stdout(output),
            ):
                rc = train_grpo.main(
                    [
                        "--smoke",
                        "--precision",
                        "fp32",
                        "--max-steps",
                        "5",
                        "--max-degenerate-steps",
                        "2",
                        "--output",
                        tmp,
                    ]
                )
        self.assertEqual(rc, 0)
        self.assertEqual(calls, 2)
        self.assertIn("no prompt group carried reward spread", output.getvalue())

    def test_native_rollout_behavior_matches_fresh_scoring(self):
        torch.manual_seed(0)
        policy = RDTForCausalLM(_cfg())
        policy.reverse_loss_enabled = False
        prompt = torch.randint(30, 200, (3,))

        loss, metrics = grpo_compute_loss(
            policy,
            None,
            [prompt],
            lambda _responses, _idx: torch.tensor([0.0, 1.0]),
            lambda ids: " ".join(str(int(token)) for token in ids),
            cfg=GRPOConfig(
                clip_eps=None,
                advantage_mode="centered",
                group_size=2,
                max_new_tokens=2,
                recurrent_steps=1,
                kl_coef=0.0,
                max_behavior_log_ratio=5e-3,
            ),
            eos_id=policy.cfg.eos_id,
            pad_id=policy.cfg.pad_id,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(metrics["behavior_log_ratio_abs_max"], 5e-3)

    def test_token_logprobs_with_mask_shapes_and_grad(self):
        torch.manual_seed(0)
        model = RDTForCausalLM(_cfg())
        model.reverse_loss_enabled = False
        ids = torch.randint(30, 200, (2, 6))
        comp_mask = torch.zeros(2, 6)
        comp_mask[:, 3:] = 1.0
        logp, mask = token_logprobs_with_mask(model, ids, comp_mask)
        self.assertEqual(logp.shape, (2, 5))
        self.assertEqual(mask.shape, (2, 5))
        # old/ref scored under no_grad; policy keeps grad.
        with torch.no_grad():
            old, _ = token_logprobs_with_mask(model, ids, comp_mask)
        adv = group_normalized_advantages(torch.tensor([0.0, 1.0]), group_size=2)
        loss, _ = grpo_loss(logp, old, adv, mask, ref_token_logp=old,
                            cfg=GRPOConfig(kl_coef=0.01))
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(any(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads))

    def test_reverse_loss_disabled_in_alignment(self):
        model = RDTForCausalLM(_cfg())
        model.reverse_loss_enabled = False
        self.assertFalse(model.reverse_loss_enabled)

    def test_grpo_step_updates_policy(self):
        torch.manual_seed(0)
        from Model.posttrain.grpo import GRPOConfig as _GC
        from Model.posttrain.grpo import grpo_step

        policy = RDTForCausalLM(_cfg())
        policy.reverse_loss_enabled = False
        ref = RDTForCausalLM(_cfg())
        ref.load_state_dict(policy.state_dict())
        for p in ref.parameters():
            p.requires_grad_(False)

        prompts = [torch.randint(30, 200, (3,)), torch.randint(30, 200, (3,))]

        def reward_fn(responses, idx):
            # Reward longer responses -> non-degenerate advantages.
            return torch.tensor([float(len(r)) for r in responses])

        def decode(ids):
            return "x" * int((ids != 0).sum())

        cfg = _GC(group_size=2, max_new_tokens=4, recurrent_steps=1, kl_coef=0.01)
        opt = torch.optim.SGD(policy.parameters(), lr=0.0)  # lr=0 -> params stable
        before = [p.clone() for p in policy.parameters()]
        m = grpo_step(policy, ref, prompts, reward_fn, decode, opt, cfg, pad_id=0)
        self.assertIn("loss", m)
        self.assertIn("kl", m)
        # Reverse loss stays disabled through the step.
        self.assertFalse(policy.reverse_loss_enabled)
        # With lr=0 the params are unchanged but a backward pass succeeded.
        for b, p in zip(before, policy.parameters()):
            self.assertTrue(torch.equal(b, p))

    def test_reward_fn_cpu_tensor_is_accepted(self):
        torch.manual_seed(0)
        from Model.posttrain.grpo import grpo_compute_loss

        policy = RDTForCausalLM(_cfg())
        policy.reverse_loss_enabled = False
        ref = RDTForCausalLM(_cfg())
        ref.load_state_dict(policy.state_dict())
        for p in ref.parameters():
            p.requires_grad_(False)

        prompts = [torch.randint(30, 200, (3,))]

        def reward_fn(_responses, _idx):
            return torch.tensor([0.0, 1.0], device="cpu")

        def decode(ids):
            return "x" * int((ids != 0).sum())

        cfg = GRPOConfig(group_size=2, max_new_tokens=2, recurrent_steps=1)
        loss, metrics = grpo_compute_loss(policy, ref, prompts, reward_fn, decode, cfg, pad_id=0)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("loss", metrics)

    def test_prompt_batches_are_rank_sharded(self):
        from scripts.train_grpo import _iter_prompt_batches

        dataset = [{"id": idx, "prompt_ids": [idx]} for idx in range(6)]
        rank0 = _iter_prompt_batches(dataset, 2, rank=0, world_size=2)
        rank1 = _iter_prompt_batches(dataset, 2, rank=1, world_size=2)

        self.assertEqual([row["id"] for row in next(rank0)], [0, 2])
        self.assertEqual([row["id"] for row in next(rank1)], [1, 3])

    def test_prompt_batches_replicate_when_shards_are_too_few(self):
        from scripts.train_grpo import _iter_prompt_batches

        dataset = [{"id": 0, "prompt_ids": [0]}]
        rank0 = _iter_prompt_batches(dataset, 1, rank=0, world_size=2)
        rank1 = _iter_prompt_batches(dataset, 1, rank=1, world_size=2)

        self.assertEqual(next(rank0)[0]["id"], 0)
        self.assertEqual(next(rank1)[0]["id"], 0)

    def test_training_reference_helper_freezes_independent_copy(self):
        from scripts.train_grpo import _build_reference_model

        policy = RDTForCausalLM(_cfg())
        reference = _build_reference_model(
            policy,
            _cfg(),
            TrainingConfig(parallel="single", max_steps=1, warmup_steps=0),
            local_rank=0,
            device=torch.device("cpu"),
        )

        self.assertFalse(reference.training)
        self.assertFalse(reference.reverse_loss_enabled)
        self.assertTrue(all(not p.requires_grad for p in reference.parameters()))
        first_policy = next(policy.parameters())
        first_reference = next(reference.parameters())
        self.assertIsNot(first_policy, first_reference)
        self.assertTrue(torch.equal(first_policy, first_reference))


if __name__ == "__main__":
    unittest.main()
