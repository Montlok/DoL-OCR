# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest
import weakref
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from Model.posttrain import ocr_anyres_grpo
from Model.posttrain import ocr_anyres_reward
from Model.posttrain.grpo import GRPOConfig
from Model.posttrain.logprobs import sequence_logprobs
from Model.posttrain.ocr_anyres_grpo import (
    AnyresGRPOAdmission,
    anyres_grpo_compute_loss,
    repeat_packed_detail_prompt_major,
)
from Model.posttrain.ocr_anyres_reward import AnyresOCRRewardAdapter


class _Vision(nn.Module):
    def __init__(self, owner) -> None:
        super().__init__()
        self._owner_ref = weakref.ref(owner)
        self.global_tower = nn.Linear(1, 2, bias=False)

    def encode_visual(self, pixels):
        values = pixels["unique_global"]
        self._owner_ref().global_calls.append(
            (int(values.shape[0]), torch.is_grad_enabled())
        )
        return self.global_tower(values)


class _TinyAnyresPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cfg = SimpleNamespace(
            eos_id=3,
            pad_id=0,
            ignore_index=-100,
            vocab_size=512,
            max_seq_len=16,
            recurrent_steps=1,
        )
        self.global_calls: list[tuple[int, bool]] = []
        self.native_calls: list[tuple[int, bool]] = []
        self.vision = _Vision(self)
        self.native_tower = nn.Linear(1, 2, bias=False)
        self.bridge = nn.Linear(2, 2, bias=False)
        self.embed = nn.Embedding(512, 2)
        self.head = nn.Linear(2, 512, bias=False)
        self.reverse_loss_enabled = True

    def forward(self, input_ids, **kwargs):
        self.assert_contract(kwargs)
        global_features = kwargs["visual_features"]
        detail_memory = kwargs["detail_memory"]
        detail_cu = kwargs["detail_cu_seqlens"]
        boundaries = [int(value) for value in detail_cu.detach().cpu().tolist()]
        detail_rows = torch.stack(
            [
                detail_memory[boundaries[index]:boundaries[index + 1]].mean(dim=0)
                for index in range(input_ids.shape[0])
            ],
            dim=0,
        )
        context = global_features + self.bridge(detail_rows)
        hidden = self.embed(input_ids) + context.unsqueeze(1)
        return {"logits": self.head(hidden), "loss_parts": {}}

    @staticmethod
    def assert_contract(kwargs) -> None:
        if kwargs.get("position_contract") != "boundary_v1":
            raise AssertionError("position contract drift")
        if kwargs.get("detail_memory") is None:
            raise AssertionError("missing detail memory")


def _batch() -> dict:
    input_ids = torch.tensor(
        [[1, 2, 6, 300, 3], [1, 2, 6, 301, 3]],
        dtype=torch.long,
    )
    labels = torch.tensor(
        [[-100, -100, -100, 300, 3], [-100, -100, -100, 301, 3]],
        dtype=torch.long,
    )
    return {
        "dataset_contract_sha256": "b" * 64,
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
        "position_contract": "boundary_v1",
        "sample_metadata": [
            {
                "reference_model": {"text": "4"},
                "preprocess_contract_sha256": "4" * 64,
            },
            {
                "reference_model": {"text": "4"},
                "preprocess_contract_sha256": "4" * 64,
            },
        ],
        "unique_global": torch.tensor([[1.0], [2.0]]),
        "unique_detail": torch.tensor([[3.0], [9.0]]),
    }


def _fake_encode(model, batch, device):
    values = batch["unique_detail"].to(device)
    model.native_calls.append((int(values.shape[0]), torch.is_grad_enabled()))
    detail = model.native_tower(values)
    return {
        "global_pixel_values": {
            "unique_global": batch["unique_global"].to(device)
        },
        "detail_memory": detail,
        "detail_cu_seqlens": torch.arange(
            values.shape[0] + 1,
            dtype=torch.int32,
            device="cpu",
        ),
    }


def _deterministic_multinomial(probabilities, num_samples):
    del num_samples
    pattern = torch.tensor([300, 301, 300, 301], device=probabilities.device)
    return pattern[: probabilities.shape[0]].unsqueeze(1)


def _encode(text: str) -> list[int]:
    return [296 + int(character) for character in text]


def _decode(ids) -> str:
    return "".join(str(int(value) - 296) for value in ids)


class _RecordingAdapter(AnyresOCRRewardAdapter):
    def __init__(self) -> None:
        super().__init__(
            eos_id=3,
            valid_token_ids=range(512),
            tokenizer_encode=_encode,
            tokenizer_decode=_decode,
            tokenizer_contract_sha256="a" * 64,
            _factory_token=ocr_anyres_reward._REVIEWED_FACTORY_TOKEN,
        )
        self.observed = {}

    def score_group(self, references, completion_ids, eos_mask):
        result = super().score_group(references, completion_ids, eos_mask)
        self.observed = {
            "responses": list(result["responses"]),
            "references": list(references),
            "tails": completion_ids.detach().cpu().clone(),
            "eos_mask": eos_mask.detach().cpu().clone(),
        }
        return result


class _GradientRewardAdapter(_RecordingAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.source = torch.tensor(
            [0.0, 1.0, 0.0, 1.0],
            requires_grad=True,
        )

    def score_group(self, references, completion_ids, eos_mask):
        responses = [self.decode_completion(row) for row in completion_ids]
        return {
            "group_rewards": self.source,
            "responses": responses,
            "samples": [
                {"diagnostics": {"invalid": False, "exact": False}}
                for _ in responses
            ],
            "config": {},
        }


def _admission(
    adapter: AnyresOCRRewardAdapter,
    policy: nn.Module,
    *,
    max_new_tokens: int = 1,
    with_reference: bool,
) -> AnyresGRPOAdmission:
    return AnyresGRPOAdmission(
        policy_checkpoint_sha256="1" * 64,
        policy_metadata_sha256="7" * 64,
        reference_checkpoint_sha256="2" * 64,
        reference_metadata_sha256="7" * 64,
        joint_stage_result_sha256="8" * 64,
        tokenizer_contract_sha256="a" * 64,
        visual_contract_sha256="3" * 64,
        preprocess_contract_sha256="4" * 64,
        native_migration_receipt_sha256="5" * 64,
        trainability_contract_sha256="6" * 64,
        trainable_parameter_names_sha256=ocr_anyres_grpo._canonical_sha256(
            sorted(
                name
                for name, parameter in policy.named_parameters()
                if parameter.requires_grad
            )
        ),
        reward_contract_sha256=adapter.contract["canonical_sha256"],
        dataset_admission_report_sha256="9" * 64,
        train_dataset_contract_sha256="b" * 64,
        sft_validation_dataset_contract_sha256="c" * 64,
        kl_selection_dataset_contract_sha256="0" * 64,
        formal_monitor_dataset_contract_sha256="1" * 64,
        text_replay_train_contract_sha256="d" * 64,
        text_replay_sft_validation_contract_sha256="e" * 64,
        text_replay_kl_selection_contract_sha256="f" * 64,
        text_replay_formal_monitor_contract_sha256="0" * 64,
        runtime_source_receipt_sha256="f" * 64,
        recommended_max_new_tokens=max_new_tokens,
    )


class AnyresGRPOTest(unittest.TestCase):
    def test_admission_binds_four_distinct_image_and_text_splits(self) -> None:
        policy, _reference = self._models()
        admission = _admission(
            _RecordingAdapter(),
            policy,
            with_reference=True,
        )
        self.assertNotIn(
            "validation_dataset_contract_sha256",
            admission.canonical_payload,
        )
        self.assertEqual(
            {
                admission.train_dataset_contract_sha256,
                admission.sft_validation_dataset_contract_sha256,
                admission.kl_selection_dataset_contract_sha256,
                admission.formal_monitor_dataset_contract_sha256,
            },
            {"b" * 64, "c" * 64, "0" * 64, "1" * 64},
        )
        self.assertNotIn(
            "text_replay_validation_contract_sha256",
            admission.canonical_payload,
        )
        self.assertEqual(
            {
                admission.text_replay_train_contract_sha256,
                admission.text_replay_sft_validation_contract_sha256,
                admission.text_replay_kl_selection_contract_sha256,
                admission.text_replay_formal_monitor_contract_sha256,
            },
            {"d" * 64, "e" * 64, "f" * 64, "0" * 64},
        )
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            replace(
                admission,
                kl_selection_dataset_contract_sha256=(
                    admission.sft_validation_dataset_contract_sha256
                ),
            )
        with self.assertRaisesRegex(ValueError, "image train.*must be distinct"):
            replace(
                admission,
                train_dataset_contract_sha256=(
                    admission.sft_validation_dataset_contract_sha256
                ),
            )
        with self.assertRaisesRegex(ValueError, "text train.*must be distinct"):
            replace(
                admission,
                text_replay_kl_selection_contract_sha256=(
                    admission.text_replay_sft_validation_contract_sha256
                ),
            )

    def _models(self):
        torch.manual_seed(4)
        policy = _TinyAnyresPolicy()
        reference = _TinyAnyresPolicy()
        reference.load_state_dict(policy.state_dict())
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        return policy, reference

    def test_unique_encoding_behavior_alignment_gradients_and_reward_contract(self):
        policy, reference = self._models()
        adapter = _RecordingAdapter()

        cfg = GRPOConfig(
            group_size=2,
            max_new_tokens=1,
            recurrent_steps=1,
            clip_eps=None,
            kl_coef=0.04,
            advantage_mode="centered",
            min_reward_spread=0.005,
            max_behavior_log_ratio=1e-5,
        )
        with (
            mock.patch.object(
                ocr_anyres_grpo,
                "encode_anyres_visual_batch",
                side_effect=_fake_encode,
            ),
            mock.patch.object(
                torch,
                "multinomial",
                side_effect=_deterministic_multinomial,
            ),
        ):
            loss, metrics = anyres_grpo_compute_loss(
                policy,
                reference,
                _batch(),
                adapter,
                _admission(adapter, policy, with_reference=True),
                cfg,
                "cpu",
            )
        loss.backward()

        self.assertEqual(policy.native_calls, [(2, False), (2, True)])
        self.assertEqual(policy.global_calls, [(2, False), (2, True)])
        self.assertEqual(reference.native_calls, [(2, False)])
        self.assertEqual(reference.global_calls, [(2, False)])
        self.assertGreater(float(policy.native_tower.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(policy.bridge.weight.grad.abs().sum()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in reference.parameters()))
        self.assertLessEqual(metrics["behavior_log_ratio_abs_max"], 1e-5)
        self.assertEqual(
            adapter.observed["references"],
            ["4", "4", "4", "4"],
        )
        self.assertEqual(
            adapter.observed["tails"].tolist(),
            [[300], [301], [300], [301]],
        )
        self.assertFalse(bool(adapter.observed["eos_mask"].any()))

    def test_kl_zero_skips_reference_completely(self):
        policy, reference = self._models()
        for parameter in reference.parameters():
            parameter.requires_grad_(True)
        reference.train()
        adapter = _RecordingAdapter()
        cfg = GRPOConfig(
            group_size=2,
            max_new_tokens=1,
            recurrent_steps=1,
            clip_eps=None,
            kl_coef=0.0,
            advantage_mode="centered",
            min_reward_spread=0.005,
            max_behavior_log_ratio=1e-5,
        )
        with (
            mock.patch.object(
                ocr_anyres_grpo,
                "encode_anyres_visual_batch",
                side_effect=_fake_encode,
            ),
            mock.patch.object(
                torch,
                "multinomial",
                side_effect=_deterministic_multinomial,
            ),
        ):
            loss, _ = anyres_grpo_compute_loss(
                policy,
                reference,
                _batch(),
                adapter,
                _admission(adapter, policy, with_reference=False),
                cfg,
                "cpu",
            )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(reference.native_calls, [])
        self.assertEqual(reference.global_calls, [])
        self.assertTrue(reference.training)

    def test_context_overflow_fails_before_any_visual_or_reference_work(self):
        policy, reference = self._models()
        policy.cfg.max_seq_len = 3
        adapter = _RecordingAdapter()
        cfg = GRPOConfig(
            group_size=2,
            max_new_tokens=1,
            clip_eps=None,
            kl_coef=0.0,
            advantage_mode="centered",
            min_reward_spread=0.005,
            max_behavior_log_ratio=1e-5,
        )
        with self.assertRaisesRegex(ValueError, "exceeds policy max_seq_len"):
            anyres_grpo_compute_loss(
                policy,
                reference,
                _batch(),
                adapter,
                _admission(adapter, policy, with_reference=False),
                cfg,
                "cpu",
            )
        self.assertEqual(policy.native_calls, [])
        self.assertEqual(reference.native_calls, [])

    def test_on_policy_behavior_and_single_process_gates(self):
        policy, _ = self._models()
        adapter = _RecordingAdapter()
        cases = (
            (
                GRPOConfig(
                    group_size=2,
                    max_new_tokens=1,
                    clip_eps=0.2,
                    kl_coef=0.0,
                    advantage_mode="centered",
                    min_reward_spread=0.005,
                    max_behavior_log_ratio=1e-5,
                ),
                policy,
                "clip_eps=None",
            ),
            (
                GRPOConfig(
                    group_size=2,
                    max_new_tokens=1,
                    clip_eps=None,
                    kl_coef=0.0,
                    advantage_mode="centered",
                    min_reward_spread=0.005,
                ),
                policy,
                "behavior-log-ratio gate",
            ),
            (
                GRPOConfig(
                    group_size=2,
                    max_new_tokens=1,
                    clip_eps=None,
                    kl_coef=0.0,
                    advantage_mode="centered",
                    min_reward_spread=0.005,
                    max_behavior_log_ratio=1e-5,
                ),
                SimpleNamespace(module=policy),
                "unwrapped single-process policy",
            ),
        )
        for cfg, candidate, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    anyres_grpo_compute_loss(
                        candidate,
                        None,
                        _batch(),
                        adapter,
                        _admission(adapter, policy, with_reference=False),
                        cfg,
                        "cpu",
                    )
        self.assertEqual(policy.native_calls, [])

    def test_admission_tool_mask_and_reward_gradient_fail_closed(self):
        policy, _ = self._models()
        adapter = _RecordingAdapter()
        cfg = GRPOConfig(
            group_size=2,
            max_new_tokens=1,
            recurrent_steps=1,
            clip_eps=None,
            kl_coef=0.0,
            advantage_mode="centered",
            min_reward_spread=0.005,
            max_behavior_log_ratio=1e-5,
        )
        bad_admission = replace(
            _admission(adapter, policy, with_reference=False),
            reward_contract_sha256="0" * 64,
        )
        with self.assertRaisesRegex(ValueError, "reward adapter contract"):
            anyres_grpo_compute_loss(
                policy,
                None,
                _batch(),
                adapter,
                bad_admission,
                cfg,
                "cpu",
            )
        tool_cfg = replace(cfg, tool_result_open_ids=[1])
        with self.assertRaisesRegex(ValueError, "forbids tool-result"):
            anyres_grpo_compute_loss(
                policy,
                None,
                _batch(),
                adapter,
                _admission(adapter, policy, with_reference=False),
                tool_cfg,
                "cpu",
            )
        self.assertEqual(policy.native_calls, [])

        gradient_adapter = _GradientRewardAdapter()
        with (
            mock.patch.object(
                ocr_anyres_grpo,
                "encode_anyres_visual_batch",
                side_effect=_fake_encode,
            ),
            mock.patch.object(
                torch,
                "multinomial",
                side_effect=_deterministic_multinomial,
            ),
        ):
            loss, _ = anyres_grpo_compute_loss(
                policy,
                None,
                _batch(),
                gradient_adapter,
                _admission(gradient_adapter, policy, with_reference=False),
                cfg,
                "cpu",
            )
        loss.backward()
        self.assertIsNone(gradient_adapter.source.grad)

    def test_detail_repeat_is_prompt_major_and_second_sample_isolated(self):
        memory = torch.tensor([[10.0], [20.0], [21.0]])
        cu = torch.tensor([0, 1, 3], dtype=torch.int32)
        repeated, repeated_cu = repeat_packed_detail_prompt_major(memory, cu, 2)
        self.assertEqual(repeated[:, 0].tolist(), [10.0, 10.0, 20.0, 21.0, 20.0, 21.0])
        self.assertEqual(repeated_cu.tolist(), [0, 1, 2, 4, 6])
        repeated[0, 0] = -1.0
        self.assertEqual(repeated[2:, 0].tolist(), [20.0, 21.0, 20.0, 21.0])

    def test_completion_slice_matches_full_with_preencoded_anyres_inputs(self):
        policy, _ = self._models()
        batch = _batch()
        unique = _fake_encode(policy, batch, torch.device("cpu"))
        global_features = policy.vision.encode_visual(unique["global_pixel_values"])
        detail, cu = repeat_packed_detail_prompt_major(
            unique["detail_memory"],
            unique["detail_cu_seqlens"],
            2,
        )
        global_features = global_features.repeat_interleave(2, dim=0)
        ids = torch.tensor(
            [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 3, 4], [1, 2, 3, 5]],
            dtype=torch.long,
        )
        full = sequence_logprobs(
            policy,
            ids,
            position_contract="boundary_v1",
            visual_features=global_features,
            detail_memory=detail,
            detail_cu_seqlens=cu,
        )
        sliced = sequence_logprobs(
            policy,
            ids,
            completion_start=2,
            position_contract="boundary_v1",
            visual_features=global_features,
            detail_memory=detail,
            detail_cu_seqlens=cu,
        )
        self.assertTrue(torch.equal(sliced[:, :2], torch.zeros_like(sliced[:, :2])))
        torch.testing.assert_close(sliced[:, 2:], full[:, 2:], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
