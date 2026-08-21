# -*- coding: utf-8 -*-

from __future__ import annotations

import types
import unittest
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.ocr.position_contract import BOUNDARY_V1
from Model.posttrain import ocr_anyres_grpo, ocr_anyres_grpo_trainer
from Model.posttrain import ocr_anyres_reward
from Model.posttrain.grpo import GRPOConfig
from Model.posttrain.ocr_anyres_grpo import AnyresGRPOAdmission
from Model.posttrain.ocr_anyres_grpo_trainer import (
    AnyresGRPOKLAblationTrial,
    train_anyres_grpo_cycle,
)
from Model.posttrain.ocr_anyres_reward import AnyresOCRRewardAdapter


class _TinyVision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tower = nn.Linear(1, 2, bias=False)
        self.projector = nn.Linear(2, 2, bias=False)

    def encode_visual(self, pixels):
        return self.projector(self.tower(pixels["unique_global"]))


class _TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cfg = types.SimpleNamespace(
            bos_id=1,
            eos_id=3,
            pad_id=0,
            ignore_index=-100,
            vocab_size=512,
            max_seq_len=16,
            recurrent_steps=1,
            d_model=2,
        )
        self.vision = _TinyVision()
        self.native_tower = nn.Linear(1, 2, bias=False)
        self.bridge = nn.Linear(2, 2, bias=False)
        self.embed = nn.Embedding(512, 2)
        self.head = nn.Linear(2, 512, bias=False)
        self.reverse_loss_enabled = False
        self.text_contracts: list[tuple[str | None, bool, int | None]] = []

    def forward(
        self,
        input_ids,
        *,
        attention_mask=None,
        labels=None,
        position_contract=None,
        return_logits=True,
        loss_chunk_size=None,
        **kwargs,
    ):
        if position_contract != BOUNDARY_V1:
            raise AssertionError("position contract drift")
        if labels is not None:
            self.text_contracts.append(
                (position_contract, return_logits, loss_chunk_size)
            )
            logits = self.head(self.embed(input_ids))
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=self.cfg.ignore_index,
            )
            return {"loss": loss, "logits": logits if return_logits else None}

        global_features = kwargs.get("visual_features")
        detail_memory = kwargs.get("detail_memory")
        detail_cu = kwargs.get("detail_cu_seqlens")
        if global_features is None or detail_memory is None or detail_cu is None:
            raise AssertionError("missing anyres visual memory")
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


class _GradientRewardAdapter(AnyresOCRRewardAdapter):
    def __init__(self) -> None:
        super().__init__(
            eos_id=3,
            valid_token_ids=range(512),
            tokenizer_encode=_encode,
            tokenizer_decode=_decode,
            tokenizer_contract_sha256="a" * 64,
            _factory_token=ocr_anyres_reward._REVIEWED_FACTORY_TOKEN,
        )
        self.source = torch.tensor(
            [0.0, 1.0, 0.0, 1.0],
            requires_grad=True,
        )

    def score_group(self, references, completion_ids, eos_mask):
        del references, eos_mask
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


class _Scheduler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1


class _Scaler:
    def __init__(self) -> None:
        self.scale_calls = 0
        self.unscale_calls = 0
        self.step_calls = 0
        self.update_calls = 0

    def scale(self, loss):
        self.scale_calls += 1
        return loss

    def unscale_(self, _optimizer) -> None:
        self.unscale_calls += 1

    def step(self, optimizer) -> None:
        self.step_calls += 1
        optimizer.step()

    def update(self) -> None:
        self.update_calls += 1

    def get_scale(self) -> float:
        return 8.0


def _encode(text: str) -> list[int]:
    return [296 + int(character) for character in text]


def _decode(ids) -> str:
    return "".join(str(int(value) - 296) for value in ids)


def _ocr_batch() -> dict[str, object]:
    input_ids = torch.tensor(
        [[1, 2, 6, 300, 3], [1, 2, 6, 301, 3]],
        dtype=torch.long,
    )
    return {
        "dataset_contract_sha256": "c" * 64,
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": torch.tensor(
            [
                [-100, -100, -100, 300, 3],
                [-100, -100, -100, 301, 3],
            ],
            dtype=torch.long,
        ),
        "position_contract": BOUNDARY_V1,
        "sample_metadata": [
            {
                "reference_model": {"text": "4"},
                "preprocess_contract_sha256": "7" * 64,
            },
            {
                "reference_model": {"text": "4"},
                "preprocess_contract_sha256": "7" * 64,
            },
        ],
        "unique_global": torch.tensor([[1.0], [2.0]]),
        "unique_detail": torch.tensor([[3.0], [9.0]]),
    }


def _text_batch() -> dict[str, object]:
    ids = torch.tensor([[1, 300, 301, 3]], dtype=torch.long)
    return {
        "dataset_contract_sha256": "e" * 64,
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "labels": ids.clone(),
        "position_contract": BOUNDARY_V1,
    }


def _fake_encode(model, batch, device):
    detail_values = batch["unique_detail"].to(device)
    return {
        "global_pixel_values": {
            "unique_global": batch["unique_global"].to(device),
        },
        "detail_memory": model.native_tower(detail_values),
        "detail_cu_seqlens": torch.arange(
            detail_values.shape[0] + 1,
            dtype=torch.int32,
            device="cpu",
        ),
    }


def _deterministic_multinomial(probabilities, num_samples):
    del num_samples
    pattern = torch.tensor([300, 301, 300, 301], device=probabilities.device)
    return pattern[: probabilities.shape[0]].unsqueeze(1)


def _models() -> tuple[_TinyPolicy, _TinyPolicy]:
    torch.manual_seed(19)
    policy = _TinyPolicy()
    reference = _TinyPolicy()
    reference.load_state_dict(policy.state_dict())
    reference.requires_grad_(False)
    return policy, reference


def _optimizer(policy: _TinyPolicy) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        [
            {
                "params": [policy.embed.weight, policy.head.weight],
                "lr": 0.01,
                "ocr_joint_role": "lm",
                "ocr_joint_base_lr": 0.01,
            },
            {
                "params": [
                    policy.vision.tower.weight,
                    policy.native_tower.weight,
                ],
                "lr": 0.02,
                "ocr_joint_role": "tower",
                "ocr_joint_base_lr": 0.02,
            },
            {
                "params": [policy.vision.projector.weight],
                "lr": 0.03,
                "ocr_joint_role": "projector",
                "ocr_joint_base_lr": 0.03,
            },
            {
                "params": [policy.bridge.weight],
                "lr": 0.04,
                "ocr_joint_role": "bridge",
                "ocr_joint_base_lr": 0.04,
            },
        ]
    )


def _cfg(*, kl_coef: float = 0.04) -> GRPOConfig:
    return GRPOConfig(
        group_size=2,
        max_new_tokens=1,
        recurrent_steps=1,
        clip_eps=None,
        kl_coef=kl_coef,
        advantage_mode="centered",
        min_reward_spread=0.005,
        max_behavior_log_ratio=1e-5,
    )


def _admission(
    adapter: AnyresOCRRewardAdapter,
    policy: nn.Module,
    *,
    with_reference: bool = True,
) -> AnyresGRPOAdmission:
    return AnyresGRPOAdmission(
        policy_checkpoint_sha256="1" * 64,
        policy_metadata_sha256="2" * 64,
        reference_checkpoint_sha256="3" * 64,
        reference_metadata_sha256="4" * 64,
        joint_stage_result_sha256="5" * 64,
        tokenizer_contract_sha256="a" * 64,
        visual_contract_sha256="6" * 64,
        preprocess_contract_sha256="7" * 64,
        native_migration_receipt_sha256="8" * 64,
        trainability_contract_sha256="9" * 64,
        trainable_parameter_names_sha256=ocr_anyres_grpo._canonical_sha256(
            sorted(
                name
                for name, parameter in policy.named_parameters()
                if parameter.requires_grad
            )
        ),
        reward_contract_sha256=adapter.contract["canonical_sha256"],
        dataset_admission_report_sha256="b" * 64,
        train_dataset_contract_sha256="c" * 64,
        sft_validation_dataset_contract_sha256="d" * 64,
        kl_selection_dataset_contract_sha256="1" * 64,
        formal_monitor_dataset_contract_sha256="2" * 64,
        text_replay_train_contract_sha256="e" * 64,
        text_replay_sft_validation_contract_sha256="f" * 64,
        text_replay_kl_selection_contract_sha256="0" * 64,
        text_replay_formal_monitor_contract_sha256="a" * 64,
        runtime_source_receipt_sha256="0" * 64,
        recommended_max_new_tokens=1,
    )


class AnyresGRPOTrainerTest(unittest.TestCase):
    def test_real_grpo_and_text_cycle_has_detached_reward_and_one_step(self) -> None:
        policy, reference = _models()
        adapter = _GradientRewardAdapter()
        optimizer = _optimizer(policy)
        scheduler = _Scheduler()
        scaler = _Scaler()
        trial = AnyresGRPOKLAblationTrial()
        real_compute = ocr_anyres_grpo_trainer.anyres_grpo_compute_loss

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
            mock.patch.object(
                ocr_anyres_grpo_trainer,
                "anyres_grpo_compute_loss",
                wraps=real_compute,
            ) as compute_loss,
            mock.patch.object(optimizer, "step", wraps=optimizer.step) as step,
        ):
            metrics = train_anyres_grpo_cycle(
                policy,
                reference,
                _ocr_batch(),
                adapter,
                _admission(adapter, policy),
                _cfg(),
                optimizer,
                kl_trial=trial,
                text_batch=_text_batch(),
                scaler=scaler,
                scheduler=scheduler,
            )

        self.assertEqual(compute_loss.call_count, 1)
        self.assertEqual(step.call_count, 1)
        self.assertEqual(scaler.scale_calls, 2)
        self.assertEqual(scaler.unscale_calls, 1)
        self.assertEqual(scaler.step_calls, 1)
        self.assertEqual(scaler.update_calls, 1)
        self.assertEqual(scheduler.steps, 1)
        self.assertIsNone(adapter.source.grad)
        self.assertTrue(
            all(parameter.grad is None for parameter in reference.parameters())
        )
        self.assertEqual(
            policy.text_contracts,
            [(BOUNDARY_V1, False, 4096)],
        )
        for module in (
            policy.embed,
            policy.head,
            policy.vision.tower,
            policy.vision.projector,
            policy.native_tower,
            policy.bridge,
        ):
            self.assertGreater(
                sum(
                    float(parameter.grad.detach().abs().sum())
                    for parameter in module.parameters()
                    if parameter.grad is not None
                ),
                0.0,
            )
        self.assertEqual(
            [group["role"] for group in metrics["optimizer_groups"]],
            ["lm", "tower", "projector", "bridge"],
        )
        self.assertEqual(metrics["optimizer_steps"], 1)
        self.assertEqual(metrics["grpo_compute_calls"], 1)
        self.assertEqual(
            metrics["kl_ablation"]["selected_kl_coef"],
            _cfg().kl_coef,
        )
        self.assertEqual(
            metrics["kl_ablation"]["execution_contract"],
            "one_selected_kl_per_optimizer_cycle",
        )

    def test_nonfinite_grpo_loss_never_steps(self) -> None:
        policy, reference = _models()
        adapter = _GradientRewardAdapter()
        optimizer = _optimizer(policy)
        scheduler = _Scheduler()
        parameter = next(policy.parameters())
        nonfinite = parameter.sum() * torch.tensor(float("nan"))

        with (
            mock.patch.object(
                ocr_anyres_grpo_trainer,
                "anyres_grpo_compute_loss",
                return_value=(nonfinite, {}),
            ),
            mock.patch.object(optimizer, "step", wraps=optimizer.step) as step,
        ):
            with self.assertRaisesRegex(FloatingPointError, "non-finite"):
                train_anyres_grpo_cycle(
                    policy,
                    reference,
                    _ocr_batch(),
                    adapter,
                    _admission(adapter, policy),
                    _cfg(),
                    optimizer,
                    kl_trial=AnyresGRPOKLAblationTrial(),
                    scheduler=scheduler,
                )
        step.assert_not_called()
        self.assertEqual(scheduler.steps, 0)
        self.assertTrue(
            all(parameter.grad is None for parameter in policy.parameters())
        )

    def test_late_nonfinite_text_loss_clears_grpo_gradients_without_step(self) -> None:
        policy, _ = _models()
        adapter = _GradientRewardAdapter()
        optimizer = _optimizer(policy)
        finite_grpo = policy.embed.weight.sum().square()

        def nonfinite_text(*_args, **_kwargs):
            return policy.head.weight.sum() * torch.tensor(float("nan"))

        with (
            mock.patch.object(
                ocr_anyres_grpo_trainer,
                "anyres_grpo_compute_loss",
                return_value=(finite_grpo, {"active_group_frac": 1.0}),
            ),
            mock.patch.object(
                ocr_anyres_grpo_trainer,
                "_forward_text_ce",
                side_effect=nonfinite_text,
            ),
            mock.patch.object(optimizer, "step", wraps=optimizer.step) as step,
        ):
            with self.assertRaisesRegex(FloatingPointError, "text replay"):
                train_anyres_grpo_cycle(
                    policy,
                    None,
                    _ocr_batch(),
                    adapter,
                    _admission(adapter, policy, with_reference=False),
                    _cfg(kl_coef=0.0),
                    optimizer,
                    kl_trial=AnyresGRPOKLAblationTrial(selected_index=2),
                    text_batch=_text_batch(),
                )
        step.assert_not_called()
        self.assertTrue(
            all(parameter.grad is None for parameter in policy.parameters())
        )

    def test_zero_reward_spread_skips_weight_decay_and_scheduler(self) -> None:
        policy, _ = _models()
        adapter = _GradientRewardAdapter()
        optimizer = torch.optim.AdamW(
            [parameter for parameter in policy.parameters() if parameter.requires_grad],
            lr=0.01,
            weight_decay=0.1,
        )
        scheduler = _Scheduler()
        before = {
            name: parameter.detach().clone()
            for name, parameter in policy.named_parameters()
        }
        zero_loss = policy.embed.weight.sum() * 0.0
        with mock.patch.object(
            ocr_anyres_grpo_trainer,
            "anyres_grpo_compute_loss",
            return_value=(zero_loss, {"active_group_frac": 0.0}),
        ), mock.patch.object(optimizer, "step", wraps=optimizer.step) as step:
            metrics = train_anyres_grpo_cycle(
                policy,
                None,
                _ocr_batch(),
                adapter,
                _admission(adapter, policy, with_reference=False),
                _cfg(kl_coef=0.0),
                optimizer,
                kl_trial=AnyresGRPOKLAblationTrial(selected_index=2),
                scheduler=scheduler,
            )
        step.assert_not_called()
        self.assertEqual(scheduler.steps, 0)
        self.assertFalse(metrics["stepped"])
        self.assertEqual(metrics["skip_reason"], "no_active_reward_groups")
        for name, parameter in policy.named_parameters():
            self.assertTrue(torch.equal(parameter, before[name]), name)

    def test_contract_gates_run_before_grpo_compute(self) -> None:
        policy, reference = _models()
        adapter = _GradientRewardAdapter()
        optimizer = _optimizer(policy)
        cases = (
            (
                {**_ocr_batch(), "position_contract": "legacy_sequential_v0"},
                reference,
                AnyresGRPOKLAblationTrial(),
                "boundary_v1",
            ),
            (
                _ocr_batch(),
                reference,
                AnyresGRPOKLAblationTrial(selected_index=1),
                "selected KL",
            ),
        )
        with mock.patch.object(
            ocr_anyres_grpo_trainer,
            "anyres_grpo_compute_loss",
        ) as compute_loss:
            for batch, candidate_reference, trial, message in cases:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        train_anyres_grpo_cycle(
                            policy,
                            candidate_reference,
                            batch,
                            adapter,
                            _admission(adapter, policy),
                            _cfg(),
                            optimizer,
                            kl_trial=trial,
                        )
        compute_loss.assert_not_called()

        policy, reference = _models()
        reference.register_buffer(
            "wrong_device_buffer",
            torch.empty(1, device="meta"),
        )
        with mock.patch.object(
            ocr_anyres_grpo_trainer,
            "anyres_grpo_compute_loss",
        ) as compute_loss:
            with self.assertRaisesRegex(ValueError, "span devices"):
                train_anyres_grpo_cycle(
                    policy,
                    reference,
                    _ocr_batch(),
                    adapter,
                    _admission(adapter, policy),
                    _cfg(),
                    _optimizer(policy),
                    kl_trial=AnyresGRPOKLAblationTrial(),
                )
        compute_loss.assert_not_called()


if __name__ == "__main__":
    unittest.main()
