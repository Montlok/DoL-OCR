# -*- coding: utf-8 -*-

from __future__ import annotations

import types
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.layers.vision_cross_attention import RaggedVisionCrossAttention
from Model.ocr.position_contract import BOUNDARY_V1
from Model.posttrain.ocr_joint_trainer import (
    configure_joint_stage_trainable,
    configure_visual_stage_trainable,
    train_joint_cycle,
    train_visual_cycle,
)


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cfg = types.SimpleNamespace(ignore_index=-100)
        self.embed = nn.Embedding(16, 4)
        self.lm_body = nn.Linear(4, 4)
        self.lm_head = nn.Linear(4, 16, bias=False)
        self.vision = nn.Module()
        self.vision.encoder = nn.Linear(4, 4)
        self.vision.omvt = nn.Module()
        self.vision.omvt.tower = nn.Linear(4, 4)
        self.vision.omvt.projector = nn.Linear(4, 4)
        self.vision.native_detail_tower = nn.Linear(4, 4)
        self.vision_cross_attention = RaggedVisionCrossAttention(
            d_model=4,
            memory_dim=4,
            n_heads=1,
        )
        self.reverse_loss_enabled = False
        self.position_contracts: list[str | None] = []
        self.text_forward_contracts: list[tuple[bool, int | None]] = []

    def forward(
        self,
        *,
        input_ids,
        attention_mask=None,
        labels=None,
        position_contract=None,
        return_logits=True,
        loss_chunk_size=None,
        **_kwargs,
    ):
        self.position_contracts.append(position_contract)
        hidden = self.lm_body(self.embed(input_ids))
        logits = self.lm_head(hidden)
        if labels is None:
            return {"logits": logits}
        self.text_forward_contracts.append((return_logits, loss_chunk_size))
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            ignore_index=self.cfg.ignore_index,
        )
        return {"loss": loss, "logits": logits if return_logits else None}


def _visual_batch(seed: int = 0) -> dict[str, object]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "position_contract": BOUNDARY_V1,
        "native": torch.randn(3, 4, generator=generator),
        "h": torch.randn(1, 2, 4, generator=generator),
        "target": torch.randn(1, 2, 4, generator=generator),
        "global": torch.randn(3, 4, generator=generator),
        "cu": torch.tensor([0, 3], dtype=torch.int32),
    }


def _bridge_only_ocr(model, batch, **_kwargs):
    memory = model.vision.native_detail_tower(batch["native"])
    prediction = model.vision_cross_attention(
        batch["h"],
        memory,
        batch["cu"],
    )
    return {"loss": F.mse_loss(prediction, batch["target"])}


def _joint_ocr(model, batch, **_kwargs):
    memory = model.vision.native_detail_tower(batch["native"])
    detail = model.vision_cross_attention(
        batch["h"],
        memory,
        batch["cu"],
    )
    global_feature = model.vision.omvt.projector(
        model.vision.omvt.tower(batch["global"])
    )
    legacy_feature = model.vision.encoder(batch["global"])
    lm_feature = model.lm_body(model.embed(torch.tensor([[2, 3]]))).mean()
    prediction = (
        detail.mean()
        + global_feature.mean()
        + legacy_feature.mean()
        + lm_feature
    )
    return {"loss": (prediction - batch["target"].mean()).square()}


def _text_batch() -> dict[str, torch.Tensor]:
    ids = torch.tensor([[2, 4, 5, 3]], dtype=torch.long)
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "labels": ids.clone(),
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


class OCRJointTrainerTest(unittest.TestCase):
    def test_visual_scope_and_zero_bridge_two_step_dynamics(self) -> None:
        torch.manual_seed(7)
        model = _ToyModel()
        names = configure_visual_stage_trainable(model)
        self.assertTrue(names)
        self.assertTrue(
            all(
                name.startswith("vision.native_detail_tower.")
                or name.startswith("vision_cross_attention.")
                for name in names
            )
        )
        optimizer = torch.optim.SGD(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=0.2,
        )
        tower_weight = model.vision.native_detail_tower.weight
        output_weight = model.vision_cross_attention.output_projection.weight
        before_output = output_weight.detach().clone()
        batch = _visual_batch()
        with patch(
            "Model.posttrain.ocr_joint_trainer.forward_anyres_ocr_batch",
            side_effect=_bridge_only_ocr,
        ):
            first = train_visual_cycle(model, [batch], optimizer)
            self.assertTrue(first["stepped"])
            self.assertTrue(
                tower_weight.grad is None
                or torch.equal(tower_weight.grad, torch.zeros_like(tower_weight.grad))
            )
            self.assertFalse(torch.equal(output_weight, before_output))
            second = train_visual_cycle(model, [batch, batch], optimizer)
        self.assertEqual(second["ocr_microbatches"], 2)
        self.assertIsNotNone(tower_weight.grad)
        self.assertTrue(bool((tower_weight.grad != 0).any()))

    def test_joint_cycle_releases_each_graph_and_steps_optimizer_once(self) -> None:
        torch.manual_seed(11)
        model = _ToyModel()
        with torch.no_grad():
            model.vision_cross_attention.output_projection.weight.fill_(0.05)
        names = configure_joint_stage_trainable(model)
        self.assertTrue(names)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.vision.encoder.parameters())
        )
        optimizer = torch.optim.SGD(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=0.01,
        )
        scheduler = _Scheduler()
        scaler = _Scaler()
        batches = [_visual_batch(index) for index in range(4)]
        with patch(
            "Model.posttrain.ocr_joint_trainer.forward_anyres_ocr_batch",
            side_effect=_joint_ocr,
        ) as ocr_forward:
            metrics = train_joint_cycle(
                model,
                batches,
                _text_batch(),
                optimizer,
                text_weight=0.2,
                scheduler=scheduler,
                scaler=scaler,
            )
        self.assertEqual(ocr_forward.call_count, 4)
        self.assertEqual(scaler.scale_calls, 5)
        self.assertEqual(scaler.unscale_calls, 1)
        self.assertEqual(scaler.step_calls, 1)
        self.assertEqual(scaler.update_calls, 1)
        self.assertEqual(scheduler.steps, 1)
        self.assertEqual(model.position_contracts, [BOUNDARY_V1])
        self.assertEqual(model.text_forward_contracts, [(False, 4096)])
        self.assertAlmostEqual(
            metrics["loss"],
            metrics["ocr_loss"] + 0.2 * metrics["text_loss"],
            places=5,
        )
        expected_gradient_modules = (
            model.embed,
            model.lm_body,
            model.lm_head,
            model.vision.omvt.tower,
            model.vision.omvt.projector,
            model.vision.native_detail_tower,
            model.vision_cross_attention,
        )
        for module in expected_gradient_modules:
            self.assertTrue(
                any(parameter.grad is not None for parameter in module.parameters())
            )
        self.assertTrue(
            all(parameter.grad is None for parameter in model.vision.encoder.parameters())
        )

    def test_nonfinite_loss_never_steps_optimizer_or_scheduler(self) -> None:
        model = _ToyModel()
        configure_visual_stage_trainable(model)
        optimizer = torch.optim.SGD(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=0.1,
        )
        scheduler = _Scheduler()

        def nonfinite(model, _batch, **_kwargs):
            parameter = next(
                parameter for parameter in model.parameters() if parameter.requires_grad
            )
            return {"loss": parameter.sum() * torch.tensor(float("nan"))}

        with patch.object(optimizer, "step", wraps=optimizer.step) as step, patch(
            "Model.posttrain.ocr_joint_trainer.forward_anyres_ocr_batch",
            side_effect=nonfinite,
        ):
            with self.assertRaisesRegex(FloatingPointError, "non-finite loss"):
                train_visual_cycle(
                    model,
                    [_visual_batch()],
                    optimizer,
                    scheduler=scheduler,
                )
        step.assert_not_called()
        self.assertEqual(scheduler.steps, 0)

    def test_reverse_auxiliary_is_rejected_before_any_ocr_forward(self) -> None:
        model = _ToyModel()
        configure_visual_stage_trainable(model)
        model.reverse_loss_enabled = True
        optimizer = torch.optim.SGD(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=0.1,
        )
        with patch(
            "Model.posttrain.ocr_joint_trainer.forward_anyres_ocr_batch"
        ) as ocr_forward:
            with self.assertRaisesRegex(RuntimeError, "reverse_loss_enabled=False"):
                train_visual_cycle(model, [_visual_batch()], optimizer)
        ocr_forward.assert_not_called()

    def test_position_contract_is_rejected_before_any_ocr_forward(self) -> None:
        model = _ToyModel()
        configure_visual_stage_trainable(model)
        optimizer = torch.optim.SGD(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=0.1,
        )
        cases = (
            ({**_visual_batch(), "position_contract": "legacy_sequential_v0"},),
            ({**_visual_batch(), "word_pos": torch.zeros(1, 1)},),
        )
        with patch(
            "Model.posttrain.ocr_joint_trainer.forward_anyres_ocr_batch"
        ) as ocr_forward:
            for batches in cases:
                with self.subTest(keys=sorted(batches[0])):
                    with self.assertRaisesRegex(
                        ValueError,
                        "boundary_v1|must not materialize",
                    ):
                        train_visual_cycle(model, batches, optimizer)
        ocr_forward.assert_not_called()


if __name__ == "__main__":
    unittest.main()
