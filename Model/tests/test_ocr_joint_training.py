# -*- coding: utf-8 -*-

"""Focused contracts for joint OCR/text optimizer groups and scheduling."""

from __future__ import annotations

import json
import unittest

import torch.nn as nn

from Model.config import TrainingConfig
from Model.training.joint_schedule import (
    OCRTextMicrobatchSchedule,
    ocr_text_microbatch_kind,
)
from Model.training.optim import build_ocr_joint_adamw, build_scheduler


class _JointToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(32, 8)
        self.lm_norm = nn.LayerNorm(8)
        self.lm_body = nn.Linear(8, 8)
        self.lm_head = nn.Linear(8, 32, bias=False)
        self.lm_head.weight = self.embed.weight

        self.vision = nn.Module()
        self.vision.encoder = nn.Sequential(
            nn.LayerNorm(8),
            nn.Linear(8, 8),
        )
        self.vision.omvt = nn.Module()
        self.vision.omvt.tower = nn.Sequential(
            nn.LayerNorm(8),
            nn.Linear(8, 8),
        )
        self.vision.omvt.projector = nn.Sequential(
            nn.LayerNorm(8),
            nn.Linear(8, 8),
        )

        self.native_detail_tower = nn.Sequential(
            nn.LayerNorm(8),
            nn.Linear(8, 8),
        )
        self.native_projector = nn.Sequential(
            nn.LayerNorm(8),
            nn.Linear(8, 8),
        )
        self.vision_cross_attention = nn.Sequential(
            nn.LayerNorm(8),
            nn.Linear(8, 8),
        )


def _training_cfg() -> TrainingConfig:
    return TrainingConfig(
        train_data="unused",
        optimizer="adamw",
        weight_decay=0.2,
        warmup_steps=2,
        max_steps=10,
        min_lr_ratio=0.1,
    )


class OCRJointOptimizerTest(unittest.TestCase):
    def test_groups_are_disjoint_complete_and_keep_role_base_lrs(self) -> None:
        model = _JointToyModel()
        cfg = _training_cfg()
        expected_lrs = {
            "lm": 1e-5,
            "tower": 2e-5,
            "projector": 3e-5,
            "bridge": 4e-5,
        }
        optimizer = build_ocr_joint_adamw(
            model,
            cfg,
            lm_lr=expected_lrs["lm"],
            tower_lr=expected_lrs["tower"],
            projector_lr=expected_lrs["projector"],
            bridge_lr=expected_lrs["bridge"],
        )

        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.vision.encoder.parameters())
        )
        legacy_ids = {id(parameter) for parameter in model.vision.encoder.parameters()}
        grouped = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        grouped_ids = [id(parameter) for parameter in grouped]
        trainable_ids = {
            id(parameter) for parameter in model.parameters() if parameter.requires_grad
        }
        self.assertEqual(len(grouped_ids), len(set(grouped_ids)))
        self.assertEqual(set(grouped_ids), trainable_ids)
        self.assertTrue(legacy_ids.isdisjoint(grouped_ids))

        group_for_param = {
            id(parameter): group
            for group in optimizer.param_groups
            for parameter in group["params"]
        }

        def assert_role(module: nn.Module, role: str) -> None:
            for parameter in module.parameters():
                self.assertEqual(group_for_param[id(parameter)]["ocr_joint_role"], role)
                self.assertEqual(group_for_param[id(parameter)]["lr"], expected_lrs[role])

        assert_role(model.vision.omvt.tower, "tower")
        assert_role(model.native_detail_tower, "tower")
        assert_role(model.vision.omvt.projector, "projector")
        assert_role(model.native_projector, "projector")
        assert_role(model.vision_cross_attention, "bridge")
        self.assertEqual(group_for_param[id(model.embed.weight)]["ocr_joint_role"], "lm")
        self.assertEqual(group_for_param[id(model.lm_body.weight)]["ocr_joint_role"], "lm")

        self.assertEqual(
            group_for_param[id(model.vision.omvt.tower[1].weight)]["weight_decay"],
            cfg.weight_decay,
        )
        self.assertEqual(
            group_for_param[id(model.vision.omvt.tower[1].bias)]["weight_decay"],
            0.0,
        )
        self.assertEqual(group_for_param[id(model.embed.weight)]["weight_decay"], 0.0)

        scheduler = build_scheduler(optimizer, cfg)
        for _ in range(5):
            factors = [
                group["lr"] / group["ocr_joint_base_lr"]
                for group in optimizer.param_groups
            ]
            self.assertTrue(
                all(abs(factor - factors[0]) < 1e-12 for factor in factors[1:])
            )
            optimizer.step()
            scheduler.step()

    def test_unclassified_visual_parameters_fail_closed(self) -> None:
        model = _JointToyModel()
        model.vision.unclassified = nn.Linear(8, 8)
        with self.assertRaisesRegex(ValueError, "unclassified trainable visual"):
            build_ocr_joint_adamw(
                model,
                _training_cfg(),
                lm_lr=1e-5,
                tower_lr=2e-5,
                projector_lr=3e-5,
                bridge_lr=4e-5,
            )


class OCRTextScheduleTest(unittest.TestCase):
    def test_ratio_and_json_roundtrip_preserve_the_next_kind(self) -> None:
        expected = ["ocr", "ocr", "ocr", "ocr", "text"] * 3
        self.assertEqual(
            [ocr_text_microbatch_kind(index) for index in range(15)],
            expected,
        )

        cursor = OCRTextMicrobatchSchedule().advance(7)
        restored = OCRTextMicrobatchSchedule.from_state_dict(
            json.loads(json.dumps(cursor.state_dict()))
        )
        self.assertEqual(restored, cursor)
        self.assertEqual(restored.next_kind(), cursor.next_kind())

        uninterrupted = []
        resumed = []
        for _ in range(13):
            uninterrupted.append(cursor.next_kind())
            resumed.append(restored.next_kind())
            cursor = cursor.advance()
            restored = restored.advance()
        self.assertEqual(resumed, uninterrupted)

        bad_state = OCRTextMicrobatchSchedule().state_dict()
        bad_state["ocr_microbatches_per_cycle"] = 3
        with self.assertRaisesRegex(ValueError, "OCR ratio"):
            OCRTextMicrobatchSchedule.from_state_dict(bad_state)


if __name__ == "__main__":
    unittest.main()
