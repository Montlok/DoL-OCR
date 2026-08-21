# -*- coding: utf-8 -*-

from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from Model.posttrain import ocr_joint_eval
from Model.posttrain.ocr_joint_eval import (
    DEPLOYMENT_BUCKET_WEIGHTS,
    evaluate_ocr_joint,
    joint_eval_eligibility,
)


_TOKEN_TEXT = {
    10: "ᠠ᠋9.",
    11: "ᠡ᠎8,",
    12: "ᠢ\u202f7!",
    13: "ᠣ᠏6?",
    14: "ᠤ᠌5;",
    15: "ᠥ᠍4:",
    16: "ᠦ3",
    17: "ᠧ2",
}
_IMAGE_CONTRACT_SHA256 = "a" * 64
_TEXT_CONTRACT_SHA256 = "b" * 64


class _Vision:
    def encode_visual(self, pixel_values):
        return pixel_values["features"]


class _TinyJointModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.cfg = SimpleNamespace(eos_id=2, pad_id=0)
        self.vision = _Vision()
        self.calls: list[dict] = []
        self.text_nll = 1.01

    def forward(self, **kwargs):
        self.calls.append(dict(kwargs))
        input_ids = kwargs["input_ids"]
        labels = kwargs.get("labels")
        visual = kwargs.get("visual_features")
        if labels is not None and visual is None:
            return {
                "logits": None,
                "loss_parts": {"forward": self.text_nll},
            }
        if kwargs.get("detail_memory") is None:
            raise AssertionError("visual evaluation must carry detail memory")
        batch, length = input_ids.shape
        logits = torch.full((batch, length, 32), -8.0, device=input_ids.device)
        if length == 3:
            chosen = visual[:, 0, 0].round().to(dtype=torch.long).clamp(0, 31)
        else:
            chosen = torch.full((batch,), self.cfg.eos_id, device=input_ids.device)
        logits[torch.arange(batch, device=input_ids.device), -1, chosen] = 8.0
        return {"logits": logits, "loss_parts": {}}


def _batch() -> dict:
    buckets = [
        "print",
        "print",
        "handwritten_good",
        "handwritten_good",
        "handwritten_medium",
        "handwritten_medium",
        "handwritten_poor",
        "handwritten_poor",
    ]
    targets = torch.arange(10, 18, dtype=torch.long)
    batch_size = len(buckets)
    input_ids = torch.zeros(batch_size, 5, dtype=torch.long)
    input_ids[:, :3] = torch.tensor([1, 5, 6])
    input_ids[:, 3] = targets
    input_ids[:, 4] = 2
    labels = torch.full_like(input_ids, -100)
    labels[:, 3] = targets
    labels[:, 4] = 2
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
        "position_contract": "boundary_v1",
        "dataset_contract_sha256": _IMAGE_CONTRACT_SHA256,
        "sample_metadata": [
            {
                "sample_id": f"sample-{index}",
                "reference_model": {"text": _TOKEN_TEXT[int(token)]},
            }
            for index, token in enumerate(targets)
        ],
        "quota_buckets": buckets,
        "global_pixel_values": {"features": targets.float().view(-1, 1, 1)},
        "fake_detail_memory": targets.float().view(-1, 1),
    }


def _encode_visual(_model, batch, *, device):
    features = batch["global_pixel_values"]["features"].to(device)
    detail = batch["fake_detail_memory"].to(device)
    count = detail.shape[0]
    return {
        "global_pixel_values": {"features": features},
        "detail_memory": detail,
        "detail_cu_seqlens": torch.arange(
            count + 1,
            dtype=torch.int32,
            device=device,
        ),
    }


def _decode(ids: torch.Tensor) -> str:
    values = ids.tolist()
    if not values:
        return ""
    if len(values) != 1:
        raise ValueError("tiny decoder expects one content token")
    return _TOKEN_TEXT.get(int(values[0]), "X")


def _text_batch() -> dict:
    return {
        "input_ids": torch.tensor([[1, 10, 2, 0]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
        "labels": torch.tensor([[1, 10, 2, -100]], dtype=torch.long),
        "position_contract": "boundary_v1",
        "dataset_contract_sha256": _TEXT_CONTRACT_SHA256,
    }


class OCRJointEvalTest(unittest.TestCase):
    def test_real_controls_metrics_text_nll_and_eligibility(self) -> None:
        model = _TinyJointModel().train()
        with mock.patch.object(
            ocr_joint_eval,
            "encode_anyres_visual_batch",
            side_effect=_encode_visual,
        ):
            report = evaluate_ocr_joint(
                model,
                [_batch()],
                _decode,
                max_new_tokens=3,
                device="cpu",
                expected_image_dataset_contract_sha256=(
                    _IMAGE_CONTRACT_SHA256
                ),
                text_replay_batches=[_text_batch()],
                expected_text_replay_contract_sha256=_TEXT_CONTRACT_SHA256,
            )

        self.assertTrue(model.training)
        self.assertEqual(report["real"]["overall"]["raw_grapheme_cer"], 0.0)
        self.assertEqual(report["real"]["overall"]["raw_line_exact"], 1.0)
        self.assertEqual(report["real"]["overall"]["eos_rate"], 1.0)
        self.assertEqual(report["real"]["overall"]["hit_cap_rate"], 0.0)
        self.assertEqual(report["real"]["overall"]["invalid_count"], 0)
        self.assertEqual(set(report["real"]["buckets"]), set(DEPLOYMENT_BUCKET_WEIGHTS))
        self.assertEqual(report["deployment_weighted_cer"], 0.0)
        self.assertEqual(report["worst_bucket"]["name"], "print")
        self.assertGreater(report["grounding"]["blank_cer_gap"], 0.0)
        self.assertGreater(report["grounding"]["shuffled_cer_gap"], 0.0)
        self.assertGreater(report["grounding"]["blank_first_token_nll_gap"], 0.0)
        self.assertGreater(report["grounding"]["shuffled_first_token_nll_gap"], 0.0)
        for interval in report["grounding"]["paired_bootstrap_95ci"].values():
            self.assertGreater(interval["low"], 0.0)
        self.assertAlmostEqual(report["text_replay"]["token_nll"], 1.01)
        self.assertEqual(
            report["image_dataset_contract_sha256"],
            _IMAGE_CONTRACT_SHA256,
        )
        self.assertEqual(
            report["text_replay"]["dataset_contract_sha256"],
            _TEXT_CONTRACT_SHA256,
        )
        symbols = report["real"]["overall"]["symbol_metrics"]
        self.assertGreater(symbols["fvs"]["n_ref"], 0)
        self.assertGreater(symbols["mvs"]["n_ref"], 0)
        self.assertGreater(symbols["nnbsp"]["n_ref"], 0)
        self.assertGreater(symbols["digit"]["n_ref"], 0)
        self.assertGreater(symbols["punctuation"]["n_ref"], 0)
        self.assertEqual(len(report["selection_records"]), 8)
        self.assertEqual(
            [row["sample_id"] for row in report["selection_records"]],
            sorted(row["sample_id"] for row in report["selection_records"]),
        )

        eligibility = joint_eval_eligibility(
            report,
            baseline_bucket_cer={bucket: 0.01 for bucket in DEPLOYMENT_BUCKET_WEIGHTS},
            baseline_text_token_nll=1.0,
        )
        self.assertTrue(eligibility["eligible"], eligibility["reasons"])
        self.assertLessEqual(
            eligibility["checks"]["text_token_nll_relative_drift"],
            0.02,
        )
        self.assertTrue(
            all(call.get("position_contract") == "boundary_v1" for call in model.calls)
        )
        self.assertTrue(all("use_cache" not in call for call in model.calls))

        failed = copy.deepcopy(report)
        failed["real"]["overall"]["invalid_count"] = 1
        rejected = joint_eval_eligibility(
            failed,
            baseline_bucket_cer={bucket: 0.01 for bucket in DEPLOYMENT_BUCKET_WEIGHTS},
            baseline_text_token_nll=1.0,
        )
        self.assertFalse(rejected["eligible"])
        self.assertIn("invalid_output_nonzero", rejected["reasons"])

    def test_singleton_bucket_shuffle_configuration_fails_before_model_work(self) -> None:
        batch = _batch()
        for key in ("input_ids", "attention_mask", "labels"):
            batch[key] = batch[key][:-1]
        batch["sample_metadata"] = batch["sample_metadata"][:-1]
        batch["quota_buckets"] = batch["quota_buckets"][:-1]
        batch["global_pixel_values"]["features"] = batch["global_pixel_values"][
            "features"
        ][:-1]
        batch["fake_detail_memory"] = batch["fake_detail_memory"][:-1]
        model = _TinyJointModel()
        with self.assertRaisesRegex(ValueError, "singleton bucket"):
            evaluate_ocr_joint(
                model,
                [batch],
                _decode,
                max_new_tokens=3,
                device="cpu",
                expected_image_dataset_contract_sha256=(
                    _IMAGE_CONTRACT_SHA256
                ),
            )
        self.assertEqual(model.calls, [])

    def test_mixed_or_wrong_image_and_text_contracts_fail_closed(self) -> None:
        model = _TinyJointModel()
        wrong_image = _batch()
        wrong_image["dataset_contract_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "image dataset contract differs"):
            evaluate_ocr_joint(
                model,
                [wrong_image],
                _decode,
                max_new_tokens=3,
                device="cpu",
                expected_image_dataset_contract_sha256=(
                    _IMAGE_CONTRACT_SHA256
                ),
            )
        self.assertEqual(model.calls, [])

        mixed_image = _batch()
        second_image = _batch()
        second_image["dataset_contract_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "mix dataset contracts"):
            evaluate_ocr_joint(
                model,
                [mixed_image, second_image],
                _decode,
                max_new_tokens=3,
                device="cpu",
                expected_image_dataset_contract_sha256=(
                    _IMAGE_CONTRACT_SHA256
                ),
            )
        self.assertEqual(model.calls, [])

        mixed_text = _text_batch()
        other_text = _text_batch()
        other_text["dataset_contract_sha256"] = "d" * 64
        with mock.patch.object(
            ocr_joint_eval,
            "encode_anyres_visual_batch",
            side_effect=_encode_visual,
        ), self.assertRaisesRegex(ValueError, "mix dataset contracts"):
            evaluate_ocr_joint(
                model,
                [_batch()],
                _decode,
                max_new_tokens=3,
                device="cpu",
                expected_image_dataset_contract_sha256=(
                    _IMAGE_CONTRACT_SHA256
                ),
                text_replay_batches=[mixed_text, other_text],
                expected_text_replay_contract_sha256=(
                    _TEXT_CONTRACT_SHA256
                ),
            )

        wrong_text = _text_batch()
        with mock.patch.object(
            ocr_joint_eval,
            "encode_anyres_visual_batch",
            side_effect=_encode_visual,
        ), self.assertRaisesRegex(ValueError, "text replay dataset contract differs"):
            evaluate_ocr_joint(
                model,
                [_batch()],
                _decode,
                max_new_tokens=3,
                device="cpu",
                expected_image_dataset_contract_sha256=(
                    _IMAGE_CONTRACT_SHA256
                ),
                text_replay_batches=[wrong_text],
                expected_text_replay_contract_sha256="d" * 64,
            )


if __name__ == "__main__":
    unittest.main()
