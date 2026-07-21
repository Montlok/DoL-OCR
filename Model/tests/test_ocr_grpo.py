# -*- coding: utf-8 -*-

"""OCR-specific GRPO safety and multimodal-contract tests."""

from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from Model.config import OMVTConfig, RDTConfig, TrainingConfig
from Model.model import RDTForCausalLM
from Model.omvt import OMVTInjector, collate_omvt_batch
from Model.posttrain.checkpointing import reconstruct_policy_from_checkpoint
from Model.posttrain.checkpointing import OCR_GRPO_CONTRACT_VERSION
from Model.posttrain.grpo import (
    GRPOConfig,
    grpo_compute_loss,
    grpo_loss,
    repeat_pixel_values,
    sample_group,
)
from Model.posttrain.ocr_decode import INVALID_OCR_TOKEN, decode_ocr_completion
from Model.posttrain.preference_data import OCRPromptDataset
from Model.posttrain.rewards import (
    RewardConfig,
    compute_rewards,
    grapheme_cer_reward,
    grapheme_cer_value,
    invalid_token_penalty,
    reward_for,
)


def _tiny_rdt() -> RDTConfig:
    return RDTConfig(
        d_model=32,
        n_heads=4,
        head_dim=8,
        kv_lora_rank=8,
        rope_head_dim=4,
        nope_head_dim=4,
        ffn_hidden=64,
        ffn_multiple=32,
        n_prelude=1,
        n_coda=1,
        mamba_per_block=1,
        attn_per_block=1,
        recurrent_steps=1,
        mamba_d_state=8,
        mamba_expand=2,
        mamba_headdim=16,
        use_official_mamba=False,
        max_seq_len=24,
    )


def _tiny_omvt() -> OMVTConfig:
    return OMVTConfig(
        image_size=16,
        vertical_patch=(8, 4),
        horizontal_patch=(4, 8),
        square_patch=(4, 4),
        layout_patch=(16, 16),
        d_vision=32,
        vision_n_heads=4,
        vision_ffn_hidden=64,
        compress_to=2,
        compressor_layers=1,
        compressor_heads=4,
        n_vertical_layers=1,
        n_horizontal_layers=1,
        n_local_attn_layers=1,
        n_layout_layers=1,
    )


class OCRRewardTest(unittest.TestCase):
    def test_ocr_cli_defaults_freeze_language_and_use_on_policy_sampling(self):
        from scripts.train_grpo import _apply_task_defaults, parse_args

        args = _apply_task_defaults(parse_args(["--task", "ocr", "--smoke"]))
        self.assertEqual(args.train_scope, "vision")
        self.assertEqual(args.temperature, 1.0)
        self.assertIsNone(args.top_p)

    def test_vision_scope_freezes_every_language_parameter(self):
        from scripts.train_grpo import _configure_trainable_scope

        rdt_cfg, omvt_cfg = _tiny_rdt(), _tiny_omvt()
        policy = RDTForCausalLM(rdt_cfg)
        policy.vision._omvt_cfg = omvt_cfg
        policy.vision.omvt = OMVTInjector(rdt_cfg, omvt_cfg)
        names = _configure_trainable_scope(policy, "vision")
        self.assertTrue(names)
        self.assertTrue(all(name.startswith("vision.omvt.") for name in names))
        self.assertTrue(any(name.startswith("vision.omvt.tower.") for name in names))
        self.assertTrue(
            any(name.startswith("vision.omvt.projector.") for name in names)
        )
        self.assertTrue(
            all(
                parameter.requires_grad == name.startswith("vision.omvt.")
                for name, parameter in policy.named_parameters()
            )
        )

    def test_dense_reward_orders_non_exact_transcripts(self):
        ref = "ᠮᠣᠩᠭᠤᠯ"
        near = "ᠮᠣᠩᠤᠯ"
        far = "ᠪᠣᠷᠣᠭᠠᠨ"
        self.assertEqual(grapheme_cer_value(ref, ref), 0.0)
        self.assertLess(grapheme_cer_value(near, ref), grapheme_cer_value(far, ref))
        self.assertGreater(
            grapheme_cer_reward(near, ref),
            grapheme_cer_reward(far, ref),
        )

    def test_cer_cap_bounds_insertion_outlier(self):
        self.assertEqual(grapheme_cer_reward("x" * 100, "a", cap=2.0), -2.0)

    def test_uncapped_default_preserves_bad_sample_ordering(self):
        self.assertGreater(
            grapheme_cer_reward("x" * 10, "a"),
            grapheme_cer_reward("x" * 100, "a"),
        )

    def test_ocr_reward_requires_reference(self):
        cfg = RewardConfig(grapheme_cer_weight=1.0)
        with self.assertRaisesRegex(ValueError, "requires a reference"):
            reward_for("ᠮ", None, cfg)

    def test_reward_batch_length_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            compute_rewards(["a", "b"], ["a"], RewardConfig(exact_match_weight=1.0))

    def test_invalid_control_token_has_explicit_penalty(self):
        cfg = RewardConfig(invalid_token_penalty_weight=0.25)
        self.assertEqual(invalid_token_penalty("valid"), 0.0)
        self.assertEqual(invalid_token_penalty(f"a{INVALID_OCR_TOKEN}b"), 1.0)
        self.assertEqual(reward_for(f"a{INVALID_OCR_TOKEN}b", None, cfg), -0.25)


class OCRRewardSafeDecodeTest(unittest.TestCase):
    def test_eos_stops_and_reserved_ids_are_never_hidden(self):
        def decode_content(ids):
            return "".join({17: " ", 300: "a", 301: "b"}[idx] for idx in ids)

        self.assertEqual(
            decode_ocr_completion([300, 0, 1, 17, 301, 3, 0], decode_content),
            f"a{INVALID_OCR_TOKEN}{INVALID_OCR_TOKEN} b",
        )

    def test_unused_reserved_id_is_invalid(self):
        self.assertEqual(
            decode_ocr_completion([200, 3], lambda _ids: ""),
            INVALID_OCR_TOKEN,
        )

    def test_unassigned_content_id_is_invalid(self):
        self.assertEqual(
            decode_ocr_completion(
                [300, 500, 3],
                lambda ids: "a" if ids == [300] else "",
                valid_token_ids={3, 300},
            ),
            f"a{INVALID_OCR_TOKEN}",
        )

    def test_missing_eos_is_visible_to_reward(self):
        self.assertEqual(
            decode_ocr_completion(
                [300],
                lambda ids: "a" if ids == [300] else "",
                valid_token_ids={3, 300},
                require_eos=True,
            ),
            f"a{INVALID_OCR_TOKEN}",
        )


class OCRPromptDatasetTest(unittest.TestCase):
    def _image(self, path: Path) -> None:
        from PIL import Image

        Image.new("RGB", (8, 12), "white").save(path)

    def test_strict_prompt_geometry_and_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._image(root / "photo.png")
            digest = hashlib.sha256((root / "photo.png").read_bytes()).hexdigest()
            manifest = root / "rl.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "id": "p1",
                        "split": "rl_train",
                        "image": "photo.png",
                        "sha256": digest,
                        "reference": "ᠮᠣᠩ",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            ds = OCRPromptDataset(
                manifest,
                encode=lambda text: [99] if text else [],
                n_image_tokens=2,
                bos_id=2,
                image_start_id=6,
                image_patch_id=7,
                image_end_id=8,
                max_prompt_len=8,
            )
            row = ds[0]
            self.assertEqual(row["prompt_ids"], [2, 6, 7, 7, 8])
            self.assertTrue(Path(row["image"]).is_file())

    def test_golden_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._image(root / "photo.png")
            manifest = root / "rl.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "id": "locked",
                        "split": "rl_train",
                        "image": "photo.png",
                        "reference": "x",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "golden"):
                OCRPromptDataset(
                    manifest,
                    encode=lambda _: [],
                    n_image_tokens=1,
                    bos_id=2,
                    image_start_id=6,
                    image_patch_id=7,
                    image_end_id=8,
                    excluded_ids={"locked"},
                )

    def test_reference_uses_lossless_encoder_and_total_context_is_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._image(root / "photo.png")
            manifest = root / "rl.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "id": "p1",
                        "split": "rl_train",
                        "image": "photo.png",
                        "reference": "abc",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            ds = OCRPromptDataset(
                manifest,
                encode=lambda _text: [99],
                encode_reference=lambda _text: [300, 301, 302],
                n_image_tokens=2,
                bos_id=2,
                image_start_id=6,
                image_patch_id=7,
                image_end_id=8,
                max_completion_len=4,
                max_seq_len=9,
                require_sha256=False,
            )
            self.assertEqual(ds[0]["reference_token_count"], 3)
            with self.assertRaisesRegex(ValueError, "exceeds model max_seq_len"):
                OCRPromptDataset(
                    manifest,
                    encode=lambda _text: [],
                    encode_reference=lambda _text: [300],
                    n_image_tokens=2,
                    bos_id=2,
                    image_start_id=6,
                    image_patch_id=7,
                    image_end_id=8,
                    max_completion_len=4,
                    max_seq_len=8,
                    require_sha256=False,
                )

    def test_empty_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._image(root / "photo.png")
            manifest = root / "rl.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "id": "p1",
                        "split": "rl_train",
                        "image": "photo.png",
                        "reference": "   ",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must not be empty"):
                OCRPromptDataset(
                    manifest,
                    encode=lambda _text: [],
                    n_image_tokens=1,
                    bos_id=2,
                    image_start_id=6,
                    image_patch_id=7,
                    image_end_id=8,
                    require_sha256=False,
                )

    def test_locked_manifest_never_encodes_reference_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._image(root / "photo.png")
            manifest = root / "golden.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "id": "locked",
                        "split": "golden",
                        "image": "photo.png",
                        "reference": "secret-label",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            def forbidden_encoder(_text):
                raise AssertionError("locked golden label was tokenized")

            dataset = OCRPromptDataset(
                manifest,
                encode=forbidden_encoder,
                encode_reference=forbidden_encoder,
                n_image_tokens=1,
                bos_id=2,
                image_start_id=6,
                image_patch_id=7,
                image_end_id=8,
                required_split="golden",
                require_sha256=False,
                inspect_reference_tokens=False,
                retain_reference=False,
                inspect_prompt_tokens=False,
            )
            self.assertIsNone(dataset[0]["reference_token_count"])
            self.assertNotIn("reference", dataset[0])
            self.assertNotIn("prompt_ids", dataset[0])

    def test_same_document_group_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._image(root / "different-crop.png")
            manifest = root / "rl.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "id": "line-2",
                        "group_id": "page-1",
                        "split": "rl_train",
                        "image": "different-crop.png",
                        "reference": "abc",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "group_id.*overlaps"):
                OCRPromptDataset(
                    manifest,
                    encode=lambda _text: [],
                    n_image_tokens=1,
                    bos_id=2,
                    image_start_id=6,
                    image_patch_id=7,
                    image_end_id=8,
                    excluded_groups={"page-1"},
                    require_group_id=True,
                    require_sha256=False,
                )

    def test_corrupt_image_is_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "broken.png").write_bytes(b"not-an-image")
            manifest = root / "rl.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "id": "broken",
                        "split": "rl_train",
                        "image": "broken.png",
                        "reference": "abc",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "cannot be decoded"):
                OCRPromptDataset(
                    manifest,
                    encode=lambda _text: [],
                    n_image_tokens=1,
                    bos_id=2,
                    image_start_id=6,
                    image_patch_id=7,
                    image_end_id=8,
                    verify_image_decode=True,
                    require_sha256=False,
                )


class MultimodalGRPOTest(unittest.TestCase):
    def test_generation_attention_distinguishes_sampled_and_fill_pad(self):
        cfg = _tiny_rdt()
        model = RDTForCausalLM(cfg).eval()
        observed: list[torch.Tensor] = []
        scheduled = [
            [cfg.eos_id, cfg.pad_id],
            [cfg.pad_id, 300],
            [cfg.pad_id, cfg.eos_id],
        ]

        def fake_forward(input_ids, attention_mask=None, **_kwargs):
            observed.append(attention_mask.detach().cpu().clone())
            logits = torch.full(
                (input_ids.shape[0], input_ids.shape[1], cfg.vocab_size),
                -1000.0,
                device=input_ids.device,
            )
            for row, token in enumerate(scheduled[len(observed) - 1]):
                logits[row, -1, token] = 0.0
            return {"logits": logits}

        model.forward = fake_forward  # type: ignore[method-assign]
        output = model.generate(
            torch.tensor([[cfg.bos_id], [cfg.bos_id]]),
            max_new_tokens=4,
            greedy=True,
            eos_id=cfg.eos_id,
            pad_id=cfg.pad_id,
        )
        self.assertEqual(output[:, 1:].tolist(), [[3, 0, 0], [0, 300, 3]])
        self.assertEqual(observed[1].tolist(), [[1, 1], [1, 1]])
        self.assertEqual(observed[2].tolist(), [[1, 1, 0], [1, 1, 1]])

    def test_pre_eos_pad_is_an_action_but_post_eos_pad_is_not(self):
        class FakeModel:
            reverse_loss_enabled = False

            def generate(self, input_ids, **_kwargs):
                tail = torch.tensor(
                    [[0, 300, 3, 0], [301, 3, 0, 0]],
                    device=input_ids.device,
                )
                return torch.cat([input_ids, tail], dim=1)

        seqs, mask = sample_group(
            FakeModel(),
            torch.tensor([2]),
            GRPOConfig(group_size=2, max_new_tokens=4),
            eos_id=3,
            pad_id=0,
        )
        self.assertEqual(seqs.shape, (2, 5))
        self.assertEqual(mask[:, 1:].tolist(), [[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]])

    def test_pre_eos_pad_remains_active_context_during_scoring(self):
        cfg = _tiny_rdt()
        policy = RDTForCausalLM(cfg)
        reference = RDTForCausalLM(cfg).eval()
        reference.load_state_dict(policy.state_dict())
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        prompt = torch.tensor([cfg.bos_id, 300])

        def fake_generate(input_ids, **_kwargs):
            tails = torch.tensor(
                [[cfg.pad_id, 301, cfg.eos_id, cfg.pad_id],
                 [302, cfg.eos_id, cfg.pad_id, cfg.pad_id]],
                device=input_ids.device,
            )
            return torch.cat([input_ids, tails], dim=1)

        policy.generate = fake_generate  # type: ignore[method-assign]
        original_forward = policy.forward
        observed = []
        observed_modes = []

        def spy_forward(*args, attention_mask=None, **kwargs):
            observed.append(attention_mask.detach().cpu())
            observed_modes.append(policy.training)
            return original_forward(*args, attention_mask=attention_mask, **kwargs)

        policy.forward = spy_forward  # type: ignore[method-assign]
        self.assertTrue(policy.training)
        loss, _ = grpo_compute_loss(
            policy,
            reference,
            [prompt],
            lambda _responses, _idx: torch.tensor([0.0, 1.0]),
            lambda ids: " ".join(str(int(token)) for token in ids),
            GRPOConfig(group_size=2, max_new_tokens=4, recurrent_steps=1),
            eos_id=cfg.eos_id,
            pad_id=cfg.pad_id,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(policy.training)
        self.assertEqual(observed_modes, [False])
        self.assertEqual(
            observed[0][:, 2:].tolist(),
            [[1, 1, 1, 0], [1, 1, 0, 0]],
        )
    def test_repeat_omvt_payload_preserves_shared_bbox(self):
        pixels = {
            "images": torch.randn(1, 3, 8, 8),
            "vertical_patches": torch.randn(1, 2, 12),
            "vertical_bbox": torch.tensor([[0, 0, 4, 2], [4, 0, 4, 2]]),
        }
        repeated = repeat_pixel_values(pixels, 4)
        assert isinstance(repeated, dict)
        self.assertEqual(repeated["images"].shape[0], 4)
        self.assertEqual(repeated["vertical_patches"].shape[0], 4)
        self.assertIs(repeated["vertical_bbox"], pixels["vertical_bbox"])

    def test_loss_averages_sequences_not_tokens(self):
        logp = torch.zeros(2, 3)
        mask = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        loss, _ = grpo_loss(
            logp,
            logp,
            torch.tensor([1.0, -1.0]),
            mask,
            cfg=GRPOConfig(group_size=2, kl_coef=0.0),
        )
        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_pixels_reach_rollout_policy_and_reference_scoring(self):
        torch.manual_seed(0)
        cfg = _tiny_rdt()
        policy = RDTForCausalLM(cfg, patch_pixels=4)
        reference = RDTForCausalLM(cfg, patch_pixels=4).eval()
        reference.load_state_dict(policy.state_dict())
        for param in reference.parameters():
            param.requires_grad_(False)

        prompt = torch.tensor(
            [cfg.bos_id, cfg.image_start_id, cfg.image_patch_id, cfg.image_end_id]
        )
        pixels = torch.randn(1, 1, 4)
        seen = {"rollout": 0, "policy": 0, "reference": 0}

        def fake_generate(input_ids, *, pixel_values=None, **_kwargs):
            self.assertIsNotNone(pixel_values)
            self.assertEqual(pixel_values.shape[0], 4)
            seen["rollout"] += 1
            tails = torch.tensor(
                [[300], [301], [300], [301]], device=input_ids.device
            )
            return torch.cat([input_ids, tails], dim=1)

        policy.generate = fake_generate  # type: ignore[method-assign]
        policy_forward = policy.forward
        reference_forward = reference.forward

        def policy_spy(*args, pixel_values=None, **kwargs):
            self.assertIsNotNone(pixel_values)
            seen["policy"] += 1
            return policy_forward(*args, pixel_values=pixel_values, **kwargs)

        def reference_spy(*args, pixel_values=None, **kwargs):
            self.assertIsNotNone(pixel_values)
            seen["reference"] += 1
            return reference_forward(*args, pixel_values=pixel_values, **kwargs)

        policy.forward = policy_spy  # type: ignore[method-assign]
        reference.forward = reference_spy  # type: ignore[method-assign]

        def reward_fn(responses, _idx):
            return torch.tensor([0.0 if "300" in r else 1.0 for r in responses])

        loss, metrics = grpo_compute_loss(
            policy,
            reference,
            [prompt, prompt.clone()],
            reward_fn,
            lambda ids: " ".join(str(int(x)) for x in ids),
            GRPOConfig(group_size=2, max_new_tokens=1, recurrent_steps=1),
            pad_id=cfg.pad_id,
            eos_id=cfg.eos_id,
            pixel_values=[pixels, pixels.clone()],
        )
        loss.backward()
        self.assertEqual(seen, {"rollout": 1, "policy": 1, "reference": 1})
        self.assertGreater(metrics["reward_std"], 0.0)
        vision_grad = policy.vision.encoder.patch_embed.weight.grad
        self.assertIsNotNone(vision_grad)
        self.assertGreater(float(vision_grad.abs().sum()), 0.0)

    def test_omvt_is_encoded_once_per_generation_not_once_per_token(self):
        torch.manual_seed(0)
        rdt_cfg, omvt_cfg = _tiny_rdt(), _tiny_omvt()
        model = RDTForCausalLM(rdt_cfg).eval()
        model.vision._omvt_cfg = omvt_cfg
        model.vision.omvt = OMVTInjector(rdt_cfg, omvt_cfg)
        prompt = torch.tensor(
            [[
                rdt_cfg.bos_id,
                rdt_cfg.image_start_id,
                rdt_cfg.image_patch_id,
                rdt_cfg.image_patch_id,
                rdt_cfg.image_end_id,
            ]]
        )
        pixels = dict(collate_omvt_batch(torch.randn(1, 3, 16, 16), omvt_cfg))
        calls = []
        handle = model.vision.omvt.register_forward_hook(
            lambda *_args: calls.append(1)
        )
        try:
            model.generate(
                prompt,
                max_new_tokens=3,
                greedy=True,
                recurrent_steps=1,
                pixel_values=pixels,
            )
        finally:
            handle.remove()
        self.assertEqual(len(calls), 1)

    def test_ocr_grpo_backpropagates_into_omvt_tower_and_projector(self):
        from scripts.train_grpo import _configure_trainable_scope

        torch.manual_seed(0)
        rdt_cfg, omvt_cfg = _tiny_rdt(), _tiny_omvt()
        policy = RDTForCausalLM(rdt_cfg).eval()
        policy.vision._omvt_cfg = omvt_cfg
        policy.vision.omvt = OMVTInjector(rdt_cfg, omvt_cfg)
        reference = RDTForCausalLM(rdt_cfg).eval()
        reference.vision._omvt_cfg = omvt_cfg
        reference.vision.omvt = OMVTInjector(rdt_cfg, omvt_cfg)
        reference.load_state_dict(policy.state_dict())
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        _configure_trainable_scope(policy, "vision")

        prompt = torch.tensor(
            [
                rdt_cfg.bos_id,
                rdt_cfg.image_start_id,
                rdt_cfg.image_patch_id,
                rdt_cfg.image_patch_id,
                rdt_cfg.image_end_id,
            ]
        )
        pixels = dict(collate_omvt_batch(torch.randn(1, 3, 16, 16), omvt_cfg))

        def fake_generate(input_ids, **_kwargs):
            tails = torch.tensor([[300], [301]], device=input_ids.device)
            return torch.cat([input_ids, tails], dim=1)

        policy.generate = fake_generate  # type: ignore[method-assign]
        loss, _ = grpo_compute_loss(
            policy,
            reference,
            [prompt],
            lambda _responses, _idx: torch.tensor([0.0, 1.0]),
            lambda ids: str(int(ids[0])),
            GRPOConfig(group_size=2, max_new_tokens=1, recurrent_steps=1),
            eos_id=rdt_cfg.eos_id,
            pad_id=rdt_cfg.pad_id,
            pixel_values=[pixels],
        )
        loss.backward()
        tower_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in policy.vision.omvt.tower.parameters()
            if parameter.grad is not None
        )
        projector_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in policy.vision.omvt.projector.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(tower_grad, 0.0)
        self.assertGreater(projector_grad, 0.0)

    def test_validation_is_greedy_batched_and_reports_exact_cer(self):
        from scripts.train_grpo import _evaluate_ocr_validation

        class FakeDataset:
            rows = [
                {"prompt_ids": [2, 6, 7, 7, 8], "image": "a", "reference": "a"},
                {"prompt_ids": [2, 6, 7, 7, 8], "image": "b", "reference": "a"},
            ]

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, idx):
                return self.rows[idx]

        class FakeModel:
            def generate(self, input_ids, *, greedy=False, **_kwargs):
                self.greedy = greedy
                tail = torch.tensor([[300, 3]], device=input_ids.device).expand(
                    input_ids.shape[0], -1
                )
                return torch.cat([input_ids, tail], dim=1)

        model = FakeModel()
        metrics = _evaluate_ocr_validation(
            model,
            FakeDataset(),
            lambda images: torch.zeros(len(images), 3, 16, 16),
            _tiny_omvt(),
            lambda ids: decode_ocr_completion(
                ids,
                lambda content: "a" if content == [300] else "",
                valid_token_ids={3, 300},
            ),
            batch_size=2,
            max_new_tokens=2,
            recurrent_steps=1,
            precision="fp32",
            device=torch.device("cpu"),
            cer_backend="python",
        )
        self.assertTrue(model.greedy)
        self.assertEqual(metrics["grapheme_cer"], 0.0)
        self.assertEqual(metrics["line_exact"], 1.0)
        self.assertEqual(metrics["eos_rate"], 1.0)


class StrictCheckpointTest(unittest.TestCase):
    def _write_checkpoint(
        self,
        root: Path,
        *,
        include_omvt_meta: bool = True,
        include_rdt_meta: bool = True,
    ) -> Path:
        rdt_cfg, omvt_cfg = _tiny_rdt(), _tiny_omvt()
        model = RDTForCausalLM(rdt_cfg)
        model.vision._omvt_cfg = omvt_cfg
        model.vision.omvt = OMVTInjector(rdt_cfg, omvt_cfg)
        step = root / "step_00000001"
        step.mkdir()
        torch.save(model.state_dict(), step / "model.pt")
        metadata = {"rdt_config": asdict(rdt_cfg)} if include_rdt_meta else {}
        if include_omvt_meta:
            metadata["omvt_config"] = asdict(omvt_cfg)
        torch.save({"step": 1, "metadata": metadata}, step / "meta.pt")
        return step

    def test_reconstructs_and_strictly_loads_omvt(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = reconstruct_policy_from_checkpoint(
                self._write_checkpoint(Path(tmp)),
                require_vision=True,
            )
            self.assertIsNotNone(loaded.model.vision.omvt)
            self.assertEqual(loaded.omvt_config.compress_to, 2)

    def test_refuses_to_guess_omvt_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "no omvt_config"):
                reconstruct_policy_from_checkpoint(
                    self._write_checkpoint(Path(tmp), include_omvt_meta=False),
                    require_vision=True,
                )

    def test_refuses_to_guess_rdt_behavioral_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "rdt_config metadata"):
                reconstruct_policy_from_checkpoint(
                    self._write_checkpoint(Path(tmp), include_rdt_meta=False),
                    fallback_rdt_config=_tiny_rdt(),
                    require_vision=True,
                )

    def test_strict_load_rejects_missing_tensor(self):
        with tempfile.TemporaryDirectory() as tmp:
            step = self._write_checkpoint(Path(tmp))
            state = torch.load(step / "model.pt", map_location="cpu", weights_only=False)
            state.pop(next(iter(state)))
            torch.save(state, step / "model.pt")
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                reconstruct_policy_from_checkpoint(step, require_vision=True)

    def test_resume_reference_loads_immutable_snapshot_not_current_policy(self):
        from scripts.train_grpo import _load_immutable_reference

        with tempfile.TemporaryDirectory() as tmp:
            step = self._write_checkpoint(Path(tmp))
            meta_path = step / "meta.pt"
            payload = torch.load(meta_path, map_location="cpu", weights_only=False)
            payload["metadata"].update(
                {
                    "phase": "grpo_reference",
                    "contract_version": OCR_GRPO_CONTRACT_VERSION,
                    "source_checkpoint": "visual-source",
                    "immutable": True,
                }
            )
            torch.save(payload, meta_path)
            (step / "COMPLETE").write_text("step=1\n", encoding="ascii")
            saved_state = torch.load(
                step / "model.pt", map_location="cpu", weights_only=False
            )
            first_key = next(iter(saved_state))

            reference, resolved = _load_immutable_reference(
                str(step),
                "visual-source",
                _tiny_rdt(),
                _tiny_omvt(),
                TrainingConfig(parallel="single", max_steps=1, warmup_steps=0),
                0,
                torch.device("cpu"),
                require_vision=True,
            )
            self.assertEqual(resolved, step)
            self.assertTrue(
                torch.equal(reference.state_dict()[first_key], saved_state[first_key])
            )
            self.assertTrue(all(not param.requires_grad for param in reference.parameters()))
            with self.assertRaisesRegex(ValueError, "different source"):
                _load_immutable_reference(
                    str(step),
                    "another-source",
                    _tiny_rdt(),
                    _tiny_omvt(),
                    TrainingConfig(parallel="single", max_steps=1, warmup_steps=0),
                    0,
                    torch.device("cpu"),
                    require_vision=True,
                )

    def test_resume_rejects_model_only_checkpoint(self):
        from scripts.train_grpo import _validate_resumable_checkpoint_files

        with tempfile.TemporaryDirectory() as tmp:
            step = Path(tmp) / "step_00000001"
            step.mkdir()
            for name in ("COMPLETE", "model.pt", "meta.pt"):
                (step / name).touch()
            with self.assertRaisesRegex(ValueError, "optimizer.pt"):
                _validate_resumable_checkpoint_files(
                    step,
                    require_scaler=False,
                )

            for name in ("optimizer.pt", "scheduler.pt", "rng.pt"):
                (step / name).touch()
            self.assertEqual(
                _validate_resumable_checkpoint_files(
                    step,
                    require_scaler=False,
                ),
                step,
            )
            with self.assertRaisesRegex(ValueError, "scaler.pt"):
                _validate_resumable_checkpoint_files(
                    step,
                    require_scaler=True,
                )


if __name__ == "__main__":
    unittest.main()
