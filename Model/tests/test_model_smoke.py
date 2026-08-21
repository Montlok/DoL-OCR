# -*- coding: utf-8 -*-

import unittest

import torch
import torch.nn as nn

from Model.config import RDTConfig
from Model.config import MIN_OFFICIAL_MAMBA3_D_STATE
import Model.layers.mamba3_layer as mamba3_layer
from Model.layers.mla import MLA
from Model.layers.mamba3_layer import Mamba3Layer
from Model.model import RDTForCausalLM
from Model.training import PretrainingCollator
from Model.vision import inject_visual_features


def _test_config() -> RDTConfig:
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
        recurrent_steps=2,
        mamba_d_state=8,
        mamba_expand=2,
        mamba_headdim=16,
        use_official_mamba=False,
        max_seq_len=64,
    )


class ModelSmokeTest(unittest.TestCase):
    def test_tied_heads_share_embedding_after_init(self):
        cfg = _test_config()
        model = RDTForCausalLM(cfg)

        self.assertEqual(model.lm_head.weight.data_ptr(), model.embed.weight.data_ptr())
        self.assertIsNotNone(model.reverse_head)
        self.assertEqual(
            model.reverse_head.weight.data_ptr(),
            model.embed.weight.data_ptr(),
        )

    def test_forward_backward_with_image_patch(self):
        torch.manual_seed(0)
        cfg = _test_config()
        model = RDTForCausalLM(cfg, patch_pixels=4)

        input_ids = torch.tensor(
            [
                [
                    cfg.bos_id,
                    256,
                    cfg.image_start_id,
                    cfg.image_patch_id,
                    cfg.image_end_id,
                    cfg.eos_id,
                ]
            ]
        )
        labels = input_ids.clone()
        labels[input_ids == cfg.image_start_id] = cfg.ignore_index
        labels[input_ids == cfg.image_patch_id] = cfg.ignore_index
        labels[input_ids == cfg.image_end_id] = cfg.ignore_index

        out = model(
            input_ids=input_ids,
            labels=labels,
            pixel_values=torch.randn(1, 4),
        )

        self.assertEqual(out["logits"].shape, (1, 6, cfg.vocab_size))
        self.assertTrue(torch.isfinite(out["loss"]))

        out["loss"].backward()
        self.assertIsNotNone(model.embed.weight.grad)

    def test_chunked_loss_matches_full_logits_loss(self):
        torch.manual_seed(0)
        cfg = _test_config()
        model = RDTForCausalLM(cfg)

        input_ids = torch.tensor([[cfg.bos_id, 256, 257, cfg.eos_id]])
        labels = input_ids.clone()
        word_pos = torch.tensor([[0, 0, 0, 1]])
        morph_depth = torch.tensor([[0, 0, 1, 0]])

        full = model(
            input_ids=input_ids,
            labels=labels,
            word_pos=word_pos,
            morph_depth=morph_depth,
        )
        chunked = model(
            input_ids=input_ids,
            labels=labels,
            word_pos=word_pos,
            morph_depth=morph_depth,
            return_logits=False,
            loss_chunk_size=2,
            bptt_window=1,
        )

        self.assertIsNone(chunked["logits"])
        self.assertTrue(
            torch.allclose(full["loss"], chunked["loss"], atol=1e-5, rtol=1e-5)
        )

    def test_pretraining_collator_outputs_model_inputs(self):
        cfg = _test_config()
        rows = [
            {
                "input_ids": [cfg.bos_id, 256, cfg.eos_id],
                "attention_mask": [1, 1, 1],
                "labels": [cfg.ignore_index, 256, cfg.eos_id],
                "word_pos": [0, 0, 1],
                "morph_depth": [0, 1, 0],
            },
            {
                "input_ids": [cfg.bos_id, cfg.eos_id],
                "attention_mask": [1, 1],
                "labels": [cfg.ignore_index, cfg.eos_id],
                "token_offsets": [(-1, -1), (-1, -1)],
            },
        ]
        batch = PretrainingCollator(
            pad_id=cfg.pad_id,
            ignore_index=cfg.ignore_index,
            pad_to_multiple_of=4,
        )(rows)

        self.assertEqual(batch["input_ids"].shape, (2, 4))
        self.assertEqual(batch["attention_mask"][1].tolist(), [1, 1, 0, 0])
        self.assertEqual(batch["labels"][1, -1].item(), cfg.ignore_index)
        self.assertEqual(batch["word_pos"].shape, batch["input_ids"].shape)

    def test_all_ignored_labels_return_zero_loss(self):
        torch.manual_seed(0)
        cfg = _test_config()
        model = RDTForCausalLM(cfg)

        input_ids = torch.tensor([[cfg.bos_id]])
        labels = torch.full_like(input_ids, cfg.ignore_index)

        out = model(input_ids=input_ids, labels=labels)

        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertEqual(float(out["loss"].detach()), 0.0)

        out["loss"].backward()
        self.assertIsNotNone(model.embed.weight.grad)

    def test_mamba_mask_skips_left_padding_state(self):
        torch.manual_seed(0)
        cfg = _test_config()
        layer = Mamba3Layer(cfg)
        layer.eval()

        tokens = torch.randn(1, 2, cfg.d_model)
        padded = torch.cat([torch.zeros(1, 2, cfg.d_model), tokens], dim=1)
        mask = torch.tensor([[0, 0, 1, 1]])

        with torch.no_grad():
            padded_out = layer(padded, attn_mask=mask)
            short_out = layer(tokens)

        self.assertTrue(
            torch.allclose(padded_out[:, 2:], short_out, atol=1e-5, rtol=1e-5)
        )
        self.assertTrue(
            torch.equal(padded_out[:, :2], torch.zeros_like(padded_out[:, :2]))
        )

    def test_fallback_mamba_honors_configured_state_size(self):
        cfg = _test_config()
        cfg.mamba_d_state = 24

        layer = Mamba3Layer(cfg)

        self.assertEqual(layer.backend, "fallback")
        self.assertEqual(layer.mamba.d_state, 24)

    def test_official_mamba_rejects_too_small_state_size(self):
        with self.assertRaisesRegex(ValueError, "official Mamba3 requires"):
            RDTConfig(
                d_model=32,
                n_heads=4,
                head_dim=8,
                kv_lora_rank=8,
                rope_head_dim=4,
                nope_head_dim=4,
                ffn_hidden=64,
                ffn_multiple=32,
                mamba_d_state=MIN_OFFICIAL_MAMBA3_D_STATE - 1,
                mamba_headdim=16,
                use_official_mamba=True,
            )

    def test_official_mamba_rejects_masked_state_updates(self):
        class FakeOfficialMamba3(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()

            def forward(self, x):
                return x

        old = mamba3_layer.OfficialMamba3
        mamba3_layer.OfficialMamba3 = FakeOfficialMamba3
        try:
            cfg = _test_config()
            cfg.mamba_d_state = MIN_OFFICIAL_MAMBA3_D_STATE
            cfg.use_official_mamba = True
            layer = Mamba3Layer(cfg)

            x = torch.randn(1, 3, cfg.d_model)
            layer(x, attn_mask=torch.ones(1, 3, dtype=torch.long))
            out = layer(x, attn_mask=torch.tensor([[1, 1, 0]]))
            self.assertTrue(torch.equal(out[:, 2:], torch.zeros_like(out[:, 2:])))

            with self.assertRaisesRegex(ValueError, "official Mamba backend"):
                layer(x, attn_mask=torch.tensor([[0, 1, 1]]))
        finally:
            mamba3_layer.OfficialMamba3 = old

    def test_mla_sdpa_matches_math_attention(self):
        torch.manual_seed(0)
        cfg_sdpa = _test_config()
        cfg_sdpa.use_sdpa_attention = True
        cfg_math = _test_config()
        cfg_math.use_sdpa_attention = False

        sdpa = MLA(cfg_sdpa)
        math = MLA(cfg_math)
        math.load_state_dict(sdpa.state_dict())
        sdpa.eval()
        math.eval()

        x = torch.randn(2, 5, cfg_sdpa.d_model)
        word_pos = torch.arange(5).unsqueeze(0).expand(2, 5)
        morph_depth = torch.zeros(2, 5, dtype=torch.long)
        attn_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])

        with torch.no_grad():
            out_sdpa = sdpa(
                x,
                word_pos=word_pos,
                morph_depth=morph_depth,
                attn_mask=attn_mask,
            )
            out_math = math(
                x,
                word_pos=word_pos,
                morph_depth=morph_depth,
                attn_mask=attn_mask,
            )

        self.assertTrue(torch.allclose(out_sdpa, out_math, atol=1e-5, rtol=1e-5))

    def test_unbatched_visual_features_require_single_batch(self):
        cfg = _test_config()
        inputs = torch.zeros(2, 3, cfg.d_model)
        input_ids = torch.tensor(
            [
                [cfg.bos_id, cfg.image_patch_id, cfg.eos_id],
                [cfg.bos_id, cfg.image_patch_id, cfg.eos_id],
            ]
        )
        features = torch.randn(2, cfg.d_model)

        with self.assertRaisesRegex(ValueError, "batch size is 1"):
            inject_visual_features(inputs, input_ids, features, cfg.image_patch_id)

    def test_batched_visual_features_must_match_patch_slots_exactly(self):
        cfg = _test_config()
        inputs = torch.zeros(2, 4, cfg.d_model)
        input_ids = torch.tensor(
            [
                [cfg.bos_id, cfg.image_patch_id, cfg.image_patch_id, cfg.eos_id],
                [cfg.bos_id, 300, cfg.image_patch_id, cfg.eos_id],
            ]
        )
        features = torch.randn(2, 2, cfg.d_model)

        with self.assertRaisesRegex(ValueError, "must equal image_patch count"):
            inject_visual_features(inputs, input_ids, features, cfg.image_patch_id)

    def test_config_rejects_invalid_dropout(self):
        with self.assertRaisesRegex(ValueError, "dropout"):
            RDTConfig(dropout=1.0)


if __name__ == "__main__":
    unittest.main()
