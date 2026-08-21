# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest

import torch

from Model.config import RDTConfig
from Model.model import RDTForCausalLM


def _tiny_config() -> RDTConfig:
    return RDTConfig(
        d_model=16,
        n_heads=2,
        head_dim=8,
        kv_lora_rank=8,
        rope_head_dim=4,
        nope_head_dim=4,
        ffn_hidden=32,
        ffn_multiple=16,
        n_prelude=1,
        n_coda=0,
        mamba_per_block=0,
        attn_per_block=1,
        recurrent_steps=1,
        mamba_d_state=8,
        mamba_expand=2,
        mamba_headdim=8,
        use_official_mamba=False,
        max_seq_len=16,
        bidirectional=False,
        tie_word_embeddings=False,
        dropout=0.0,
    )


class VisionCrossAttentionIntegrationTest(unittest.TestCase):
    def test_install_is_exact_noop_then_trains_and_isolates_samples(self) -> None:
        torch.manual_seed(23)
        model = RDTForCausalLM(_tiny_config()).eval()
        input_ids = torch.tensor(
            [
                [2, 256, 257, 3],
                [2, 258, 259, 3],
            ],
            dtype=torch.long,
        )
        attention_mask = torch.ones_like(input_ids)
        word_pos = torch.tensor([[0, 0, 0, 1], [0, 0, 0, 1]])
        morph_depth = torch.tensor([[0, 0, 1, 0], [0, 0, 1, 0]])
        common = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_pos": word_pos,
            "morph_depth": morph_depth,
        }

        self.assertIsNone(model.vision_cross_attention)
        self.assertFalse(
            any(key.startswith("vision_cross_attention.") for key in model.state_dict())
        )
        with torch.no_grad():
            before_install = model(**common)["logits"]

        bridge = model.install_vision_cross_attention(memory_dim=12, n_heads=2)
        self.assertTrue(
            torch.equal(
                bridge.output_projection.weight,
                torch.zeros_like(bridge.output_projection.weight),
            )
        )
        memory = torch.randn(5, 12)
        cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)
        with torch.no_grad():
            detail_off = model(**common)["logits"]
            detail_on = model(
                **common,
                detail_memory=memory,
                detail_cu_seqlens=cu_seqlens,
            )["logits"]
        self.assertTrue(torch.equal(before_install, detail_off))
        self.assertTrue(torch.equal(before_install, detail_on))

        # A zero output projection intentionally opens the path in two stages:
        # the first update learns the residual write; subsequent updates carry
        # gradient through the attention projections into OMVT detail memory.
        model.train()
        optimizer = torch.optim.SGD([bridge.output_projection.weight], lr=0.5)
        first = model(
            **common,
            detail_memory=memory,
            detail_cu_seqlens=cu_seqlens,
        )["logits"]
        first_loss = first[0, -1, 256] + 0.7 * first[1, -1, 258]
        first_loss.backward()
        self.assertTrue(bool((bridge.output_projection.weight.grad != 0).any()))
        optimizer.step()

        model.zero_grad(set_to_none=True)
        trained_memory = memory.detach().clone().requires_grad_(True)
        second = model(
            **common,
            detail_memory=trained_memory,
            detail_cu_seqlens=cu_seqlens,
        )["logits"]
        second_loss = second[0, -1, 256] + 0.7 * second[1, -1, 258]
        second_loss.backward()
        self.assertIsNotNone(trained_memory.grad)
        self.assertTrue(bool((trained_memory.grad != 0).any()))
        self.assertIsNotNone(bridge.value_projection.weight.grad)
        self.assertTrue(bool((bridge.value_projection.weight.grad != 0).any()))

        model.eval()
        changed_memory = trained_memory.detach().clone()
        changed_memory[2:] = changed_memory[2:] + 5.0 * torch.randn_like(
            changed_memory[2:]
        )
        with torch.no_grad():
            original = model(
                **common,
                detail_memory=trained_memory.detach(),
                detail_cu_seqlens=cu_seqlens,
            )["logits"]
            changed = model(
                **common,
                detail_memory=changed_memory,
                detail_cu_seqlens=cu_seqlens,
            )["logits"]
        self.assertTrue(torch.equal(original[0], changed[0]))
        self.assertFalse(torch.equal(original[1], changed[1]))

    def test_ragged_boundaries_fail_closed(self) -> None:
        model = RDTForCausalLM(_tiny_config())
        model.install_vision_cross_attention(memory_dim=4)
        with self.assertRaisesRegex(ValueError, "end at sumM"):
            model(
                input_ids=torch.tensor([[2, 3], [2, 3]]),
                detail_memory=torch.randn(3, 4),
                detail_cu_seqlens=torch.tensor([0, 1, 2], dtype=torch.int32),
            )
        with self.assertRaisesRegex(ValueError, "supplied together"):
            model(
                input_ids=torch.tensor([[2, 3]]),
                detail_memory=torch.randn(1, 4),
            )

    def test_cache_free_generation_consumes_detail_memory(self) -> None:
        torch.manual_seed(29)
        model = RDTForCausalLM(_tiny_config()).eval()
        prompt = torch.tensor([[2, 256, 3]], dtype=torch.long)
        before = model.generate(prompt, max_new_tokens=2, greedy=True)
        model.install_vision_cross_attention(memory_dim=6, n_heads=2)
        memory = torch.randn(3, 6)
        offsets = torch.tensor([0, 3], dtype=torch.int32)
        after = model.generate(
            prompt,
            max_new_tokens=2,
            greedy=True,
            detail_memory=memory,
            detail_cu_seqlens=offsets,
        )
        self.assertTrue(torch.equal(before, after))
        with self.assertRaisesRegex(ValueError, "supplied together"):
            model.generate(
                prompt,
                max_new_tokens=1,
                greedy=True,
                detail_memory=memory,
            )


if __name__ == "__main__":
    unittest.main()
