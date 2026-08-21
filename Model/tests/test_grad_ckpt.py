# -*- coding: utf-8 -*-

import unittest

import torch

from Model.config import RDTConfig
from Model.model import RDTForCausalLM


def _cfg(grad_ckpt: bool) -> RDTConfig:
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
        recurrent_steps=3,
        mamba_d_state=8,
        mamba_expand=2,
        mamba_headdim=16,
        use_official_mamba=False,
        max_seq_len=64,
        grad_ckpt_recurrent=grad_ckpt,
        grad_ckpt_prelude_coda=grad_ckpt,
    )


class GradCkptEquivalenceTest(unittest.TestCase):
    def test_loss_and_grad_match_with_without_checkpointing(self):
        torch.manual_seed(0)
        cfg_plain = _cfg(False)
        plain = RDTForCausalLM(cfg_plain)
        plain.train()

        torch.manual_seed(0)
        cfg_ckpt = _cfg(True)
        ckpt = RDTForCausalLM(cfg_ckpt)
        ckpt.train()
        ckpt.load_state_dict(plain.state_dict())

        input_ids = torch.tensor(
            [[cfg_plain.bos_id, 300, 301, 302, cfg_plain.eos_id]]
        )
        labels = input_ids.clone()

        out_plain = plain(input_ids=input_ids, labels=labels)
        out_ckpt = ckpt(input_ids=input_ids, labels=labels)

        self.assertTrue(
            torch.allclose(out_plain["loss"], out_ckpt["loss"], atol=1e-5, rtol=1e-5),
            msg=f"loss mismatch: {out_plain['loss']} vs {out_ckpt['loss']}",
        )

        out_plain["loss"].backward()
        out_ckpt["loss"].backward()

        plain_grad = plain.embed.weight.grad
        ckpt_grad = ckpt.embed.weight.grad
        self.assertIsNotNone(plain_grad)
        self.assertIsNotNone(ckpt_grad)
        self.assertTrue(
            torch.allclose(plain_grad, ckpt_grad, atol=1e-4, rtol=1e-4),
            msg="embed grad mismatch under grad checkpointing",
        )


if __name__ == "__main__":
    unittest.main()
