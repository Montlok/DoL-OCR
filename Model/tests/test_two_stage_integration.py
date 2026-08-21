# -*- coding: utf-8 -*-

import unittest

import torch

from Model.config import tiny_config, two_stage_tiny_config
from Model.model import RDTForCausalLM
from Model.two_stage import TwoStageCore


class TwoStageIntegrationTest(unittest.TestCase):
    def test_two_stage_forward_backward_is_finite(self):
        torch.manual_seed(0)
        cfg = two_stage_tiny_config()
        model = RDTForCausalLM(cfg)

        self.assertIsInstance(model.recurrent, TwoStageCore)

        input_ids = torch.randint(256, cfg.vocab_size, (2, 24))
        input_ids[:, 0] = cfg.bos_id
        labels = input_ids.clone()

        out = model(input_ids=input_ids, labels=labels)

        self.assertEqual(out["logits"].shape, (2, 24, cfg.vocab_size))
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertEqual(out["rec_info"]["steps_used"], cfg.recurrent_steps)

        out["loss"].backward()
        self.assertIsNotNone(model.embed.weight.grad)
        self.assertTrue(torch.isfinite(model.embed.weight.grad).all())

    def test_two_stage_is_strictly_causal(self):
        """Logits at positions 0..t must not depend on tokens after t."""

        torch.manual_seed(0)
        cfg = two_stage_tiny_config()
        model = RDTForCausalLM(cfg)
        model.eval()

        seq_len = 12
        split = 6

        base = torch.randint(256, cfg.vocab_size, (1, seq_len))
        base[:, 0] = cfg.bos_id

        other = base.clone()
        # Perturb every token strictly after ``split``.
        other[:, split + 1 :] = torch.randint(
            256, cfg.vocab_size, (1, seq_len - split - 1)
        )

        # Supply identical, explicit morph info so the default derivation (which
        # scans the whole row) cannot itself introduce a prefix difference.
        word_pos = torch.arange(seq_len).unsqueeze(0)
        morph_depth = torch.zeros(1, seq_len, dtype=torch.long)

        with torch.no_grad():
            out_base = model(
                input_ids=base,
                word_pos=word_pos,
                morph_depth=morph_depth,
            )["logits"]
            out_other = model(
                input_ids=other,
                word_pos=word_pos,
                morph_depth=morph_depth,
            )["logits"]

        prefix_base = out_base[:, : split + 1]
        prefix_other = out_other[:, : split + 1]

        self.assertTrue(
            torch.allclose(prefix_base, prefix_other, rtol=0.0, atol=1e-5),
            msg=(
                "prefix logits diverged: max diff "
                f"{(prefix_base - prefix_other).abs().max().item():.3e}"
            ),
        )

        # Sanity: the suffix actually changed, so the test is non-trivial.
        suffix_diff = (out_base[:, split + 1 :] - out_other[:, split + 1 :]).abs().max()
        self.assertGreater(suffix_diff.item(), 1e-4)

    def test_interleaved_core_unaffected(self):
        torch.manual_seed(0)
        cfg = tiny_config()
        model = RDTForCausalLM(cfg)

        self.assertEqual(type(model.recurrent).__name__, "RecurrentCore")

        input_ids = torch.randint(256, 24576, (2, 16))
        input_ids[:, 0] = cfg.bos_id
        labels = input_ids.clone()

        out = model(input_ids=input_ids, labels=labels)

        self.assertTrue(torch.isfinite(out["loss"]))

        out["loss"].backward()
        self.assertIsNotNone(model.embed.weight.grad)
        self.assertTrue(torch.isfinite(model.embed.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
