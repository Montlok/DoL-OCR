# -*- coding: utf-8 -*-

"""Tests for per-token log-prob helpers (Model/posttrain/logprobs.py).

The headline test pins the RL-critical contract: log-probs from a single full
forward must match those implied by the incremental decode cache used for
sampling. This reuses the bit-exact cache machinery exercised by
``test_decode_cache``.
"""

import unittest

import torch
import torch.nn.functional as F

from Model.config import RDTConfig
from Model.inference.cache import DecodeCache
from Model.model import RDTForCausalLM
from Model.ocr.position_contract import BOUNDARY_V1
from Model.posttrain.logprobs import (
    completion_logprobs,
    logits_to_token_logprobs,
    sequence_logprobs,
)

ATOL = 1e-4


def _cfg() -> RDTConfig:
    return RDTConfig(
        d_model=32,
        n_heads=4,
        head_dim=8,
        kv_lora_rank=8,
        rope_head_dim=4,
        nope_head_dim=4,
        ffn_hidden=64,
        ffn_multiple=32,
        n_prelude=2,
        n_coda=2,
        recurrent_steps=3,
        mamba_d_state=8,
        mamba_expand=2,
        mamba_headdim=8,
        use_official_mamba=False,
        max_seq_len=64,
        core_type="two_stage",
        stage1_mamba_layers=3,
        stage2_attn_layers=2,
        recurrent_drift_mode="mhc",
    )


def _incremental_logprobs(model, ids, prefill=2):
    """Per-token log-probs computed via the incremental decode cache."""
    cache = DecodeCache()
    mask = torch.ones_like(ids)
    wp, md = model._default_morph_info(ids, mask)
    length = ids.shape[1]
    logits_steps = []
    lg = model._forward_decode(ids[:, :prefill], wp[:, :prefill], md[:, :prefill], cache)
    logits_steps.append(lg)
    for t in range(prefill, length):
        lg1 = model._forward_decode(
            ids[:, t : t + 1], wp[:, t : t + 1], md[:, t : t + 1], cache
        )
        logits_steps.append(lg1)
    logits = torch.cat(logits_steps, dim=1)
    return logits_to_token_logprobs(logits[:, :-1, :], ids[:, 1:])


class LogprobTest(unittest.TestCase):
    def _model_ids(self):
        torch.manual_seed(0)
        cfg = _cfg()
        model = RDTForCausalLM(cfg).eval()
        ids = torch.randint(300, cfg.vocab_size, (2, 11))
        return model, ids

    def test_full_forward_matches_incremental_decode(self):
        model, ids = self._model_ids()
        with torch.no_grad():
            full = sequence_logprobs(model, ids)
            incr = _incremental_logprobs(model, ids)
        self.assertEqual(full.shape, incr.shape)
        self.assertLess((full - incr).abs().max().item(), ATOL)

    def test_logprobs_are_log_softmax_fp32(self):
        torch.manual_seed(1)
        logits = torch.randn(2, 5, 7)
        targets = torch.randint(0, 7, (2, 5))
        got = logits_to_token_logprobs(logits, targets)
        ref = F.log_softmax(logits.float(), dim=-1).gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1)
        self.assertTrue(torch.allclose(got, ref))
        # Log-probs are non-positive.
        self.assertTrue((got <= 1e-5).all())

    def test_completion_mask_zeroes_prompt(self):
        model, ids = self._model_ids()
        completion_mask = torch.zeros_like(ids)
        completion_mask[:, 6:] = 1  # only the tail counts as the response
        with torch.no_grad():
            summed, token = completion_logprobs(model, ids, completion_mask)
        # Prompt positions (target index < 5 after the shift) must be exactly 0.
        self.assertTrue(torch.equal(token[:, :5], torch.zeros_like(token[:, :5])))
        # Sum equals the masked token log-probs.
        self.assertTrue(torch.allclose(summed, token.sum(dim=-1), atol=ATOL))

    def test_completion_slice_matches_full_path_exactly(self):
        model, ids = self._model_ids()
        start = 5
        with torch.no_grad():
            full = sequence_logprobs(
                model,
                ids,
                position_contract=BOUNDARY_V1,
            )
            sliced = sequence_logprobs(
                model,
                ids,
                completion_start=start,
                position_contract=BOUNDARY_V1,
            )
        self.assertTrue(
            torch.equal(sliced[:, :start], torch.zeros_like(sliced[:, :start]))
        )
        torch.testing.assert_close(
            sliced[:, start:],
            full[:, start:],
            rtol=0.0,
            atol=0.0,
        )
        with torch.no_grad():
            empty_tail = sequence_logprobs(
                model,
                ids,
                completion_start=ids.shape[1] - 1,
                position_contract=BOUNDARY_V1,
            )
        self.assertTrue(torch.equal(empty_tail, torch.zeros_like(empty_tail)))

    def test_completion_slice_rejects_invalid_start(self):
        model, ids = self._model_ids()
        for value, error in ((True, TypeError), (-1, ValueError), (11, ValueError)):
            with self.subTest(value=value):
                with self.assertRaises(error):
                    sequence_logprobs(model, ids, completion_start=value)

    def test_depth_knob_changes_but_is_deterministic(self):
        model, ids = self._model_ids()
        with torch.no_grad():
            shallow = sequence_logprobs(model, ids, recurrent_steps=1)
            deep = sequence_logprobs(model, ids, recurrent_steps=5)
            deep_again = sequence_logprobs(model, ids, recurrent_steps=5)
        self.assertFalse(torch.allclose(shallow, deep))
        self.assertTrue(torch.allclose(deep, deep_again))


if __name__ == "__main__":
    unittest.main()
