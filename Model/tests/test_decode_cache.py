# -*- coding: utf-8 -*-

"""Bit-exactness tests for the incremental KV/state decode cache.

The whole-model cache is correct only if it is *numerically identical* to a
fresh full forward over the growing prefix. These tests pin that contract at
three levels (single layer, whole-model logits, ``generate()`` API) and probe
the adversarial corners surfaced from user / engineer / peer perspectives:
single-token prompts, EOS-triggered padding, batch > 1, non-morphological RoPE
(exercising ``pos_offset``), the plain (non-mHC) refinement loop, and the
official-Mamba prefill/step adapter contract.
"""

import unittest
from unittest import mock

import torch
import torch.nn as nn

from Model.blocks import MambaSubLayer
from Model.config import MIN_OFFICIAL_MAMBA3_D_STATE, RDTConfig
from Model.inference.cache import DecodeCache, MambaCache, MLACache
from Model.layers import mamba3_layer as mamba3_module
from Model.layers.mla import MLA
from Model.model import RDTForCausalLM

ATOL = 1e-4


class _FakeInferenceParams:
    def __init__(
        self,
        max_seqlen,
        max_batch_size,
        seqlen_offset=0,
        batch_size_offset=0,
        key_value_memory_dict=None,
        lengths_per_sample=None,
    ):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = seqlen_offset
        self.batch_size_offset = batch_size_offset
        self.key_value_memory_dict = (
            {} if key_value_memory_dict is None else key_value_memory_dict
        )
        self.lengths_per_sample = lengths_per_sample


class _FakeOfficialMamba3(nn.Module):
    """CPU contract double; it does not emulate the CUDA Mamba-3 kernel."""

    def __init__(self, d_model, layer_idx=None, **_kwargs):
        super().__init__()
        self.d_model = d_model
        self.layer_idx = layer_idx
        self.events = []

    def allocate_inference_cache(
        self,
        batch_size,
        max_seqlen,
        device=None,
        dtype=None,
        **_kwargs,
    ):
        self.events.append(("allocate", batch_size, max_seqlen))
        return (
            torch.zeros(batch_size, self.d_model, device=device, dtype=dtype),
            torch.zeros(batch_size, 1, device=device, dtype=dtype),
            torch.zeros(batch_size, 1, device=device, dtype=dtype),
            torch.zeros(batch_size, 1, device=device, dtype=dtype),
        )

    @staticmethod
    def _scan(u, state):
        outputs = []
        for token in u.unbind(dim=1):
            state.add_(token)
            outputs.append(state.clone())
        return torch.stack(outputs, dim=1)

    def forward(
        self,
        u,
        seq_idx=None,
        cu_seqlens=None,
        inference_params=None,
    ):
        del seq_idx, cu_seqlens
        if inference_params is None:
            state = torch.zeros_like(u[:, 0, :])
            return self._scan(u, state)
        if inference_params.seqlen_offset != 0:
            raise AssertionError("adapter must call step() after prefill")
        self.events.append(("prefill", u.shape[1], id(inference_params)))
        state = inference_params.key_value_memory_dict[self.layer_idx][0]
        return self._scan(u, state)

    def step(self, u, angle_state, ssm_state, k_state, v_state):
        if u.ndim != 2:
            raise AssertionError("official step input must be [B, d_model]")
        self.events.append(("step", u.shape[0]))
        angle_state.add_(u)
        ssm_state.add_(1)
        k_state.add_(1)
        v_state.add_(1)
        return angle_state.clone(), angle_state, ssm_state, k_state, v_state


def _two_stage_cfg(drift_mode: str = "mhc", morph_rope: bool = True) -> RDTConfig:
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
        mamba_d_conv=4,
        use_official_mamba=False,
        use_morphological_rope=morph_rope,
        max_seq_len=64,
        core_type="two_stage",
        stage1_mamba_layers=3,
        stage2_attn_layers=2,
        recurrent_drift_mode=drift_mode,
        mhc_n_streams=4,
        mhc_sinkhorn_iters=10,
    )


def _official_cfg(max_seq_len: int = 16) -> RDTConfig:
    cfg = _two_stage_cfg()
    cfg.mamba_d_state = MIN_OFFICIAL_MAMBA3_D_STATE
    cfg.use_official_mamba = True
    cfg.max_seq_len = max_seq_len
    return cfg


class MLACacheTest(unittest.TestCase):
    def test_mla_cached_matches_full(self):
        torch.manual_seed(0)
        cfg = _two_stage_cfg()
        mla = MLA(cfg).eval()
        b, length = 2, 9
        x = torch.randn(b, length, cfg.d_model)
        wp = torch.arange(length).unsqueeze(0).expand(b, length).contiguous()
        md = torch.zeros(b, length, dtype=torch.long)

        with torch.no_grad():
            ref = mla(x, word_pos=wp, morph_depth=md)
            cache = MLACache()
            parts = [mla(x[:, :3], word_pos=wp[:, :3], morph_depth=md[:, :3],
                         cache=cache)]
            for t in range(3, length):
                parts.append(
                    mla(x[:, t:t + 1], word_pos=wp[:, t:t + 1],
                        morph_depth=md[:, t:t + 1], cache=cache)
                )
            cached = torch.cat(parts, dim=1)

        self.assertTrue(torch.allclose(ref, cached, atol=ATOL))

    def test_mla_cached_non_morph_rope_uses_pos_offset(self):
        torch.manual_seed(1)
        cfg = _two_stage_cfg(morph_rope=False)
        mla = MLA(cfg).eval()
        b, length = 2, 8
        x = torch.randn(b, length, cfg.d_model)

        with torch.no_grad():
            ref = mla(x)
            cache = MLACache()
            parts = [mla(x[:, :2], cache=cache, pos_offset=0)]
            for t in range(2, length):
                parts.append(mla(x[:, t:t + 1], cache=cache, pos_offset=t))
            cached = torch.cat(parts, dim=1)

        self.assertTrue(torch.allclose(ref, cached, atol=ATOL))


class MambaCacheTest(unittest.TestCase):
    def test_mamba_cached_matches_full(self):
        torch.manual_seed(0)
        cfg = _two_stage_cfg()
        layer = MambaSubLayer(cfg, layer_idx=0).eval()
        b, length = 2, 9
        x = torch.randn(b, length, cfg.d_model)

        with torch.no_grad():
            ref = layer(x)
            cache = MambaCache()
            parts = [layer(x[:, :4], cache=cache)]
            for t in range(4, length):
                parts.append(layer(x[:, t:t + 1], cache=cache))
            cached = torch.cat(parts, dim=1)

        self.assertTrue(torch.allclose(ref, cached, atol=ATOL))
        self.assertEqual(cache.backend, "naive")
        self.assertIsNone(cache.official_inference_params)


class OfficialMambaCacheAdapterTest(unittest.TestCase):
    def _patch_official(self, *, version="2.3.2.post1"):
        return mock.patch.multiple(
            mamba3_module,
            OfficialMamba3=_FakeOfficialMamba3,
            OfficialInferenceParams=_FakeInferenceParams,
            _OFFICIAL_MAMBA_VERSION=version,
        )

    def test_prefill_then_step_matches_full_and_isolates_call_sites(self):
        torch.manual_seed(2)
        with self._patch_official():
            layer = MambaSubLayer(_official_cfg(), layer_idx=0).eval()
            x = torch.randn(2, 6, layer.mamba.cfg.d_model)

            with torch.no_grad():
                reference = layer(x)
                first_run = DecodeCache()
                cache = first_run.mamba_cache("stage1.0")
                pieces = [layer(x[:, :3], cache=cache)]
                for token_idx in range(3, x.shape[1]):
                    pieces.append(layer(x[:, token_idx:token_idx + 1], cache=cache))
                incremental = torch.cat(pieces, dim=1)

            self.assertTrue(torch.allclose(reference, incremental, atol=ATOL))
            self.assertEqual(cache.backend, "official")
            params = cache.official_inference_params
            self.assertEqual(params.seqlen_offset, x.shape[1])
            self.assertEqual(set(params.key_value_memory_dict), {0})
            self.assertEqual(
                [event[0] for event in layer.mamba.mamba.events],
                ["allocate", "prefill", "step", "step", "step"],
            )

            second_cache = DecodeCache().mamba_cache("stage1.0")
            with torch.no_grad():
                second_prefill = layer(x[:, :3], cache=second_cache)
            self.assertTrue(torch.allclose(reference[:, :3], second_prefill, atol=ATOL))
            self.assertIsNot(
                cache.official_inference_params,
                second_cache.official_inference_params,
            )
            first_state = params.key_value_memory_dict[0][0]
            second_state = second_cache.official_inference_params.key_value_memory_dict[0][0]
            self.assertIsNot(first_state, second_state)

    def test_binding_batch_capacity_backend_and_version_fail_closed(self):
        x = torch.randn(2, 3, _official_cfg().d_model)

        with self._patch_official(version="2.3.1"):
            layer = MambaSubLayer(_official_cfg(), layer_idx=0).eval()
            with self.assertRaisesRegex(RuntimeError, "2.3.2.post1"):
                layer(x[:, :2], cache=MambaCache())

        with self._patch_official():
            layer = MambaSubLayer(_official_cfg(max_seq_len=4), layer_idx=0).eval()
            cache = MambaCache()
            with torch.no_grad():
                layer(x[:, :2], cache=cache)

            with self.assertRaisesRegex(ValueError, "batch size is fixed"):
                layer(x[:1, :1], cache=cache)
            with self.assertRaisesRegex(ValueError, "exactly one token"):
                layer(x[:, :2], cache=cache)

            other_layer = MambaSubLayer(_official_cfg(max_seq_len=4), layer_idx=0).eval()
            with self.assertRaisesRegex(RuntimeError, "different Mamba call site"):
                other_layer(x[:, :1], cache=cache)

            with self.assertRaisesRegex(RuntimeError, "backend='naive'"):
                layer(x[:, :2], cache=MambaCache(backend="naive"))

            with self.assertRaisesRegex(ValueError, "exceeds max_seq_len"):
                layer(
                    torch.randn(2, 5, x.shape[-1]),
                    cache=MambaCache(),
                )


class WholeModelCacheTest(unittest.TestCase):
    def _gold(self, cfg: RDTConfig, prefill: int = 4):
        torch.manual_seed(0)
        model = RDTForCausalLM(cfg).eval()
        b, length = 2, 11
        ids = torch.randint(300, cfg.vocab_size, (b, length))

        with torch.no_grad():
            ref = model(ids, return_logits=True)["logits"]
            cache = DecodeCache()
            mask = torch.ones_like(ids)
            wp, md = model._default_morph_info(ids, mask)
            diffs = []
            lg = model._forward_decode(
                ids[:, :prefill], wp[:, :prefill], md[:, :prefill], cache
            )
            for t in range(prefill):
                diffs.append((ref[:, t, :] - lg[:, t, :]).abs().max().item())
            for t in range(prefill, length):
                lg1 = model._forward_decode(
                    ids[:, t:t + 1], wp[:, t:t + 1], md[:, t:t + 1], cache
                )
                diffs.append((ref[:, t, :] - lg1[:, 0, :]).abs().max().item())
        return max(diffs)

    def test_incremental_logits_bit_exact_mhc(self):
        self.assertLess(self._gold(_two_stage_cfg("mhc")), ATOL)

    def test_incremental_logits_bit_exact_plain(self):
        self.assertLess(self._gold(_two_stage_cfg("none")), ATOL)

    def test_incremental_logits_bit_exact_decay(self):
        self.assertLess(self._gold(_two_stage_cfg("decay")), ATOL)

    def test_cache_seq_len_tracks_tokens(self):
        torch.manual_seed(0)
        cfg = _two_stage_cfg()
        model = RDTForCausalLM(cfg).eval()
        ids = torch.randint(300, cfg.vocab_size, (1, 5))
        cache = DecodeCache()
        mask = torch.ones_like(ids)
        wp, md = model._default_morph_info(ids, mask)
        with torch.no_grad():
            model._forward_decode(ids, wp, md, cache)
        self.assertEqual(cache.seq_len, 5)


class GenerateCacheEquivalenceTest(unittest.TestCase):
    def _check(self, ids, **kw):
        torch.manual_seed(0)
        cfg = _two_stage_cfg()
        model = RDTForCausalLM(cfg).eval()
        with torch.no_grad():
            a = model.generate(ids, greedy=True, use_cache=False, **kw)
            b = model.generate(ids, greedy=True, use_cache=True, **kw)
        self.assertTrue(torch.equal(a, b), msg=f"{a}\n!=\n{b}")

    def test_batch_equivalence(self):
        self._check(torch.randint(300, 320, (2, 7)), max_new_tokens=12)

    def test_single_token_prompt(self):
        self._check(torch.randint(300, 320, (1, 1)), max_new_tokens=10)

    def test_eos_padding_equivalence(self):
        self._check(torch.randint(300, 320, (3, 5)), max_new_tokens=15, eos_id=305)

    def test_repetition_penalty_equivalence(self):
        self._check(
            torch.randint(300, 320, (2, 6)),
            max_new_tokens=10,
            repetition_penalty=1.3,
        )

    def test_special_tokens_in_prompt(self):
        ids = torch.tensor([[1, 2, 300, 301, 302, 3, 303]])
        self._check(ids, max_new_tokens=8)


class CacheRejectionTest(unittest.TestCase):
    def test_use_cache_rejects_non_two_stage(self):
        cfg = RDTConfig(
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
            max_seq_len=16,
        )
        model = RDTForCausalLM(cfg).eval()
        ids = torch.randint(300, cfg.vocab_size, (1, 3))
        with self.assertRaises(NotImplementedError):
            model.generate(ids, max_new_tokens=2, use_cache=True)

    def test_cached_generation_max_seq_len_guard(self):
        torch.manual_seed(0)
        cfg = _two_stage_cfg()
        cfg_small = cfg
        object.__setattr__(cfg_small, "max_seq_len", 8)
        model = RDTForCausalLM(cfg_small).eval()
        ids = torch.randint(300, 320, (1, 6))
        with self.assertRaises(ValueError):
            model.generate(ids, max_new_tokens=10, greedy=True, use_cache=True)

    def test_cached_generation_rejects_prompt_at_capacity(self):
        # Prompt already at max_seq_len with max_new_tokens=1 must be rejected
        # up front (it would otherwise return a sequence of max_seq_len + 1).
        torch.manual_seed(0)
        cfg = _two_stage_cfg()
        object.__setattr__(cfg, "max_seq_len", 8)
        model = RDTForCausalLM(cfg).eval()
        ids = torch.randint(300, 320, (1, 8))
        with self.assertRaises(ValueError):
            model.generate(ids, max_new_tokens=1, greedy=True, use_cache=True)

    def test_cached_generation_rejects_padded_prompt(self):
        # The incremental cache cannot mask padding, so a padded prompt must be
        # rejected rather than silently corrupting cached keys/values.
        torch.manual_seed(0)
        cfg = _two_stage_cfg()
        model = RDTForCausalLM(cfg).eval()
        ids = torch.randint(300, 320, (1, 4))
        ids[0, -1] = cfg.pad_id
        with self.assertRaises(ValueError):
            model.generate(ids, max_new_tokens=2, greedy=True, use_cache=True)


class ActCacheExclusionTest(unittest.TestCase):
    """ACT (PonderNet) and the decode cache are mutually exclusive by
    construction: ``use_cache`` requires the two_stage/segmented cores
    (model.py) and both of those cores reject ``use_act=True``. Pin the
    guards from both sides so a refactor that lifts either one must handle
    the combination explicitly — variable ponder depth would silently
    desync a cache that assumes one refinement call per token.
    """

    def test_two_stage_core_rejects_act(self):
        cfg = _two_stage_cfg()
        object.__setattr__(cfg, "use_act", True)
        object.__setattr__(cfg, "act_max_steps", 4)
        with self.assertRaisesRegex(ValueError, "use_act"):
            RDTForCausalLM(cfg)

    def test_segmented_core_rejects_act(self):
        from Model.segmented import SegmentedCore  # noqa: F401  (guard lives there)

        cfg = _two_stage_cfg()
        object.__setattr__(cfg, "core_type", "segmented")
        object.__setattr__(cfg, "use_act", True)
        object.__setattr__(cfg, "act_max_steps", 4)
        with self.assertRaisesRegex(ValueError, "use_act"):
            RDTForCausalLM(cfg)


if __name__ == "__main__":
    unittest.main()
