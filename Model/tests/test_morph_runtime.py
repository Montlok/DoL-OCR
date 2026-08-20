# -*- coding: utf-8 -*-

import unittest

import torch

from Model.config import RDTConfig
from Model.model import RDTForCausalLM
from Tokenizer.pretraining import derive_morph_info_from_boundary_ids
from Tokenizer.pretraining import derive_morph_info_from_track_ids


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


class DefaultMorphInfoTest(unittest.TestCase):
    def test_matches_python_reference(self):
        cfg = _cfg()
        model = RDTForCausalLM(cfg)
        wb = cfg.word_boundary_id
        mb = cfg.morpheme_boundary_id
        bos = cfg.bos_id
        eos = cfg.eos_id

        seqs = [
            [bos, wb, 300, mb, 301, wb, 302, eos],
            [bos, 300, wb, 301, mb, 302, mb, eos],
        ]
        input_ids = torch.tensor(seqs, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        word_pos, morph_depth = model._default_morph_info(input_ids, attention_mask)

        for i, seq in enumerate(seqs):
            ref_wp, ref_md = derive_morph_info_from_boundary_ids(
                seq, wb, mb
            )
            self.assertEqual(word_pos[i].tolist(), ref_wp, msg=f"row {i}")
            self.assertEqual(morph_depth[i].tolist(), ref_md, msg=f"row {i}")

    def test_no_boundaries_yields_word_zero(self):
        cfg = _cfg()
        model = RDTForCausalLM(cfg)
        input_ids = torch.tensor([[300, 301, 302, 303]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        word_pos, morph_depth = model._default_morph_info(input_ids, attention_mask)
        self.assertEqual(word_pos.tolist(), [[0, 0, 0, 0]])
        self.assertEqual(morph_depth.tolist(), [[0, 0, 0, 0]])

    def test_morph_depth_is_not_preclipped_before_rope(self):
        cfg = _cfg()
        cfg.max_morph_depth = 3
        model = RDTForCausalLM(cfg)
        mb = cfg.morpheme_boundary_id
        wb = cfg.word_boundary_id
        ids = [[wb, mb, mb, mb, mb, mb, 300]]
        input_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        _, morph_depth = model._default_morph_info(input_ids, attention_mask)
        self.assertGreater(int(morph_depth.max().item()), cfg.max_morph_depth)

    def test_long_word_track_and_explicit_features_have_identical_logits(self):
        cfg = _cfg()
        cfg.max_morph_depth = 3
        model = RDTForCausalLM(cfg).eval()
        table = torch.zeros(cfg.vocab_size, dtype=torch.long)
        table[300] = 1
        ids = torch.tensor(
            [[cfg.bos_id, 300, 300, 300, 300, 300, cfg.eos_id]],
            dtype=torch.long,
        )
        tracks = [int(table[token_id]) for token_id in ids[0].tolist()]
        expected_word_pos, expected_morph_depth = (
            derive_morph_info_from_track_ids(tracks)
        )
        self.assertGreater(max(expected_morph_depth), cfg.max_morph_depth)

        with torch.no_grad():
            explicit = model(
                ids,
                word_pos=torch.tensor([expected_word_pos]),
                morph_depth=torch.tensor([expected_morph_depth]),
            )["logits"]
            runtime = model(
                ids,
                morphology_track_table=table,
            )["logits"]
        torch.testing.assert_close(explicit, runtime, rtol=0, atol=0)

    def test_token_track_runtime_matches_python_contract(self):
        cfg = _cfg()
        model = RDTForCausalLM(cfg)
        table = torch.zeros(cfg.vocab_size, dtype=torch.long)
        table[300:303] = 1
        table[400:402] = 2
        rows = [
            [cfg.bos_id, 300, 301, cfg.word_boundary_id, 302, cfg.eos_id],
            [cfg.bos_id, 400, 401, 300, 301, cfg.eos_id],
        ]
        ids = torch.tensor(rows, dtype=torch.long)
        word_pos, morph_depth = model._morph_info_from_track_table(ids, table)
        for index, row in enumerate(rows):
            tracks = [int(table[token_id]) for token_id in row]
            expected = derive_morph_info_from_track_ids(tracks)
            self.assertEqual(word_pos[index].tolist(), expected[0])
            self.assertEqual(morph_depth[index].tolist(), expected[1])


if __name__ == "__main__":
    unittest.main()
