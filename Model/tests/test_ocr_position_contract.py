# -*- coding: utf-8 -*-

"""Focused regression tests for the versioned OCR position contract."""

from __future__ import annotations

import unittest

import torch

from Model.config import RDTConfig, TrainingConfig
from Model.model import RDTForCausalLM
from Model.ocr.data import build_ocr_row, split_ocr_row
from Model.ocr.position_contract import (
    BOUNDARY_V1,
    LEGACY_SEQUENTIAL_V0,
    OCR_POSITION_CONTRACT_METADATA_VERSION,
    resolve_checkpoint_ocr_position_contract,
)
from Model.training.data import PretrainingCollator
from Model.training.loop import TrainState, evaluate, train_one_step


def _cuda_bf16_test_supported() -> bool:
    """Avoid kernels when the installed PyTorch predates the GPU architecture."""

    if not torch.cuda.is_available():
        return False
    compiled = []
    for arch in torch.cuda.get_arch_list():
        if arch.startswith("sm_") and arch[3:].isdigit():
            digits = arch[3:]
            compiled.append((int(digits[:-1]), int(digits[-1])))
    return bool(compiled) and torch.cuda.get_device_capability() <= max(compiled)


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
        dropout=0.0,
    )


def _row(cfg: RDTConfig, target: list[int]) -> dict:
    return build_ocr_row(
        target,
        4,
        "unused.png",
        bos_id=cfg.bos_id,
        image_start_id=cfg.image_start_id,
        image_patch_id=cfg.image_patch_id,
        image_end_id=cfg.image_end_id,
        eos_id=cfg.eos_id,
    )


class OCRPositionContractTest(unittest.TestCase):
    def test_boundary_training_omits_positions_and_legacy_reproduces_fallback(self):
        cfg = _cfg()
        rows = [
            _row(
                cfg,
                [300, cfg.word_boundary_id, 301, cfg.morpheme_boundary_id, 302],
            ),
            _row(cfg, [303]),
        ]
        boundary = PretrainingCollator(position_contract=BOUNDARY_V1)(rows)
        self.assertNotIn("word_pos", boundary)
        self.assertNotIn("morph_depth", boundary)
        self.assertEqual(boundary["position_contract"], BOUNDARY_V1)

        legacy = PretrainingCollator(position_contract=LEGACY_SEQUENTIAL_V0)(rows)
        self.assertEqual(
            legacy["position_contract"], LEGACY_SEQUENTIAL_V0
        )
        model = RDTForCausalLM(cfg)
        expected_wp, expected_md = model._morph_info_for_position_contract(
            legacy["input_ids"], legacy["attention_mask"], LEGACY_SEQUENTIAL_V0
        )
        self.assertTrue(torch.equal(legacy["word_pos"], expected_wp))
        self.assertTrue(torch.equal(legacy["morph_depth"], expected_md))
        self.assertEqual(legacy["word_pos"][1, -1].item(), 0)  # padded position

        boundary_wp, boundary_md = model._morph_info_for_position_contract(
            boundary["input_ids"], boundary["attention_mask"], BOUNDARY_V1
        )
        self.assertEqual(boundary_wp[0, 1:6].unique().tolist(), [0])
        self.assertGreater(int(legacy["word_pos"][0, 5]), int(boundary_wp[0, 5]))
        self.assertGreater(int(boundary_md[0].max()), 0)

    def test_boundary_rows_reject_ambiguous_precomputed_positions(self):
        cfg = _cfg()
        row = _row(cfg, [300])
        row["word_pos"] = list(range(len(row["input_ids"])))
        row["morph_depth"] = [0] * len(row["input_ids"])
        with self.assertRaisesRegex(ValueError, "must omit"):
            PretrainingCollator(position_contract=BOUNDARY_V1)([row])

    def test_training_and_evaluation_forward_the_batch_contract(self):
        cfg = _cfg()
        batch = PretrainingCollator(position_contract=BOUNDARY_V1)(
            [_row(cfg, [300])]
        )

        class SpyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.cfg = cfg
                self.seen: list[str | None] = []

            def forward(self, *, position_contract=None, **_kwargs):
                self.seen.append(position_contract)
                loss = self.weight.square()
                return {"loss": loss, "loss_parts": {"forward": loss.detach()}}

        model = SpyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda _step: 1.0,
        )
        train_cfg = TrainingConfig(
            precision="fp32",
            grad_clip=1.0,
            grad_accum_steps=1,
        )
        train_one_step(
            model,
            iter([batch]),
            optimizer,
            scheduler,
            train_cfg,
            TrainState(),
            device=torch.device("cpu"),
        )
        evaluate(
            model,
            [batch],
            train_cfg,
            device=torch.device("cpu"),
            max_batches=1,
        )
        self.assertEqual(model.seen, [BOUNDARY_V1, BOUNDARY_V1])

    def test_explicit_contract_rejects_track_table(self):
        cfg = _cfg()
        model = RDTForCausalLM(cfg).eval()
        ids = torch.tensor([[cfg.bos_id, 300]], dtype=torch.long)
        table = torch.zeros(cfg.vocab_size, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            model(
                ids,
                morphology_track_table=table,
                position_contract=BOUNDARY_V1,
            )
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            model.generate(
                ids,
                max_new_tokens=1,
                morphology_track_table=table,
                position_contract=BOUNDARY_V1,
            )

    def test_boundary_contract_preserves_production_depth_preclip(self):
        cfg = _cfg()
        cfg.max_morph_depth = 2
        model = RDTForCausalLM(cfg).eval()
        ids = torch.tensor(
            [[
                300,
                cfg.morpheme_boundary_id,
                301,
                cfg.morpheme_boundary_id,
                302,
                cfg.morpheme_boundary_id,
                303,
            ]],
            dtype=torch.long,
        )
        attention = torch.ones_like(ids)
        _generic_word, generic_depth = model._default_morph_info(ids, attention)
        _boundary_word, boundary_depth = model._morph_info_for_position_contract(
            ids,
            attention,
            BOUNDARY_V1,
        )
        self.assertGreater(int(generic_depth.max()), cfg.max_morph_depth - 1)
        self.assertEqual(int(boundary_depth.max()), cfg.max_morph_depth - 1)

    def test_track_table_is_not_boundary_v1(self):
        cfg = _cfg()
        model = RDTForCausalLM(cfg).eval()
        ids = torch.tensor(
            [[300, 301, cfg.word_boundary_id, 302]],
            dtype=torch.long,
        )
        attention = torch.ones_like(ids)
        track_table = torch.zeros(cfg.vocab_size, dtype=torch.long)
        track_table[300] = 1
        track_table[301] = 1
        track_table[302] = 1
        track_word, _ = model._morph_info_from_track_table(ids, track_table)
        boundary_word, _ = model._morph_info_for_position_contract(
            ids,
            attention,
            BOUNDARY_V1,
        )
        self.assertNotEqual(
            int(track_word[0, 2]),
            int(boundary_word[0, 2]),
        )

    def test_boundary_and_legacy_position_semantics_are_exact(self):
        cfg = _cfg()
        row = _row(
            cfg,
            [300, cfg.word_boundary_id, 301, cfg.morpheme_boundary_id, 302],
        )
        ids = torch.tensor(
            [[*row["input_ids"], cfg.pad_id]], dtype=torch.long
        )
        attention = torch.tensor(
            [[*[1] * len(row["input_ids"]), 0]], dtype=torch.long
        )
        model = RDTForCausalLM(cfg)

        boundary_wp, boundary_md = model._morph_info_for_position_contract(
            ids, attention, BOUNDARY_V1
        )
        target_start = len(row["input_ids"]) - 6
        self.assertEqual(
            boundary_wp[0, target_start:].tolist(),
            [0, 1, 1, 1, 1, 1, 1],
        )
        self.assertEqual(
            boundary_md[0, target_start:].tolist(),
            [0, 0, 0, 1, 1, 0, 0],
        )

        legacy_wp, legacy_md = model._morph_info_for_position_contract(
            ids, attention, LEGACY_SEQUENTIAL_V0
        )
        self.assertEqual(
            legacy_wp[0, :-1].tolist(), list(range(ids.shape[1] - 1))
        )
        self.assertEqual(int(legacy_wp[0, -1]), 0)
        self.assertEqual(legacy_md.unique().tolist(), [0])

    def test_fp32_teacher_forced_and_generation_first_step_are_identical(self):
        torch.manual_seed(7)
        cfg = _cfg()
        model = RDTForCausalLM(cfg).eval()
        row = _row(
            cfg,
            [300, cfg.word_boundary_id, 301, cfg.morpheme_boundary_id, 302],
        )
        prompt, _target, _image = split_ocr_row(row, eos_id=cfg.eos_id)
        full_ids = torch.tensor([row["input_ids"]], dtype=torch.long)
        prompt_ids = torch.tensor([prompt], dtype=torch.long)

        for contract in (BOUNDARY_V1, LEGACY_SEQUENTIAL_V0):
            with self.subTest(contract=contract), torch.no_grad():
                full_mask = torch.ones_like(full_ids)
                prompt_mask = torch.ones_like(prompt_ids)
                full_wp, full_md = model._morph_info_for_position_contract(
                    full_ids, full_mask, contract
                )
                prompt_wp, prompt_md = model._morph_info_for_position_contract(
                    prompt_ids, prompt_mask, contract
                )
                self.assertTrue(
                    torch.equal(full_wp[:, : len(prompt)], prompt_wp)
                )
                self.assertTrue(
                    torch.equal(full_md[:, : len(prompt)], prompt_md)
                )
                full_logits = model(
                    full_ids,
                    attention_mask=full_mask,
                    word_pos=full_wp,
                    morph_depth=full_md,
                    steps=cfg.recurrent_steps,
                )["logits"][0, len(prompt) - 1].float()
                prompt_logits = model(
                    prompt_ids,
                    attention_mask=prompt_mask,
                    word_pos=prompt_wp,
                    morph_depth=prompt_md,
                    steps=cfg.recurrent_steps,
                )["logits"][0, -1].float()
                self.assertLessEqual(
                    float((full_logits - prompt_logits).abs().max()), 1e-5
                )
                generated = model.generate(
                    prompt_ids,
                    max_new_tokens=1,
                    greedy=True,
                    recurrent_steps=cfg.recurrent_steps,
                    position_contract=contract,
                )
                self.assertEqual(
                    int(generated[0, -1]), int(prompt_logits.argmax().item())
                )

    @unittest.skipUnless(
        _cuda_bf16_test_supported(),
        "CUDA BF16 parity requires a kernel compiled for this GPU architecture",
    )
    def test_bf16_teacher_forced_and_generation_top1_are_identical(self):
        torch.manual_seed(11)
        cfg = _cfg()
        model = RDTForCausalLM(cfg).cuda().eval()
        row = _row(cfg, [300, cfg.word_boundary_id, 301])
        prompt, _target, _image = split_ocr_row(row, eos_id=cfg.eos_id)
        full_ids = torch.tensor([row["input_ids"]], dtype=torch.long, device="cuda")
        prompt_ids = torch.tensor([prompt], dtype=torch.long, device="cuda")
        for contract in (BOUNDARY_V1, LEGACY_SEQUENTIAL_V0):
            with self.subTest(contract=contract), torch.no_grad(), torch.autocast(
                "cuda", dtype=torch.bfloat16
            ):
                full_mask = torch.ones_like(full_ids)
                prompt_mask = torch.ones_like(prompt_ids)
                full_wp, full_md = model._morph_info_for_position_contract(
                    full_ids, full_mask, contract
                )
                prompt_wp, prompt_md = model._morph_info_for_position_contract(
                    prompt_ids, prompt_mask, contract
                )
                full_top1 = model(
                    full_ids,
                    attention_mask=full_mask,
                    word_pos=full_wp,
                    morph_depth=full_md,
                    steps=cfg.recurrent_steps,
                )["logits"][0, len(prompt) - 1].argmax()
                prompt_top1 = model(
                    prompt_ids,
                    attention_mask=prompt_mask,
                    word_pos=prompt_wp,
                    morph_depth=prompt_md,
                    steps=cfg.recurrent_steps,
                )["logits"][0, -1].argmax()
                self.assertEqual(int(full_top1), int(prompt_top1))

    def test_checkpoint_contract_is_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "no ocr_position_contract"):
            resolve_checkpoint_ocr_position_contract({}, None)
        with self.assertRaisesRegex(ValueError, "legacy_sequential_v0"):
            resolve_checkpoint_ocr_position_contract({}, BOUNDARY_V1)
        self.assertEqual(
            resolve_checkpoint_ocr_position_contract(
                {}, LEGACY_SEQUENTIAL_V0
            ),
            LEGACY_SEQUENTIAL_V0,
        )

        metadata = {
            "ocr_position_contract": BOUNDARY_V1,
            "ocr_position_contract_version": (
                OCR_POSITION_CONTRACT_METADATA_VERSION
            ),
        }
        self.assertEqual(
            resolve_checkpoint_ocr_position_contract(metadata, None), BOUNDARY_V1
        )
        with self.assertRaisesRegex(ValueError, "mismatch"):
            resolve_checkpoint_ocr_position_contract(
                metadata, LEGACY_SEQUENTIAL_V0
            )


if __name__ == "__main__":
    unittest.main()
