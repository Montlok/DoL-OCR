# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from Model.blocks import RecurrentBlock


class RecurrentCore(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.block = RecurrentBlock(cfg)
        self.inject = cfg.inject_embedding
        self.inject_scale = cfg.inject_scale
        self.grad_ckpt = bool(getattr(cfg, "grad_ckpt_recurrent", False))

        if self.inject_scale < 0:
            raise ValueError("inject_scale must be non-negative")
        if cfg.recurrent_steps <= 0:
            raise ValueError("recurrent_steps must be positive")

    def forward(
        self,
        e0: torch.Tensor,
        word_pos: torch.Tensor | None = None,
        morph_depth: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        causal: bool = True,
        steps: int | None = None,
        bptt_window: int | None = None,
    ) -> tuple[torch.Tensor, dict]:
        self._check_inputs(e0, word_pos, morph_depth, attn_mask)

        return self._forward_fixed(
            e0=e0,
            word_pos=word_pos,
            morph_depth=morph_depth,
            attn_mask=attn_mask,
            causal=causal,
            steps=steps,
            bptt_window=bptt_window,
        )

    def _forward_fixed(
        self,
        e0: torch.Tensor,
        word_pos: torch.Tensor | None,
        morph_depth: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
        causal: bool,
        steps: int | None,
        bptt_window: int | None,
    ) -> tuple[torch.Tensor, dict]:
        total_steps = int(steps if steps is not None else self.cfg.recurrent_steps)

        if total_steps <= 0:
            raise ValueError("steps must be positive")

        if bptt_window is not None:
            if bptt_window <= 0:
                raise ValueError("bptt_window must be positive")
            bptt_window = min(bptt_window, total_steps)

        h = e0

        for idx in range(total_steps):
            if bptt_window is not None and idx < total_steps - bptt_window:
                h = h.detach()

            if self.inject:
                h = h + self.inject_scale * e0

            h = self._run_block(
                h,
                word_pos=word_pos,
                morph_depth=morph_depth,
                attn_mask=attn_mask,
                causal=causal,
            )

        return h, {
            "steps_used": total_steps,
        }

    def _run_block(
        self,
        h: torch.Tensor,
        word_pos: torch.Tensor | None,
        morph_depth: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
        causal: bool,
    ) -> torch.Tensor:
        if self.grad_ckpt and self.training and h.requires_grad:
            def _fn(h_in):
                return self.block(
                    h_in,
                    word_pos=word_pos,
                    morph_depth=morph_depth,
                    attn_mask=attn_mask,
                    causal=causal,
                )

            return checkpoint(_fn, h, use_reentrant=False)
        return self.block(
            h,
            word_pos=word_pos,
            morph_depth=morph_depth,
            attn_mask=attn_mask,
            causal=causal,
        )

    def _check_inputs(
        self,
        e0: torch.Tensor,
        word_pos: torch.Tensor | None,
        morph_depth: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
    ) -> None:
        if e0.ndim != 3:
            raise ValueError("e0 must have shape [B, L, d_model]")

        bsz, seq_len, dim = e0.shape

        if dim != self.cfg.d_model:
            raise ValueError(f"expected d_model={self.cfg.d_model}, got {dim}")

        if word_pos is not None and word_pos.shape != (bsz, seq_len):
            raise ValueError("word_pos must have shape [B, L]")

        if morph_depth is not None and morph_depth.shape != (bsz, seq_len):
            raise ValueError("morph_depth must have shape [B, L]")

        if attn_mask is not None and attn_mask.shape != (bsz, seq_len):
            raise ValueError("attn_mask must have shape [B, L]")


def _grad_norm(x: torch.Tensor) -> float:
    if x.grad is None:
        return 0.0
    return x.grad.norm().item()


def _check() -> None:
    from Model.config import tiny_config

    torch.manual_seed(0)

    cfg = tiny_config()
    core = RecurrentCore(cfg)
    core.eval()

    bsz, seq_len = 2, 16
    e0 = torch.randn(bsz, seq_len, cfg.d_model)
    word_pos = torch.arange(seq_len).unsqueeze(0).expand(bsz, seq_len)
    morph_depth = torch.zeros(bsz, seq_len, dtype=torch.long)

    h, info = core(e0, word_pos=word_pos, morph_depth=morph_depth)

    print("RecurrentCore")
    print(f"  fixed_steps: {cfg.recurrent_steps}")
    print(f"  shape: {tuple(e0.shape)} -> {tuple(h.shape)}")
    print(f"  steps_used: {info['steps_used']}")

    e0g = torch.randn(bsz, seq_len, cfg.d_model, requires_grad=True)
    h, _info = core(e0g, word_pos=word_pos, morph_depth=morph_depth)
    h.sum().backward()

    print(f"  grad_norm: {_grad_norm(e0g):.6f}")

    for steps in [2, 4, 8, 16]:
        with torch.no_grad():
            h, _info = core(
                e0,
                word_pos=word_pos,
                morph_depth=morph_depth,
                steps=steps,
            )
        print(f"  steps={steps}, out_norm={h.norm().item():.6f}")

    e0b = torch.randn(bsz, seq_len, cfg.d_model, requires_grad=True)
    h, _info = core(
        e0b,
        word_pos=word_pos,
        morph_depth=morph_depth,
        bptt_window=2,
    )
    h.sum().backward()

    print(f"  bptt_window=2, grad_norm={_grad_norm(e0b):.6f}")

if __name__ == "__main__":
    _check()
