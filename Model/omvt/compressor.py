# -*- coding: utf-8 -*-

"""Perceiver-style compressor for OMVT."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _CrossAttn(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, d_model * 2)
        self.out = nn.Linear(d_model, d_model)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

    def forward(self, latents: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        bsz, n_lat, d = latents.shape
        q = self.q_proj(self.norm_q(latents))
        kv = self.kv_proj(self.norm_kv(ctx))
        k, v = kv.chunk(2, dim=-1)

        q = q.reshape(bsz, n_lat, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, ctx.shape[1], self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, ctx.shape[1], self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(bsz, n_lat, d)
        return self.out(attn)


class PerceiverCompressor(nn.Module):
    def __init__(self, omvt_cfg) -> None:
        super().__init__()
        self.cfg = omvt_cfg
        self.latents = nn.Parameter(
            torch.randn(omvt_cfg.compress_to, omvt_cfg.d_vision) * 0.02
        )
        self.blocks = nn.ModuleList()
        for _ in range(omvt_cfg.compressor_layers):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "cross": _CrossAttn(omvt_cfg.d_vision, omvt_cfg.compressor_heads),
                        "ffn": nn.Sequential(
                            nn.LayerNorm(omvt_cfg.d_vision),
                            nn.Linear(omvt_cfg.d_vision, omvt_cfg.vision_ffn_hidden),
                            nn.GELU(),
                            nn.Linear(omvt_cfg.vision_ffn_hidden, omvt_cfg.d_vision),
                        ),
                    }
                )
            )
        self.final_norm = nn.LayerNorm(omvt_cfg.d_vision)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        bsz = patches.shape[0]
        latents = self.latents.unsqueeze(0).expand(bsz, -1, -1)
        for block in self.blocks:
            latents = latents + block["cross"](latents, patches)
            latents = latents + block["ffn"](latents)
        return self.final_norm(latents)


__all__ = ["PerceiverCompressor"]
