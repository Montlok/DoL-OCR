# -*- coding: utf-8 -*-

"""Ragged, sample-isolated detail-memory compression for OMVT-v2."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PackedDetailMemory:
    """Packed visual memory emitted for a later ragged cross-attention bridge."""

    memory: torch.Tensor
    cu_seqlens: torch.Tensor
    sample_ids: torch.Tensor
    token_counts: torch.Tensor
    source_token_counts: torch.Tensor


def _validate_packed_features(
    features: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError("packed features must have shape [N, D]")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must have shape [batch + 1]")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("cu_seqlens must use an integer dtype")
    if cu_seqlens.device.type != "cpu":
        raise ValueError("cu_seqlens must remain on CPU")
    if int(cu_seqlens[0].item()) != 0:
        raise ValueError("cu_seqlens must start at zero")
    if int(cu_seqlens[-1].item()) != int(features.shape[0]):
        raise ValueError("cu_seqlens must end at the packed feature count")
    if bool((cu_seqlens[1:] < cu_seqlens[:-1]).any()):
        raise ValueError("cu_seqlens must be non-decreasing")
    return (cu_seqlens[1:] - cu_seqlens[:-1]).to(dtype=torch.long)


class _SampleCrossAttention(nn.Module):
    """One-sample cross attention; callers own ragged segmentation."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = int(n_heads)
        self.head_dim = d_model // n_heads
        self.norm_query = nn.LayerNorm(d_model)
        self.norm_context = nn.LayerNorm(d_model)
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_value_proj = nn.Linear(d_model, d_model * 2)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if query.ndim != 2 or context.ndim != 2:
            raise ValueError("query and context must have shape [tokens, D]")
        if context.shape[0] == 0:
            raise ValueError("detail compression requires non-empty sample context")
        n_query, d_model = query.shape
        projected_query = self.query_proj(self.norm_query(query))
        key, value = self.key_value_proj(self.norm_context(context)).chunk(2, dim=-1)
        projected_query = projected_query.reshape(
            n_query, self.n_heads, self.head_dim
        ).transpose(0, 1).unsqueeze(0)
        key = key.reshape(-1, self.n_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
        value = value.reshape(-1, self.n_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
        attended = F.scaled_dot_product_attention(projected_query, key, value)
        attended = attended.squeeze(0).transpose(0, 1).reshape(n_query, d_model)
        return self.out(attended)


class NativeDetailCompressor(nn.Module):
    """Compress each sample to a monotonic, hard-capped ragged token budget.

    The current implementation is deliberately a correctness kernel: it loops
    over samples and never pads them into a shared dense context.  A later
    optimized kernel must preserve this exact isolation contract.
    """

    def __init__(
        self,
        omvt_cfg,
        *,
        max_detail_tokens_per_sample: int | None = None,
        source_tokens_per_detail_token: int = 4,
    ) -> None:
        super().__init__()
        cap = (
            int(omvt_cfg.compress_to)
            if max_detail_tokens_per_sample is None
            else int(max_detail_tokens_per_sample)
        )
        if cap <= 0:
            raise ValueError("max_detail_tokens_per_sample must be positive")
        if source_tokens_per_detail_token <= 0:
            raise ValueError("source_tokens_per_detail_token must be positive")
        self.max_detail_tokens_per_sample = cap
        self.source_tokens_per_detail_token = int(source_tokens_per_detail_token)
        self.latents = nn.Parameter(
            torch.randn(cap, omvt_cfg.d_vision) * 0.02
        )
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "cross": _SampleCrossAttention(
                            omvt_cfg.d_vision,
                            omvt_cfg.compressor_heads,
                        ),
                        "ffn": nn.Sequential(
                            nn.LayerNorm(omvt_cfg.d_vision),
                            nn.Linear(
                                omvt_cfg.d_vision,
                                omvt_cfg.vision_ffn_hidden,
                            ),
                            nn.GELU(),
                            nn.Linear(
                                omvt_cfg.vision_ffn_hidden,
                                omvt_cfg.d_vision,
                            ),
                        ),
                    }
                )
                for _ in range(omvt_cfg.compressor_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(omvt_cfg.d_vision)

    def detail_token_budget(self, source_token_count: int) -> int:
        count = int(source_token_count)
        if count < 0:
            raise ValueError("source_token_count must be non-negative")
        if count == 0:
            return 0
        requested = (
            count + self.source_tokens_per_detail_token - 1
        ) // self.source_tokens_per_detail_token
        return min(self.max_detail_tokens_per_sample, requested)

    def _one_sample(self, context: torch.Tensor, output_tokens: int) -> torch.Tensor:
        if output_tokens <= 0 or output_tokens > self.max_detail_tokens_per_sample:
            raise ValueError("output token count violates the detail-memory cap")
        latents = self.latents[:output_tokens]
        for block in self.blocks:
            latents = latents + block["cross"](latents, context)
            latents = latents + block["ffn"](latents)
        return self.final_norm(latents)

    def forward(
        self,
        features: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> PackedDetailMemory:
        source_counts = _validate_packed_features(features, cu_seqlens)
        output_counts = torch.tensor(
            [self.detail_token_budget(int(count.item())) for count in source_counts],
            dtype=torch.long,
            device="cpu",
        )
        if bool((output_counts > self.max_detail_tokens_per_sample).any()):
            raise RuntimeError("detail-memory token budget exceeded its hard cap")
        if bool((source_counts == 0).any()):
            raise ValueError("native detail compression does not accept empty samples")

        memories: list[torch.Tensor] = []
        sample_id_rows: list[torch.Tensor] = []
        output_cumulative = [0]
        for sample_index in range(int(source_counts.numel())):
            start = int(cu_seqlens[sample_index].item())
            end = int(cu_seqlens[sample_index + 1].item())
            output_count = int(output_counts[sample_index].item())
            memories.append(self._one_sample(features[start:end], output_count))
            sample_id_rows.append(
                torch.full(
                    (output_count,),
                    sample_index,
                    dtype=torch.long,
                    device=features.device,
                )
            )
            output_cumulative.append(output_cumulative[-1] + output_count)

        memory = torch.cat(memories, dim=0)
        packed = PackedDetailMemory(
            memory=memory,
            cu_seqlens=torch.tensor(
                output_cumulative,
                dtype=torch.int32,
                device="cpu",
            ),
            sample_ids=torch.cat(sample_id_rows, dim=0),
            token_counts=output_counts,
            source_token_counts=source_counts,
        )
        if int(packed.cu_seqlens[-1].item()) != int(memory.shape[0]):
            raise RuntimeError("packed detail-memory offsets are inconsistent")
        return packed


__all__ = ["NativeDetailCompressor", "PackedDetailMemory"]
