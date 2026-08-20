# -*- coding: utf-8 -*-

"""Ragged, sample-isolated visual-detail cross-attention.

This bridge is intentionally a teacher-forced training primitive.  It does
not define an incremental-cache or generation contract.  Detail memory is
packed across the batch and ``cu_seqlens`` is the only authority that maps
memory rows back to samples; no query can attend across those boundaries.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RaggedVisionCrossAttention(nn.Module):
    """Residual cross-attention from RDT tokens to packed visual memory.

    Args:
        d_model: Width of the RDT residual stream.
        memory_dim: Width of one OMVT-v2 detail-memory token.
        n_heads: Number of cross-attention heads.
        dropout: Attention-probability dropout used only while training.

    ``output_projection`` is zero-initialized.  Installing the bridge is
    therefore an exact no-op until optimization updates that projection.
    """

    def __init__(
        self,
        *,
        d_model: int,
        memory_dim: int,
        n_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model <= 0 or memory_dim <= 0 or n_heads <= 0:
            raise ValueError("d_model, memory_dim, and n_heads must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.d_model = int(d_model)
        self.memory_dim = int(memory_dim)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.dropout = float(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.query_norm = nn.LayerNorm(self.d_model)
        self.memory_norm = nn.LayerNorm(self.memory_dim)
        self.query_projection = nn.Linear(self.d_model, self.d_model, bias=False)
        self.key_projection = nn.Linear(self.memory_dim, self.d_model, bias=False)
        self.value_projection = nn.Linear(self.memory_dim, self.d_model, bias=False)
        self.output_projection = nn.Linear(self.d_model, self.d_model, bias=False)
        self.zero_init_output_projection()

    @torch.no_grad()
    def zero_init_output_projection(self) -> None:
        """Make installation bit-exact with the pre-bridge residual path."""

        nn.init.zeros_(self.output_projection.weight)

    def forward(
        self,
        h: torch.Tensor,
        memory: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        query_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``h + CrossAttention(h, memory)`` without sample mixing.

        Shapes are ``h[B, T, D]``, ``memory[sumM, Dv]``, and
        ``cu_seqlens[B + 1]``.  Empty per-sample memory slices are permitted
        and contribute an exact zero residual.
        """

        batch, tokens, offsets = self._validate_inputs(
            h,
            memory,
            cu_seqlens,
            query_mask=query_mask,
        )
        query = self.query_projection(self.query_norm(h)).view(
            batch,
            tokens,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        normalized_memory = self.memory_norm(memory)
        key = self.key_projection(normalized_memory).view(
            memory.shape[0],
            self.n_heads,
            self.head_dim,
        )
        value = self.value_projection(normalized_memory).view(
            memory.shape[0],
            self.n_heads,
            self.head_dim,
        )

        # A per-sample loop is deliberate: it makes the isolation boundary
        # structural instead of relying on a large, error-prone block mask.
        contexts: list[torch.Tensor] = []
        for sample_index, (start, end) in enumerate(
            zip(offsets[:-1], offsets[1:])
        ):
            if start == end:
                contexts.append(
                    query[sample_index].new_zeros(query[sample_index].shape)
                )
                continue
            sample_key = key[start:end].transpose(0, 1)
            sample_value = value[start:end].transpose(0, 1)
            scores = torch.matmul(
                query[sample_index],
                sample_key.transpose(-2, -1),
            ) * self.scale
            probabilities = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
            if self.dropout:
                probabilities = F.dropout(
                    probabilities,
                    p=self.dropout,
                    training=self.training,
                )
            contexts.append(torch.matmul(probabilities, sample_value))

        context = torch.stack(contexts, dim=0).transpose(1, 2).reshape(
            batch,
            tokens,
            self.d_model,
        )
        residual = self.output_projection(context)
        if query_mask is not None:
            residual = residual * query_mask.to(residual.dtype).unsqueeze(-1)
        return h + residual

    def _validate_inputs(
        self,
        h: torch.Tensor,
        memory: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        query_mask: torch.Tensor | None,
    ) -> tuple[int, int, tuple[int, ...]]:
        if h.ndim != 3 or h.shape[-1] != self.d_model:
            raise ValueError(f"h must have shape [B, T, {self.d_model}]")
        if memory.ndim != 2 or memory.shape[-1] != self.memory_dim:
            raise ValueError(
                f"memory must have shape [sumM, {self.memory_dim}]"
            )
        batch, tokens, _ = h.shape
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() != batch + 1:
            raise ValueError("cu_seqlens must have shape [B + 1]")
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise TypeError("cu_seqlens must be int32 or int64")
        if h.device != memory.device:
            raise ValueError("h and memory must be on one device")
        if h.dtype != memory.dtype:
            raise ValueError("h and memory must have the same dtype")
        if cu_seqlens.device.type != "cpu":
            raise ValueError(
                "cu_seqlens must remain on CPU to avoid decode-loop DtoH sync"
            )
        offsets = tuple(int(value) for value in cu_seqlens.tolist())
        if offsets[0] != 0 or offsets[-1] != memory.shape[0]:
            raise ValueError("cu_seqlens must start at 0 and end at sumM")
        if any(right < left for left, right in zip(offsets[:-1], offsets[1:])):
            raise ValueError("cu_seqlens must be nondecreasing")
        if query_mask is not None:
            if query_mask.shape != (batch, tokens):
                raise ValueError("query_mask must have shape [B, T]")
            if query_mask.device != h.device:
                raise ValueError("query_mask must be on the same device as h")
        return batch, tokens, offsets


__all__ = ["RaggedVisionCrossAttention"]
