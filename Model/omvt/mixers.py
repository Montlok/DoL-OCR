# -*- coding: utf-8 -*-

"""Direction-specific mixers used by OMVT."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ffn(d: int, hidden: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.LayerNorm(d),
        nn.Linear(d, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, d),
        nn.Dropout(dropout),
    )


class _DirectionalSSM(nn.Module):
    """Diagonal-state linear recurrence used for vertical / horizontal."""

    def __init__(self, d_model: int, state_dim: int = 16, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.in_proj = nn.Linear(d_model, state_dim * 2 + d_model)
        self.out_proj = nn.Linear(state_dim, d_model)
        self.A_log = nn.Parameter(torch.log(torch.linspace(0.1, 0.9, state_dim)))
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        proj = self.in_proj(x)
        B_, dt, gate = proj.split([self.state_dim, self.state_dim, self.d_model], dim=-1)
        gate = F.silu(gate)

        a = (-self.A_log.exp()).exp().to(x.dtype)
        h = x.new_zeros(x.shape[0], self.state_dim)
        outputs = []
        for t in range(x.shape[1]):
            h = h * a + B_[:, t] * F.softplus(dt[:, t])
            outputs.append(self.out_proj(h))
        out = torch.stack(outputs, dim=1)
        return self.dropout(out * gate)


class _GridMixerBase(nn.Module):
    def __init__(
        self,
        d_model: int,
        ffn_hidden: int,
        direction: str,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if direction not in {"vertical", "horizontal"}:
            raise ValueError("direction must be 'vertical' or 'horizontal'")
        self.direction = direction
        self.ssm = _DirectionalSSM(d_model, dropout=dropout)
        self.ffn = _ffn(d_model, ffn_hidden, dropout)

    def _scan(self, x: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
        if self.direction == "vertical":
            key = bbox[:, 0] * 100000 + bbox[:, 1]
        else:
            key = bbox[:, 1] * 100000 + bbox[:, 0]
        order = torch.argsort(key)
        inv = torch.argsort(order)
        x_sorted = x.index_select(1, order)
        x_sorted = self.ssm(x_sorted)
        return x_sorted.index_select(1, inv)

    def forward(self, x: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
        x = x + self._scan(x, bbox)
        x = x + self.ffn(x)
        return x


class VerticalSSM(_GridMixerBase):
    def __init__(self, d_model: int, ffn_hidden: int, dropout: float = 0.0) -> None:
        super().__init__(d_model, ffn_hidden, "vertical", dropout)


class HorizontalSSM(_GridMixerBase):
    def __init__(self, d_model: int, ffn_hidden: int, dropout: float = 0.0) -> None:
        super().__init__(d_model, ffn_hidden, "horizontal", dropout)


class LocalAttention(nn.Module):
    """Windowed self-attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_hidden: int,
        window: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window = window
        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.ffn = _ffn(d_model, ffn_hidden, dropout)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        bsz, n, d = h.shape
        qkv = self.qkv(h).reshape(bsz, n, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        idx = torch.arange(n, device=x.device)
        block = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() <= self.window
        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=block,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).reshape(bsz, n, d)
        x = x + self.out(attn)
        x = x + self.ffn(x)
        return x


class LayoutMixer(nn.Module):
    """Coarse mixer that injects bounding-box features into patch embeddings."""

    def __init__(self, d_model: int, ffn_hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.bbox_proj = nn.Linear(4, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.mixer = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, d_model),
        )

    def forward(self, x: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
        bbox_f = bbox.to(x.dtype)
        bbox_f = bbox_f / (bbox_f.max() + 1.0)
        pos = self.bbox_proj(bbox_f).unsqueeze(0)
        return x + self.mixer(self.norm(x + pos))


__all__ = ["HorizontalSSM", "LayoutMixer", "LocalAttention", "VerticalSSM"]
