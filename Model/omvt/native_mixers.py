# -*- coding: utf-8 -*-

"""Sample-isolated mixers for the packed native-resolution OMVT-v2 path.

These modules intentionally favour a small, explicit per-sample/per-window
implementation over pretending that the packed path already has a fused
kernel.  In particular, no module below constructs a batch-global ``N x N``
attention mask.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.omvt.native_patcher import PackedPatchStream


def _ffn(d_model: int, hidden: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, d_model),
        nn.Dropout(dropout),
    )


def _segment_bounds(cu_seqlens: torch.Tensor) -> Iterator[tuple[int, int, int]]:
    for sample_index in range(int(cu_seqlens.numel()) - 1):
        start = int(cu_seqlens[sample_index].item())
        end = int(cu_seqlens[sample_index + 1].item())
        yield sample_index, start, end


def validate_packed_patch_stream(
    stream: PackedPatchStream,
    *,
    expected_tokens: int | None = None,
) -> int:
    """Validate the packed geometry contract and return the sample count."""

    token_count = int(stream.patches.shape[0])
    if expected_tokens is not None and token_count != int(expected_tokens):
        raise ValueError("packed stream token count does not match its features")
    if stream.patches.ndim != 2:
        raise ValueError("packed patches must have shape [N, patch_pixels]")
    expected_leading_shapes = {
        "bbox_px_yxxy": (token_count, 4),
        "bbox_norm_yxxy": (token_count, 4),
        "valid_fraction": (token_count,),
        "sample_ids": (token_count,),
        "grid_yx": (token_count, 2),
    }
    for name, expected in expected_leading_shapes.items():
        if tuple(getattr(stream, name).shape) != expected:
            raise ValueError(f"{name} must have shape {expected}")
    if stream.cu_seqlens.ndim != 1 or stream.cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must have shape [batch + 1]")
    if stream.cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("cu_seqlens must use an integer dtype")
    if int(stream.cu_seqlens[0].item()) != 0:
        raise ValueError("cu_seqlens must start at zero")
    if int(stream.cu_seqlens[-1].item()) != token_count:
        raise ValueError("cu_seqlens must end at the packed token count")
    if bool((stream.cu_seqlens[1:] < stream.cu_seqlens[:-1]).any()):
        raise ValueError("cu_seqlens must be non-decreasing")
    if bool((stream.valid_fraction <= 0).any()) or bool(
        (stream.valid_fraction > 1).any()
    ):
        raise ValueError("valid_fraction must be in (0, 1]")
    if bool((stream.bbox_norm_yxxy < 0).any()) or bool(
        (stream.bbox_norm_yxxy > 1).any()
    ):
        raise ValueError("bbox_norm_yxxy must be normalized per sample")
    if bool(
        (stream.bbox_norm_yxxy[:, 2:] < stream.bbox_norm_yxxy[:, :2]).any()
    ):
        raise ValueError("normalized patch boxes must have non-negative extent")

    batch_size = int(stream.cu_seqlens.numel()) - 1
    for sample_index, start, end in _segment_bounds(stream.cu_seqlens):
        expected_ids = torch.full(
            (end - start,),
            sample_index,
            dtype=stream.sample_ids.dtype,
            device=stream.sample_ids.device,
        )
        if not torch.equal(stream.sample_ids[start:end], expected_ids):
            raise ValueError("sample_ids must agree with contiguous cu_seqlens segments")
    return batch_size


def stable_lexicographic_order(
    grid_yx: torch.Tensor,
    *,
    direction: str,
) -> torch.Tensor:
    """Return a deterministic stable 2-D order without scalar key packing."""

    if grid_yx.ndim != 2 or grid_yx.shape[1] != 2:
        raise ValueError("grid_yx must have shape [N, 2]")
    if direction == "vertical":
        primary, secondary = 0, 1
    elif direction == "horizontal":
        primary, secondary = 1, 0
    else:
        raise ValueError("direction must be 'vertical' or 'horizontal'")

    order = torch.arange(grid_yx.shape[0], device=grid_yx.device)
    secondary_order = torch.argsort(
        grid_yx.index_select(0, order)[:, secondary], stable=True
    )
    order = order.index_select(0, secondary_order)
    primary_order = torch.argsort(
        grid_yx.index_select(0, order)[:, primary], stable=True
    )
    return order.index_select(0, primary_order)


class _PackedDirectionalRecurrence(nn.Module):
    """Diagonal recurrence whose hidden state is reset at every sample."""

    def __init__(
        self,
        d_model: int,
        *,
        direction: str,
        state_dim: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if direction not in {"vertical", "horizontal"}:
            raise ValueError("direction must be 'vertical' or 'horizontal'")
        self.direction = direction
        self.state_dim = int(state_dim)
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, state_dim * 2 + d_model)
        self.out_proj = nn.Linear(state_dim, d_model)
        self.A_log = nn.Parameter(torch.log(torch.linspace(0.1, 0.9, state_dim)))
        self.dropout = nn.Dropout(dropout)

    def _one_sample(
        self,
        x: torch.Tensor,
        grid_yx: torch.Tensor,
    ) -> torch.Tensor:
        if x.shape[0] == 0:
            return x
        order = stable_lexicographic_order(grid_yx, direction=self.direction)
        inverse = torch.argsort(order, stable=True)
        sorted_x = x.index_select(0, order)
        projected = self.in_proj(self.norm(sorted_x))
        drive, dt, gate = projected.split(
            [self.state_dim, self.state_dim, x.shape[-1]], dim=-1
        )
        gate = F.silu(gate)
        decay = (-self.A_log.exp()).exp().to(dtype=x.dtype, device=x.device)
        state = x.new_zeros((self.state_dim,))
        outputs: list[torch.Tensor] = []
        for token_index in range(sorted_x.shape[0]):
            state = state * decay + drive[token_index] * F.softplus(dt[token_index])
            outputs.append(self.out_proj(state) * gate[token_index])
        scanned = torch.stack(outputs, dim=0)
        return self.dropout(scanned).index_select(0, inverse)

    def forward(self, x: torch.Tensor, stream: PackedPatchStream) -> torch.Tensor:
        validate_packed_patch_stream(stream, expected_tokens=x.shape[0])
        sample_outputs = [
            self._one_sample(x[start:end], stream.grid_yx[start:end])
            for _, start, end in _segment_bounds(stream.cu_seqlens)
        ]
        return torch.cat(sample_outputs, dim=0)


class NativeDirectionalMixer(nn.Module):
    def __init__(
        self,
        d_model: int,
        ffn_hidden: int,
        *,
        direction: str,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ssm = _PackedDirectionalRecurrence(
            d_model,
            direction=direction,
            dropout=dropout,
        )
        self.ffn = _ffn(d_model, ffn_hidden, dropout)

    def forward(self, x: torch.Tensor, stream: PackedPatchStream) -> torch.Tensor:
        x = x + self.ssm(x, stream)
        return x + self.ffn(x)


class NativeVerticalSSM(NativeDirectionalMixer):
    def __init__(self, d_model: int, ffn_hidden: int, dropout: float = 0.0) -> None:
        super().__init__(
            d_model,
            ffn_hidden,
            direction="vertical",
            dropout=dropout,
        )


class NativeHorizontalSSM(NativeDirectionalMixer):
    def __init__(self, d_model: int, ffn_hidden: int, dropout: float = 0.0) -> None:
        super().__init__(
            d_model,
            ffn_hidden,
            direction="horizontal",
            dropout=dropout,
        )


class NativeSquareWindowAttention(nn.Module):
    """Self-attention in real 8x8 patch-grid windows, one sample at a time."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_hidden: int,
        *,
        window_yx: tuple[int, int] = (8, 8),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if window_yx[0] <= 0 or window_yx[1] <= 0:
            raise ValueError("window_yx entries must be positive")
        self.n_heads = int(n_heads)
        self.head_dim = d_model // n_heads
        self.window_yx = (int(window_yx[0]), int(window_yx[1]))
        self.max_window_tokens = self.window_yx[0] * self.window_yx[1]
        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.ffn = _ffn(d_model, ffn_hidden, dropout)
        self.dropout = float(dropout)

    def _attend_group(self, x: torch.Tensor) -> torch.Tensor:
        token_count, d_model = x.shape
        if token_count > self.max_window_tokens:
            raise ValueError("a native square window contains duplicate grid cells")
        qkv = self.qkv(self.norm(x)).reshape(
            token_count, 3, self.n_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=1)
        query = query.transpose(0, 1).unsqueeze(0)
        key = key.transpose(0, 1).unsqueeze(0)
        value = value.transpose(0, 1).unsqueeze(0)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return attended.squeeze(0).transpose(0, 1).reshape(token_count, d_model)

    def _one_sample(self, x: torch.Tensor, grid_yx: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            return x
        window_scale = grid_yx.new_tensor(self.window_yx)
        window_ids = torch.div(grid_yx, window_scale, rounding_mode="floor")
        window_keys = sorted(
            {tuple(int(value) for value in row) for row in window_ids.detach().cpu().tolist()}
        )
        group_indices: list[torch.Tensor] = []
        group_outputs: list[torch.Tensor] = []
        for window_y, window_x in window_keys:
            in_window = (window_ids[:, 0] == window_y) & (
                window_ids[:, 1] == window_x
            )
            indices = torch.nonzero(in_window, as_tuple=False).flatten()
            local_order = stable_lexicographic_order(
                grid_yx.index_select(0, indices), direction="vertical"
            )
            indices = indices.index_select(0, local_order)
            group_indices.append(indices)
            group_outputs.append(self._attend_group(x.index_select(0, indices)))

        packed_indices = torch.cat(group_indices, dim=0)
        packed_outputs = torch.cat(group_outputs, dim=0)
        restore = torch.argsort(packed_indices, stable=True)
        return packed_outputs.index_select(0, restore)

    def forward(self, x: torch.Tensor, stream: PackedPatchStream) -> torch.Tensor:
        validate_packed_patch_stream(stream, expected_tokens=x.shape[0])
        attended = torch.cat(
            [
                self._one_sample(x[start:end], stream.grid_yx[start:end])
                for _, start, end in _segment_bounds(stream.cu_seqlens)
            ],
            dim=0,
        )
        x = x + self.out(attended)
        return x + self.ffn(x)


class NativeLayoutMixer(nn.Module):
    """Inject per-sample normalized boxes and a sample-local layout summary."""

    def __init__(self, d_model: int, ffn_hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.bbox_proj = nn.Linear(4, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.context_proj = nn.Linear(d_model, d_model)
        self.mixer = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, stream: PackedPatchStream) -> torch.Tensor:
        validate_packed_patch_stream(stream, expected_tokens=x.shape[0])
        positioned = self.norm(
            x + self.bbox_proj(stream.bbox_norm_yxxy.to(dtype=x.dtype))
        )
        outputs: list[torch.Tensor] = []
        for _, start, end in _segment_bounds(stream.cu_seqlens):
            sample = positioned[start:end]
            weights = stream.valid_fraction[start:end].to(dtype=x.dtype).unsqueeze(-1)
            denominator = weights.sum().clamp_min(torch.finfo(x.dtype).eps)
            summary = (sample * weights).sum(dim=0, keepdim=True) / denominator
            outputs.append(x[start:end] + self.mixer(sample + self.context_proj(summary)))
        return torch.cat(outputs, dim=0)


__all__ = [
    "NativeHorizontalSSM",
    "NativeLayoutMixer",
    "NativeSquareWindowAttention",
    "NativeVerticalSSM",
    "stable_lexicographic_order",
    "validate_packed_patch_stream",
]
