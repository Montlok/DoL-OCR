# -*- coding: utf-8 -*-

"""Self-supervised heads for OMVT Phase-1 pretraining."""

from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Module:
    return nn.Sequential(
        nn.LayerNorm(d_in),
        nn.Linear(d_in, d_hidden),
        nn.GELU(),
        nn.Linear(d_hidden, d_out),
    )


class OCRReconstructionHead(nn.Module):
    def __init__(self, d_vision: int, vocab_size: int, hidden: int | None = None) -> None:
        super().__init__()
        h = hidden or d_vision * 2
        self.head = _mlp(d_vision, h, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(tokens)


class MaskedPatchHead(nn.Module):
    def __init__(self, d_vision: int, patch_pixels: int, hidden: int | None = None) -> None:
        super().__init__()
        h = hidden or d_vision * 2
        self.head = _mlp(d_vision, h, patch_pixels)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(tokens)


class OrientationHead(nn.Module):
    def __init__(self, d_vision: int, hidden: int | None = None) -> None:
        super().__init__()
        h = hidden or d_vision
        self.head = _mlp(d_vision, h, 4)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(tokens.mean(dim=1))


class LayoutOrderHead(nn.Module):
    def __init__(self, d_vision: int, max_positions: int, hidden: int | None = None) -> None:
        super().__init__()
        h = hidden or d_vision
        self.head = _mlp(d_vision, h, max_positions)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(tokens)


__all__ = [
    "LayoutOrderHead",
    "MaskedPatchHead",
    "OCRReconstructionHead",
    "OrientationHead",
]
