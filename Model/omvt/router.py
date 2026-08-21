# -*- coding: utf-8 -*-

"""Geometric (LM-free) router for OMVT streams."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_gray(images: torch.Tensor) -> torch.Tensor:
    if images.shape[1] == 1:
        return images.squeeze(1)
    weights = images.new_tensor([0.2989, 0.5870, 0.1140])
    return torch.einsum("bchw,c->bhw", images, weights)


def _sobel_directions(gray: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    kx = gray.new_tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]])
    ky = gray.new_tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]])
    gx = F.conv2d(gray.unsqueeze(1), kx, padding=1).squeeze(1)
    gy = F.conv2d(gray.unsqueeze(1), ky, padding=1).squeeze(1)
    return gx.abs(), gy.abs()


class GeometricRouter(nn.Module):
    def __init__(self, omvt_cfg) -> None:
        super().__init__()
        self.cfg = omvt_cfg
        self.bias = nn.Parameter(torch.zeros(4))
        self.log_temperature = nn.Parameter(torch.zeros(()))
        self.min_prob = float(omvt_cfg.router_min_route_prob)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError("images must have shape [B, C, H, W]")

        gray = _to_gray(images)
        gx, gy = _sobel_directions(gray)

        h_density = gx.mean(dim=(1, 2))
        v_density = gy.mean(dim=(1, 2))
        total = h_density + v_density + 1e-6

        vertical = v_density / total
        horizontal = h_density / total
        square = 1.0 - (vertical - horizontal).abs()
        layout = 1.0 / (1.0 + total)

        logits = torch.stack([vertical, horizontal, square, layout], dim=-1)
        logits = logits + self.bias
        logits = logits / (
            self.log_temperature.exp() * self.cfg.router_temperature + 1e-4
        )

        probs = F.softmax(logits, dim=-1)
        if self.min_prob > 0:
            probs = probs + self.min_prob
            probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs


__all__ = ["GeometricRouter"]
