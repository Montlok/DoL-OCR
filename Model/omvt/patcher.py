# -*- coding: utf-8 -*-

"""Multi-scale patching for OMVT."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

PATCH_KINDS: tuple[str, ...] = ("vertical", "horizontal", "square", "layout")


def _unfold_patches(
    images: torch.Tensor,
    patch_h: int,
    patch_w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if images.ndim != 4:
        raise ValueError("images must have shape [B, C, H, W]")

    bsz, ch, h, w = images.shape
    if patch_h <= 0 or patch_w <= 0:
        raise ValueError("patch dimensions must be positive")

    pad_h = (-h) % patch_h
    pad_w = (-w) % patch_w
    if pad_h or pad_w:
        images = F.pad(images, (0, pad_w, 0, pad_h))
    H = h + pad_h
    W = w + pad_w

    nh = H // patch_h
    nw = W // patch_w

    tiles = images.reshape(bsz, ch, nh, patch_h, nw, patch_w)
    tiles = tiles.permute(0, 2, 4, 1, 3, 5).contiguous()
    patches = tiles.reshape(bsz, nh * nw, ch * patch_h * patch_w)

    ys = torch.arange(nh, device=images.device) * patch_h
    xs = torch.arange(nw, device=images.device) * patch_w
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    bboxes = torch.stack(
        [
            grid_y.reshape(-1),
            grid_x.reshape(-1),
            torch.full_like(grid_y.reshape(-1), patch_h),
            torch.full_like(grid_x.reshape(-1), patch_w),
        ],
        dim=-1,
    ).to(torch.long)
    return patches, bboxes


class MultiScalePatcher(nn.Module):
    def __init__(self, omvt_cfg) -> None:
        super().__init__()
        self.cfg = omvt_cfg

    def patch_shapes(self) -> dict[str, tuple[int, int]]:
        return {
            "vertical": tuple(self.cfg.vertical_patch),
            "horizontal": tuple(self.cfg.horizontal_patch),
            "square": tuple(self.cfg.square_patch),
            "layout": tuple(self.cfg.layout_patch),
        }

    def forward(self, images: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        out: dict[str, dict[str, torch.Tensor]] = {}
        for kind, (ph, pw) in self.patch_shapes().items():
            patches, bboxes = _unfold_patches(images, ph, pw)
            out[kind] = {"patches": patches, "bbox": bboxes}
        return out


def patch_pixels_for(kind: str, omvt_cfg) -> int:
    ph, pw = MultiScalePatcher(omvt_cfg).patch_shapes()[kind]
    return omvt_cfg.in_channels * ph * pw


def collate_omvt_batch(
    images: torch.Tensor,
    omvt_cfg,
) -> Mapping[str, torch.Tensor]:
    patcher = MultiScalePatcher(omvt_cfg)
    streams = patcher(images)
    flat: dict[str, torch.Tensor] = {"images": images}
    for kind, item in streams.items():
        flat[f"{kind}_patches"] = item["patches"]
        flat[f"{kind}_bbox"] = item["bbox"]
    return flat


__all__ = [
    "MultiScalePatcher",
    "PATCH_KINDS",
    "collate_omvt_batch",
    "patch_pixels_for",
]
