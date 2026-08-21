# -*- coding: utf-8 -*-

"""OMVT vision tower assembly."""

from __future__ import annotations

import torch
import torch.nn as nn

from Model.omvt.compressor import PerceiverCompressor
from Model.omvt.mixers import (
    HorizontalSSM,
    LayoutMixer,
    LocalAttention,
    VerticalSSM,
)
from Model.omvt.patcher import (
    PATCH_KINDS,
    MultiScalePatcher,
    collate_omvt_batch,
    patch_pixels_for,
)
from Model.omvt.router import GeometricRouter


class _StreamEncoder(nn.Module):
    def __init__(self, kind: str, omvt_cfg) -> None:
        super().__init__()
        self.kind = kind
        d = omvt_cfg.d_vision
        ffn = omvt_cfg.vision_ffn_hidden
        drop = omvt_cfg.vision_dropout

        self.embed = nn.Linear(patch_pixels_for(kind, omvt_cfg), d)

        layers: list[nn.Module] = []
        if kind == "vertical":
            for _ in range(omvt_cfg.n_vertical_layers):
                layers.append(VerticalSSM(d, ffn, drop))
        elif kind == "horizontal":
            for _ in range(omvt_cfg.n_horizontal_layers):
                layers.append(HorizontalSSM(d, ffn, drop))
        elif kind == "square":
            for _ in range(omvt_cfg.n_local_attn_layers):
                layers.append(LocalAttention(d, omvt_cfg.vision_n_heads, ffn, dropout=drop))
        elif kind == "layout":
            for _ in range(omvt_cfg.n_layout_layers):
                layers.append(LayoutMixer(d, ffn, drop))
        else:
            raise ValueError(f"unknown patch kind: {kind}")

        self.layers = nn.ModuleList(layers)

    def forward(self, patches: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
        x = self.embed(patches)
        for layer in self.layers:
            x = layer(x, bbox)
        return x


class OMVTVisionTower(nn.Module):
    def __init__(self, omvt_cfg) -> None:
        super().__init__()
        self.cfg = omvt_cfg
        self.patcher = MultiScalePatcher(omvt_cfg)
        self.router = GeometricRouter(omvt_cfg)
        self.encoders = nn.ModuleDict(
            {kind: _StreamEncoder(kind, omvt_cfg) for kind in PATCH_KINDS}
        )
        self.fuse_norm = nn.LayerNorm(omvt_cfg.d_vision)
        self.compressor = PerceiverCompressor(omvt_cfg)

    def forward(
        self,
        inputs: torch.Tensor | dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if isinstance(inputs, torch.Tensor):
            batch = dict(collate_omvt_batch(inputs, self.cfg))
        else:
            batch = dict(inputs)

        images = batch["images"]
        weights = self.router(images)

        scaled_streams: list[torch.Tensor] = []
        encoded: dict[str, torch.Tensor] = {}
        for idx, kind in enumerate(PATCH_KINDS):
            patches = batch[f"{kind}_patches"]
            bbox = batch[f"{kind}_bbox"]
            feats = self.encoders[kind](patches, bbox)
            w = weights[:, idx].view(-1, 1, 1)
            scaled_streams.append(feats * w)
            encoded[kind] = feats

        fused = torch.cat(scaled_streams, dim=1)
        fused = self.fuse_norm(fused)
        compressed = self.compressor(fused)

        return {
            "compressed": compressed,
            "fused": fused,
            "router_weights": weights,
            "streams": encoded,
        }


__all__ = ["OMVTVisionTower"]
