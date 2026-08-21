# -*- coding: utf-8 -*-

"""Project OMVT compressed tokens into RDT embedding space."""

from __future__ import annotations

import torch
import torch.nn as nn

from Model.omvt.tower import OMVTVisionTower


class OMVTInjector(nn.Module):
    def __init__(self, rdt_cfg, omvt_cfg) -> None:
        super().__init__()
        self.rdt_cfg = rdt_cfg
        self.omvt_cfg = omvt_cfg
        self.tower = OMVTVisionTower(omvt_cfg)

        if omvt_cfg.projector_hidden and omvt_cfg.projector_hidden > 0:
            self.projector = nn.Sequential(
                nn.LayerNorm(omvt_cfg.d_vision),
                nn.Linear(omvt_cfg.d_vision, omvt_cfg.projector_hidden),
                nn.GELU(),
                nn.Linear(omvt_cfg.projector_hidden, rdt_cfg.d_model),
            )
        else:
            self.projector = nn.Sequential(
                nn.LayerNorm(omvt_cfg.d_vision),
                nn.Linear(omvt_cfg.d_vision, rdt_cfg.d_model),
            )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        pixel_repeats: int = 1,
    ) -> torch.Tensor:
        if type(pixel_repeats) is not int or pixel_repeats <= 0:
            raise ValueError("pixel_repeats must be a positive integer")
        tower_out = self.tower(batch)
        projected = self.projector(tower_out["compressed"])
        if pixel_repeats != 1:
            projected = projected.repeat_interleave(pixel_repeats, dim=0)
        return projected


__all__ = ["OMVTInjector"]
