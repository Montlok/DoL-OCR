# -*- coding: utf-8 -*-

"""Loss helpers for OMVT SSL + joint VLM training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class OMVTSSLOutputs:
    total: torch.Tensor
    ocr: torch.Tensor
    masked_patch: torch.Tensor
    orientation: torch.Tensor
    layout_order: torch.Tensor


def ocr_reconstruction_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_ids.reshape(-1),
        ignore_index=ignore_index,
    )


def masked_patch_loss(
    predicted_pixels: torch.Tensor,
    target_pixels: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    diff = (predicted_pixels - target_pixels) ** 2
    if mask is None:
        return diff.mean()
    m = mask.to(diff.dtype).unsqueeze(-1)
    return (diff * m).sum() / (m.sum() * diff.shape[-1] + 1e-6)


def orientation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target)


def layout_order_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target.reshape(-1),
        ignore_index=ignore_index,
    )


def patch_text_contrastive_loss(
    visual: torch.Tensor,
    text: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    v = F.normalize(visual, dim=-1)
    t = F.normalize(text, dim=-1)
    logits = v @ t.t() / temperature
    target = torch.arange(v.shape[0], device=v.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))


__all__ = [
    "OMVTSSLOutputs",
    "layout_order_loss",
    "masked_patch_loss",
    "ocr_reconstruction_loss",
    "orientation_loss",
    "patch_text_contrastive_loss",
]
