# -*- coding: utf-8 -*-

"""Packed per-sample patch geometry for native-resolution OMVT-v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from Model.omvt.patcher import PATCH_KINDS, MultiScalePatcher


@dataclass(frozen=True)
class PackedPatchStream:
    patches: torch.Tensor
    bbox_px_yxxy: torch.Tensor
    bbox_norm_yxxy: torch.Tensor
    valid_fraction: torch.Tensor
    sample_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    grid_yx: torch.Tensor


@dataclass(frozen=True)
class PackedNativeOMVTBatch:
    streams: Mapping[str, PackedPatchStream]
    original_hw: torch.Tensor
    raw_patch_tokens: torch.Tensor


def _packed_stream(
    samples: Sequence[Any],
    *,
    patch_h: int,
    patch_w: int,
    pad_value: torch.Tensor,
) -> PackedPatchStream:
    patch_rows: list[torch.Tensor] = []
    bbox_rows: list[torch.Tensor] = []
    bbox_norm_rows: list[torch.Tensor] = []
    fraction_rows: list[torch.Tensor] = []
    sample_rows: list[torch.Tensor] = []
    grid_rows: list[torch.Tensor] = []
    cumulative = [0]

    for sample_index, sample in enumerate(samples):
        pixels = sample.pixels
        pixel_mask = sample.pixel_valid_mask
        if pixels.ndim != 3:
            raise ValueError("native pixels must have shape [C,H,W]")
        channels, height, width = pixels.shape
        if tuple(pixel_mask.shape) != (height, width):
            raise ValueError("pixel_valid_mask must match native image H/W")
        if tuple(sample.original_hw) != (height, width):
            raise ValueError("original_hw must match the decoded native tensor")
        if pad_value.numel() != channels:
            raise ValueError("normalized_white must have one value per channel")

        rows = (height + patch_h - 1) // patch_h
        cols = (width + patch_w - 1) // patch_w
        padded_h = rows * patch_h
        padded_w = cols * patch_w
        canvas = pad_value.to(device=pixels.device, dtype=pixels.dtype).view(
            channels, 1, 1
        ).expand(channels, padded_h, padded_w).clone()
        canvas[:, :height, :width] = pixels
        valid = torch.zeros((padded_h, padded_w), dtype=torch.bool, device=pixels.device)
        valid[:height, :width] = pixel_mask.to(device=pixels.device, dtype=torch.bool)

        patches = canvas.reshape(
            channels, rows, patch_h, cols, patch_w
        ).permute(1, 3, 0, 2, 4).reshape(
            rows * cols, channels * patch_h * patch_w
        )
        fractions = valid.reshape(rows, patch_h, cols, patch_w).permute(
            0, 2, 1, 3
        ).reshape(rows * cols, patch_h * patch_w).float().mean(dim=1)

        ys = torch.arange(rows, device=pixels.device, dtype=torch.long)
        xs = torch.arange(cols, device=pixels.device, dtype=torch.long)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        y0 = grid_y.reshape(-1) * patch_h
        x0 = grid_x.reshape(-1) * patch_w
        y1 = torch.clamp(y0 + patch_h, max=height)
        x1 = torch.clamp(x0 + patch_w, max=width)
        bbox = torch.stack((y0, x0, y1, x1), dim=-1)
        scale = pixels.new_tensor((height, width, height, width))
        bbox_norm = bbox.to(dtype=pixels.dtype) / scale
        keep = fractions > 0

        patch_rows.append(patches[keep])
        bbox_rows.append(bbox[keep])
        bbox_norm_rows.append(bbox_norm[keep])
        fraction_rows.append(fractions[keep])
        count = int(keep.sum())
        sample_rows.append(
            torch.full((count,), sample_index, dtype=torch.long, device=pixels.device)
        )
        grid_rows.append(torch.stack((grid_y.reshape(-1), grid_x.reshape(-1)), dim=-1)[keep])
        cumulative.append(cumulative[-1] + count)

    return PackedPatchStream(
        patches=torch.cat(patch_rows, dim=0),
        bbox_px_yxxy=torch.cat(bbox_rows, dim=0),
        bbox_norm_yxxy=torch.cat(bbox_norm_rows, dim=0),
        valid_fraction=torch.cat(fraction_rows, dim=0),
        sample_ids=torch.cat(sample_rows, dim=0),
        cu_seqlens=torch.tensor(
            cumulative,
            dtype=torch.int32,
            device=patch_rows[0].device,
        ),
        grid_yx=torch.cat(grid_rows, dim=0),
    )


def pack_native_omvt_batch(
    samples: Sequence[Any],
    omvt_cfg,
    *,
    normalized_white: Sequence[float],
    max_raw_patch_tokens_per_sample: int | None = None,
) -> PackedNativeOMVTBatch:
    if not samples:
        raise ValueError("native OMVT batch must not be empty")
    if max_raw_patch_tokens_per_sample is not None and (
        max_raw_patch_tokens_per_sample <= 0
    ):
        raise ValueError("max_raw_patch_tokens_per_sample must be positive")
    first_pixels = samples[0].pixels
    pad_value = torch.tensor(
        tuple(float(value) for value in normalized_white),
        device=first_pixels.device,
        dtype=first_pixels.dtype,
    )
    patch_shapes = MultiScalePatcher(omvt_cfg).patch_shapes()
    streams = {
        kind: _packed_stream(
            samples,
            patch_h=patch_shapes[kind][0],
            patch_w=patch_shapes[kind][1],
            pad_value=pad_value,
        )
        for kind in PATCH_KINDS
    }
    counts = torch.zeros(
        (len(samples),), dtype=torch.long, device=first_pixels.device
    )
    for stream in streams.values():
        counts.scatter_add_(
            0,
            stream.sample_ids,
            torch.ones_like(stream.sample_ids),
        )
    if (
        max_raw_patch_tokens_per_sample is not None
        and bool((counts > max_raw_patch_tokens_per_sample).any())
    ):
        raise ValueError(
            "native image exceeds max_raw_patch_tokens_per_sample; the "
            "reviewed streamed macro-window planner is required"
        )
    original_hw = torch.tensor(
        [tuple(int(value) for value in sample.original_hw) for sample in samples],
        dtype=torch.long,
        device=first_pixels.device,
    )
    return PackedNativeOMVTBatch(
        streams=streams,
        original_hw=original_hw,
        raw_patch_tokens=counts,
    )


__all__ = [
    "PackedNativeOMVTBatch",
    "PackedPatchStream",
    "pack_native_omvt_batch",
]
