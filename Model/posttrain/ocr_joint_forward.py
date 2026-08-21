# -*- coding: utf-8 -*-

"""Canonical OMVT-v2 detail encoding and RDT teacher-forced forward."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import torch

from Model.ocr.position_contract import BOUNDARY_V1
from Model.omvt.native_patcher import PackedNativeOMVTBatch, PackedPatchStream
from Model.posttrain.ocr_anyres_collator import merge_packed_detail_views


def move_native_batch(
    batch: PackedNativeOMVTBatch,
    device: torch.device,
    *,
    dtype: torch.dtype | None = None,
) -> PackedNativeOMVTBatch:
    def move(value: torch.Tensor) -> torch.Tensor:
        target_dtype = dtype if dtype is not None and value.is_floating_point() else None
        return value.to(
            device,
            dtype=target_dtype,
            non_blocking=True,
        )

    streams = {
        kind: PackedPatchStream(
            patches=move(stream.patches),
            bbox_px_yxxy=move(stream.bbox_px_yxxy),
            bbox_norm_yxxy=move(stream.bbox_norm_yxxy),
            valid_fraction=move(stream.valid_fraction),
            sample_ids=move(stream.sample_ids),
            cu_seqlens=stream.cu_seqlens.to("cpu"),
            grid_yx=move(stream.grid_yx),
        )
        for kind, stream in batch.streams.items()
    }
    return replace(
        batch,
        streams=streams,
        original_hw=move(batch.original_hw),
        raw_patch_tokens=batch.raw_patch_tokens.to("cpu"),
    )


def forward_anyres_ocr_batch(
    model,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
    return_logits: bool = False,
    loss_chunk_size: int | None = None,
) -> dict[str, Any]:
    if batch.get("position_contract") != BOUNDARY_V1:
        raise ValueError("anyres OCR forward requires boundary_v1")
    forbidden = {"word_pos", "morph_depth", "token_offsets"}.intersection(batch)
    if forbidden:
        raise ValueError(
            "anyres OCR forward must not materialize position fields: "
            f"{sorted(forbidden)}"
        )
    visual = encode_anyres_visual_batch(model, batch, device=device)
    output = model(
        input_ids=batch["input_ids"].to(device, non_blocking=True),
        attention_mask=batch["attention_mask"].to(device, non_blocking=True),
        labels=batch["labels"].to(device, non_blocking=True),
        pixel_values=visual["global_pixel_values"],
        position_contract=batch["position_contract"],
        detail_memory=visual["detail_memory"],
        detail_cu_seqlens=visual["detail_cu_seqlens"],
        return_logits=return_logits,
        loss_chunk_size=loss_chunk_size,
    )
    return {**output, **visual}


def encode_anyres_visual_batch(
    model,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    native_tower = getattr(model.vision, "native_detail_tower", None)
    if native_tower is None:
        raise RuntimeError("anyres OCR forward requires an installed native detail tower")
    if model.vision_cross_attention is None:
        raise RuntimeError("anyres OCR forward requires an installed vision bridge")
    native_packed = batch.get("native_packed")
    if not isinstance(native_packed, PackedNativeOMVTBatch):
        raise TypeError("batch.native_packed must be a PackedNativeOMVTBatch")
    view_to_sample = batch.get("view_to_sample")
    if not isinstance(view_to_sample, torch.Tensor):
        raise TypeError("batch.view_to_sample must be a tensor")

    target_parameter = next(native_tower.parameters(), None)
    if target_parameter is None:
        raise RuntimeError("native detail tower has no parameter dtype contract")
    target_dtype = target_parameter.dtype
    native_packed = move_native_batch(
        native_packed,
        device,
        dtype=target_dtype,
    )
    detail = native_tower(native_packed)
    logical_batch = int(batch["input_ids"].shape[0])
    detail_memory, detail_cu = merge_packed_detail_views(
        detail["detail_memory"],
        detail["detail_cu_seqlens"],
        view_to_sample.to("cpu"),
        logical_batch,
    )
    global_pixels = {}
    for key, value in dict(batch["global_pixel_values"]).items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"global_pixel_values[{key!r}] must be a tensor")
        global_pixels[key] = value.to(
            device,
            dtype=target_dtype if value.is_floating_point() else None,
            non_blocking=True,
        )
    return {
        "global_pixel_values": global_pixels,
        "detail_memory": detail_memory,
        "detail_cu_seqlens": detail_cu,
        "detail_token_counts": detail["detail_token_counts"],
        "source_token_counts": detail["source_token_counts"],
    }


__all__ = [
    "encode_anyres_visual_batch",
    "forward_anyres_ocr_batch",
    "move_native_batch",
]
