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
) -> PackedNativeOMVTBatch:
    streams = {
        kind: PackedPatchStream(
            patches=stream.patches.to(device, non_blocking=True),
            bbox_px_yxxy=stream.bbox_px_yxxy.to(device, non_blocking=True),
            bbox_norm_yxxy=stream.bbox_norm_yxxy.to(device, non_blocking=True),
            valid_fraction=stream.valid_fraction.to(device, non_blocking=True),
            sample_ids=stream.sample_ids.to(device, non_blocking=True),
            cu_seqlens=stream.cu_seqlens.to("cpu"),
            grid_yx=stream.grid_yx.to(device, non_blocking=True),
        )
        for kind, stream in batch.streams.items()
    }
    return replace(
        batch,
        streams=streams,
        original_hw=batch.original_hw.to(device, non_blocking=True),
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

    native_packed = move_native_batch(native_packed, device)
    detail = native_tower(native_packed)
    logical_batch = int(batch["input_ids"].shape[0])
    detail_memory, detail_cu = merge_packed_detail_views(
        detail["detail_memory"],
        detail["detail_cu_seqlens"],
        view_to_sample.to("cpu"),
        logical_batch,
    )
    global_pixels = {
        key: value.to(device, non_blocking=True)
        for key, value in dict(batch["global_pixel_values"]).items()
    }
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
