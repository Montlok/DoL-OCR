# -*- coding: utf-8 -*-

"""First complete packed native-resolution OMVT-v2 detail tower."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.omvt.native_compressor import NativeDetailCompressor
from Model.omvt.native_mixers import (
    NativeHorizontalSSM,
    NativeLayoutMixer,
    NativeSquareWindowAttention,
    NativeVerticalSSM,
    validate_packed_patch_stream,
)
from Model.omvt.native_patcher import PackedNativeOMVTBatch, PackedPatchStream
from Model.omvt.patcher import PATCH_KINDS, patch_pixels_for


class _NativeGeometryRouter(nn.Module):
    """Sample-local geometry router; it never normalizes across a batch."""

    def __init__(self, omvt_cfg) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(len(PATCH_KINDS)))
        self.log_temperature = nn.Parameter(torch.zeros(()))
        self.base_temperature = float(omvt_cfg.router_temperature)
        self.min_prob = float(omvt_cfg.router_min_route_prob)
        if self.base_temperature <= 0:
            raise ValueError("router_temperature must be positive")
        if self.min_prob < 0:
            raise ValueError("router_min_route_prob must be non-negative")

    def forward(self, original_hw: torch.Tensor) -> torch.Tensor:
        if original_hw.ndim != 2 or original_hw.shape[1] != 2:
            raise ValueError("original_hw must have shape [batch, 2]")
        if bool((original_hw <= 0).any()):
            raise ValueError("native image height and width must be positive")
        height, width = original_hw.to(dtype=self.bias.dtype).unbind(dim=-1)
        log_aspect = torch.log(height / width)
        log_area = torch.log1p(height * width)
        logits = torch.stack(
            (
                log_aspect,
                -log_aspect,
                -log_aspect.abs(),
                0.125 * log_area,
            ),
            dim=-1,
        )
        logits = (logits + self.bias) / (
            self.log_temperature.exp() * self.base_temperature + 1e-4
        )
        probabilities = F.softmax(logits, dim=-1)
        if self.min_prob > 0:
            probabilities = probabilities + self.min_prob
            probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
        return probabilities


class _NativeStreamEncoder(nn.Module):
    def __init__(self, kind: str, omvt_cfg) -> None:
        super().__init__()
        if kind not in PATCH_KINDS:
            raise ValueError(f"unknown native patch kind: {kind}")
        self.kind = kind
        d_model = int(omvt_cfg.d_vision)
        ffn_hidden = int(omvt_cfg.vision_ffn_hidden)
        dropout = float(omvt_cfg.vision_dropout)
        self.patch_embed = nn.Linear(patch_pixels_for(kind, omvt_cfg), d_model)
        self.bbox_embed = nn.Linear(4, d_model)
        self.validity_embed = nn.Linear(2, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        if kind == "vertical":
            layers = [
                NativeVerticalSSM(d_model, ffn_hidden, dropout)
                for _ in range(omvt_cfg.n_vertical_layers)
            ]
        elif kind == "horizontal":
            layers = [
                NativeHorizontalSSM(d_model, ffn_hidden, dropout)
                for _ in range(omvt_cfg.n_horizontal_layers)
            ]
        elif kind == "square":
            layers = [
                NativeSquareWindowAttention(
                    d_model,
                    omvt_cfg.vision_n_heads,
                    ffn_hidden,
                    window_yx=(8, 8),
                    dropout=dropout,
                )
                for _ in range(omvt_cfg.n_local_attn_layers)
            ]
        else:
            layers = [
                NativeLayoutMixer(d_model, ffn_hidden, dropout)
                for _ in range(omvt_cfg.n_layout_layers)
            ]
        self.layers = nn.ModuleList(layers)

    def forward(self, stream: PackedPatchStream) -> torch.Tensor:
        validate_packed_patch_stream(stream)
        fraction = stream.valid_fraction.to(dtype=stream.patches.dtype)
        validity = torch.stack((fraction, 1.0 - fraction), dim=-1)
        features = self.patch_embed(stream.patches)
        features = features * fraction.sqrt().unsqueeze(-1)
        features = features + self.bbox_embed(
            stream.bbox_norm_yxxy.to(dtype=features.dtype)
        )
        features = features + self.validity_embed(validity.to(dtype=features.dtype))
        features = self.input_norm(features)
        for layer in self.layers:
            features = layer(features, stream)
        return features


def _validate_native_batch(inputs: PackedNativeOMVTBatch) -> int:
    if not isinstance(inputs.streams, Mapping):
        raise TypeError("native OMVT streams must be a mapping")
    if set(inputs.streams) != set(PATCH_KINDS):
        raise ValueError("native OMVT batch must provide all four patch streams")
    if inputs.original_hw.ndim != 2 or inputs.original_hw.shape[1] != 2:
        raise ValueError("original_hw must have shape [batch, 2]")
    batch_size = int(inputs.original_hw.shape[0])
    if tuple(inputs.raw_patch_tokens.shape) != (batch_size,):
        raise ValueError("raw_patch_tokens must have shape [batch]")

    reconstructed_counts = torch.zeros_like(inputs.raw_patch_tokens)
    reference_device = inputs.original_hw.device
    for kind in PATCH_KINDS:
        stream = inputs.streams[kind]
        stream_batch_size = validate_packed_patch_stream(stream)
        if stream_batch_size != batch_size:
            raise ValueError("all native streams must describe the same batch")
        if stream.patches.device != reference_device:
            raise ValueError("all native geometry and patches must share one device")
        reconstructed_counts = reconstructed_counts + (
            stream.cu_seqlens[1:] - stream.cu_seqlens[:-1]
        ).to(dtype=reconstructed_counts.dtype)
    if not torch.equal(reconstructed_counts, inputs.raw_patch_tokens):
        raise ValueError("raw_patch_tokens does not match packed stream offsets")
    return batch_size


class NativeOMVTDetailTower(nn.Module):
    """Packed OMVT-v2 tower with ragged, hard-capped detail memory.

    This class is intentionally separate from :class:`OMVTVisionTower`; the
    v1 square-input checkpoint contract is therefore unchanged.
    """

    def __init__(
        self,
        omvt_cfg,
        *,
        max_detail_tokens_per_sample: int | None = None,
        source_tokens_per_detail_token: int = 4,
    ) -> None:
        super().__init__()
        self.cfg = omvt_cfg
        self.router = _NativeGeometryRouter(omvt_cfg)
        self.encoders = nn.ModuleDict(
            {
                kind: _NativeStreamEncoder(kind, omvt_cfg)
                for kind in PATCH_KINDS
            }
        )
        self.fuse_norm = nn.LayerNorm(omvt_cfg.d_vision)
        self.compressor = NativeDetailCompressor(
            omvt_cfg,
            max_detail_tokens_per_sample=max_detail_tokens_per_sample,
            source_tokens_per_detail_token=source_tokens_per_detail_token,
        )

    def _fuse_samples(
        self,
        inputs: PackedNativeOMVTBatch,
        encoded: Mapping[str, torch.Tensor],
        router_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sample_rows: list[torch.Tensor] = []
        cumulative = [0]
        batch_size = int(inputs.original_hw.shape[0])
        for sample_index in range(batch_size):
            stream_rows: list[torch.Tensor] = []
            for kind_index, kind in enumerate(PATCH_KINDS):
                stream = inputs.streams[kind]
                start = int(stream.cu_seqlens[sample_index].item())
                end = int(stream.cu_seqlens[sample_index + 1].item())
                stream_rows.append(
                    encoded[kind][start:end] * router_weights[sample_index, kind_index]
                )
            sample = torch.cat(stream_rows, dim=0)
            sample_rows.append(sample)
            cumulative.append(cumulative[-1] + int(sample.shape[0]))
        fused = self.fuse_norm(torch.cat(sample_rows, dim=0))
        return fused, torch.tensor(
            cumulative,
            dtype=torch.int32,
            device="cpu",
        )

    def forward(self, inputs: PackedNativeOMVTBatch) -> dict[str, object]:
        batch_size = _validate_native_batch(inputs)
        router_weights = self.router(inputs.original_hw)
        if tuple(router_weights.shape) != (batch_size, len(PATCH_KINDS)):
            raise RuntimeError("native router returned an invalid shape")
        encoded = {
            kind: self.encoders[kind](inputs.streams[kind])
            for kind in PATCH_KINDS
        }
        fused, fused_cu_seqlens = self._fuse_samples(
            inputs,
            encoded,
            router_weights,
        )
        detail = self.compressor(fused, fused_cu_seqlens)
        return {
            "detail_memory": detail.memory,
            "detail_cu_seqlens": detail.cu_seqlens,
            "detail_sample_ids": detail.sample_ids,
            "detail_token_counts": detail.token_counts,
            "source_token_counts": detail.source_token_counts,
            "fused": fused,
            "fused_cu_seqlens": fused_cu_seqlens,
            "router_weights": router_weights,
            "streams": encoded,
        }


__all__ = ["NativeOMVTDetailTower"]
