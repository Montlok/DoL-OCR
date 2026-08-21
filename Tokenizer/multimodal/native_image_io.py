# -*- coding: utf-8 -*-

"""Aspect-preserving image decoding for native-resolution vision paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .image_io import _DEFAULT_MEAN, _DEFAULT_STD, _open_to_rgb


@dataclass(frozen=True)
class NativeImageTensor:
    pixels: torch.Tensor
    pixel_valid_mask: torch.Tensor
    original_hw: tuple[int, int]


class NativeImageProcessorV2:
    """Decode images without cropping, squaring, or changing aspect ratio."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        mean: Sequence[float] | None = _DEFAULT_MEAN,
        std: Sequence[float] | None = _DEFAULT_STD,
        max_decode_pixels: int = 40_000_000,
    ) -> None:
        if in_channels not in (1, 3):
            raise ValueError("in_channels must be 1 or 3")
        if max_decode_pixels <= 0:
            raise ValueError("max_decode_pixels must be positive")
        if mean is not None and len(mean) != in_channels:
            raise ValueError("mean must have one entry per channel")
        if std is not None and len(std) != in_channels:
            raise ValueError("std must have one entry per channel")
        if (mean is None) ^ (std is None):
            raise ValueError("mean and std must be set together or both None")
        self.in_channels = int(in_channels)
        self.mean = tuple(float(value) for value in mean) if mean is not None else None
        self.std = tuple(float(value) for value in std) if std is not None else None
        self.max_decode_pixels = int(max_decode_pixels)

    @property
    def normalized_white(self) -> tuple[float, ...]:
        if self.mean is None or self.std is None:
            return (1.0,) * self.in_channels
        return tuple(
            (1.0 - mean) / std
            for mean, std in zip(self.mean, self.std, strict=True)
        )

    def _single(self, spec: Any) -> NativeImageTensor:
        image = _open_to_rgb(
            spec,
            channels=self.in_channels,
            max_pixels=self.max_decode_pixels,
        )
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("image has zero width or height")
        pixels_count = width * height
        if pixels_count > self.max_decode_pixels:
            raise ValueError(
                "image exceeds max_decode_pixels; use the reviewed streamed "
                f"native-resolution path: {pixels_count} > {self.max_decode_pixels}"
            )
        raw = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        if self.in_channels == 3:
            pixels = raw.reshape(height, width, 3).permute(2, 0, 1)
        else:
            pixels = raw.reshape(1, height, width)
        pixels = pixels.to(torch.float32) / 255.0
        if self.mean is not None and self.std is not None:
            mean = torch.tensor(self.mean, dtype=pixels.dtype).view(-1, 1, 1)
            std = torch.tensor(self.std, dtype=pixels.dtype).view(-1, 1, 1)
            pixels = (pixels - mean) / std
        return NativeImageTensor(
            pixels=pixels,
            pixel_valid_mask=torch.ones((height, width), dtype=torch.bool),
            original_hw=(height, width),
        )

    def __call__(self, images: Sequence[Any]) -> list[NativeImageTensor]:
        return [self._single(image) for image in images]


__all__ = ["NativeImageProcessorV2", "NativeImageTensor"]
