# -*- coding: utf-8 -*-
"""Shared preprocessing for the legacy line-level OCR visual contract."""

from __future__ import annotations

import io
import os
from typing import Any

from PIL import Image, ImageOps


def letterbox_grayscale_to_square(
    image: Any,
    image_size: int = 224,
) -> Image.Image:
    """Return an EXIF-corrected L image letterboxed on white with LANCZOS."""

    if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("image_size must be a positive integer")

    if isinstance(image, Image.Image):
        source = ImageOps.exif_transpose(image).convert("L")
    elif isinstance(image, (bytes, bytearray, memoryview)):
        with Image.open(io.BytesIO(bytes(image))) as raw:
            raw.load()
            source = ImageOps.exif_transpose(raw).convert("L")
    elif isinstance(image, (str, os.PathLike)):
        with Image.open(image) as raw:
            raw.load()
            source = ImageOps.exif_transpose(raw).convert("L")
    else:
        raise TypeError(
            f"unsupported image type {type(image).__name__}; expected bytes, "
            "path, or PIL.Image"
        )

    width, height = source.size
    if width <= 0 or height <= 0:
        raise ValueError("image has zero width or height")
    side = max(width, height)
    canvas = Image.new("L", (side, side), 255)
    canvas.paste(source, ((side - width) // 2, (side - height) // 2))
    if side != image_size:
        resample = (
            Image.Resampling.LANCZOS
            if hasattr(Image, "Resampling")
            else Image.LANCZOS
        )
        canvas = canvas.resize((image_size, image_size), resample)
    return canvas


__all__ = ["letterbox_grayscale_to_square"]
