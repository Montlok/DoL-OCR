# -*- coding: utf-8 -*-
"""Image placeholder expansion helpers."""

from __future__ import annotations

import math

from .tokens import IMAGE_END, IMAGE_PATCH, IMAGE_PLACEHOLDER, IMAGE_START


def image_patch_count(
    width: int, height: int, patch_size: int = 14, merge_size: int = 2
) -> int:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if patch_size <= 0 or merge_size <= 0:
        raise ValueError("patch_size and merge_size must be positive")

    patch_w = math.ceil(width / patch_size)
    patch_h = math.ceil(height / patch_size)
    merged_w = math.ceil(patch_w / merge_size)
    merged_h = math.ceil(patch_h / merge_size)
    return max(1, merged_w * merged_h)


def image_placeholder_tokens(patches_per_image: int) -> list[str]:
    if patches_per_image < 1:
        raise ValueError("patches_per_image must be positive")
    return [IMAGE_START] + [IMAGE_PATCH] * patches_per_image + [IMAGE_END]


def expand_image_placeholders(text: str, patches_per_image: int) -> str:
    return "".join(image_placeholder_tokens(patches_per_image)).join(
        text.split(IMAGE_PLACEHOLDER)
    )


def expand_image_placeholders_by_sizes(
    text: str,
    image_sizes: list[tuple[int, int]] | list[dict],
    patch_size: int = 14,
    merge_size: int = 2,
) -> str:
    """Expand <image> placeholders per-image by inferred patch count.

    image_sizes can be a list of (width, height) tuples or {"width","height"}
    dicts. Number of <image> markers in text must equal len(image_sizes).
    """
    parts = text.split(IMAGE_PLACEHOLDER)
    expected = len(parts) - 1
    if expected != len(image_sizes):
        raise ValueError(
            f"<image> placeholder count ({expected}) does not match "
            f"image_sizes length ({len(image_sizes)})"
        )
    out = [parts[0]]
    for i, size in enumerate(image_sizes):
        if isinstance(size, dict):
            w = int(size.get("width") or size.get("w"))
            h = int(size.get("height") or size.get("h"))
        else:
            w, h = int(size[0]), int(size[1])
        n = image_patch_count(w, h, patch_size, merge_size)
        out.append("".join(image_placeholder_tokens(n)))
        out.append(parts[i + 1])
    return "".join(out)
