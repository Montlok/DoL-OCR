# -*- coding: utf-8 -*-
"""Video placeholder expansion helpers.

Mirrors Qwen2-VL: video tokens = ceil(num_frames/temporal_patch_size) *
(ceil(width/patch_size)/merge_size) * (ceil(height/patch_size)/merge_size).
"""

from __future__ import annotations

import math

from .tokens import VIDEO_END, VIDEO_PATCH, VIDEO_PLACEHOLDER, VIDEO_START


def video_patch_count(
    num_frames: int,
    width: int,
    height: int,
    patch_size: int = 14,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
) -> int:
    if num_frames <= 0 or width <= 0 or height <= 0:
        raise ValueError("frames/width/height must be positive")
    if patch_size <= 0 or temporal_patch_size <= 0 or merge_size <= 0:
        raise ValueError("patch sizes must be positive")
    t = math.ceil(num_frames / temporal_patch_size)
    w = math.ceil(math.ceil(width / patch_size) / merge_size)
    h = math.ceil(math.ceil(height / patch_size) / merge_size)
    return max(1, t * w * h)


def video_placeholder_tokens(patches_per_video: int) -> list[str]:
    if patches_per_video < 1:
        raise ValueError("patches_per_video must be positive")
    return [VIDEO_START] + [VIDEO_PATCH] * patches_per_video + [VIDEO_END]


def expand_video_placeholders_by_sizes(
    text: str,
    video_sizes: list[tuple[int, int, int]] | list[dict],
    patch_size: int = 14,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
) -> str:
    parts = text.split(VIDEO_PLACEHOLDER)
    expected = len(parts) - 1
    if expected != len(video_sizes):
        raise ValueError(
            f"<video> placeholder count ({expected}) does not match "
            f"video_sizes length ({len(video_sizes)})"
        )
    out = [parts[0]]
    for i, size in enumerate(video_sizes):
        if isinstance(size, dict):
            f = int(size.get("frames") or size.get("num_frames"))
            w = int(size.get("width") or size.get("w"))
            h = int(size.get("height") or size.get("h"))
        else:
            f, w, h = int(size[0]), int(size[1]), int(size[2])
        n = video_patch_count(f, w, h, patch_size, temporal_patch_size, merge_size)
        out.append("".join(video_placeholder_tokens(n)))
        out.append(parts[i + 1])
    return "".join(out)
