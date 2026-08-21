# -*- coding: utf-8 -*-
"""Bounding-box token helpers.

We use a string-token scheme: <bbox_xxx_yyy_xxx_yyy> with each coordinate
quantized to `bins` (default 1000). If the resulting token isn't in vocab,
byte fallback will handle it cleanly.
"""

from __future__ import annotations


def normalize_bbox(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    bins: int = 1000,
) -> tuple[int, int, int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if bins <= 1:
        raise ValueError("bins must be > 1")
    x0, y0, x1, y1 = bbox
    if x0 > x1 or y0 > y1:
        raise ValueError("bbox must be (x0,y0,x1,y1) with x0<=x1, y0<=y1")

    def q(v: float, dim: int) -> int:
        r = v / dim
        r = max(0.0, min(1.0, r))
        return min(bins - 1, int(round(r * (bins - 1))))

    return (q(x0, width), q(y0, height), q(x1, width), q(y1, height))


def encode_bbox_tokens(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    bins: int = 1000,
) -> str:
    x0, y0, x1, y1 = normalize_bbox(bbox, width, height, bins)
    return f"<bbox_{x0:03d}_{y0:03d}_{x1:03d}_{y1:03d}>"


def decode_bbox_tokens(
    token: str, width: int, height: int, bins: int = 1000
) -> tuple[float, float, float, float]:
    if not (token.startswith("<bbox_") and token.endswith(">")):
        raise ValueError(f"not a bbox token: {token!r}")
    body = token[len("<bbox_"):-1]
    parts = body.split("_")
    if len(parts) != 4:
        raise ValueError(f"bbox token must have 4 coords: {token!r}")
    x0, y0, x1, y1 = (int(p) for p in parts)
    scale = bins - 1
    return (
        x0 / scale * width,
        y0 / scale * height,
        x1 / scale * width,
        y1 / scale * height,
    )
