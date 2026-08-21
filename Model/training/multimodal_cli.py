# -*- coding: utf-8 -*-

"""Shared multimodal CLI helpers for training entry points.

Centralises the construction of :class:`PILImageProcessor` /
:class:`OMVTConfig` so that ``train_rdt`` / ``train_vlm_align`` /
``train_omvt_ssl`` expose a consistent ``--multimodal`` + ``--image-size``
+ ``--n-image-tokens`` surface and remain in lockstep on defaults.
"""

from __future__ import annotations

import argparse
from typing import Any

from Model.config import OMVTConfig
from Tokenizer.multimodal.image_placeholders import image_patch_count


def add_multimodal_args(p: argparse.ArgumentParser, *, default_image_size: int = 56) -> None:
    """Register the standard multimodal flags on a CLI parser."""

    p.add_argument(
        "--multimodal",
        action="store_true",
        help="enable pixel-aware path: loads images from JSONL rows via PIL",
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=default_image_size,
        help="square image edge fed to OMVT (must be divisible by 4)",
    )
    p.add_argument(
        "--n-image-tokens",
        type=int,
        default=None,
        help="OMVT compress_to; defaults to patch count implied by --image-size",
    )
    p.add_argument(
        "--d-vision",
        type=int,
        default=64,
        help="OMVT hidden width (also the projector input dim)",
    )


def build_image_processor(args: argparse.Namespace) -> Any | None:
    """Construct a :class:`PILImageProcessor`, deferring the import.

    Returns ``None`` when ``--multimodal`` is not set so call sites can
    short-circuit. PIL is imported lazily to keep text-only runs free of
    the Pillow dependency.
    """

    if not getattr(args, "multimodal", False):
        return None
    from Tokenizer.multimodal import PILImageProcessor  # local import

    return PILImageProcessor(image_size=args.image_size)


def make_omvt_cfg(
    image_size: int,
    d_vision: int,
    compress_to: int | None = None,
    *,
    preset: str = "derived",
) -> OMVTConfig:
    """Single source of truth for CLI-driven OMVT tower geometry.

    ``preset="derived"`` is the legacy smoke layout (patch grids scaled from
    ``image_size``: half-half vertical/horizontal split, quarter-square grid,
    one layout-level macro patch). ``preset="prod"`` keeps the OMVTConfig
    dataclass multi-scale defaults (32x8 / 8x32 / 16x16 / 56x56), which is
    what real-page training uses. Every trainer CLI builds its tower through
    here so the entry points cannot drift apart again.
    """

    if image_size <= 0 or image_size % 4 != 0:
        raise ValueError(
            f"image_size must be a positive multiple of 4, got {image_size}"
        )
    if preset not in ("derived", "prod"):
        raise ValueError(f"unknown OMVT preset: {preset!r}")
    if compress_to is None:
        compress_to = image_patch_count(image_size, image_size)
    if preset == "prod":
        return OMVTConfig(
            image_size=image_size,
            d_vision=d_vision,
            compress_to=compress_to,
        )
    s = image_size
    return OMVTConfig(
        image_size=s,
        d_vision=d_vision,
        vertical_patch=(s // 2, s // 4),
        horizontal_patch=(s // 4, s // 2),
        square_patch=(s // 4, s // 4),
        layout_patch=(s, s),
        compress_to=compress_to,
    )


def build_omvt_cfg(args: argparse.Namespace) -> OMVTConfig | None:
    """Construct an :class:`OMVTConfig` from CLI args (None when not multimodal)."""

    if not getattr(args, "multimodal", False):
        return None
    return make_omvt_cfg(
        args.image_size,
        args.d_vision,
        args.n_image_tokens,
        preset=getattr(args, "patch_preset", "derived"),
    )


__all__ = [
    "add_multimodal_args",
    "build_image_processor",
    "build_omvt_cfg",
    "make_omvt_cfg",
]
