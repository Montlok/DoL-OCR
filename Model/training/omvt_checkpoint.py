# -*- coding: utf-8 -*-

"""Shared OMVT SSL checkpoint helpers.

``scripts/train_omvt_ssl`` writes step checkpoints as::

    <output>/step_XXXXXXXX/omvt_ssl.pt   (+ a ``latest`` symlink)

with a payload of ``{"step", "omvt_config", "tower", "ocr_head", ...,
"tower_ema"?}``. Resolving that layout and overlaying EMA tower weights used
to be re-implemented by every consumer (the SSL trainer itself, the SSL
evaluator, the VLM-alignment trainer); this module is the single copy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

_CHECKPOINT_NAME = "omvt_ssl.pt"


def resolve_omvt_checkpoint_path(path: str | Path) -> Path:
    """Resolve a file, run dir, or ``latest`` layout to the checkpoint file."""

    p = Path(path)
    if p.is_file():
        return p
    for candidate in (p / _CHECKPOINT_NAME, p / "latest" / _CHECKPOINT_NAME):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"OMVT SSL checkpoint not found: {path}")


def load_omvt_payload(path: str | Path, *, weights_only: bool = True) -> Any:
    """Load the checkpoint payload behind ``path`` (file or run dir)."""

    return torch.load(
        resolve_omvt_checkpoint_path(path),
        map_location="cpu",
        weights_only=weights_only,
    )


def tower_state_from_payload(payload: Any, *, use_ema: bool = False) -> Any:
    """Extract the tower state dict, optionally overlaying EMA weights.

    EMA shadows cover only floating-point entries; they are overlaid on the
    raw tower state so non-float buffers keep their trained values. Payloads
    that are already bare state dicts pass through unchanged.
    """

    if not (isinstance(payload, dict) and "tower" in payload):
        return payload
    state = payload["tower"]
    if use_ema and payload.get("tower_ema"):
        state = dict(state)
        state.update(payload["tower_ema"])
    return state


__all__ = [
    "load_omvt_payload",
    "resolve_omvt_checkpoint_path",
    "tower_state_from_payload",
]
