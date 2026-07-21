# -*- coding: utf-8 -*-

"""Strict checkpoint reconstruction for post-training.

OCR alignment must reconstruct the exact RDT and OMVT geometry *before* loading
weights.  Loading a multimodal checkpoint into the default lazy vision module
with ``strict=False`` silently drops ``vision.omvt.*`` tensors, producing a text-
only policy while the trainer appears to run normally.  This module makes that
state impossible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from Model.config import OMVTConfig, RDTConfig
from Model.model import RDTForCausalLM
from Model.omvt import OMVTInjector
from Model.training.checkpoint import (
    load_checkpoint_metadata,
    resolve_checkpoint_dir,
)


OCR_GRPO_CONTRACT_VERSION = 1


@dataclass
class ReconstructedPolicy:
    model: RDTForCausalLM
    rdt_config: RDTConfig
    omvt_config: OMVTConfig | None
    metadata: dict[str, Any]
    checkpoint_dir: Path


def _rdt_config_from_metadata(
    metadata: dict[str, Any],
    fallback: RDTConfig | None,
) -> RDTConfig:
    raw = metadata.get("rdt_config")
    if raw is None:
        if fallback is None:
            raise ValueError(
                "checkpoint has no rdt_config metadata; exact post-training "
                "reconstruction requires an explicit fallback config"
            )
        return fallback
    if not isinstance(raw, dict):
        raise TypeError("checkpoint rdt_config metadata must be a dict")
    return RDTConfig(**raw)


def _omvt_config_from_metadata(metadata: dict[str, Any]) -> OMVTConfig | None:
    raw = metadata.get("omvt_config")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TypeError("checkpoint omvt_config metadata must be a dict")
    values = dict(raw)
    for key in ("vertical_patch", "horizontal_patch", "square_patch", "layout_patch"):
        if key in values:
            values[key] = tuple(values[key])
    return OMVTConfig(**values)


def reconstruct_policy_from_checkpoint(
    path: str | Path,
    *,
    fallback_rdt_config: RDTConfig | None = None,
    require_vision: bool = False,
) -> ReconstructedPolicy:
    """Build and strictly load an RDT/OMVT policy from checkpoint metadata."""

    checkpoint_dir = resolve_checkpoint_dir(path)
    metadata = load_checkpoint_metadata(checkpoint_dir)
    if require_vision and "rdt_config" not in metadata:
        raise ValueError(
            "image-conditioned OCR post-training requires checkpoint "
            "rdt_config metadata; tensor shapes alone do not recover behavioral "
            "settings such as recurrent depth and context length"
        )
    rdt_cfg = _rdt_config_from_metadata(metadata, fallback_rdt_config)
    omvt_cfg = _omvt_config_from_metadata(metadata)

    state = torch.load(
        checkpoint_dir / "model.pt",
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint model state must be a dict: {checkpoint_dir}")
    has_omvt_weights = any(str(key).startswith("vision.omvt.") for key in state)
    if has_omvt_weights and omvt_cfg is None:
        raise ValueError(
            "checkpoint contains vision.omvt weights but has no omvt_config metadata; "
            "refusing to guess tower geometry"
        )
    if require_vision and not has_omvt_weights:
        raise ValueError(
            "image-conditioned OCR post-training requires vision.omvt weights"
        )

    model = RDTForCausalLM(rdt_cfg)
    if has_omvt_weights:
        assert omvt_cfg is not None
        model.vision._omvt_cfg = omvt_cfg
        model.vision.omvt = OMVTInjector(rdt_cfg, omvt_cfg)

    # strict=True is the safety property of this loader.  A missing projector,
    # silently skipped tower, wrong tokenizer geometry, or backend mismatch must
    # stop before an optimizer is allocated.
    model.load_state_dict(state, strict=True)
    return ReconstructedPolicy(
        model=model,
        rdt_config=rdt_cfg,
        omvt_config=omvt_cfg,
        metadata=metadata,
        checkpoint_dir=checkpoint_dir,
    )


__all__ = [
    "OCR_GRPO_CONTRACT_VERSION",
    "ReconstructedPolicy",
    "reconstruct_policy_from_checkpoint",
]
