# -*- coding: utf-8 -*-
"""Versioned OCR position semantics shared by training and inference.

``boundary_v1`` is the production contract: OCR batches omit synthetic
position tensors and let :class:`Model.model.RDTForCausalLM` derive the same
word/morpheme positions from boundary token IDs in both teacher forcing and
generation.

``legacy_sequential_v0`` exists only to reproduce checkpoints trained before
the contract was made explicit.  Those runs received ``word_pos=arange(L)``
and zero ``morph_depth`` from the generic pretraining collator.
"""

from __future__ import annotations

from collections.abc import Mapping


BOUNDARY_V1 = "boundary_v1"
LEGACY_SEQUENTIAL_V0 = "legacy_sequential_v0"
OCR_POSITION_CONTRACT_CHOICES = (BOUNDARY_V1, LEGACY_SEQUENTIAL_V0)
OCR_POSITION_CONTRACT_METADATA_VERSION = 1


def validate_ocr_position_contract(value: str) -> str:
    """Return a known contract value or fail closed."""

    value = str(value)
    if value not in OCR_POSITION_CONTRACT_CHOICES:
        raise ValueError(
            "unsupported OCR position contract "
            f"{value!r}; expected one of {OCR_POSITION_CONTRACT_CHOICES}"
        )
    return value


def resolve_checkpoint_ocr_position_contract(
    metadata: Mapping[str, object],
    requested: str | None,
) -> str:
    """Resolve an inference/resume contract against checkpoint metadata.

    New checkpoints are self-describing.  Historical VLM checkpoints have no
    position field; they are accepted only when the caller explicitly selects
    ``legacy_sequential_v0``.  This prevents a missing field from silently
    reproducing the train/inference mismatch that motivated this contract.
    """

    saved = metadata.get("ocr_position_contract")
    if saved is None:
        if requested == LEGACY_SEQUENTIAL_V0:
            return LEGACY_SEQUENTIAL_V0
        raise ValueError(
            "checkpoint has no ocr_position_contract metadata; historical "
            "OCR checkpoints were trained with legacy sequential positions. "
            "Pass --ocr-position-contract legacy_sequential_v0 explicitly "
            "after verifying the checkpoint provenance"
        )

    saved_value = validate_ocr_position_contract(str(saved))
    saved_version = metadata.get("ocr_position_contract_version")
    if saved_version != OCR_POSITION_CONTRACT_METADATA_VERSION:
        raise ValueError(
            "checkpoint OCR position metadata version mismatch: "
            f"checkpoint={saved_version!r} supported="
            f"{OCR_POSITION_CONTRACT_METADATA_VERSION}"
        )
    if requested is None:
        return saved_value
    requested_value = validate_ocr_position_contract(requested)
    if requested_value != saved_value:
        raise ValueError(
            "OCR position contract mismatch: "
            f"checkpoint={saved_value!r} requested={requested_value!r}"
        )
    return saved_value


__all__ = [
    "BOUNDARY_V1",
    "LEGACY_SEQUENTIAL_V0",
    "OCR_POSITION_CONTRACT_CHOICES",
    "OCR_POSITION_CONTRACT_METADATA_VERSION",
    "resolve_checkpoint_ocr_position_contract",
    "validate_ocr_position_contract",
]
