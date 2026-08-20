# -*- coding: utf-8 -*-
"""Fail-closed visual-input contracts for OCR checkpoints."""

from __future__ import annotations

from collections.abc import Mapping


DOL_OCR_LINE_LETTERBOX_224_V1 = "dol_ocr_line_letterbox_224_v1"
DOL_OCR_ANYRES_V2 = "dol_ocr_anyres_v2"

OCR_VISUAL_INPUT_CONTRACT_METADATA_KEY = "ocr_visual_input_contract"
OCR_VISUAL_INPUT_CONTRACT_VERSION_METADATA_KEY = (
    "ocr_visual_input_contract_version"
)

_CONTRACT_VERSIONS = {
    DOL_OCR_LINE_LETTERBOX_224_V1: 1,
    DOL_OCR_ANYRES_V2: 2,
}
OCR_VISUAL_INPUT_CONTRACT_CHOICES = tuple(_CONTRACT_VERSIONS)


def validate_ocr_visual_input_contract(value: object) -> str:
    """Return a supported contract name without accepting aliases."""

    if not isinstance(value, str) or value not in _CONTRACT_VERSIONS:
        raise ValueError(
            "unsupported OCR visual input contract "
            f"{value!r}; expected one of {OCR_VISUAL_INPUT_CONTRACT_CHOICES}"
        )
    return value


def ocr_visual_input_contract_version(contract: object) -> int:
    """Return the metadata version bound to an exact contract name."""

    return _CONTRACT_VERSIONS[validate_ocr_visual_input_contract(contract)]


def resolve_checkpoint_ocr_visual_input_contract(
    metadata: Mapping[str, object],
    requested: str | None = None,
) -> str:
    """Resolve an explicitly declared checkpoint contract.

    There is deliberately no historical default.  A reviewed legacy artifact
    may be bound externally by its immutable release identity, but ordinary
    checkpoints must carry both fields below and requests must match them.
    """

    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint metadata must be an object")

    has_contract = OCR_VISUAL_INPUT_CONTRACT_METADATA_KEY in metadata
    has_version = OCR_VISUAL_INPUT_CONTRACT_VERSION_METADATA_KEY in metadata
    if not has_contract and not has_version:
        raise ValueError(
            "checkpoint has no OCR visual input contract metadata; refusing "
            "to infer line-letterbox or anyres semantics"
        )
    if has_contract != has_version:
        raise ValueError(
            "checkpoint OCR visual input contract metadata is incomplete; "
            "both contract and version are required"
        )

    saved = validate_ocr_visual_input_contract(
        metadata[OCR_VISUAL_INPUT_CONTRACT_METADATA_KEY]
    )
    saved_version = metadata[OCR_VISUAL_INPUT_CONTRACT_VERSION_METADATA_KEY]
    expected_version = ocr_visual_input_contract_version(saved)
    if (
        isinstance(saved_version, bool)
        or not isinstance(saved_version, int)
        or saved_version != expected_version
    ):
        raise ValueError(
            "checkpoint OCR visual input contract version mismatch: "
            f"contract={saved!r} checkpoint={saved_version!r} "
            f"supported={expected_version}"
        )

    if requested is None:
        return saved
    requested_value = validate_ocr_visual_input_contract(requested)
    if requested_value != saved:
        raise ValueError(
            "OCR visual input contract mismatch: "
            f"checkpoint={saved!r} requested={requested_value!r}"
        )
    return saved


__all__ = [
    "DOL_OCR_ANYRES_V2",
    "DOL_OCR_LINE_LETTERBOX_224_V1",
    "OCR_VISUAL_INPUT_CONTRACT_CHOICES",
    "OCR_VISUAL_INPUT_CONTRACT_METADATA_KEY",
    "OCR_VISUAL_INPUT_CONTRACT_VERSION_METADATA_KEY",
    "ocr_visual_input_contract_version",
    "resolve_checkpoint_ocr_visual_input_contract",
    "validate_ocr_visual_input_contract",
]
