# -*- coding: utf-8 -*-
"""Traditional-Mongolian OCR utilities (metrics, row contract, segmentation)."""

from Model.ocr.anyres_preprocess_contract import (
    build_anyres_preprocess_contract,
    validate_anyres_preprocess_contract,
)
from Model.ocr.data import build_ocr_row
from Model.ocr.metrics import (
    OCRReport,
    cer,
    edit_distance,
    grapheme_clusters,
    nominal_normalize,
    ocr_report,
    wer,
)
from Model.ocr.position_contract import (
    BOUNDARY_V1,
    LEGACY_SEQUENTIAL_V0,
    OCR_POSITION_CONTRACT_CHOICES,
    resolve_checkpoint_ocr_position_contract,
)
from Model.ocr.segment import (
    chunk_column_by_height,
    detect_columns,
    detect_line_columns,
    is_plausible_line,
    lines_from_column,
    trim_scan_borders,
)
from Model.ocr.visual_input_contract import (
    DOL_OCR_ANYRES_V2,
    DOL_OCR_LINE_LETTERBOX_224_V1,
    OCR_VISUAL_INPUT_CONTRACT_CHOICES,
    ocr_visual_input_contract_version,
    resolve_checkpoint_ocr_visual_input_contract,
)

__all__ = [
    "OCRReport",
    "BOUNDARY_V1",
    "DOL_OCR_ANYRES_V2",
    "DOL_OCR_LINE_LETTERBOX_224_V1",
    "LEGACY_SEQUENTIAL_V0",
    "OCR_POSITION_CONTRACT_CHOICES",
    "OCR_VISUAL_INPUT_CONTRACT_CHOICES",
    "build_ocr_row",
    "build_anyres_preprocess_contract",
    "cer",
    "chunk_column_by_height",
    "detect_columns",
    "detect_line_columns",
    "edit_distance",
    "grapheme_clusters",
    "is_plausible_line",
    "lines_from_column",
    "nominal_normalize",
    "ocr_report",
    "ocr_visual_input_contract_version",
    "resolve_checkpoint_ocr_position_contract",
    "resolve_checkpoint_ocr_visual_input_contract",
    "trim_scan_borders",
    "validate_anyres_preprocess_contract",
    "wer",
]
