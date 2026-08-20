# -*- coding: utf-8 -*-

"""Orientation-aware Multiscript Vision Tower (OMVT) for RDT."""

from Model.omvt.compressor import PerceiverCompressor
from Model.omvt.heads import (
    LayoutOrderHead,
    MaskedPatchHead,
    OCRReconstructionHead,
    OrientationHead,
)
from Model.omvt.injector import OMVTInjector
from Model.omvt.losses import (
    OMVTSSLOutputs,
    layout_order_loss,
    masked_patch_loss,
    ocr_reconstruction_loss,
    orientation_loss,
    patch_text_contrastive_loss,
)
from Model.omvt.mixers import (
    HorizontalSSM,
    LayoutMixer,
    LocalAttention,
    VerticalSSM,
)
from Model.omvt.native_patcher import (
    PackedNativeOMVTBatch,
    PackedPatchStream,
    pack_native_omvt_batch,
)
from Model.omvt.native_compressor import NativeDetailCompressor, PackedDetailMemory
from Model.omvt.native_mixers import (
    NativeHorizontalSSM,
    NativeLayoutMixer,
    NativeSquareWindowAttention,
    NativeVerticalSSM,
)
from Model.omvt.native_migration import (
    NativeMigrationMapping,
    NativeMigrationReceipt,
    initialize_fresh_native_detail,
    migrate_legacy_omvt_to_native_detail,
)
from Model.omvt.native_planner import (
    NativeCoverageProof,
    NativeMacroWindow,
    NativeMacroWindowPlan,
    plan_native_macro_windows,
    prove_native_macro_window_coverage,
    raw_patch_token_count,
)
from Model.omvt.native_tower import NativeOMVTDetailTower
from Model.omvt.patcher import (
    PATCH_KINDS,
    MultiScalePatcher,
    collate_omvt_batch,
    patch_pixels_for,
)
from Model.omvt.router import GeometricRouter
from Model.omvt.tower import OMVTVisionTower

__all__ = [
    "GeometricRouter",
    "HorizontalSSM",
    "LayoutMixer",
    "LayoutOrderHead",
    "LocalAttention",
    "MaskedPatchHead",
    "MultiScalePatcher",
    "NativeDetailCompressor",
    "NativeCoverageProof",
    "NativeHorizontalSSM",
    "NativeLayoutMixer",
    "NativeMacroWindow",
    "NativeMacroWindowPlan",
    "NativeMigrationMapping",
    "NativeMigrationReceipt",
    "NativeOMVTDetailTower",
    "NativeSquareWindowAttention",
    "NativeVerticalSSM",
    "OCRReconstructionHead",
    "OMVTInjector",
    "OMVTSSLOutputs",
    "OMVTVisionTower",
    "OrientationHead",
    "PackedNativeOMVTBatch",
    "PackedDetailMemory",
    "PackedPatchStream",
    "PATCH_KINDS",
    "PerceiverCompressor",
    "VerticalSSM",
    "collate_omvt_batch",
    "layout_order_loss",
    "masked_patch_loss",
    "migrate_legacy_omvt_to_native_detail",
    "ocr_reconstruction_loss",
    "orientation_loss",
    "initialize_fresh_native_detail",
    "pack_native_omvt_batch",
    "patch_pixels_for",
    "patch_text_contrastive_loss",
    "plan_native_macro_windows",
    "prove_native_macro_window_coverage",
    "raw_patch_token_count",
]
