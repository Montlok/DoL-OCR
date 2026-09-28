# -*- coding: utf-8 -*-
"""Compatibility exports for pretraining packing and windowing."""

from .producer import (
    _empty_text_pack,
    _has_modality,
    _next_word_pos,
    _pad,
    _trim,
    iter_pack_samples,
    pack_samples,
)

__all__ = [
    "iter_pack_samples", "pack_samples", "_empty_text_pack",
    "_has_modality", "_next_word_pos", "_pad", "_trim",
]
