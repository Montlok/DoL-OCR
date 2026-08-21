# -*- coding: utf-8 -*-
"""Morphological rule helpers for traditional Mongolian tokenization."""

try:
    from .unicode_norm import strip_all
except ImportError:  # pragma: no cover - supports direct script execution.
    from unicode_norm import strip_all  # type: ignore[no-redef]


MASCULINE_VOWELS = {"ᠠ", "ᠣ", "ᠤ"}
FEMININE_VOWELS = {"ᠡ", "ᠥ", "ᠦ"}
NEUTRAL_VOWELS = {"ᠢ"}


def get_harmony(text: str) -> str:
    s = strip_all(text)
    has_m = any(ch in MASCULINE_VOWELS for ch in s)
    has_f = any(ch in FEMININE_VOWELS for ch in s)

    if has_m and not has_f:
        return "masculine"
    if has_f and not has_m:
        return "feminine"
    if not has_m and not has_f:
        return "neutral"
    return "mixed"


def harmony_ok(stem_harmony: str, suffix_harmony: str, declared_harmony: str) -> bool:
    if declared_harmony in {"none", "neutral"}:
        return True
    if stem_harmony in {"neutral", "mixed"}:
        return True
    if suffix_harmony in {"neutral", "mixed"}:
        return True
    if declared_harmony == "all_variants":
        return stem_harmony == suffix_harmony
    return stem_harmony == declared_harmony


LEGAL_OUTER = {
    "voice": {
        "voice",
        "tense",
        "mood",
        "aspect",
        "converb",
        "participle",
        "negation",
        "derivational",
    },
    "tense": {"negation", "possessive", "case"},
    "mood": {"negation", "possessive"},
    "aspect": {"tense", "mood", "converb", "participle", "negation"},
    "converb": {"possessive", "negation", "case"},
    "participle": {"case", "plural", "possessive", "negation", "derivational"},
    "plural": {"case", "possessive"},
    "case": {"possessive"},
    "possessive": {"case"},
    "negation": {"case", "possessive", "tense"},
    "particle": set(),
    "derivational": {
        "voice",
        "tense",
        "mood",
        "aspect",
        "converb",
        "participle",
        "plural",
        "case",
        "possessive",
        "negation",
        "derivational",
    },
}


def stacking_ok(inner_type: str, outer_type: str) -> bool:
    return outer_type in LEGAL_OUTER.get(inner_type, set())


def sequence_sane(types: list[str]) -> bool:
    """Reject obviously runaway suffix stacks while keeping duplicate surfaces legal."""
    limits = {
        "plural": 1,
        "case": 2,
        "possessive": 2,
    }
    for suffix_type, limit in limits.items():
        if types.count(suffix_type) > limit:
            return False
    return True
