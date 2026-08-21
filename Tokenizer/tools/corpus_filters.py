#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Corpus-wide deduplication and quality filtering for the training mix.

The per-source cleaning in ``build_corpus_mix.py`` (``min_chars``,
``strip_url_lines``, ``keep_lines``) operates on one document at a time and does
not look across documents or apply quality heuristics. This module adds the two
missing pieces, designed to run as a pre-stage *after* per-source cleaning and
*before* token-weighted mixing:

  * :class:`Deduper` -- global exact (SHA-1) + near-duplicate (SimHash + banded
    LSH) document dedup. Short, boilerplate-heavy documents fall back to
    exact-only matching so distinct stubs that share a long masthead are not
    wrongly merged.
  * :func:`quality_ok` -- a Gopher/C4-style heuristic filter (length, script
    purity, repeated-line ratio, symbol-to-word ratio, mean word length,
    boilerplate blocklist) returning ``(ok, reason)``.

Everything is dependency-free (stdlib only) and deterministic so the pipeline
stays reproducible and the unit tests are stable.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

# --- script ranges (kept in sync with build_corpus_mix._SCRIPT_RANGES) --------
_SCRIPT_RANGES = {
    "zh": ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF)),
    "mongolian": ((0x1800, 0x18AF), (0x11660, 0x1167F)),
    "cyrillic": ((0x0400, 0x04FF), (0x0500, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)),
}

_WS_RE = re.compile(r"\s+")
_MONG_WORD_RE = re.compile(r"[\u1800-\u18AF\U00011660-\U0001167F]+")
_WORD_RE = re.compile(r"\S+")
# CJK / kana are not space-delimited: a whole Chinese sentence is one ``\S+``
# run, which breaks whitespace-based word metrics. Tokenize such scripts per
# character so mean-word-length / symbol-per-word stay meaningful.
_CJK = r"\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\u3040-\u309F\u30A0-\u30FF"
_TOKEN_RE = re.compile(rf"[{_CJK}]|[^\s{_CJK}]+")
_CJK_RE = re.compile(rf"[{_CJK}]")


def _tokens(text: str) -> list[str]:
    """Script-aware word tokens: CJK/kana per-char, others whitespace-delimited."""

    return _TOKEN_RE.findall(text)


def _dedup_terms(text: str) -> list[str]:
    """Terms used for SimHash; preserve Mongolian word matching, fallback broad."""

    mongolian_words = _MONG_WORD_RE.findall(text)
    return mongolian_words if mongolian_words else _tokens(text)


# ===========================================================================
# Deduplication
# ===========================================================================
SIMHASH_BITS = 64
BANDS = 4
BAND_BITS = SIMHASH_BITS // BANDS
BAND_MASK = (1 << BAND_BITS) - 1


def normalize(text: str) -> str:
    """Collapse whitespace so trivial formatting changes don't defeat dedup."""

    return _WS_RE.sub(" ", text or "").strip()


def exact_key(text: str) -> str:
    """SHA-1 hex of the normalized text (verbatim-repost fingerprint)."""

    return hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()


def _shingles(text: str, k: int = 4):
    """k-word Mongolian shingles; falls back to single words for short docs."""

    words = _dedup_terms(text)
    if len(words) < k:
        return [" ".join(words)] if words else []
    return [" ".join(words[i : i + k]) for i in range(len(words) - k + 1)]


def simhash(text: str) -> int:
    """64-bit SimHash over word shingles (Charikar)."""

    shingles = _shingles(text)
    if not shingles:
        return 0
    votes = [0] * SIMHASH_BITS
    for sh in shingles:
        h = int.from_bytes(
            hashlib.blake2b(sh.encode("utf-8"), digest_size=8).digest(), "big"
        )
        for b in range(SIMHASH_BITS):
            votes[b] += 1 if (h >> b) & 1 else -1
    out = 0
    for b in range(SIMHASH_BITS):
        if votes[b] > 0:
            out |= 1 << b
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _bands(sig: int):
    return [(i, (sig >> (i * BAND_BITS)) & BAND_MASK) for i in range(BANDS)]


class Deduper:
    """Persistable exact + near-dup filter.

    ``seen(text)`` returns True when ``text`` is NEW (and records it) or False
    when it duplicates / near-duplicates something already seen. ``thresh`` is
    the max SimHash Hamming distance treated as a near-dup; with 4 bands the
    pigeonhole principle guarantees any pair within ``thresh <= 3`` collides in
    at least one band, keeping candidate lookup sub-linear.
    """

    def __init__(self, thresh: int = 3, min_simhash_words: int = 120):
        if not 0 <= thresh < SIMHASH_BITS:
            raise ValueError("thresh out of range")
        self.thresh = thresh
        self.min_simhash_words = min_simhash_words
        self._exact: set[str] = set()
        self._sigs: list[int] = []
        self._buckets: dict[str, list[int]] = {}

    # ---- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "thresh": self.thresh,
            "min_simhash_words": self.min_simhash_words,
            "exact": sorted(self._exact),
            "sigs": [str(s) for s in self._sigs],
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> "Deduper":
        dd = cls(
            thresh=int((d or {}).get("thresh", 3)),
            min_simhash_words=int((d or {}).get("min_simhash_words", 120)),
        )
        if not d:
            return dd
        dd._exact = set(d.get("exact", []))
        for s in d.get("sigs", []):
            dd._index(int(s))
        return dd

    # ---- internals ---------------------------------------------------------
    def _index(self, sig: int) -> None:
        idx = len(self._sigs)
        self._sigs.append(sig)
        for b, val in _bands(sig):
            self._buckets.setdefault(f"{b}:{val}", []).append(idx)

    def _is_near_dup(self, sig: int) -> bool:
        seen_candidates: set[int] = set()
        for b, val in _bands(sig):
            for idx in self._buckets.get(f"{b}:{val}", ()):
                if idx in seen_candidates:
                    continue
                seen_candidates.add(idx)
                if hamming(sig, self._sigs[idx]) <= self.thresh:
                    return True
        return False

    # ---- public API --------------------------------------------------------
    def seen(self, text: str) -> bool:
        """Record ``text``; return True if NEW, False if a (near-)duplicate."""

        key = exact_key(text)
        if key in self._exact:
            return False
        self._exact.add(key)
        if len(_dedup_terms(text)) < self.min_simhash_words:
            # short / boilerplate-heavy: exact-only to avoid false merges
            return True
        sig = simhash(text)
        if self._sigs and self._is_near_dup(sig):
            return False
        self._index(sig)
        return True

    def __len__(self) -> int:
        return len(self._sigs)


# ===========================================================================
# Quality filtering
# ===========================================================================
_DEFAULT_BOILERPLATE = (
    "京ICP备",
    "蒙ICP备",
    "版权所有",
    "All Rights Reserved",
    "未经授权",
)


def script_ratio(text: str, script: str) -> float:
    """Fraction of non-space chars that belong to ``script`` (0..1)."""

    ranges = _SCRIPT_RANGES[script]
    non_space = [ch for ch in text if not ch.isspace()]
    if not non_space:
        return 0.0
    hit = sum(any(lo <= ord(ch) <= hi for lo, hi in ranges) for ch in non_space)
    return hit / len(non_space)


def repeated_line_fraction(text: str) -> float:
    """Fraction of non-empty lines that are exact duplicates of an earlier one."""

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return 0.0
    seen: set[str] = set()
    dup = 0
    for ln in lines:
        if ln in seen:
            dup += 1
        else:
            seen.add(ln)
    return dup / len(lines)


def symbol_to_word_ratio(text: str) -> float:
    """Ratio of 'junk' symbol chars (#, …, bullets, pipes) to word count."""

    words = _tokens(text)
    if not words:
        return 1.0
    symbols = sum(text.count(c) for c in "#…•·|►▶■□▪")
    return symbols / len(words)


def mean_word_len(text: str) -> float | None:
    """Mean length of space-delimited words, or ``None`` if not applicable.

    The mean-word-length heuristic only makes sense for space-delimited scripts
    (Latin, Mongolian). CJK text has no word spaces, so a whole paragraph is one
    ``\\S+`` token; we exclude tokens containing CJK and return ``None`` when too
    few Latin/Mongolian words remain for the check to be meaningful.
    """

    words = [
        w
        for w in _WORD_RE.findall(text)
        if not _CJK_RE.search(w) and any(ch.isalpha() for ch in w)
    ]
    if len(words) < 5:
        return None
    return sum(len(w) for w in words) / len(words)


@dataclass
class QualityConfig:
    """Thresholds for :func:`quality_ok`. Defaults are permissive; tune per run.

    ``script`` + ``min_script_ratio`` enforce language purity (e.g. require a
    document to be mostly traditional Mongolian). Leave ``script`` empty to skip
    the purity check.
    """

    min_chars: int = 40
    script: str = ""
    min_script_ratio: float = 0.5
    max_repeated_line_fraction: float = 0.4
    max_symbol_to_word_ratio: float = 0.5
    min_mean_word_len: float = 1.5
    max_mean_word_len: float = 40.0
    boilerplate: tuple[str, ...] = field(default_factory=lambda: _DEFAULT_BOILERPLATE)


def quality_ok(text: str, cfg: QualityConfig) -> tuple[bool, str]:
    """Return ``(ok, reason)``; ``reason`` is "" when the doc passes."""

    t = text or ""
    if len(t.strip()) < cfg.min_chars:
        return False, "too_short"
    if cfg.script:
        if cfg.script not in _SCRIPT_RANGES:
            raise ValueError(f"unknown script {cfg.script!r}")
        if script_ratio(t, cfg.script) < cfg.min_script_ratio:
            return False, "script_impurity"
    if repeated_line_fraction(t) > cfg.max_repeated_line_fraction:
        return False, "repeated_lines"
    if symbol_to_word_ratio(t) > cfg.max_symbol_to_word_ratio:
        return False, "symbol_spam"
    mwl = mean_word_len(t)
    if mwl is not None and (mwl < cfg.min_mean_word_len or mwl > cfg.max_mean_word_len):
        return False, "word_len"
    low = t
    for bp in cfg.boilerplate:
        if bp in low:
            return False, "boilerplate"
    return True, ""
