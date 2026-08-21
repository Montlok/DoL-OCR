# -*- coding: utf-8 -*-

"""OCR accuracy metrics for traditional Mongolian.

The core measurement problem for Mongolian OCR is that the same *visual* text
can map to several different code-point sequences: free variation selectors
(FVS, U+180B-180D), the Mongolian vowel separator (MVS, U+180E), narrow no-break
space (NNBSP, U+202F) and positional/presentation variants. Comparing raw
code points therefore conflates genuine recognition errors with harmless
encoding/rendering differences.

A second, independent measurement problem is that a single character can be a
base letter plus one or more Unicode combining marks; comparing Python
code points then charges a single missed/extra mark as an error on top of
whatever else differs at that position, over-counting relative to what a
human reader perceives as one mistake.

We report three character error rates:

- **grapheme CER** (headline, ``OCRReport.grapheme_cer``): both sides are
  clustered into user-perceived graphemes (see :func:`grapheme_clusters`) on
  the RAW (unfolded) text, then compared. This is the number that best
  matches "how many visual mistakes did the model make" — it still counts
  genuine rendering-variant differences (unlike normalized CER below) but
  does not fragment one combining-mark miss into several code-point errors
  (unlike raw CER below).
- **normalized CER**: both prediction and reference are first folded by the
  repository's deterministic Unicode-only Python implementation, then compared.
  The fold strips FVS/zero-width transport noise and maps NNBSP to MVS.
- **raw CER**: compares the unmodified code points, exposing the true
  encoding gap.

Plus word accuracy (WER over whitespace tokens) and an exact line-match rate.

A single blended CER hides *where* the errors are: a checkpoint mistranscribing
every CJK gloss but nailing the Mongolian body text looks identical, in the
headline number, to the reverse. :func:`script_bucket_cer` (surfaced on
``OCRReport.script_cer``) splits grapheme CER by script — ``"mn"``/``"cjk"``/
``"latin"``/``"other"`` (see :func:`script_of`) — by projecting each side down
to one script's characters and scoring the projections independently; see
that function's docstring for the method's cross-script-substitution caveat.

``OCRReport.symbol_metrics`` provides a second, raw-codepoint diagnostic for
digits, punctuation, FVS, MVS, and NNBSP.  Unlike the script projection, it
backtraces the full Levenshtein alignment before attributing matches and edits,
so a selector or punctuation error remains tied to its position in the line.
FVS is reported both as one aggregate class and separately for FVS1--FVS4.

This module is intentionally torch-free so it can be unit-tested without loading
a model.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

# Code points that are pure encoding/rendering variation and must not count as
# recognition errors in nominal folding.
_FVS = {0x180B, 0x180C, 0x180D, 0x180F}  # free variation selectors (incl. FVS4)
_MVS = {0x180E}  # Mongolian vowel separator
_ZERO_WIDTH_NOISE = {0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF}
_NNBSP = 0x202F  # The canonical nominal fold maps NNBSP to MVS.

_FALLBACK_DELETE = _FVS | _ZERO_WIDTH_NOISE

# Categories that attach to the preceding base character as combining marks:
# nonspacing (Mn), spacing-combining (Mc), enclosing (Me). FVS is deliberately
# checked as an explicit code-point set rather than relying on its Unicode
# category: FVS4 (U+180F) was only formally assigned category Mn in a later
# Unicode revision, and depending on the Python interpreter's bundled
# `unicodedata` version it can report as Cn (unassigned) instead. Grapheme
# clustering must not silently split FVS4 off as its own grapheme just
# because an older Unicode database has not caught up — that would make the
# metric's behavior depend on the interpreter it happens to run under.
_COMBINING_CATEGORIES = ("Mn", "Mc", "Me")


def grapheme_clusters(text: str) -> list[str]:
    """Split ``text`` into user-perceived grapheme clusters (Mongolian-scoped).

    A cluster is a base character followed by any run of trailing combining
    marks: characters in ``_FVS`` (explicit set, see above) or characters
    whose ``unicodedata.category`` is one of Mn/Mc/Me. Everything else starts
    a new cluster.

    MVS (U+180E, category Cf — format control) and NNBSP (U+202F, category
    Zs — space separator) are deliberately treated as their own standalone
    clusters, not attached to a neighbor: both are morpheme/word boundary
    markers in Mongolian text, not decorations on the character next to them,
    so an OCR miss on either is exactly one grapheme error — not zero (if it
    silently attached) and not conflated with the neighboring letter.

    This is intentionally *not* a full UAX #29 extended-grapheme-cluster
    implementation: no emoji ZWJ-sequence handling, no Hangul jamo
    composition, no regional-indicator pairing. Traditional Mongolian OCR
    output does not produce those sequences, so the narrower FVS/Mn/Mc/Me
    rule above covers what this corpus actually needs without pulling in a
    dependency or a large exception table.
    """
    clusters: list[str] = []
    for ch in text:
        attaches = clusters and (
            ord(ch) in _FVS or unicodedata.category(ch) in _COMBINING_CATEGORIES
        )
        if attaches:
            clusters[-1] += ch
        else:
            clusters.append(ch)
    return clusters


def _python_fold(text: str) -> str:
    """Deterministic nominal fold for already-Unicode OCR text."""
    out = []
    for ch in text:
        cp = ord(ch)
        if cp in _FALLBACK_DELETE:
            continue
        if cp == _NNBSP:
            out.append("\u180e")
            continue
        out.append(ch)
    return "".join(out)


def nominal_normalize(
    texts: Sequence[str], *, backend: str = "auto"
) -> list[str]:
    """Fold ``texts`` to nominal Mongolian Unicode for normalized CER.

    ``backend`` remains for call compatibility. ``"auto"`` and ``"python"``
    both select the one deterministic Python implementation. Removed backends
    fail explicitly instead of silently changing evaluation semantics.
    """
    folded, _ = _fold_with_backend(texts, backend=backend)
    return folded


def _fold_with_backend(
    texts: Sequence[str], *, backend: str = "auto"
) -> tuple[list[str], str]:
    """Like :func:`nominal_normalize` but also return the implementation name."""
    texts = list(texts)
    if backend in {"auto", "python"}:
        return [_python_fold(t) for t in texts], "python"
    raise ValueError(
        f"unknown or removed nominal-normalization backend {backend!r}; "
        "use 'python' or 'auto'"
    )


def _fold_pair(
    preds: Sequence[str], refs: Sequence[str], *, backend: str = "auto"
) -> tuple[list[str], list[str], str]:
    """Fold ``preds`` and ``refs`` through a *single* backend decision.

    Concatenating before folding keeps the pair on one normalization decision
    and preserves the historical helper contract.
    """
    preds = list(preds)
    refs = list(refs)
    combined, used = _fold_with_backend(preds + refs, backend=backend)
    n = len(preds)
    return combined[:n], combined[n:], used


def edit_distance(a: Sequence, b: Sequence) -> int:
    """Levenshtein edit distance between two sequences (O(len(a)*len(b)) time,
    O(min) space)."""
    if a is b or a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def _levenshtein_alignment(
    pred: str,
    ref: str,
) -> list[tuple[str, str | None, str | None]]:
    """Return a deterministic minimum-edit raw-codepoint alignment.

    Each item is ``(operation, reference_character, prediction_character)``.
    ``operation`` is one of ``"match"``, ``"substitution"``, ``"deletion"``,
    or ``"insertion"``.  Exact diagonal matches are preferred during
    backtracing, followed by substitutions, deletions, and insertions.  The
    preference only resolves equal-cost paths; it never changes the
    Levenshtein distance.
    """

    n_ref, n_pred = len(ref), len(pred)
    distance = [[0] * (n_pred + 1) for _ in range(n_ref + 1)]
    for i in range(1, n_ref + 1):
        distance[i][0] = i
    for j in range(1, n_pred + 1):
        distance[0][j] = j
    for i, ref_ch in enumerate(ref, start=1):
        for j, pred_ch in enumerate(pred, start=1):
            substitution_cost = 0 if ref_ch == pred_ch else 1
            distance[i][j] = min(
                distance[i - 1][j] + 1,
                distance[i][j - 1] + 1,
                distance[i - 1][j - 1] + substitution_cost,
            )

    reverse_alignment: list[tuple[str, str | None, str | None]] = []
    i, j = n_ref, n_pred
    while i or j:
        if (
            i
            and j
            and ref[i - 1] == pred[j - 1]
            and distance[i][j] == distance[i - 1][j - 1]
        ):
            reverse_alignment.append(("match", ref[i - 1], pred[j - 1]))
            i -= 1
            j -= 1
            continue
        if i and j and distance[i][j] == distance[i - 1][j - 1] + 1:
            reverse_alignment.append(("substitution", ref[i - 1], pred[j - 1]))
            i -= 1
            j -= 1
            continue
        if i and distance[i][j] == distance[i - 1][j] + 1:
            reverse_alignment.append(("deletion", ref[i - 1], None))
            i -= 1
            continue
        if j and distance[i][j] == distance[i][j - 1] + 1:
            reverse_alignment.append(("insertion", None, pred[j - 1]))
            j -= 1
            continue
        raise RuntimeError("invalid Levenshtein backtrace state")

    reverse_alignment.reverse()
    return reverse_alignment


_SYMBOL_CLASS_NAMES = ("digit", "punctuation", "fvs", "mvs", "nnbsp")
_FVS_VARIANT_NAMES = {
    0x180B: "fvs1",
    0x180C: "fvs2",
    0x180D: "fvs3",
    0x180F: "fvs4",
}
_SYMBOL_COUNT_FIELDS = (
    "n_ref",
    "n_pred",
    "correct",
    "substitutions",
    "deletions",
    "insertions",
    "line_support",
)


def _new_symbol_counts() -> dict[str, int]:
    return {field: 0 for field in _SYMBOL_COUNT_FIELDS}


def _symbol_classes(ch: str) -> tuple[str, ...]:
    """Return non-overlapping base classes plus an exact FVS variant class."""

    cp = ord(ch)
    if cp in _FVS_VARIANT_NAMES:
        return ("fvs", _FVS_VARIANT_NAMES[cp])
    if cp in _MVS:
        return ("mvs",)
    if cp == _NNBSP:
        return ("nnbsp",)
    category = unicodedata.category(ch)
    if category == "Nd":
        return ("digit",)
    if category.startswith("P"):
        return ("punctuation",)
    return ()


def _finalize_symbol_counts(
    counts: dict[str, int],
) -> dict[str, int | float | None]:
    result: dict[str, int | float | None] = dict(counts)
    n_ref = counts["n_ref"]
    if n_ref == 0:
        result["error_rate"] = None
    else:
        errors = counts["substitutions"] + counts["deletions"] + counts["insertions"]
        result["error_rate"] = errors / n_ref
    return result


def symbol_metrics(
    preds: Sequence[str],
    refs: Sequence[str],
) -> dict[str, dict[str, object]]:
    """Measure raw Unicode symbol recognition from full line alignments.

    The five top-level classes are:

    - ``digit``: Unicode decimal digits (general category ``Nd``);
    - ``punctuation``: every Unicode punctuation category (``P*``);
    - ``fvs``: U+180B/U+180C/U+180D/U+180F;
    - ``mvs``: U+180E;
    - ``nnbsp``: U+202F.

    Every class reports ``n_ref``, ``n_pred``, ``correct``,
    ``substitutions``, ``deletions``, ``insertions``, ``error_rate``, and
    ``line_support``. ``line_support`` counts reference lines containing at
    least one member of the class. ``error_rate`` is ``None`` when ``n_ref`` is
    zero, leaving prediction-only insertions visible without manufacturing a
    denominator.

    A substitution is charged to the reference character's class. If the
    predicted character belongs to a *different* tracked class, it is also an
    insertion for that predicted class. This exposes cross-class false
    positives while counting an in-class substitution (for example ``1`` to
    ``2``) exactly once. The aggregate ``fvs`` entry contains a ``variants``
    mapping with the same fields for FVS1--FVS4.
    """

    preds = list(preds)
    refs = list(refs)
    if len(preds) != len(refs):
        raise ValueError(
            f"preds/refs length mismatch: {len(preds)} != {len(refs)}"
        )

    class_names = (*_SYMBOL_CLASS_NAMES, *_FVS_VARIANT_NAMES.values())
    totals = {name: _new_symbol_counts() for name in class_names}

    for pred, ref in zip(preds, refs):
        ref_support: set[str] = set()
        for ch in ref:
            for name in _symbol_classes(ch):
                totals[name]["n_ref"] += 1
                ref_support.add(name)
        for name in ref_support:
            totals[name]["line_support"] += 1
        for ch in pred:
            for name in _symbol_classes(ch):
                totals[name]["n_pred"] += 1

        for operation, ref_ch, pred_ch in _levenshtein_alignment(pred, ref):
            ref_classes = set(_symbol_classes(ref_ch)) if ref_ch is not None else set()
            pred_classes = (
                set(_symbol_classes(pred_ch)) if pred_ch is not None else set()
            )
            if operation == "match":
                for name in ref_classes:
                    totals[name]["correct"] += 1
            elif operation == "substitution":
                for name in ref_classes:
                    totals[name]["substitutions"] += 1
                for name in pred_classes - ref_classes:
                    totals[name]["insertions"] += 1
            elif operation == "deletion":
                for name in ref_classes:
                    totals[name]["deletions"] += 1
            elif operation == "insertion":
                for name in pred_classes:
                    totals[name]["insertions"] += 1
            else:  # pragma: no cover - private aligner has a closed operation set.
                raise RuntimeError(f"unknown alignment operation {operation!r}")

    result: dict[str, dict[str, object]] = {
        name: _finalize_symbol_counts(totals[name]) for name in _SYMBOL_CLASS_NAMES
    }
    result["fvs"]["variants"] = {
        name: _finalize_symbol_counts(totals[name])
        for name in _FVS_VARIANT_NAMES.values()
    }
    return result


def _corpus_rate(
    preds: Sequence[Sequence], refs: Sequence[Sequence]
) -> tuple[float, int, int]:
    """Micro-averaged error rate: sum(edit distance) / sum(max(len(ref), 1)).

    Using ``max(len(ref), 1)`` per sample ensures pure-insertion errors against
    an empty reference are still reflected (a plain ``sum(len(ref))`` denominator
    would add the insertions to the numerator but nothing to the denominator,
    silently under-penalizing them — and would be 0/0 if every ref were empty).
    """
    total_dist = 0
    total_len = 0
    for p, r in zip(preds, refs):
        total_dist += edit_distance(p, r)
        total_len += max(len(r), 1)
    rate = total_dist / total_len if total_len else 0.0
    return rate, total_dist, total_len


# Script-bucket boundaries for script_of/script_bucket_cer. Traditional
# Mongolian ranges: the main Mongolian block (U+1800-18AF, which also holds
# FVS/MVS) and the Mongolian Supplement block added for GB/T 25914-2023
# (U+11660-1167F). NNBSP (U+202F) is bucketed as "mn" too even though its code
# point is outside both blocks: it is the Mongolian-specific narrow no-break
# space used as a word-boundary marker inside Mongolian text (see NNBSP in the
# module docstring), so charging it to "other" would misattribute a
# Mongolian-text error to an unrelated bucket.
_MN_RANGES = ((0x1800, 0x18AF), (0x11660, 0x1167F))
_MN_EXTRA = _FVS | _MVS | {_NNBSP}

# CJK: unified ideograph blocks (BMP + extensions on common planes), CJK
# punctuation, and fullwidth forms (fullwidth Latin/digits/punctuation used in
# CJK typesetting count as "cjk", not "latin" — they are visually and
# functionally CJK-context characters).
_CJK_RANGES = (
    (0x2E80, 0x2EFF),  # CJK Radicals Supplement
    (0x3000, 0x303F),  # CJK Symbols and Punctuation
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0xFF00, 0xFFEF),  # Halfwidth and Fullwidth Forms
    (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
)

# Latin: ASCII letters/digits plus Latin-1 Supplement and Latin Extended-A/B
# letters (accented Latin text). ASCII punctuation/space/symbols are "other",
# not "latin" — only letters and digits count as this bucket's content.
_LATIN_RANGES = (
    (0x0041, 0x005A),  # ASCII A-Z
    (0x0061, 0x007A),  # ASCII a-z
    (0x0030, 0x0039),  # ASCII 0-9
    (0x00C0, 0x00FF),  # Latin-1 Supplement letters (excludes × U+00D7, ÷ U+00F7)
    (0x0100, 0x024F),  # Latin Extended-A + Extended-B
)
_LATIN_EXCLUDE = {0x00D7, 0x00F7}  # multiplication/division signs, not letters


def _in_ranges(cp: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(lo <= cp <= hi for lo, hi in ranges)


def script_of(ch: str) -> str:
    """Classify one character into a script bucket for error-rate breakdowns.

    Returns one of:

    - ``"mn"``: traditional Mongolian letters (main block + GB/T 25914-2023
      Supplement block) plus FVS/MVS/NNBSP, which only occur embedded in
      Mongolian text.
    - ``"cjk"``: CJK unified ideographs (+ common extensions/compatibility
      blocks), CJK punctuation, and fullwidth forms.
    - ``"latin"``: ASCII letters/digits and accented Latin letters
      (Latin-1 Supplement, Extended-A/B).
    - ``"other"``: everything else — ASCII punctuation, whitespace, symbols,
      and any script not called out above.

    Classifies a single ``str`` character (one Python code point), not a
    grapheme cluster; callers doing grapheme-unit bucketing classify by the
    cluster's first character (its base letter carries the script identity —
    combining marks/FVS do not change what script a cluster belongs to).
    """
    if len(ch) != 1:
        raise ValueError(f"script_of expects a single character, got {ch!r}")
    cp = ord(ch)
    if cp in _MN_EXTRA or _in_ranges(cp, _MN_RANGES):
        return "mn"
    if _in_ranges(cp, _CJK_RANGES):
        return "cjk"
    if cp not in _LATIN_EXCLUDE and _in_ranges(cp, _LATIN_RANGES):
        return "latin"
    return "other"


def _cluster_script(cluster: str) -> str:
    """Script bucket for a grapheme cluster: the base (first) character's script."""
    return script_of(cluster[0])


def _project_bucket(units: list, bucket: str, unit: str) -> list:
    """Keep only the elements of ``units`` (chars or grapheme clusters) whose
    script is ``bucket``, preserving order."""
    if unit == "grapheme":
        return [u for u in units if _cluster_script(u) == bucket]
    return [u for u in units if script_of(u) == bucket]


def script_bucket_cer(
    preds: Sequence[str],
    refs: Sequence[str],
    *,
    normalize: bool = True,
    backend: str = "auto",
    unit: str = "grapheme",
) -> dict[str, dict[str, float | int]]:
    """Per-script-bucket CER: where do the errors live (mn/cjk/latin/other)?

    For each bucket, both ``preds`` and ``refs`` are *projected* down to only
    the characters/clusters classified into that bucket by :func:`script_of`
    (order preserved within each sample), then scored with the same
    micro-averaged corpus rate as :func:`cer` (:func:`_corpus_rate`).

    This is a **filtering/projection** method, not an alignment of the full
    pred/ref edit path split by bucket. It is deterministic and needs no
    backtrace over the Levenshtein DP (this module has none), and it directly
    answers "how well does the model transcribe bucket-B content" — but it
    has one honest caveat: a cross-script substitution (e.g. a Mongolian
    letter OCR'd as a similar-looking CJK character) is *not* attributed to a
    single bucket's error count the way a true alignment would. Instead each
    side contributes its own character to its own bucket's projected
    sequence, so the error surfaces as an edit in *both* buckets' projections
    (a "phantom" deletion in one, insertion in the other) rather than being
    charged once to whichever bucket is "correct". For content that is
    overwhelmingly single-script per bucket (the expected case for this
    corpus) this does not matter; for heavily-interleaved or confused
    cross-script content this can inflate two buckets' error rates for what
    is really one mistake. Buckets with zero reference units are omitted
    entirely (there is nothing to compute a rate over) rather than reported
    as a misleading 0.0.

    ``unit``: ``"grapheme"`` (default) buckets on :func:`grapheme_clusters`
    output, classifying each cluster by its base character's script (see
    :func:`_cluster_script`) — this matches the ``grapheme_cer`` headline
    convention. ``"codepoint"`` buckets on raw characters instead.

    Returns ``{bucket: {"cer": float, "n_ref": int}}``, one entry per bucket
    with ``n_ref > 0`` among ``"mn"``, ``"cjk"``, ``"latin"``, ``"other"``.
    """
    if unit not in ("codepoint", "grapheme"):
        raise ValueError(f"unknown unit {unit!r}; expected 'codepoint' or 'grapheme'")
    if normalize:
        preds, refs, _ = _fold_pair(preds, refs, backend=backend)
    else:
        preds, refs = list(preds), list(refs)

    if unit == "grapheme":
        pred_units = [grapheme_clusters(p) for p in preds]
        ref_units = [grapheme_clusters(r) for r in refs]
    else:
        pred_units = [list(p) for p in preds]
        ref_units = [list(r) for r in refs]

    return _bucket_cer_from_units(pred_units, ref_units, unit=unit)


def _bucket_cer_from_units(
    pred_units: list[list[str]], ref_units: list[list[str]], *, unit: str
) -> dict[str, dict[str, float | int]]:
    """Shared core of :func:`script_bucket_cer`: bucket already-split
    (grapheme or codepoint) unit lists and score each bucket's projection.

    Split out so :func:`ocr_report` can reuse the grapheme clusters it
    already computed for ``grapheme_cer`` instead of re-clustering the same
    raw text a second time.
    """
    out: dict[str, dict[str, float | int]] = {}
    for bucket in ("mn", "cjk", "latin", "other"):
        bucket_preds = [_project_bucket(p, bucket, unit) for p in pred_units]
        bucket_refs = [_project_bucket(r, bucket, unit) for r in ref_units]
        n_ref = sum(len(r) for r in bucket_refs)
        if n_ref == 0:
            continue
        rate, _, _ = _corpus_rate(bucket_preds, bucket_refs)
        out[bucket] = {"cer": rate, "n_ref": n_ref}
    return out


def cer(
    preds: Sequence[str],
    refs: Sequence[str],
    *,
    normalize: bool = True,
    backend: str = "auto",
    unit: str = "codepoint",
) -> float:
    """Corpus character error rate. When ``normalize`` is true, both sides are
    folded to nominal Unicode first (the primary, render-robust metric).

    ``unit``: ``"codepoint"`` (default) compares raw Python characters.
    ``"grapheme"`` clusters each side with :func:`grapheme_clusters` *after*
    the optional fold and compares clusters instead — a missed/extra
    combining mark then counts as one error, not one error per constituent
    code point. Any other value raises ``ValueError``.
    """
    if unit not in ("codepoint", "grapheme"):
        raise ValueError(f"unknown unit {unit!r}; expected 'codepoint' or 'grapheme'")
    if normalize:
        preds, refs, _ = _fold_pair(preds, refs, backend=backend)
    if unit == "grapheme":
        preds = [grapheme_clusters(p) for p in preds]
        refs = [grapheme_clusters(r) for r in refs]
    rate, _, _ = _corpus_rate(preds, refs)
    return rate


def wer(
    preds: Sequence[str],
    refs: Sequence[str],
    *,
    normalize: bool = True,
    backend: str = "auto",
) -> float:
    """Corpus word error rate over whitespace-split tokens."""
    if normalize:
        preds, refs, _ = _fold_pair(preds, refs, backend=backend)
    rate, _, _ = _corpus_rate([p.split() for p in preds], [r.split() for r in refs])
    return rate


@dataclass
class OCRReport:
    n: int
    norm_cer: float
    raw_cer: float
    grapheme_cer: float
    wer: float
    line_exact: float
    raw_line_exact: float
    normalized_line_exact: float
    rejection_rate: float
    backend: str
    script_cer: dict[str, dict[str, float | int]] | None = None
    symbol_metrics: dict[str, dict[str, object]] | None = None


def ocr_report(
    preds: Sequence[str],
    refs: Sequence[str],
    *,
    backend: str = "auto",
    rejected: Sequence[bool] | None = None,
) -> OCRReport:
    """Full OCR quality report over aligned ``preds``/``refs``.

    ``rejected`` (optional): per-sample mask of predictions withheld by a
    confidence gate; only the *kept* samples are scored, and the rejection rate
    is reported separately (high-precision corpus ingestion).
    """
    preds = list(preds)
    refs = list(refs)
    if len(preds) != len(refs):
        raise ValueError(f"preds/refs length mismatch: {len(preds)} != {len(refs)}")
    total = len(preds)
    if rejected is not None:
        rejected = list(rejected)
        if len(rejected) != total:
            raise ValueError("rejected mask length must match preds")
        keep = [i for i in range(total) if not rejected[i]]
    else:
        keep = list(range(total))
    rejection_rate = (total - len(keep)) / total if total else 0.0

    kp = [preds[i] for i in keep]
    kr = [refs[i] for i in keep]

    norm_p, norm_r, used_backend = _fold_pair(kp, kr, backend=backend)

    norm_cer, _, _ = _corpus_rate(norm_p, norm_r)
    raw_cer, _, _ = _corpus_rate(kp, kr)
    # Grapheme CER is the OCR headline number: clustered on the RAW (unfolded)
    # text, so it reflects what the model actually emitted (rendering-variant
    # differences still count, unlike norm_cer) while still not double-charging
    # a single missed combining mark as multiple code-point errors (unlike
    # raw_cer).
    grapheme_p = [grapheme_clusters(p) for p in kp]
    grapheme_r = [grapheme_clusters(r) for r in kr]
    grapheme_cer_rate, _, _ = _corpus_rate(grapheme_p, grapheme_r)
    wer_rate, _, _ = _corpus_rate(
        [p.split() for p in norm_p], [r.split() for r in norm_r]
    )
    normalized_exact = sum(1 for p, r in zip(norm_p, norm_r) if p == r)
    normalized_line_exact = normalized_exact / len(keep) if keep else 0.0
    raw_exact = sum(1 for p, r in zip(kp, kr) if p == r)
    raw_line_exact = raw_exact / len(keep) if keep else 0.0
    # Reuse the same raw-text grapheme clusters as grapheme_cer above (same
    # convention: unfolded text) rather than re-clustering via a second
    # script_bucket_cer(..., unit="grapheme") call.
    script_cer_map = _bucket_cer_from_units(grapheme_p, grapheme_r, unit="grapheme")
    symbol_metrics_map = symbol_metrics(kp, kr)

    return OCRReport(
        n=len(keep),
        norm_cer=norm_cer,
        raw_cer=raw_cer,
        grapheme_cer=grapheme_cer_rate,
        wer=wer_rate,
        # Compatibility alias: historically line_exact was normalized.
        line_exact=normalized_line_exact,
        raw_line_exact=raw_line_exact,
        normalized_line_exact=normalized_line_exact,
        rejection_rate=rejection_rate,
        backend=used_backend,
        script_cer=script_cer_map,
        symbol_metrics=symbol_metrics_map,
    )


__all__ = [
    "OCRReport",
    "cer",
    "edit_distance",
    "grapheme_clusters",
    "nominal_normalize",
    "ocr_report",
    "script_bucket_cer",
    "script_of",
    "symbol_metrics",
    "wer",
]
