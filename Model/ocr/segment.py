# -*- coding: utf-8 -*-

"""Ink-profile page segmentation for vertical traditional-Mongolian pages.

Single implementation, by mandate: deployment consumers
(:mod:`scripts.ocr_infer`) and the L2 deployment evaluator
(:mod:`scripts.eval_l2_deploy`) MUST segment through the functions in this
module — never a private copy — so the CER the evaluator reports is produced
by exactly the segmentation the product ships. The annotation-pack builder
(:mod:`scripts.build_annotation_pack`) shares the same detection core through
:func:`detect_line_columns`.

Reading order: traditional Mongolian runs top-to-bottom within a column, and
columns advance **left-to-right** across the page. In-repo evidence: the
production page renderer uses CSS ``writing-mode: vertical-lr``
(:mod:`scripts.render_mn_pages`, whose block-progression direction is
left-to-right by the CSS Writing Modes spec), and
:func:`scripts.build_ocr_data.render_vertical_line` states "layout columns
left-to-right is the script's natural flow". Boxes returned here are ordered
left-to-right; consumers apply their own ``--column-order`` override.

Two distinct concerns live here, deliberately:

- **Detection** (:func:`detect_columns`, wrapped by
  :func:`detect_line_columns`): find the vertical text columns via the page's
  smoothed per-x ink profile; valleys (near-zero smoothed ink) are the
  inter-column gutters. A detected column is one text line of the script.
- **Scale normalization** (:func:`lines_from_column`): the OCR model trains
  on ~64x400-900 px strips (see :mod:`scripts.build_ocr_data_from_pairs`); a
  full-page column can be far taller, and letterboxing it whole would shrink
  glyphs below the training scale. ``lines_from_column`` cuts one tall column
  into strips of roughly ``target_px`` height along the writing axis,
  preferring row-ink valleys (whitespace between words) so glyphs are only
  bisected when no valley exists in the allowed window. This is *not*
  horizontal-script "line detection within a column" — it is chunking along
  the script's own flow, at whitespace when possible.
  :func:`chunk_column_by_height` is the arithmetic-only variant the
  annotation-pack builder uses where deterministic, content-independent chunk
  geometry matters more than clean cuts.

All detection is torch-free (numpy; 2-D grayscale arrays, white background,
dark ink, uint8 0..255 convention) and returns integer boxes, not crops, so
callers keep provenance of every strip.

Known failure modes (tune the knobs rather than adding code):

- A background darker than ``ink_threshold`` reads as solid ink: one giant
  column, hard cuts every ``target_px`` rows. Binarize or raise
  ``ink_threshold`` upstream for such scans.
- A column with no interior valley (dense text, no word gaps wider than the
  smoothing window) falls back to hard cuts at ``target_px``, which can
  bisect a glyph. Included in L2 scores by design.
- Skewed scans smear the gutters; deskew upstream — no rotation estimation
  here.
"""

from __future__ import annotations

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised only without numpy
    np = None  # type: ignore[assignment]
    _NUMPY_IMPORT_ERROR: Exception | None = exc
else:
    _NUMPY_IMPORT_ERROR = None

Box = tuple[int, int, int, int]  # (x0, y0, x1, y1), pixel coords, half-open
Span = tuple[int, int]  # (start, end), half-open


def _require_numpy() -> None:
    if np is None:
        raise ImportError(
            "numpy is required for Model.ocr.segment (it ships with the "
            "[model] extra's torch install)."
        ) from _NUMPY_IMPORT_ERROR


def _ink_mask(arr, ink_threshold: int):
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError(
            f"expected a 2-D grayscale array (white bg, dark ink), got ndim={a.ndim}"
        )
    return a < ink_threshold


def _moving_average(profile, window: int):
    profile = profile.astype(np.float64)
    if window <= 1:
        return profile
    window = int(window) | 1  # force odd so mode="same" stays centered
    kernel = np.full(window, 1.0 / window)
    return np.convolve(profile, kernel, mode="same")


def _runs(mask) -> list[Span]:
    """Half-open ``(start, end)`` spans of consecutive True in a 1-D mask."""
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def detect_columns(
    page,
    *,
    ink_threshold: int = 200,
    smooth: int = 9,
    valley_frac: float = 0.05,
    min_width: int = 12,
    min_gap: int = 6,
    min_ink_count: float = 0.0,
) -> list[Box]:
    """Find vertical text columns on a page via column-ink-profile valleys.

    Args:
        page: 2-D grayscale array ``[H, W]``, white background, dark ink.
        ink_threshold: pixels strictly below this count as ink.
        smooth: moving-average window (px) applied to the ink profile.
        valley_frac: a profile position is a valley when its smoothed ink
            count is <= ``valley_frac * profile.max()``.
        min_width: discard detected columns narrower than this (px).
        min_gap: valleys narrower than this do not split a column — the two
            ink runs around them are merged (bridges anti-aliasing slivers).
        min_ink_count: absolute floor on the valley threshold, in smoothed
            ink-pixel counts (used by :func:`detect_line_columns` to
            reproduce its historical absolute-density semantics; leave 0 for
            the relative rule alone).

    Returns:
        Full-height boxes ``(x0, y0, x1, y1)`` (half-open, ``y0=0``,
        ``y1=H``), ordered left to right. A blank page returns ``[]``.
    """
    _require_numpy()
    if smooth < 1:
        raise ValueError("smooth must be >= 1")
    if not 0.0 <= valley_frac < 1.0:
        raise ValueError("valley_frac must be in [0, 1)")
    if min_width < 1:
        raise ValueError("min_width must be >= 1")
    if min_gap < 0:
        raise ValueError("min_gap must be >= 0")
    if min_ink_count < 0:
        raise ValueError("min_ink_count must be >= 0")

    mask = _ink_mask(page, ink_threshold)
    height = mask.shape[0]
    smoothed = _moving_average(mask.sum(axis=0), smooth)
    peak = float(smoothed.max(initial=0.0))
    if peak <= 0.0:
        return []
    ink_cols = smoothed > max(valley_frac * peak, float(min_ink_count))

    merged: list[Span] = []
    for start, end in _runs(ink_cols):
        if merged and start - merged[-1][1] < min_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return [
        (int(x0), 0, int(x1), int(height))
        for x0, x1 in merged
        if x1 - x0 >= min_width
    ]


def lines_from_column(
    column,
    target_px: int,
    *,
    ink_threshold: int = 200,
    smooth: int = 5,
    valley_frac: float = 0.05,
    min_height: int = 8,
    overshoot: float = 1.35,
    pad: int = 2,
) -> list[Span]:
    """Cut one tall column into model-sized line strips at row-ink valleys.

    Walks the column top to bottom. Each strip may grow to at most
    ``target_px * overshoot`` rows; the cut lands on the deepest row-ink
    valley inside the allowed window (ties broken toward ``target_px``), or,
    when the window holds no valley at all, hard-cuts at ``target_px``.
    Each strip is then trimmed to its ink extent plus ``pad`` rows.

    Args:
        column: 2-D grayscale array ``[H, W]`` of one column (white bg).
        target_px: preferred strip height in source pixels. Choose it so a
            strip letterboxed to the model's square keeps glyphs near the
            training scale (training strips were ~400-900 px tall).
        ink_threshold/smooth/valley_frac: as in :func:`detect_columns`
            (profile is per-row here).
        min_height: strips shorter than this after trimming are dropped.
        overshoot: window factor; ``target_px * overshoot`` must exceed
            ``min_height``.
        pad: white rows kept above/below each trimmed strip.

    Returns:
        ``(y0, y1)`` half-open row spans relative to ``column``, in reading
        order (top to bottom). A blank column returns ``[]``.
    """
    _require_numpy()
    if target_px < 1:
        raise ValueError("target_px must be >= 1")
    if smooth < 1:
        raise ValueError("smooth must be >= 1")
    if not 0.0 <= valley_frac < 1.0:
        raise ValueError("valley_frac must be in [0, 1)")
    if min_height < 1:
        raise ValueError("min_height must be >= 1")
    if overshoot < 1.0:
        raise ValueError("overshoot must be >= 1.0")
    if pad < 0:
        raise ValueError("pad must be >= 0")
    max_len = max(int(round(target_px * overshoot)), 1)
    if max_len <= min_height:
        raise ValueError("target_px * overshoot must exceed min_height")

    mask2d = _ink_mask(column, ink_threshold)
    n_rows = mask2d.shape[0]
    smoothed = _moving_average(mask2d.sum(axis=1), smooth)
    peak = float(smoothed.max(initial=0.0))
    if peak <= 0.0:
        return []
    thr = valley_frac * peak
    ink_rows = smoothed > thr
    # Smoothing exists to make valley detection robust; extent measurement
    # (per-strip trim + min_height dust filtering) must use the raw mask, or
    # the window would smear a 2px speck into a strip that clears min_height.
    raw_ink_rows = mask2d.any(axis=1)
    runs = _runs(ink_rows)
    if not runs:
        return []
    content_start, content_end = runs[0][0], runs[-1][1]

    cuts: list[Span] = []
    y = content_start
    while content_end - y > max_len:
        lo = y + min_height
        hi = y + max_len  # < content_end by the loop condition
        window = smoothed[lo:hi]
        valley_pos = np.flatnonzero(window <= thr)
        if valley_pos.size:
            depths = window[valley_pos]
            deepest = valley_pos[depths <= depths.min() + 1e-12]
            cut = int(deepest[np.argmin(np.abs(deepest + lo - (y + target_px)))]) + lo
        else:
            cut = y + target_px  # no valley in window: hard cut (documented)
        cuts.append((y, cut))
        rest = np.flatnonzero(ink_rows[cut:content_end])
        if rest.size == 0:
            y = content_end
            break
        y = cut + int(rest[0])
    if content_end > y:
        cuts.append((y, content_end))

    boxes: list[Span] = []
    for a, b in cuts:
        seg = np.flatnonzero(raw_ink_rows[a:b])
        if seg.size == 0:
            continue
        y0 = max(a + int(seg[0]) - pad, 0)
        y1 = min(a + int(seg[-1]) + 1 + pad, n_rows)
        if y1 - y0 >= min_height:
            boxes.append((y0, y1))
    return boxes


def detect_line_columns(
    page_img,
    *,
    min_ink_density: float = 0.02,
    min_col_width_px: int = 8,
    min_gap_px: int = 3,
) -> list[Box]:
    """PIL-facing wrapper over :func:`detect_columns` for the annotation pack.

    Behavioral contract kept from the original implementation (see
    :mod:`scripts.build_annotation_pack`): PIL image in; ink is ``gray <
    160``; a pixel column is ink-bearing when its dark-pixel *fraction*
    exceeds ``min_ink_density`` (absolute rule — mapped onto
    ``detect_columns``'s ``min_ink_count`` floor as ``min_ink_density *
    height``, with no smoothing and no relative-valley rule); runs separated
    by a gap <= ``min_gap_px`` merge; and a page with no qualifying run falls
    back to the whole page as a single column (noisy/low-contrast scans),
    rather than returning ``[]``.
    """
    _require_numpy()
    from PIL import Image

    if not isinstance(page_img, Image.Image):
        raise TypeError("page_img must be a PIL.Image.Image")
    if min_col_width_px < 1:
        raise ValueError("min_col_width_px must be >= 1")

    gray = np.asarray(page_img.convert("L"))
    height, width = gray.shape
    if width == 0 or height == 0:
        return []
    boxes = detect_columns(
        gray,
        ink_threshold=160,
        smooth=1,
        valley_frac=0.0,
        min_width=min_col_width_px,
        min_gap=min_gap_px + 1,  # detect_columns merges strictly-narrower gaps
        min_ink_count=min_ink_density * height,
    )
    if not boxes:
        # Whole-page fallback: noisy scan with no clean column gaps found.
        return [(0, 0, width, height)]
    return boxes


def chunk_column_by_height(col_box: Box, *, max_chunk_px: int = 900) -> list[Box]:
    """Split a column box into height-bounded chunks, top-to-bottom.

    Pure arithmetic slicing — deliberately NOT ink-gap detection: the
    annotation-pack builder needs deterministic, content-independent chunk
    geometry (stable ids, reproducible sampling), so it trades clean cuts for
    determinism. The deployment path uses :func:`lines_from_column` instead,
    which cuts at row-ink valleys. A column shorter than ``max_chunk_px`` is
    returned unchanged as a single chunk. The final chunk may be shorter than
    ``max_chunk_px`` (no padding, no dropped remainder).
    """
    if max_chunk_px < 1:
        raise ValueError("max_chunk_px must be >= 1")
    x0, y0, x1, y1 = col_box
    if y1 <= y0:
        raise ValueError(f"col_box must have y1 > y0, got {col_box!r}")
    height = y1 - y0
    if height <= max_chunk_px:
        return [col_box]

    chunks: list[Box] = []
    y = y0
    while y < y1:
        y_end = min(y + max_chunk_px, y1)
        chunks.append((x0, y, x1, y_end))
        y = y_end
    return chunks


__all__ = [
    "Box",
    "Span",
    "chunk_column_by_height",
    "detect_columns",
    "detect_line_columns",
    "lines_from_column",
]
