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

Real scanned-book input (as opposed to the clean synthetic ink bars the
detectors above were tuned against) adds two failure modes neither
``ink_threshold`` nor ``valley_frac`` can absorb, because they are not
gutter/glyph problems, they are page-geometry problems:

- **Scan borders**: a solid black strip from the scanner bed or a page-edge
  shadow reads as a very-high-ink-density region that is not a text column
  at all. :func:`trim_scan_borders` strips these from a page BEFORE column
  detection runs, so they never enter the ink profile that
  :func:`detect_columns` smooths over.
- **Implausible line crops**: near-blank slivers (a column-detection sliver
  landing on margin whitespace) and solid ink bars (a border/gutter-shadow
  region that survived detection) are geometrically distinguishable from
  real text at the crop level — real glyph strokes cover a narrow ink-
  fraction band and have plausible width/height, borders and blanks do not.
  :func:`is_plausible_line` is that per-crop check;
  :func:`lines_from_column` applies it inline (as does the annotation-pack
  builder, on its own :func:`chunk_column_by_height` output) and reports
  rejections by reason so a bad batch is visible in counts, not silently
  half-empty.
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


def trim_scan_borders(
    page,
    *,
    ink_threshold: int = 200,
    border_ink_frac: float = 0.5,
    safety_margin: int = 3,
    max_trim_frac: float = 0.15,
) -> tuple:
    """Strip solid scan-border/scanner-bed rows and columns from a page edge.

    Real scanned books carry artifacts a clean synthetic ink bar never has:
    a solid black strip from the scanner bed, a gutter shadow, or a page-edge
    smear. Each reads as a near-100%-ink edge run that is not text, and left
    in place it either seeds a false column (:func:`detect_columns` finds a
    "column" that is actually the border) or, worse, IS the whole detected
    column when it dominates the ink profile.

    Iteratively inspects the outermost row/column on each of the four edges;
    while its ink fraction exceeds ``border_ink_frac`` it is trimmed, one
    row/column at a time, alternating edges so a page bordered on all four
    sides trims evenly rather than eating one side down to the cap before
    touching the others. An additional ``safety_margin`` rows/columns are
    trimmed once a border stops qualifying (the anti-aliased fringe just
    inside a hard border is usually still darker than body-text margins).
    Trimming per side is capped at ``max_trim_frac`` of that side's original
    extent, so a dense-text page (or a page with no border at all) cannot
    have its actual content eaten: a page with no qualifying border edge is a
    no-op and returns the input unchanged (as a copy-free view where
    possible).

    Args:
        page: 2-D grayscale array ``[H, W]``, white background, dark ink.
        ink_threshold: pixels strictly below this count as ink (as in
            :func:`detect_columns`).
        border_ink_frac: an edge row/column qualifies as border while its
            ink fraction (dark pixels / length) exceeds this.
        safety_margin: extra rows/columns trimmed past the last qualifying
            border row/column, per side.
        max_trim_frac: hard cap on trim depth per side, as a fraction of
            that side's original height/width — protects page content from
            an over-eager border rule on an unusually dense page.

    Returns:
        ``(cropped, (y_offset, x_offset))`` where ``cropped`` is the
        border-trimmed array and ``(y_offset, x_offset)`` is the top-left
        corner of ``cropped`` in the original page's coordinates (add it back
        to any box detected in ``cropped`` to recover original-page
        coordinates).
    """
    _require_numpy()
    if not 0.0 < border_ink_frac <= 1.0:
        raise ValueError("border_ink_frac must be in (0, 1]")
    if safety_margin < 0:
        raise ValueError("safety_margin must be >= 0")
    if not 0.0 <= max_trim_frac < 0.5:
        raise ValueError("max_trim_frac must be in [0, 0.5)")

    mask = _ink_mask(page, ink_threshold)
    height, width = mask.shape
    cap_h = int(height * max_trim_frac)
    cap_w = int(width * max_trim_frac)

    # Track *amount trimmed so far* per side (not an absolute boundary), so
    # "trimmed nothing" is unambiguous (trimmed == 0) and the safety margin
    # below can be skipped cleanly on a side that never qualified -- a page
    # with no border must be a true no-op, not "no-op plus a flat
    # safety-margin nibble".
    trimmed_top = trimmed_bottom = trimmed_left = trimmed_right = 0

    def _row_ink_frac(y: int) -> float:
        left, right = trimmed_left, width - trimmed_right
        if right - left <= 0:
            return 0.0
        return float(mask[y, left:right].mean())

    def _col_ink_frac(x: int) -> float:
        top, bottom = trimmed_top, height - trimmed_bottom
        if bottom - top <= 0:
            return 0.0
        return float(mask[top:bottom, x].mean())

    # Alternate edges so all four sides get a fair shot before any one side
    # exhausts its cap: each pass trims at most one row/column per side.
    while (height - trimmed_top - trimmed_bottom) > 1 and (
        width - trimmed_left - trimmed_right
    ) > 1:
        progressed = False
        if trimmed_top < cap_h and _row_ink_frac(trimmed_top) > border_ink_frac:
            trimmed_top += 1
            progressed = True
        if (
            trimmed_bottom < cap_h
            and (height - trimmed_top - trimmed_bottom) > 1
            and _row_ink_frac(height - trimmed_bottom - 1) > border_ink_frac
        ):
            trimmed_bottom += 1
            progressed = True
        if trimmed_left < cap_w and _col_ink_frac(trimmed_left) > border_ink_frac:
            trimmed_left += 1
            progressed = True
        if (
            trimmed_right < cap_w
            and (width - trimmed_left - trimmed_right) > 1
            and _col_ink_frac(width - trimmed_right - 1) > border_ink_frac
        ):
            trimmed_right += 1
            progressed = True
        if not progressed:
            break

    # Safety margin only applies past a side that actually qualified as a
    # border (trimmed > 0): a side with trimmed == 0 had no border and must
    # stay untouched, or a fully-clean page would lose a flat safety_margin
    # of content on every side for nothing.
    if trimmed_top > 0:
        trimmed_top = min(trimmed_top + safety_margin, cap_h)
    if trimmed_bottom > 0:
        trimmed_bottom = min(trimmed_bottom + safety_margin, cap_h)
    if trimmed_left > 0:
        trimmed_left = min(trimmed_left + safety_margin, cap_w)
    if trimmed_right > 0:
        trimmed_right = min(trimmed_right + safety_margin, cap_w)

    # Degenerate-overlap guard: if the two opposing margins would now cross
    # (only possible on a very small or adversarial page), fall back to no
    # trim on that axis rather than returning an empty/invalid array.
    if trimmed_top + trimmed_bottom >= height:
        trimmed_top = trimmed_bottom = 0
    if trimmed_left + trimmed_right >= width:
        trimmed_left = trimmed_right = 0

    top, bottom = trimmed_top, height - trimmed_bottom
    left, right = trimmed_left, width - trimmed_right
    cropped = page[top:bottom, left:right]
    return cropped, (top, left)


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


def is_plausible_line(
    crop_array,
    page_w: int | None = None,
    page_h: int | None = None,
    *,
    ink_threshold: int = 200,
    min_ink_fraction: float = 0.015,
    max_ink_fraction: float = 0.55,
    min_line_width: int = 18,
    max_width_frac: float = 0.35,
    min_line_height: int = 60,
) -> tuple[bool, str]:
    """Is ``crop_array`` a plausible text-line crop, or a segmentation artifact?

    A real glyph-bearing line strip occupies a narrow, predictable band on
    every axis this checks; the failure modes seen on real scanned-book
    input each blow past exactly one of these bands:

    - **Near-blank slivers** (a detected "line" landing on margin
      whitespace, or so thin it caught only anti-aliasing fringe): ink
      fraction below ``min_ink_fraction``.
    - **Solid bars** (a scan-border remnant or gutter shadow that survived
      upstream trimming, or a photo/figure the detector mistook for text):
      ink fraction above ``max_ink_fraction`` — real traditional-Mongolian
      glyph strokes, even in dense text, do not fill more than roughly half
      their bounding crop.
    - **Geometry outliers**: ``crop_array``'s width is the ACROSS-line
      (cross-writing-axis) dimension for a vertical script strip — training
      strips are ~64px wide (see :mod:`scripts.build_ocr_data_from_pairs`),
      so anything under ``min_line_width`` is a sliver, and anything over
      ``max_width_frac * page_w`` is wider than a single text column has any
      business being (a column-detection failure swallowing most of the
      page, e.g. :func:`detect_line_columns`'s whole-page fallback on a
      noisy scan). The width-vs-page-width check is skipped when
      ``page_w`` is not given, since some callers (synthetic
      reconstructions with no real "page") have no meaningful page width to
      compare against. Height under ``min_line_height`` is likewise a
      sliver along the writing axis; ``page_h`` is accepted for symmetry
      and future use but not currently compared against (a full-height
      column chunk is never implausibly TALL the way a runaway column chunk
      can be implausibly WIDE).

    Args:
        crop_array: 2-D grayscale array ``[H, W]`` of one candidate line
            crop, white background, dark ink (same convention as the rest
            of this module).
        page_w: the source page's full width in pixels, for the
            width-vs-page-width outlier check. ``None`` skips that check.
        page_h: accepted for signature symmetry with ``page_w``; unused
            today (see above).
        ink_threshold: pixels strictly below this count as ink.
        min_ink_fraction/max_ink_fraction: the plausible ink-density band.
        min_line_width/max_width_frac: the plausible width band (px floor,
            page-fraction ceiling).
        min_line_height: height floor, px.

    Returns:
        ``(is_plausible, reason)``. ``reason`` is ``"ok"`` when plausible,
        else one of ``"blank"``, ``"solid_bar"``, ``"too_narrow"``,
        ``"too_wide"``, ``"too_short"`` — the first failing check, checked
        in that order.
    """
    _require_numpy()
    if not 0.0 <= min_ink_fraction < max_ink_fraction <= 1.0:
        raise ValueError(
            "require 0 <= min_ink_fraction < max_ink_fraction <= 1, got "
            f"{min_ink_fraction!r}, {max_ink_fraction!r}"
        )
    if min_line_width < 1:
        raise ValueError("min_line_width must be >= 1")
    if not 0.0 < max_width_frac <= 1.0:
        raise ValueError("max_width_frac must be in (0, 1]")
    if min_line_height < 1:
        raise ValueError("min_line_height must be >= 1")

    a = np.asarray(crop_array)
    if a.ndim != 2:
        raise ValueError(
            f"expected a 2-D grayscale array (white bg, dark ink), got ndim={a.ndim}"
        )
    height, width = a.shape
    if height == 0 or width == 0:
        return False, "blank"

    ink_fraction = float((a < ink_threshold).mean())
    if ink_fraction < min_ink_fraction:
        return False, "blank"
    if ink_fraction > max_ink_fraction:
        return False, "solid_bar"
    if width < min_line_width:
        return False, "too_narrow"
    if page_w is not None and width > max_width_frac * page_w:
        return False, "too_wide"
    if height < min_line_height:
        return False, "too_short"
    return True, "ok"


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
    page_w: int | None = None,
    page_h: int | None = None,
    min_ink_fraction: float = 0.015,
    max_ink_fraction: float = 0.55,
    min_line_width: int = 18,
    max_width_frac: float = 0.35,
    min_line_height: int = 60,
) -> tuple[list[Span], dict[str, int]]:
    """Cut one tall column into model-sized line strips at row-ink valleys.

    Walks the column top to bottom. Each strip may grow to at most
    ``target_px * overshoot`` rows; the cut lands on the deepest row-ink
    valley inside the allowed window (ties broken toward ``target_px``), or,
    when the window holds no valley at all, hard-cuts at ``target_px``.
    Each strip is then trimmed to its ink extent plus ``pad`` rows, then
    passed through :func:`is_plausible_line` (blank/solid-bar/geometry
    reject) before being kept.

    Args:
        column: 2-D grayscale array ``[H, W]`` of one column (white bg).
        target_px: preferred strip height in source pixels. Choose it so a
            strip letterboxed to the model's square keeps glyphs near the
            training scale (training strips were ~400-900 px tall).
        ink_threshold/smooth/valley_frac: as in :func:`detect_columns`
            (profile is per-row here).
        min_height: strips shorter than this after trimming are dropped
            (upstream of the plausibility filter's own ``min_line_height``
            — the two are independent knobs; keep ``min_height <=
            min_line_height`` or the filter's height reject never fires).
        overshoot: window factor; ``target_px * overshoot`` must exceed
            ``min_height``.
        pad: white rows kept above/below each trimmed strip.
        page_w/page_h: the source page's full dimensions, forwarded to
            :func:`is_plausible_line` for its width-vs-page-width check.
            ``page_w=None`` (the default) skips that one check only —
            ink-fraction and absolute width/height rejects still apply
            unconditionally, since they need no page context. Existing
            callers that never passed page dimensions keep getting a
            column-relative-only quality filter, not the pre-filter
            passthrough behavior (this function always quality-filters;
            what changes without ``page_w`` is only which checks run).
        min_ink_fraction/max_ink_fraction/min_line_width/max_width_frac/
            min_line_height: forwarded to :func:`is_plausible_line`
            verbatim; see its docstring.

    Returns:
        ``(boxes, rejections)``: ``boxes`` are ``(y0, y1)`` half-open row
        spans relative to ``column``, in reading order (top to bottom) —
        a blank column returns ``[]``. ``rejections`` counts dropped
        candidate strips by reason (``"blank"``, ``"solid_bar"``,
        ``"too_narrow"``, ``"too_wide"``, ``"too_short"``), always present
        with all five keys even when 0, for observability without a
        truthiness check on every key.
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

    empty_rejections = {
        "blank": 0,
        "solid_bar": 0,
        "too_narrow": 0,
        "too_wide": 0,
        "too_short": 0,
    }

    mask2d = _ink_mask(column, ink_threshold)
    n_rows = mask2d.shape[0]
    smoothed = _moving_average(mask2d.sum(axis=1), smooth)
    peak = float(smoothed.max(initial=0.0))
    if peak <= 0.0:
        return [], dict(empty_rejections)
    thr = valley_frac * peak
    ink_rows = smoothed > thr
    # Smoothing exists to make valley detection robust; extent measurement
    # (per-strip trim + min_height dust filtering) must use the raw mask, or
    # the window would smear a 2px speck into a strip that clears min_height.
    raw_ink_rows = mask2d.any(axis=1)
    runs = _runs(ink_rows)
    if not runs:
        return [], dict(empty_rejections)
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
    rejections = dict(empty_rejections)
    for a, b in cuts:
        seg = np.flatnonzero(raw_ink_rows[a:b])
        if seg.size == 0:
            continue
        y0 = max(a + int(seg[0]) - pad, 0)
        y1 = min(a + int(seg[-1]) + 1 + pad, n_rows)
        if y1 - y0 < min_height:
            continue
        plausible, reason = is_plausible_line(
            column[y0:y1, :],
            page_w,
            page_h,
            ink_threshold=ink_threshold,
            min_ink_fraction=min_ink_fraction,
            max_ink_fraction=max_ink_fraction,
            min_line_width=min_line_width,
            max_width_frac=max_width_frac,
            min_line_height=min_line_height,
        )
        if not plausible:
            rejections[reason] += 1
            continue
        boxes.append((y0, y1))
    return boxes, rejections


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
    "is_plausible_line",
    "lines_from_column",
    "trim_scan_borders",
]
