# -*- coding: utf-8 -*-

"""Build a human-annotation package from a directory of scanned-book PDFs.

The project needs hundreds-to-thousands of human-transcribed real lines for
(a) a frozen L3 real-scan benchmark and (b) leakage-controlled OCR
reinforcement learning. This tool
turns a directory of scanned-book PDFs into a ready-to-transcribe package: it
samples pages, segments each into traditional-Mongolian text-line columns
(:mod:`Model.ocr.segment`), samples lines, crops + letterboxes each one, and
emits TSV annotation sheets plus a manifest tracing every line back to its
source pixels.

Pipeline per document::

    render sampled pages (pdftoppm subprocess, fallback PyMuPDF)
        -> trim_scan_borders (Model.ocr.segment; strips solid scan-border/
           scanner-bed strips BEFORE column detection)
        -> detect_line_columns (Model.ocr.segment)
        -> chunk_column_by_height (splits page-spanning columns)
        -> is_plausible_line (Model.ocr.segment; rejects blank/solid-bar/
           geometry-outlier chunks BEFORE sampling, so lines_per_page fills
           from real candidates, not garbage; rejections counted per page)
        -> sample --lines-per-page chunks
        -> crop raw strip + letterbox_to_square 224px preview
        -> assign id doc__page__col__chunk

Real scanned-book PDFs (as opposed to clean synthetic pages) carry solid
black scan-border strips, gutter shadows, and wide blank margins that a
column/chunk detector tuned on clean ink bars will happily segment as if
they were text -- see :mod:`Model.ocr.segment`'s module docstring for the
failure modes this causes and why border trimming + per-chunk plausibility
filtering, not smarter column detection alone, is the fix. Every document's
run also emits ``qa_montage_<doc_stem>.png`` under ``--out`` -- up to
:data:`_QA_MONTAGE_MAX_CROPS` sampled KEPT crops composed into one strip, so
a human can eyeball pack quality without opening files under ``lines/``.

Two input modes, mutually exclusive:

- ``--pdf-dir``: a directory of PDF files. Pages are rasterized via
  ``pdftoppm`` (subprocess) if present on ``$PATH``, else PyMuPDF (``fitz``).
  Both are verified to produce pixel-identical output dimensions at the same
  DPI (``pdftoppm -r 300`` vs. ``fitz`` at ``zoom = 300/72``), so which
  renderer handled a given PDF does not affect the geometry recorded in the
  manifest.
- ``--image-dir``: a directory of pre-rendered page images, treated as if
  they were one document's already-rasterized pages (sorted by filename).
  ``--capture-session`` is mandatory and assigns every crop in that directory
  to one leakage-safe split group. Repeat both flags, in matching order, to
  combine several genuinely independent capture sessions in one pack.
  This is both the escape hatch for a host with neither ``pdftoppm`` nor a
  real PDF corpus available, and the mode this module's own tests drive end
  to end.

Determinism: every random draw (page sampling, line sampling, double-annotate
subset) is made from a **fresh** ``random.Random(...)`` instance seeded from
``--seed`` plus a context-specific hash (per-document, per-page) -- never a
single shared stream reused across contexts. Concretely:

    Random(seed ^ _stable_hash(doc_stem))              -- which pages to sample
    Random(seed ^ _stable_hash(doc_stem, page_index))  -- which chunks to sample
    Random(seed)                                       -- global double-annotate draw

``_stable_hash`` (SHA-256-based), not Python's built-in ``hash()``, because
``hash(str)`` is randomized per PROCESS by default (PYTHONHASHSEED salting) --
using it here would make "deterministic given --seed" false across separate
runs despite being true within a single run, which is exactly the kind of
bug that looks fine until you diff two runs (caught by re-running this
tool's own smoke test twice and diffing the manifests before trusting it).

Each ``Random(...)`` instance is instantiated once and used immediately, so
reusing the bare ``seed`` value across these three call sites does not
couple them: there is no shared mutable RNG state for one draw's consumption
to perturb another's.

Robustness: a page whose segmentation finds nothing usable is logged and
skipped (counted in the manifest, not fatal); a corrupt/unreadable page or
PDF is likewise logged and skipped rather than crashing the run. Two input
documents that sanitize to the same id stem are a hard error raised BEFORE
any output is written (see :func:`check_stem_collisions`) -- silently
aliasing two source documents under one id prefix would corrupt provenance,
which is this tool's entire reason to exist.

Usage::

    python3 -m scripts.build_annotation_pack \\
        --pdf-dir ../corpus/RAW/scans/some_shelf --out ../corpus/outputs/annopack_v1 \\
        --pages-per-doc 3 --lines-per-page 8 --dpi 300 --seed 0

    # Smoke / no-PDF-access mode:
    python3 -m scripts.build_annotation_pack \\
        --image-dir /tmp/rendered_pages --capture-session smoke-session \\
        --image-dir /tmp/rendered_pages_2 --capture-session smoke-session-2 \\
        --out /tmp/annopack_smoke --limit 20
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Model.ocr.segment import (  # noqa: E402
    Box,
    chunk_column_by_height,
    detect_line_columns,
    is_plausible_line,
    trim_scan_borders,
)
from scripts.build_ocr_data_from_pairs import letterbox_to_square  # noqa: E402

_LETTERBOX_SIZE = 224

# Cap on how many kept crops from one document are composed into its
# qa_montage_<doc_stem>.png -- enough for a human to eyeball pack quality at
# a glance without opening every file under lines/, not an exhaustive dump.
_QA_MONTAGE_MAX_CROPS = 12

# Pages within this fraction of the start/end of a document are treated as
# covers/TOC and never sampled (mirrors common front/back-matter placement in
# scanned books).
_EDGE_EXCLUDE_FRAC = 0.05

# A tiny document (fewer than this many pages) skips the edge exclusion
# entirely -- excluding 5% of, say, a 6-page document would leave nothing to
# sample from.
_MIN_PAGES_FOR_EDGE_EXCLUDE = 20


# ===========================================================================
# git metadata (copied from scripts/train_rdt.py's _git_metadata pattern:
# importing it directly would pull in torch + Model.model.RDTForCausalLM at
# module scope for a 15-line subprocess helper, an avoidable second heavy
# import trigger on top of the one letterbox_to_square already costs).
# ===========================================================================


def _git_metadata() -> dict[str, Any]:
    def _run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=str(_REPO_ROOT),
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return ""

    return {
        "commit": _run("rev-parse", "HEAD"),
        "branch": _run("branch", "--show-current"),
        "dirty": bool(_run("status", "--short")),
    }


# ===========================================================================
# Stem sanitization + collision guard
# ===========================================================================


def sanitize_stem(name: str) -> str:
    """Alnum+underscore sanitization, matching scripts.ingest_scan_pdfs's stem rule.

    Unlike that module, no sha1-of-path suffix is appended: annotation ids
    need to be short and human-readable in a spreadsheet, and cross-document
    collisions are instead caught upfront by :func:`check_stem_collisions`
    (a hard error before any output is written) rather than silently
    disambiguated by an opaque hash tail.
    """
    stem = unicodedata.normalize("NFC", name)
    return "".join(ch if ch.isalnum() else "_" for ch in stem)[:64]


def check_stem_collisions(paths: list[Path]) -> None:
    """Raise ValueError if two inputs sanitize to the same stem.

    Called before any output is written. Aliasing two different source
    documents under one id prefix would corrupt provenance -- this tool's
    entire purpose is unambiguous traceability from an id back to its source
    pixels, so this is a hard stop, not a logged warning.
    """
    by_stem: dict[str, list[Path]] = {}
    for p in paths:
        by_stem.setdefault(sanitize_stem(p.stem), []).append(p)
    collisions = {stem: ps for stem, ps in by_stem.items() if len(ps) > 1}
    if collisions:
        lines = [
            f"  {stem!r}: {[str(p) for p in ps]}" for stem, ps in collisions.items()
        ]
        raise ValueError(
            "input documents collide on sanitized stem (would alias distinct "
            "documents under one id prefix):\n" + "\n".join(lines)
        )


# ===========================================================================
# Deterministic page/line sampling
# ===========================================================================


def _stable_hash(*parts: object) -> int:
    """A cross-process-stable replacement for Python's built-in ``hash()``.

    Python randomizes ``hash(str)`` per process (PYTHONHASHSEED salting, on
    by default) for security -- ``hash(doc_stem)`` therefore differs between
    two separate runs of this tool even with an identical ``--seed`` and
    identical input, silently breaking the "deterministic given --seed"
    contract this tool exists to provide. This function hashes a
    ``\\x1f``-joined string encoding of ``parts`` via SHA-256 and takes the
    first 8 bytes as a big-endian int, which is stable across processes,
    interpreter versions, and machines.
    """
    key = "\x1f".join(str(p) for p in parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def sample_page_indices(
    n_pages: int, pages_per_doc: int, *, seed: int, doc_stem: str
) -> list[int]:
    """Deterministically pick which 0-based page indices to sample.

    Stratified across ``[low, high)``, excluding the first/last
    ``_EDGE_EXCLUDE_FRAC`` of the page range (covers/TOC) unless the document
    is too short for that exclusion to leave any pages
    (``< _MIN_PAGES_FOR_EDGE_EXCLUDE``). Uses a fresh
    ``random.Random(seed ^ _stable_hash(doc_stem))`` -- distinct per
    document, so adding/removing one PDF from the input directory does not
    reshuffle any other document's sample; and stable across process
    invocations (see :func:`_stable_hash`), unlike Python's built-in
    ``hash()``.
    """
    if n_pages <= 0:
        return []
    if n_pages >= _MIN_PAGES_FOR_EDGE_EXCLUDE:
        exclude = max(1, round(n_pages * _EDGE_EXCLUDE_FRAC))
        low, high = exclude, n_pages - exclude
    else:
        low, high = 0, n_pages
    if high <= low:
        low, high = 0, n_pages  # degenerate: exclusion ate the whole range

    candidates = list(range(low, high))
    rng = random.Random(seed ^ _stable_hash(doc_stem))
    k = min(pages_per_doc, len(candidates))
    return sorted(rng.sample(candidates, k))


def sample_chunks(
    chunks: list[Box], lines_per_page: int, *, seed: int, doc_stem: str, page_index: int
) -> list[int]:
    """Deterministically pick which flattened-chunk indices to keep for a page.

    Fresh ``random.Random(seed ^ _stable_hash(doc_stem, page_index))`` per
    page, so resampling is scoped to exactly this (document, page) pair and
    stable across process invocations (see :func:`_stable_hash`).
    """
    if not chunks:
        return []
    rng = random.Random(seed ^ _stable_hash(doc_stem, page_index))
    k = min(lines_per_page, len(chunks))
    return sorted(rng.sample(range(len(chunks)), k))


# ===========================================================================
# Page rendering: pdftoppm subprocess primary, PyMuPDF fallback
# ===========================================================================


def render_pdf_pages(pdf_path: Path, page_indices: list[int], dpi: int, tmp_dir: Path):
    """Render the given 0-based page indices of ``pdf_path`` to PIL Images.

    Tries ``pdftoppm`` (subprocess) first if it is on ``$PATH``; falls back
    to PyMuPDF (``fitz``) if ``pdftoppm`` is absent or errors on this file.
    Both renderers are verified to produce pixel-identical output dimensions
    for the same DPI (``pdftoppm -r <dpi>`` == ``fitz`` at
    ``zoom = dpi / 72``), so column/line geometry in the manifest is
    consistent regardless of which one handled a given PDF. Returns a dict
    ``{page_index: PIL.Image}`` -- a page that fails to render under BOTH
    paths is simply absent from the returned dict (caller logs + skips it,
    per this tool's per-page fault-tolerance contract).
    """
    from PIL import Image

    out: dict[int, Any] = {}
    if shutil.which("pdftoppm") is not None:
        for pi in page_indices:
            prefix = tmp_dir / f"page_{pi:04d}"
            try:
                subprocess.run(
                    [
                        "pdftoppm",
                        "-r",
                        str(dpi),
                        "-f",
                        str(pi + 1),
                        "-l",
                        str(pi + 1),
                        "-png",
                        "-singlefile",
                        str(pdf_path),
                        str(prefix),
                    ],
                    check=True,
                    capture_output=True,
                )
                png_path = prefix.with_suffix(".png")
                if png_path.exists():
                    img = Image.open(png_path)
                    img.load()
                    out[pi] = img
            except Exception as exc:
                print(
                    f"[annopack] pdftoppm failed on {pdf_path.name} page {pi}: {exc}; "
                    "will try PyMuPDF fallback",
                    file=sys.stderr,
                )
        missing = [pi for pi in page_indices if pi not in out]
        if not missing:
            return out
        page_indices = missing  # fall through to fitz for whatever's left

    try:
        import fitz  # PyMuPDF
    except ImportError:
        if not out:
            print(
                f"[annopack] SKIP {pdf_path.name}: neither pdftoppm nor PyMuPDF "
                "(fitz) is available",
                file=sys.stderr,
            )
        return out

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        print(f"[annopack] SKIP {pdf_path.name}: cannot open ({exc})", file=sys.stderr)
        return out
    zoom = dpi / 72.0
    for pi in page_indices:
        try:
            if pi >= doc.page_count:
                continue
            page = doc[pi]
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert(
                "L"
            )
            out[pi] = img
        except Exception as exc:
            print(
                f"[annopack] page render error {pdf_path.name}:{pi}: {exc}",
                file=sys.stderr,
            )
    doc.close()
    return out


def load_image_dir_pages(image_dir: Path) -> dict[int, Any]:
    """Load a directory of pre-rendered page images, sorted by filename.

    ``--image-dir`` mode's page source: every file directly under
    ``image_dir`` (non-recursive) with an image-like suffix, sorted, indexed
    0..N-1 in sorted order -- treated exactly like a rendered PDF's pages.
    """
    from PIL import Image

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    files = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)
    out: dict[int, Any] = {}
    for i, p in enumerate(files):
        try:
            img = Image.open(p)
            img.load()
            out[i] = img.convert("L")
        except Exception as exc:
            print(f"[annopack] SKIP unreadable image {p.name}: {exc}", file=sys.stderr)
    return out


# ===========================================================================
# Per-document counters (manifest provenance)
# ===========================================================================


@dataclass
class DocResult:
    doc_stem: str
    source_name: str
    n_pages_total: int
    sampled_pages: list[int] = field(default_factory=list)
    pages_no_segmentation: list[int] = field(default_factory=list)
    lines: list[dict[str, Any]] = field(default_factory=list)
    # {page_index: {"blank": n, "solid_bar": n, "too_narrow": n, "too_wide":
    # n, "too_short": n}} -- per-page counts of chunks dropped by
    # is_plausible_line, keyed the same as Model.ocr.segment.lines_from_column's
    # rejection dict, for the manifest and end-of-run observability.
    rejections_by_page: dict[int, dict[str, int]] = field(default_factory=dict)
    # Kept-crop PIL images sampled for this doc's qa_montage.png (not every
    # kept line -- see _QA_MONTAGE_MAX_CROPS -- just enough to eyeball pack
    # quality without opening the whole lines/ directory).
    montage_crops: list[Any] = field(default_factory=list)


# ===========================================================================
# QA montage: one glance at pack quality, per document
# ===========================================================================


def write_qa_montage(crops: list[Any], out_path: Path, *, strip_height: int = 96) -> bool:
    """Compose up to ``_QA_MONTAGE_MAX_CROPS`` kept line crops into one PNG.

    Each crop is letterboxed to a fixed height (``strip_height``, preserving
    aspect ratio, white-padded) and laid out top to bottom in one tall strip
    so a human can eyeball a document's line-crop quality in one glance
    rather than opening individual files in ``lines/``. Returns ``False``
    (writes nothing) when ``crops`` is empty -- a document that produced zero
    kept lines has nothing to montage, which is itself visible in the
    manifest's counts, not something this function should paper over with a
    blank image.
    """
    from PIL import Image

    if not crops:
        return False
    if strip_height < 1:
        raise ValueError("strip_height must be >= 1")

    resample = (
        Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    )
    resized = []
    for crop in crops:
        w, h = crop.size
        if h <= 0:
            continue
        new_w = max(1, round(w * (strip_height / h)))
        resized.append(crop.convert("L").resize((new_w, strip_height), resample))
    if not resized:
        return False

    pad = 4
    montage_w = max(im.width for im in resized) + 2 * pad
    montage_h = sum(im.height for im in resized) + pad * (len(resized) + 1)
    montage = Image.new("L", (montage_w, montage_h), 255)
    y = pad
    for im in resized:
        montage.paste(im, (pad, y))
        y += im.height + pad
    out_path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(out_path)
    return True


# ===========================================================================
# Core per-document pipeline
# ===========================================================================


def process_document(
    doc_stem: str,
    source_name: str,
    n_pages_total: int,
    sampled_pages: list[int],
    pages: dict[int, Any],
    *,
    out_dir: Path,
    lines_per_page: int,
    dpi: int,
    seed: int,
    limit_remaining: int | None,
    group_id_override: str | None = None,
) -> DocResult:
    """Segment + sample lines from already-rendered, already-sampled pages.

    Page SAMPLING is decided exactly once by the caller (:func:`build_pack`,
    via :func:`sample_page_indices` against the document's true page count)
    and passed in as ``sampled_pages`` -- this function must not re-derive or
    re-sample page indices itself. Re-sampling here against ``pages.keys()``
    (only the subset that was rendered) would silently sample from the wrong
    -- much smaller and differently-indexed -- population than the true
    document, since a PDF only has its pre-selected candidate pages rendered
    (never all pages), producing a plausible-looking but wrong stratified
    sample. ``pages`` therefore only ever needs to be indexed by the ids in
    ``sampled_pages``; a requested page absent from ``pages`` (render
    failure) is reported via ``pages_no_segmentation``, not silently dropped.

    ``limit_remaining`` (``None`` = unlimited) caps how many more lines this
    call may write in total across all its pages -- ``--limit`` is enforced
    as a running budget threaded through every document, not a per-document
    cap, so the smoke flag reliably bounds the whole run's line count.

    Each page is border-trimmed (:func:`Model.ocr.segment.trim_scan_borders`)
    BEFORE column detection, and every candidate chunk is quality-filtered
    (:func:`Model.ocr.segment.is_plausible_line`) BEFORE line sampling --
    filtering before, not after, sampling means a page with N good chunks and
    a pile of blank/border-artifact chunks still tries to fill
    ``lines_per_page`` from the N good ones, rather than possibly sampling
    mostly garbage and yielding far fewer than requested. Per-page rejection
    counts land in ``result.rejections_by_page``; a sample of kept crops
    (see ``_QA_MONTAGE_MAX_CROPS``) lands in ``result.montage_crops`` for
    :func:`build_pack` to render as ``qa_montage.png``.
    """
    from PIL import Image

    result = DocResult(doc_stem=doc_stem, source_name=source_name, n_pages_total=n_pages_total)
    result.sampled_pages = list(sampled_pages)
    if not sampled_pages:
        return result

    lines_dir = out_dir / "lines"
    lines_dir.mkdir(parents=True, exist_ok=True)

    for page_idx in sampled_pages:
        if limit_remaining is not None and limit_remaining <= 0:
            break
        if page_idx not in pages:
            # Requested but failed to render (corrupt page / renderer error):
            # logged already by the renderer; count as empty-segmentation
            # rather than silently vanishing from provenance.
            result.pages_no_segmentation.append(page_idx)
            continue
        page_img = pages[page_idx]

        # Real scanned-book pages carry solid black scan-border/scanner-bed
        # strips that a clean synthetic page never has; trimmed BEFORE
        # column detection so they never seed a false column or dominate the
        # ink profile. y_off/x_off map trimmed-local boxes back to this
        # page's own pixel coordinates for the manifest and for cropping out
        # of the untouched original `page_img` (crop from the original, not
        # a numpy round-trip of it, to avoid any lossy re-encode).
        page_arr = np.asarray(page_img.convert("L"))
        trimmed_arr, (y_off, x_off) = trim_scan_borders(page_arr)
        trimmed_img = Image.fromarray(trimmed_arr, mode="L")
        page_h, page_w = page_arr.shape

        try:
            columns = detect_line_columns(trimmed_img)
        except Exception as exc:
            print(
                f"[annopack] segmentation error {source_name} page {page_idx}: {exc}",
                file=sys.stderr,
            )
            result.pages_no_segmentation.append(page_idx)
            continue

        flat_chunks: list[Box] = []
        chunk_owner_col: list[int] = []
        chunk_crops: list[Any] = []
        page_rejections = {
            "blank": 0,
            "solid_bar": 0,
            "too_narrow": 0,
            "too_wide": 0,
            "too_short": 0,
        }
        for col_idx, col_box in enumerate(columns):
            for chunk_box in chunk_column_by_height(col_box):
                # chunk_box is in trimmed-local coordinates (detect_line_columns
                # ran on trimmed_img); add the trim offset back to crop from
                # the original, untrimmed page_img and to record
                # original-page-space geometry in the manifest.
                cx0, cy0, cx1, cy1 = chunk_box
                orig_box = (cx0 + x_off, cy0 + y_off, cx1 + x_off, cy1 + y_off)
                crop = page_img.crop(orig_box).convert("L")
                plausible, reason = is_plausible_line(
                    np.asarray(crop), page_w, page_h
                )
                if not plausible:
                    page_rejections[reason] += 1
                    continue
                flat_chunks.append(orig_box)
                chunk_owner_col.append(col_idx)
                chunk_crops.append(crop)
        result.rejections_by_page[page_idx] = page_rejections

        if not flat_chunks:
            result.pages_no_segmentation.append(page_idx)
            continue

        keep_idx = sample_chunks(
            flat_chunks,
            lines_per_page,
            seed=seed,
            doc_stem=doc_stem,
            page_index=page_idx,
        )
        if limit_remaining is not None:
            keep_idx = keep_idx[: max(0, limit_remaining)]

        for chunk_pos, flat_i in enumerate(keep_idx):
            chunk_box = flat_chunks[flat_i]
            col_idx = chunk_owner_col[flat_i]
            line_id = f"{doc_stem}__p{page_idx:04d}__c{col_idx:02d}__l{chunk_pos:03d}"

            crop = chunk_crops[flat_i]

            raw_path = lines_dir / f"{line_id}_raw.png"
            crop.save(raw_path)

            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            preview = letterbox_to_square(buf.getvalue(), _LETTERBOX_SIZE)
            preview_path = lines_dir / f"{line_id}_224.png"
            preview.save(preview_path)

            if len(result.montage_crops) < _QA_MONTAGE_MAX_CROPS:
                result.montage_crops.append(crop)

            col_box_orig = (
                columns[col_idx][0] + x_off,
                columns[col_idx][1] + y_off,
                columns[col_idx][2] + x_off,
                columns[col_idx][3] + y_off,
            )
            result.lines.append(
                {
                    "id": line_id,
                    "pdf": source_name,
                    "group_id": (
                        group_id_override or f"document:{source_name}"
                    ),
                    "page": page_idx,
                    "col_box": list(col_box_orig),
                    "line_box": list(chunk_box),
                    "dpi": dpi,
                    "raw_path": str(raw_path.relative_to(out_dir)),
                    "preview_path": str(preview_path.relative_to(out_dir)),
                }
            )
            if limit_remaining is not None:
                limit_remaining -= 1

    return result


# ===========================================================================
# CLI
# ===========================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--pdf-dir", default=None, help="directory of PDF files")
    ap.add_argument(
        "--image-dir",
        action="append",
        default=None,
        help="repeatable directory of pre-rendered page images; each occurrence "
        "requires one matching --capture-session (mutually exclusive with "
        "--pdf-dir)",
    )
    ap.add_argument(
        "--capture-session",
        action="append",
        default=[],
        help="repeatable stable camera/session provenance, paired by order with "
        "--image-dir and used as one leakage-safe split group",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--pages-per-doc", type=int, default=3)
    ap.add_argument("--lines-per-page", type=int, default=8)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--double-annotate-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--limit", type=int, default=0, help="cap total lines across the whole run; 0 = unlimited"
    )
    args = ap.parse_args(argv)

    if (args.pdf_dir is None) == (args.image_dir is None):
        ap.error("exactly one of --pdf-dir or --image-dir must be given")
    if args.image_dir is not None and len(args.image_dir) != len(
        args.capture_session
    ):
        ap.error(
            "each --image-dir requires exactly one matching --capture-session"
        )
    if any(not value.strip() for value in args.capture_session):
        ap.error("--capture-session must be non-empty")
    if args.pdf_dir is not None and args.capture_session:
        ap.error("--capture-session is only valid with --image-dir")
    if args.pages_per_doc < 1:
        ap.error("--pages-per-doc must be >= 1")
    if args.lines_per_page < 1:
        ap.error("--lines-per-page must be >= 1")
    if args.dpi < 1:
        ap.error("--dpi must be >= 1")
    if not (0.0 <= args.double_annotate_frac <= 1.0):
        ap.error("--double-annotate-frac must be in [0, 1]")
    if args.limit < 0:
        ap.error("--limit must be >= 0")
    return args


def _iter_pdf_paths(pdf_dir: Path) -> list[Path]:
    return sorted(p for p in pdf_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")


def build_pack(args: argparse.Namespace) -> dict[str, Any]:
    """Drive the whole pipeline; returns the manifest dict (also written to disk)."""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lines").mkdir(parents=True, exist_ok=True)
    (out_dir / "sheets").mkdir(parents=True, exist_ok=True)

    limit_remaining: int | None = args.limit if args.limit > 0 else None
    results: list[DocResult] = []

    if args.image_dir is not None:
        image_dirs = [Path(value) for value in args.image_dir]
        check_stem_collisions(image_dirs)
        for image_dir, capture_session in zip(
            image_dirs,
            args.capture_session,
        ):
            doc_stem = sanitize_stem(image_dir.name)
            pages = load_image_dir_pages(image_dir)
            # n_pages_total for --image-dir mode IS the loaded-image count
            # (unlike PDF mode, there is no independent true page count).
            n_pages_total = len(pages)
            sampled_pages = sample_page_indices(
                n_pages_total,
                args.pages_per_doc,
                seed=args.seed,
                doc_stem=doc_stem,
            )
            result = process_document(
                doc_stem,
                image_dir.name,
                n_pages_total,
                sampled_pages,
                pages,
                out_dir=out_dir,
                lines_per_page=args.lines_per_page,
                dpi=args.dpi,
                seed=args.seed,
                limit_remaining=limit_remaining,
                group_id_override=f"capture:{capture_session.strip()}",
            )
            if limit_remaining is not None:
                limit_remaining -= len(result.lines)
            results.append(result)
    else:
        pdf_dir = Path(args.pdf_dir)
        pdf_paths = _iter_pdf_paths(pdf_dir)
        check_stem_collisions(pdf_paths)
        for pdf_path in pdf_paths:
            if limit_remaining is not None and limit_remaining <= 0:
                results.append(
                    DocResult(
                        doc_stem=sanitize_stem(pdf_path.stem),
                        source_name=pdf_path.name,
                        n_pages_total=0,
                    )
                )
                continue
            doc_stem = sanitize_stem(pdf_path.stem)
            with tempfile.TemporaryDirectory(prefix=f"annopack_{doc_stem}_") as tmp:
                try:
                    import fitz

                    n_pages_total = fitz.open(str(pdf_path)).page_count
                except Exception as exc:
                    print(
                        f"[annopack] SKIP PDF {pdf_path.name}: cannot read page "
                        f"count ({exc})",
                        file=sys.stderr,
                    )
                    results.append(
                        DocResult(
                            doc_stem=doc_stem, source_name=pdf_path.name, n_pages_total=0
                        )
                    )
                    continue

                # Page sampling happens exactly ONCE, here, against the
                # document's TRUE total page count -- render_pdf_pages is
                # then asked to rasterize only these already-decided
                # indices. process_document must not re-sample: it only
                # ever receives this same sampled_pages list, so a page
                # present in sampled_pages but absent from the rendered
                # `pages` dict (render failure) is reported as
                # empty-segmentation rather than silently vanishing or
                # (worse) triggering a second, differently-scoped sample.
                sampled_pages = sample_page_indices(
                    n_pages_total, args.pages_per_doc, seed=args.seed, doc_stem=doc_stem
                )
                pages = render_pdf_pages(
                    pdf_path, sampled_pages, args.dpi, Path(tmp)
                )
                result = process_document(
                    doc_stem,
                    pdf_path.name,
                    n_pages_total,
                    sampled_pages,
                    pages,
                    out_dir=out_dir,
                    lines_per_page=args.lines_per_page,
                    dpi=args.dpi,
                    seed=args.seed,
                    limit_remaining=limit_remaining,
                )
                if limit_remaining is not None:
                    limit_remaining -= len(result.lines)
                results.append(result)

    all_lines: list[dict[str, Any]] = []
    montage_paths: dict[str, str] = {}
    for r in results:
        all_lines.extend(r.lines)
        # One flat file per document directly under out_dir (not nested in
        # lines/), so it is the first thing a human sees browsing the pack.
        # Single-document runs (--image-dir mode) produce exactly one file
        # named "qa_montage_<doc_stem>.png"; multi-document --pdf-dir runs
        # get one per PDF, disambiguated the same way.
        montage_out = out_dir / f"qa_montage_{r.doc_stem}.png"
        if write_qa_montage(r.montage_crops, montage_out):
            montage_paths[r.doc_stem] = str(montage_out.relative_to(out_dir))

    # Global double-annotate draw: a single fresh Random(seed) over the
    # WHOLE run's kept lines, computed once at the end -- not per-page, so
    # the fraction is accurate over the corpus rather than possibly zero on
    # small per-page slices. This Random(seed) instance is independent of
    # the per-doc/per-page Random(seed ^ _stable_hash(...)) instances used during
    # sampling above: each call site constructs its own fresh instance, so
    # there is no shared mutable stream for this draw to interact with.
    rng = random.Random(args.seed)
    n_double = round(len(all_lines) * args.double_annotate_frac)
    double_ids: set[str] = set()
    if all_lines and n_double > 0:
        chosen = rng.sample(all_lines, min(n_double, len(all_lines)))
        double_ids = {row["id"] for row in chosen}
    for row in all_lines:
        row["double_annotate"] = row["id"] in double_ids

    # Sheets
    sheet_a_path = out_dir / "sheets" / "annotator_A.tsv"
    sheet_b_path = out_dir / "sheets" / "annotator_B.tsv"
    header = "id\timage_relpath\ttranscription\tnotes\n"
    with sheet_a_path.open("w", encoding="utf-8") as fa:
        fa.write(header)
        for row in all_lines:
            fa.write(f"{row['id']}\t{row['preview_path']}\t\t\n")
    with sheet_b_path.open("w", encoding="utf-8") as fb:
        fb.write(header)
        for row in all_lines:
            if row["double_annotate"]:
                fb.write(f"{row['id']}\t{row['preview_path']}\t\t\n")

    # Aggregate is_plausible_line rejection reasons across every page of
    # every document, for a run-level "how much garbage did the filter
    # catch" summary alongside the per-page breakdown each doc entry carries.
    total_rejections = {
        "blank": 0,
        "solid_bar": 0,
        "too_narrow": 0,
        "too_wide": 0,
        "too_short": 0,
    }
    for r in results:
        for page_counts in r.rejections_by_page.values():
            for reason, n in page_counts.items():
                total_rejections[reason] += n

    manifest = {
        "tool_git_rev": _git_metadata(),
        "seed": args.seed,
        "dpi": args.dpi,
        "pages_per_doc": args.pages_per_doc,
        "lines_per_page": args.lines_per_page,
        "double_annotate_frac": args.double_annotate_frac,
        "docs": [
            {
                "pdf": r.source_name,
                "n_pages_total": r.n_pages_total,
                "sampled_pages": r.sampled_pages,
                "pages_no_segmentation": r.pages_no_segmentation,
                # JSON object keys are always strings; page indices are
                # stringified here explicitly (rather than left to json.dump's
                # implicit int-key coercion) so the on-disk shape matches what
                # round-trips back through json.load without surprises.
                "rejections_by_page": {
                    str(page_idx): counts
                    for page_idx, counts in r.rejections_by_page.items()
                },
                "qa_montage_path": montage_paths.get(r.doc_stem),
            }
            for r in results
        ],
        "lines": all_lines,
        "counts": {
            "n_docs": len(results),
            "n_pages_sampled": sum(len(r.sampled_pages) for r in results),
            "n_pages_empty": sum(len(r.pages_no_segmentation) for r in results),
            "n_lines": len(all_lines),
            "n_double_annotate": len(double_ids),
            "n_rejected_total": sum(total_rejections.values()),
            "rejections_by_reason": total_rejections,
        },
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    print(
        f"[annopack] done: docs={manifest['counts']['n_docs']} "
        f"pages_sampled={manifest['counts']['n_pages_sampled']} "
        f"pages_empty={manifest['counts']['n_pages_empty']} "
        f"lines={manifest['counts']['n_lines']} "
        f"double_annotate={manifest['counts']['n_double_annotate']} "
        f"rejected={manifest['counts']['n_rejected_total']} {total_rejections} "
        f"qa_montages={len(montage_paths)} "
        f"under {out_dir}",
        flush=True,
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        build_pack(args)
    except ValueError as exc:
        print(f"[annopack] ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
