# -*- coding: utf-8 -*-

"""Ingest real scanned PDFs into the OMVT SSL data format.

Counterpart of :mod:`scripts.render_mn_pages` for the *real* corpus: book
scans have no usable ground-truth text, so rows are emitted WITHOUT
``ocr_labels`` — ``scripts.train_omvt_ssl`` already zeroes the OCR loss for
such rows, while the label-free SSL tasks (masked patch, orientation, layout)
train as usual. That is exactly what domain adaptation on noisy scans needs.

Production behaviors (deliberately not toy-grade):

- **Spread splitting**: landscape pages (aspect > ``--split-aspect``) are
  scanned double-page spreads; they are split into left/right halves so glyphs
  keep usable resolution. Reading order for Mongolian books is left page
  first, which matches the emitted order.
- **Aspect-preserving raster**: the long side is rendered to ``--page-px``
  and the result is padded (not stretched) to a square canvas, so the square
  resize inside ``PILImageProcessor`` cannot distort the vertical script.
- **Resume**: pages already recorded in ``meta.jsonl`` are skipped, so the
  tool is idempotent across reruns and new PDFs can be dropped into the same
  output dir later.
- **Per-page fault tolerance**: a corrupt page or PDF logs and moves on.
- **Text-layer flag**: pages whose PDF text layer contains Mongolian script
  are flagged in ``meta.jsonl`` (born-digital sources can later provide real
  OCR supervision); the text itself is stored only when Mongolian is present.

Usage::

    PYTHONPATH=. python3 -m scripts.ingest_scan_pdfs \
        --pdfs ../corpus/RAW/scans --out ../corpus/outputs/omvt_scan_v1 \
        --page-px 1024
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from pathlib import Path

_MN_LO, _MN_HI = "ᠠ", "ᢪ"


def _iter_pdfs(root: Path):
    if root.is_file() and root.suffix.lower() == ".pdf":
        yield root
        return
    yield from sorted(p for p in root.rglob("*.pdf") if p.is_file())


def _pad_square(img, fill: int):
    from PIL import Image

    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    canvas = Image.new("L", (side, side), fill)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def _ink_fraction(img) -> float:
    """Dark-pixel fraction at FULL resolution via histogram.

    Never measure this on a downsampled copy: averaging erases thin text
    strokes entirely (a page with 7% dark pixels at full res reads as 0%
    after a 64x64 bicubic resize), which silently discards real text pages.
    """
    hist = img.histogram()
    total = img.width * img.height
    return sum(hist[:160]) / max(total, 1)


def _mn_count(text: str) -> int:
    return sum(1 for ch in text if _MN_LO <= ch <= _MN_HI)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--pdfs", required=True, help="a PDF file or a directory of PDFs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--page-px", type=int, default=1024,
                    help="long-side raster size (before square padding)")
    ap.add_argument("--max-pages-per-pdf", type=int, default=0, help="0 = all")
    ap.add_argument("--min-ink", type=float, default=0.002,
                    help="skip pages whose dark-pixel fraction is below this; "
                    "0 disables")
    ap.add_argument("--split-aspect", type=float, default=1.15,
                    help="pages wider than this W/H ratio are split into "
                    "left/right halves (double-page spreads); 0 disables")
    ap.add_argument("--pad-fill", type=int, default=247)
    ap.add_argument("--rotate-map", default="",
                    help="JSON object {filename_substring: degrees} applied "
                    "right after rasterization (PIL convention: positive = "
                    "counter-clockwise), BEFORE aspect/split/ink checks. For "
                    "books scanned sideways, e.g. '{\"cd8a00cd\": -90}'.")
    args = ap.parse_args()
    rotate_map: dict[str, int] = (
        json.loads(args.rotate_map) if args.rotate_map else {}
    )

    import fitz  # PyMuPDF
    from PIL import Image

    root = Path(args.pdfs).expanduser()
    if not root.exists():
        raise FileNotFoundError(root)
    out_dir = Path(args.out).expanduser()
    page_dir = out_dir / "pages"
    page_dir.mkdir(parents=True, exist_ok=True)

    # Resume: anything already in meta.jsonl is complete.
    done: set[str] = set()
    meta_path = out_dir / "meta.jsonl"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["image"])
                except (json.JSONDecodeError, KeyError):
                    continue
    if done:
        print(f"[ingest] resume: {len(done)} page images already ingested")

    n_pdf = n_page = n_skip = n_split = n_err = 0
    with (out_dir / "ssl.jsonl").open("a", encoding="utf-8") as ssl_fh, \
            meta_path.open("a", encoding="utf-8") as meta_fh:
        for pdf_path in _iter_pdfs(root):
            try:
                doc = fitz.open(pdf_path)
                page_count = doc.page_count
            except Exception as exc:
                print(f"[ingest] SKIP PDF {pdf_path.name}: {exc}", file=sys.stderr)
                n_err += 1
                continue
            n_pdf += 1
            stem = unicodedata.normalize("NFC", pdf_path.stem)
            stem = "".join(ch if ch.isalnum() else "_" for ch in stem)[:64]
            # Sanitized+truncated stems can collide across books (same title in
            # different dirs would silently skip or overwrite each other); a
            # path digest keeps page names — and the resume keys derived from
            # them — unique per source file.
            stem += "_" + hashlib.sha1(
                str(pdf_path.resolve()).encode("utf-8")
            ).hexdigest()[:8]
            limit = args.max_pages_per_pdf or page_count
            rotation = 0
            for key, deg in rotate_map.items():
                if key in pdf_path.name or key in stem:
                    rotation = int(deg)
                    break
            if rotation:
                print(f"[ingest] {pdf_path.name[:40]}: rotate {rotation}°")

            for pi in range(min(page_count, limit)):
                try:
                    page = doc[pi]
                    rect = page.rect
                    zoom = args.page_px / max(rect.width, rect.height, 1)
                    pix = page.get_pixmap(
                        matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY
                    )
                    img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
                    if rotation:
                        # Paper-colored fill: non-right-angle rotations must
                        # not introduce black corners ahead of the ink filter.
                        img = img.rotate(
                            rotation, expand=True, fillcolor=args.pad_fill
                        )

                    halves: list[tuple[str, Image.Image]] = []
                    aspect = img.width / max(img.height, 1)
                    if args.split_aspect and aspect > args.split_aspect:
                        mid = img.width // 2
                        halves.append(("L", img.crop((0, 0, mid, img.height))))
                        halves.append(("R", img.crop((mid, 0, img.width, img.height))))
                        n_split += 1
                    else:
                        halves.append(("", img))

                    text = page.get_text() or ""
                    mn_chars = _mn_count(text)

                    for tag, half in halves:
                        name = f"{stem}_p{pi:04d}{tag}.png"
                        if name in done:
                            continue
                        if args.min_ink > 0 and _ink_fraction(half) < args.min_ink:
                            n_skip += 1
                            continue
                        half = _pad_square(half, args.pad_fill)
                        png_path = page_dir / name
                        half.save(png_path)
                        # The ssl row is flushed before its meta row so that
                        # the resume key (meta presence) implies the ssl row
                        # is on disk; a crash in between re-emits the page on
                        # rerun — a duplicate row at worst, never a lost one.
                        ssl_fh.write(json.dumps({
                            "images": [str(png_path.resolve())],
                            "image_sizes": [[half.height, half.width]],
                        }, ensure_ascii=False) + "\n")
                        ssl_fh.flush()
                        meta_row = {
                            "image": name,
                            "pdf": str(pdf_path.resolve()),
                            "page_index": pi,
                            "half": tag or None,
                            "kind": "scan",
                            "mn_text_chars": mn_chars,
                        }
                        if mn_chars >= 20:
                            # Born-digital Mongolian text layer: future OCR
                            # supervision; stored once per page (both halves
                            # carry it — alignment happens downstream).
                            meta_row["text"] = text
                        meta_fh.write(
                            json.dumps(meta_row, ensure_ascii=False) + "\n"
                        )
                        meta_fh.flush()
                        n_page += 1
                except Exception as exc:
                    n_err += 1
                    print(
                        f"[ingest] page error {pdf_path.name}:{pi}: {exc}",
                        file=sys.stderr,
                    )
            doc.close()
            print(f"[ingest] {pdf_path.name[:40]}: 累计 {n_page} 页", flush=True)

    print(
        f"[ingest] done: {n_pdf} pdfs -> {n_page} page images "
        f"({n_split} spreads split, {n_skip} near-blank skipped, {n_err} errors) "
        f"under {out_dir}"
    )
    return 0 if n_page else 1


if __name__ == "__main__":
    raise SystemExit(main())
