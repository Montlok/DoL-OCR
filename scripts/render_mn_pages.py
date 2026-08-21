# -*- coding: utf-8 -*-

"""Render traditional-Mongolian text into vertical-script page images + PDFs
for OMVT SSL training.

Why Chrome: this host's Pillow lacks libraqm, so ``scripts/build_ocr_data.py``'s
``direction="ttb"`` path cannot shape cursive Mongolian here. Headless Chrome
ships HarfBuzz and implements CSS ``writing-mode: vertical-lr`` (designed for
Mongolian), giving correct contextual glyph forms and NNBSP suffix joining.

Why screenshots, not ``--print-to-pdf``: Chrome's print path lays vertical
writing-mode lines out in their pre-rotation horizontal form (verified on
Chrome 149), so pages are captured with ``--screenshot`` and the per-document
PDF is assembled from the captured PNGs with PyMuPDF instead. PNG (training
sample) and PDF page are therefore pixel-identical; degradation is applied to
the PNGs only, after PDF assembly.

Outputs under ``--out``:

    pdf/doc_XXXXXX.pdf          one multi-page PDF per source document (clean)
    pages/doc_XXXXXX_pXXX.png   one grayscale PNG per page (square, maybe degraded)
    ssl.jsonl                   rows for ``scripts.train_omvt_ssl --data``:
                                {"images": [path], "image_sizes": [[H, W]],
                                 "ocr_labels": [[id, ...]]}
    meta.jsonl                  page -> source text map (for later VLM align)

Usage (from repo root)::

    PYTHONPATH=. python3 -m scripts.render_mn_pages \
        --input ../corpus/cleaned/mn_traditional/main/mn_traditional.clean.v3.jsonl \
        --font ../corpus/ARTIFACTS/fonts/OnonSoninSans.ttf \
        --tokenizer-bundle ../corpus/outputs/tok_build_v2/tokenizer/bundle \
        --out ../corpus/outputs/omvt_synth_v1 --docs 40
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import queue
import random
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

CHROME_DEFAULT = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# Suffix-joining NNBSP must stay glued to its word: split on plain spaces only
# (str.split() with no args would also split U+202F).
_WORD_SEP = " "

_MVS = "᠎"
_NNBSP = " "
_SEP_VOWELS = ("ᠠ", "ᠡ")  # ᠠ ᠡ — the only separated-final vowels
_MN_LETTER_LO, _MN_LETTER_HI = "ᠠ", "ᢪ"


def _to_presentation(text: str) -> str:
    """Nominal Unicode -> display form for rendering.

    Some normalized corpora use MVS (U+180E) for the historical suffix
    separator, while reviewed real-world fonts may key suffix shaping on NNBSP
    (U+202F).
    Rendering nominal text directly therefore shows wrong forms on *every*
    suffix. This maps suffix-separator MVS back to NNBSP while keeping genuine
    separated-vowel MVS (a single word-final ᠠ/ᠡ, e.g. ᠭᠠᠵᠠᠷ᠎ᠠ) intact.
    OCR labels must keep encoding the nominal text — only the pixels change.
    """
    out = []
    n = len(text)
    for i, ch in enumerate(text):
        if ch != _MVS:
            out.append(ch)
            continue
        nxt = text[i + 1] if i + 1 < n else ""
        nxt2 = text[i + 2] if i + 2 < n else ""
        is_sep_vowel = nxt in _SEP_VOWELS and not (
            nxt2 and _MN_LETTER_LO <= nxt2 <= _MN_LETTER_HI
        )
        out.append(_MVS if is_sep_vowel else _NNBSP)
    return "".join(out)


def _iter_docs(path: str, min_chars: int):
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text")
            if isinstance(text, str) and len(text) >= min_chars:
                yield text


def _normalize(text: str) -> str:
    # Collapse runs of layout whitespace but keep NNBSP (suffix separator).
    out = []
    prev_space = False
    for ch in text:
        if ch in ("\n", "\r", "\t", " "):
            if not prev_space:
                out.append(" ")
            prev_space = True
        else:
            out.append(ch)
            prev_space = False
    return "".join(out).strip()


def _chunk_words(words: list[str], budget_chars: int) -> list[str]:
    chunks, cur, cur_len = [], [], 0
    for w in words:
        add = len(w) + 1
        if cur and cur_len + add > budget_chars:
            chunks.append(_WORD_SEP.join(cur))
            cur, cur_len = [], 0
        cur.append(w)
        cur_len += add
    if cur:
        chunks.append(_WORD_SEP.join(cur))
    return chunks


def _page_budget(
    page_px: int, font_px: int, margin_px: int, line_height: float, fill: float
) -> int:
    inner = page_px - 2 * margin_px
    columns = max(1, int(inner / (font_px * line_height)))
    chars_per_col = max(1, int(inner / (font_px * 0.62)))
    return max(40, int(columns * chars_per_col * fill))


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _page_html(font_uri: str, page_px: int, p: dict) -> str:
    """Build one page's HTML. Text is converted to presentation form here —
    this is the single choke point between nominal data and rendered pixels."""
    title_html = (
        f'<div class="title" style="font-size:{p["font_px"] + 10}px">'
        f"{_esc(_to_presentation(p['title']))}</div>"
        if p.get("title")
        else ""
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
@font-face {{ font-family: "MnFont"; src: url("{font_uri}"); }}
* {{ margin: 0; box-sizing: border-box; }}
body {{
  width: {page_px}px; height: {page_px}px; overflow: hidden;
  writing-mode: vertical-lr;
  font-family: "MnFont";
  padding: {p["margin_px"]}px;
  font-size: {p["font_px"]}px;
  line-height: {p["line_height"]};
  color: {p["fg"]}; background: {p["bg"]};
}}
.title {{ margin-left: 0.6em; }}
</style></head><body>{title_html}<div>{_esc(_to_presentation(p["text"]))}</div></body></html>"""


class _ChromeCDP:
    """One persistent headless Chrome driven over the DevTools protocol.

    Spawning a fresh Chrome per page proved pathologically flaky on this host
    (first attempts routinely hung until the watchdog timeout, retries passed);
    a long-lived browser per worker renders a page in ~100-300ms instead and
    is the only design that scales to tens of thousands of pages.
    """

    def __init__(self, chrome: str, page_px: int) -> None:
        import urllib.request

        import websocket

        self.chrome = chrome
        self.page_px = page_px
        self.profile = tempfile.mkdtemp(prefix="mnrender_cdp_")
        self.proc = subprocess.Popen(
            [
                chrome,
                "--headless=new",
                "--disable-gpu",
                "--no-first-run",
                "--hide-scrollbars",
                "--mute-audio",
                "--force-device-scale-factor=1",
                f"--user-data-dir={self.profile}",
                f"--window-size={page_px},{page_px}",
                "--remote-debugging-port=0",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        port_file = Path(self.profile) / "DevToolsActivePort"
        deadline = time.time() + 30
        while not port_file.exists() or port_file.stat().st_size == 0:
            if time.time() > deadline or self.proc.poll() is not None:
                self.close()
                raise RuntimeError("chrome CDP endpoint did not come up")
            time.sleep(0.05)
        port = int(port_file.read_text().splitlines()[0])
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/json/new?about:blank", method="PUT"
        )
        info = json.loads(urllib.request.urlopen(req, timeout=10).read())
        self.ws = websocket.create_connection(
            info["webSocketDebuggerUrl"], timeout=60, suppress_origin=True
        )
        self._id = 0
        self._cmd("Page.enable", {})
        self._cmd("Runtime.enable", {})
        # --window-size includes browser chrome, so the viewport comes up
        # short (e.g. 800x713); force the exact page geometry instead.
        self._cmd(
            "Emulation.setDeviceMetricsOverride",
            {
                "width": page_px,
                "height": page_px,
                "deviceScaleFactor": 1,
                "mobile": False,
            },
        )

    def _cmd(self, method: str, params: dict) -> dict:
        self._id += 1
        mid = self._id
        self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        # Events arriving while we wait for our reply are irrelevant here
        # (readiness is polled via Runtime.evaluate), so just skip them.
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"CDP {method}: {msg['error']}")
                return msg.get("result", {})

    def _eval(self, expression: str, await_promise: bool = False):
        res = self._cmd(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": True,
                "awaitPromise": await_promise,
            },
        )
        return res.get("result", {}).get("value")

    def render(self, html_path: Path, png_path: Path) -> None:
        url = html_path.as_uri()
        self._cmd("Page.navigate", {"url": url})
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                if (
                    self._eval("location.href") == url
                    and self._eval("document.readyState") == "complete"
                ):
                    break
            except RuntimeError:
                pass  # execution context torn down mid-navigation; retry
            time.sleep(0.02)
        else:
            raise RuntimeError(f"navigation timeout for {html_path}")
        # Fonts load async even after readyState=complete; the glyphs are the
        # whole point of these images, so block on them explicitly.
        self._eval("document.fonts.ready.then(() => true)", await_promise=True)
        shot = self._cmd("Page.captureScreenshot", {"format": "png"})
        png_path.write_bytes(base64.b64decode(shot["data"]))
        if png_path.stat().st_size == 0:
            raise RuntimeError(f"empty screenshot for {html_path}")

    def close(self) -> None:
        try:
            if hasattr(self, "ws"):
                self.ws.close()
        except Exception:
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        shutil.rmtree(self.profile, ignore_errors=True)


def _render_job(
    sessions: "queue.Queue[_ChromeCDP]",
    chrome: str,
    page_px: int,
    html_path: Path,
    png_path: Path,
) -> None:
    sess = sessions.get()
    try:
        try:
            sess.render(html_path, png_path)
        except Exception:
            sess.close()
            sess = _ChromeCDP(chrome, page_px)
            sess.render(html_path, png_path)
    finally:
        sessions.put(sess)


def _degrade(img, rng: random.Random):
    from PIL import Image, ImageEnhance, ImageFilter

    ops = rng.sample(["blur", "jpeg", "rotate", "contrast"], k=rng.choice([1, 2]))
    if "rotate" in ops:
        img = img.rotate(rng.uniform(-0.9, 0.9), fillcolor=250, expand=False)
    if "blur" in ops:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.1)))
    if "contrast" in ops:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.75, 0.95))
    if "jpeg" in ops:
        buf = io.BytesIO()
        img.convert("L").save(buf, "JPEG", quality=rng.randint(35, 70))
        buf.seek(0)
        img = Image.open(buf).convert("L")
        img.load()
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--input", required=True, help="cleaned JSONL with 'text'")
    ap.add_argument("--font", required=True, help="Mongolian .ttf/.otf")
    ap.add_argument("--tokenizer-bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--docs", type=int, default=40, help="documents to render")
    ap.add_argument("--skip-docs", type=int, default=0, help="offset for sharding")
    ap.add_argument("--max-pages-per-doc", type=int, default=16)
    ap.add_argument("--snippet-ratio", type=float, default=0.3,
                    help="fraction of extra short-text pages (complete OCR label)")
    ap.add_argument("--degrade-ratio", type=float, default=0.5)
    ap.add_argument("--page-px", type=int, default=800, help="page side in px")
    ap.add_argument("--max-label-tokens", type=int, default=512)
    ap.add_argument("--min-doc-chars", type=int, default=400)
    ap.add_argument("--chrome", default=CHROME_DEFAULT)
    ap.add_argument("--workers", type=int, default=4, help="parallel chrome renders")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import fitz  # PyMuPDF
    from PIL import Image

    from Tokenizer.unified.bundle import TokenizerBundle

    if not Path(args.chrome).exists():
        raise FileNotFoundError(f"chrome not found: {args.chrome}")
    font_path = Path(args.font).resolve()
    if not font_path.exists():
        raise FileNotFoundError(f"font not found: {font_path}")

    bundle = TokenizerBundle.from_dir(args.tokenizer_bundle)
    rng = random.Random(args.seed)

    out_dir = Path(args.out)
    pdf_dir = out_dir / "pdf"
    page_dir = out_dir / "pages"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    page_dir.mkdir(parents=True, exist_ok=True)

    # Resume support: a doc with any row in meta.jsonl is complete — meta rows
    # are written and flushed only after the doc's PDF, every page PNG and all
    # of its (already flushed) ssl rows — so skip it.
    done_ids: set[str] = set()
    meta_path = out_dir / "meta.jsonl"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    done_ids.add(json.loads(line)["doc_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    if done_ids:
        print(f"[render-mn] resume: {len(done_ids)} docs already done, skipping")

    n_docs = n_pages = n_snippets = 0
    ssl_fh = open(out_dir / "ssl.jsonl", "a", encoding="utf-8")
    meta_fh = open(out_dir / "meta.jsonl", "a", encoding="utf-8")
    pool = ThreadPoolExecutor(max_workers=args.workers)
    sessions: "queue.Queue[_ChromeCDP]" = queue.Queue()
    for _ in range(args.workers):
        sessions.put(_ChromeCDP(args.chrome, args.page_px))

    try:
        for doc_idx, raw_text in enumerate(_iter_docs(args.input, args.min_doc_chars)):
            if doc_idx < args.skip_docs:
                continue
            if n_docs >= args.docs:
                break
            doc_id = f"doc_{doc_idx:06d}"
            if doc_id in done_ids:
                continue
            text = _normalize(raw_text)
            words = text.split(_WORD_SEP)

            font_px = rng.randint(20, 34)
            margin_px = rng.randint(48, 90)
            line_height = round(rng.uniform(1.35, 1.8), 2)
            fg = rng.choice(["#101010", "#1a1a2a", "#262626", "#30281e"])
            bg = rng.choice(["#ffffff", "#faf6ee", "#f4ead8", "#f0e6d2", "#efefef"])
            budget = _page_budget(
                args.page_px, font_px, margin_px, line_height, fill=0.80
            )

            chunks = _chunk_words(words, budget)[: args.max_pages_per_doc]
            pages = []
            for ci, chunk in enumerate(chunks):
                pages.append({
                    "kind": "page",
                    "text": chunk,
                    "title": _WORD_SEP.join(words[:4]) if ci == 0 else "",
                    "font_px": font_px,
                    "margin_px": margin_px,
                    "line_height": line_height,
                    "fg": fg,
                    "bg": bg,
                })
            # Short snippet pages: big type, few words, OCR label fully covers them.
            if rng.random() < args.snippet_ratio and len(words) > 24:
                start = rng.randrange(0, len(words) - 12)
                snip = _WORD_SEP.join(words[start : start + rng.randint(6, 14)])
                pages.append({
                    "kind": "snippet",
                    "text": snip,
                    "title": "",
                    "font_px": rng.randint(40, 56),
                    "margin_px": rng.randint(90, 150),
                    "line_height": round(rng.uniform(1.6, 2.2), 2),
                    "fg": fg,
                    "bg": bg,
                })

            # Render every page concurrently via chrome screenshots.
            tmp_dir = Path(tempfile.mkdtemp(prefix=f"mnrender_{doc_id}_"))
            futures = []
            try:
                for pi, p in enumerate(pages):
                    html_path = tmp_dir / f"p{pi:03d}.html"
                    html_path.write_text(
                        _page_html(font_path.as_uri(), args.page_px, p),
                        encoding="utf-8",
                    )
                    png_path = page_dir / f"{doc_id}_p{pi:03d}.png"
                    futures.append(pool.submit(
                        _render_job, sessions, args.chrome, args.page_px,
                        html_path, png_path,
                    ))
                for fut in futures:
                    fut.result()
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

            # Assemble the clean PDF first, then degrade training PNGs in place.
            pdf_path = pdf_dir / f"{doc_id}.pdf"
            pdf = fitz.open()
            for pi in range(len(pages)):
                png_path = page_dir / f"{doc_id}_p{pi:03d}.png"
                page = pdf.new_page(width=args.page_px, height=args.page_px)
                page.insert_image(page.rect, filename=str(png_path))
            pdf.save(pdf_path, deflate=True)
            pdf.close()

            doc_ssl_rows: list[str] = []
            doc_meta_rows: list[str] = []
            for pi, p in enumerate(pages):
                png_path = page_dir / f"{doc_id}_p{pi:03d}.png"
                img = Image.open(png_path).convert("L")
                if rng.random() < args.degrade_ratio:
                    img = _degrade(img, rng)
                img.save(png_path)

                ids = bundle.encode(p["text"], add_bos=False, add_eos=False)
                ids = ids[: args.max_label_tokens]
                doc_ssl_rows.append(json.dumps({
                    "images": [str(png_path.resolve())],
                    "image_sizes": [[img.height, img.width]],
                    "ocr_labels": [ids],
                }, ensure_ascii=False))
                doc_meta_rows.append(json.dumps({
                    "doc_id": doc_id,
                    "pdf": str(pdf_path.resolve()),
                    "page_index": pi,
                    "kind": p["kind"],
                    "n_label_tokens": len(ids),
                    "text": p["text"],
                }, ensure_ascii=False))
                n_pages += 1
                n_snippets += p["kind"] == "snippet"

            # Commit the whole doc at once, ssl rows flushed BEFORE the meta
            # rows that mark it done for resume. A crash inside this window
            # can only re-render the doc on the next run (duplicate ssl rows
            # at worst); it can never mark a doc done whose ssl rows were
            # still sitting in a lost write buffer.
            ssl_fh.write("".join(row + "\n" for row in doc_ssl_rows))
            ssl_fh.flush()
            meta_fh.write("".join(row + "\n" for row in doc_meta_rows))
            meta_fh.flush()
            n_docs += 1
            if n_docs % 10 == 0:
                print(f"[render-mn] {n_docs} docs, {n_pages} pages", flush=True)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        while not sessions.empty():
            try:
                sessions.get_nowait().close()
            except queue.Empty:
                break
        ssl_fh.close()
        meta_fh.close()

    print(
        f"[render-mn] done: {n_docs} docs -> {n_pages} pages "
        f"({n_snippets} snippets) under {out_dir}"
    )
    return 0 if n_pages else 1


if __name__ == "__main__":
    raise SystemExit(main())
