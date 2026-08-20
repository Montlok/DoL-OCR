# -*- coding: utf-8 -*-

"""Single-command OCR inference for DoL-OCR checkpoints (deployment path).

Takes one line-strip image, one full page image (``--page``), or one PDF page
(``--pdf --pdf-page N``) and prints the transcription, one line of text per
detected line strip, in reading order. The checkpoint is loaded exactly the
way :mod:`scripts.eval_vlm_ocr` loads it (same arch resolution via
``CONFIG_CHOICES`` + ``_resolve_mamba_backend``, same tower construction via
``make_omvt_cfg``, same ``model.pt``/step-dir/output-root resolution) and
decoding reuses that module's ``_decode_batches`` — this CLI adds
segmentation and I/O around the proven eval path, it does not reinvent it.

Pipeline (page mode)::

    page -> Model.ocr.segment.detect_columns          (column boxes)
         -> Model.ocr.segment.lines_from_column       (strip spans per column)
         -> letterbox_to_square(image_size)           (REUSED from the data builder)
         -> [BOS] <image_start> <image_patch>*N <image_end>  prompt rows
         -> model.generate (greedy, cache-free; kl_exit early-exit active
            whenever the config sets kl_exit_threshold, e.g. two_stage_pretrain)
         -> tokenizer-bundle decode -> stdout, reading order

Reading order: traditional Mongolian reads top-to-bottom within a column,
columns advancing LEFT-TO-RIGHT. Evidence in this repo: the production page
renderer uses CSS ``writing-mode: vertical-lr`` (left-to-right block
progression per the CSS Writing Modes spec; :mod:`scripts.render_mn_pages`)
and :func:`scripts.build_ocr_data.render_vertical_line` states "layout
columns left-to-right is the script's natural flow". ``--column-order ltr``
is therefore the default; ``rtl`` exists for atypical material only.

Geometry: ``--image-size 224 --d-vision 512 --n-image-tokens 256
--patch-preset prod`` are pinned to the production OCR data build
(:mod:`scripts.build_ocr_data_from_pairs` defaults). They MUST mirror the
flags of the training run that produced ``--checkpoint`` — note
``--n-image-tokens 256`` is an explicit override (224px would derive 64, not
256), and a mismatched tower fails ``load_state_dict`` loudly rather than
decoding garbage. If loading fails on shape mismatch, read the training run's
recorded metadata instead of guessing flags.

PDF input renders the requested page via ``pdftoppm`` (poppler) when on
``$PATH``, else PyMuPDF (``fitz``) when importable, else exits with an
actionable install hint. Both tiers are implemented here.

Usage::

    PYTHONPATH=. python3 -m scripts.ocr_infer \
        --image line_strip.png \
        --checkpoint /path/to/vlm_align_out --tokenizer-bundle /path/to/bundle

    PYTHONPATH=. python3 -m scripts.ocr_infer \
        --pdf book.pdf --pdf-page 3 --column-order ltr \
        --checkpoint /path/to/vlm_align_out --tokenizer-bundle /path/to/bundle

    # CPU pipeline smoke, random weights (garbage text BY CONSTRUCTION):
    PYTHONPATH=. python3 -m scripts.ocr_infer --image page.png --page \
        --config two_stage_tiny --allow-random-init --max-new-tokens 6
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from Model.config import (  # noqa: E402
    BOS_ID,
    EOS_ID,
    IMAGE_END_ID,
    IMAGE_PATCH_ID,
    IMAGE_START_ID,
)
from Model.ocr.data import build_ocr_row, split_ocr_row  # noqa: E402
from Model.ocr.image_preprocess import (  # noqa: E402
    letterbox_grayscale_to_square,
)
from Model.ocr.position_contract import (  # noqa: E402
    BOUNDARY_V1,
    OCR_POSITION_CONTRACT_CHOICES,
)
from Model.ocr.segment import detect_columns, lines_from_column  # noqa: E402
from Tokenizer.multimodal import PILImageProcessor  # noqa: E402
from scripts.eval_vlm_ocr import (  # noqa: E402
    _build_model,
    _decode_batches,
    _load_model_state,
    _pixel_batch,
    _resolve_ocr_position_contract,
    _restore_omvt_geometry,
    _validate_checkpoint_tokenizer,
)
from scripts.train_rdt import CONFIG_CHOICES  # noqa: E402

# Any non-special vocab id; used only to satisfy build_ocr_row's non-empty
# target requirement so the prompt half can be split back out. Never decoded.
_DUMMY_TARGET_ID = 300

_RANDOM_INIT_BANNER = """\
================================================================
WARNING: --allow-random-init is set and no checkpoint was loaded.
The model weights are RANDOM. Every decoded string below is
garbage BY CONSTRUCTION. This mode exists only to smoke-test the
segmentation -> letterbox -> prompt -> generate -> decode wiring.
Never use it for real OCR; pass --checkpoint instead.
================================================================"""


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Single-command OCR inference (line strip, page, or PDF page)"
    )
    src = p.add_argument_group("input (exactly one of --image / --pdf)")
    src.add_argument("--image", default="", help="PNG/JPG line strip or full page")
    src.add_argument(
        "--page",
        action="store_true",
        help="treat --image as a full page (segment into columns/lines); "
        "without this flag --image is decoded as a single line strip",
    )
    src.add_argument("--pdf", default="", help="PDF file (implies page mode)")
    src.add_argument(
        "--pdf-page", type=int, default=1, help="1-based PDF page number (default 1)"
    )
    src.add_argument(
        "--pdf-dpi", type=int, default=200, help="PDF raster resolution (default 200)"
    )

    m = p.add_argument_group("model (must mirror the training run's flags)")
    m.add_argument("--checkpoint", default="", help="train_vlm_align output root, "
                   "step dir, or model.pt (same resolution as eval_vlm_ocr)")
    m.add_argument("--tokenizer-bundle", default="",
                   help="unified tokenizer bundle dir (id -> text)")
    m.add_argument("--config", choices=list(CONFIG_CHOICES),
                   default="two_stage_pretrain")
    m.add_argument("--mamba", choices=["auto", "official", "naive"], default="auto")
    m.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    m.add_argument("--precision", choices=("auto", "fp32", "bf16"), default="auto",
                   help="'auto' = bf16 autocast on cuda, fp32 elsewhere")
    m.add_argument("--image-size", type=int, default=224)
    m.add_argument("--d-vision", type=int, default=512)
    m.add_argument("--n-image-tokens", type=int, default=256,
                   help="visual tokens per strip; explicit override, MUST equal "
                   "the training build's value (224px would derive 64, not 256)")
    m.add_argument("--patch-preset", choices=("derived", "prod"), default="prod")
    m.add_argument("--max-new-tokens", type=int, default=256)
    m.add_argument("--batch-size", type=int, default=8)
    m.add_argument(
        "--recurrent-steps",
        type=int,
        default=None,
        help="fixed decode depth; defaults to the checkpoint's trained depth",
    )
    m.add_argument("--repetition-penalty", type=float, default=1.0)
    m.add_argument(
        "--ocr-position-contract",
        choices=OCR_POSITION_CONTRACT_CHOICES,
        default=None,
        help=(
            "must match checkpoint metadata; historical checkpoints require "
            "explicit legacy_sequential_v0"
        ),
    )
    m.add_argument("--allow-random-init", action="store_true",
                   help="run WITHOUT loading a checkpoint (random weights); "
                   "pipeline smoke only, prints a loud warning")

    s = p.add_argument_group("segmentation (page mode)")
    s.add_argument("--column-order", choices=("ltr", "rtl"), default="ltr",
                   help="column reading order; ltr is the traditional-Mongolian "
                   "convention (CSS writing-mode: vertical-lr, see module doc)")
    s.add_argument("--target-line-px", type=int, default=640,
                   help="preferred strip height in source px before letterboxing; "
                   "training strips were ~400-900 px tall (default 640)")
    s.add_argument("--ink-threshold", type=int, default=200)
    s.add_argument("--valley-frac", type=float, default=0.05)
    s.add_argument("--min-column-width", type=int, default=12)
    s.add_argument("--min-column-gap", type=int, default=6)
    s.add_argument("--min-line-height", type=int, default=8)

    p.add_argument("--out", default="",
                   help="also write {column, line, box, n_tokens, text} JSONL here")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def render_pdf_page(pdf_path: str, page_number: int, dpi: int) -> Image.Image:
    """Rasterize one PDF page to a grayscale PIL image.

    Tier 1: ``pdftoppm`` subprocess (poppler). Tier 2: PyMuPDF. Neither
    available -> actionable error. ``page_number`` is 1-based (pdftoppm
    ``-f/-l`` convention).
    """
    pdf_path = str(pdf_path)
    if not Path(pdf_path).exists():
        raise SystemExit(f"scripts/ocr_infer: PDF not found: {pdf_path}")
    if page_number < 1:
        raise SystemExit("scripts/ocr_infer: --pdf-page is 1-based (must be >= 1)")

    pdftoppm_reason = ""
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm:
        with tempfile.TemporaryDirectory() as td:
            prefix = os.path.join(td, "page")
            proc = subprocess.run(
                [pdftoppm, "-png", "-gray", "-r", str(dpi),
                 "-f", str(page_number), "-l", str(page_number), pdf_path, prefix],
                capture_output=True, text=True,
            )
            outs = sorted(Path(td).glob("page-*.png"))
            if proc.returncode == 0 and outs:
                with Image.open(outs[0]) as raw:
                    raw.load()
                    return raw.convert("L")
            pdftoppm_reason = (
                f"pdftoppm rc={proc.returncode}, {len(outs)} file(s) produced"
                f" ({proc.stderr.strip()[:200] or 'page may be out of range'})"
            )
            print(f"[infer] {pdftoppm_reason}; trying pymupdf", file=sys.stderr)
    else:
        pdftoppm_reason = "pdftoppm not on $PATH"

    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise SystemExit(
            "scripts/ocr_infer: cannot rasterize PDF: "
            f"{pdftoppm_reason}, and pymupdf is not installed. Fix either:\n"
            "  brew install poppler    # provides pdftoppm\n"
            "  pip install pymupdf     # provides fitz"
        ) from None
    doc = fitz.open(pdf_path)
    if page_number > doc.page_count:
        raise SystemExit(
            f"scripts/ocr_infer: --pdf-page {page_number} out of range "
            f"(1..{doc.page_count})"
        )
    scale = dpi / 72.0
    pix = doc[page_number - 1].get_pixmap(
        matrix=fitz.Matrix(scale, scale), colorspace=fitz.csGRAY
    )
    return Image.frombytes("L", (pix.width, pix.height), pix.samples)


def order_columns(boxes, column_order: str):
    """Reading order over column boxes: ltr (default, see module doc) or rtl."""
    boxes = sorted(boxes, key=lambda b: b[0])
    if column_order == "rtl":
        boxes.reverse()
    return boxes


def strip_to_letterboxed(strip: np.ndarray, image_size: int) -> Image.Image:
    """Convert one numpy strip with the shared legacy OCR letterbox."""

    image = Image.fromarray(np.ascontiguousarray(strip))
    return letterbox_grayscale_to_square(image, image_size)


def segment_page(
    page: np.ndarray,
    *,
    column_order: str = "ltr",
    target_line_px: int = 640,
    ink_threshold: int = 200,
    valley_frac: float = 0.05,
    min_column_width: int = 12,
    min_column_gap: int = 6,
    min_line_height: int = 8,
) -> list[dict]:
    """Full-page segmentation into reading-order strip records.

    Returns ``[{"column", "line", "box", "strip"}, ...]`` where ``box`` is
    ``[x0, y0, x1, y1]`` in page coordinates and ``strip`` is the cropped
    numpy array. Shared by :mod:`scripts.eval_l2_deploy` — single
    implementation of the deployment segmentation order. ``page``'s true
    dimensions are forwarded to :func:`Model.ocr.segment.lines_from_column`
    as ``page_w``/``page_h`` so its quality filter's width-vs-page-width
    outlier check is active here (real page geometry is available, unlike
    :mod:`scripts.eval_l2_deploy`'s synthetic reconstructed columns).
    """
    page_h, page_w = page.shape
    columns = detect_columns(
        page,
        ink_threshold=ink_threshold,
        valley_frac=valley_frac,
        min_width=min_column_width,
        min_gap=min_column_gap,
    )
    records: list[dict] = []
    for ci, (x0, y0, x1, y1) in enumerate(order_columns(columns, column_order)):
        column = page[y0:y1, x0:x1]
        spans, _rejections = lines_from_column(
            column,
            target_line_px,
            ink_threshold=ink_threshold,
            valley_frac=valley_frac,
            min_height=min_line_height,
            page_w=page_w,
            page_h=page_h,
        )
        for li, (ly0, ly1) in enumerate(spans):
            records.append(
                {
                    "column": ci,
                    "line": li,
                    "box": [int(x0), int(y0 + ly0), int(x1), int(y0 + ly1)],
                    "strip": column[ly0:ly1, :],
                }
            )
    return records


def build_inference_prompt(n_image_tokens: int) -> list[int]:
    """``[BOS] <image_start> <image_patch>*N <image_end>`` via the row contract.

    Built through :func:`Model.ocr.data.build_ocr_row` with a throwaway
    target and split back out with :func:`split_ocr_row` (the pattern
    ``eval_vlm_ocr._smoke`` uses), so the prompt can never drift from the
    training rows.
    """
    row = build_ocr_row(
        [_DUMMY_TARGET_ID],
        n_image_tokens,
        "inference",
        bos_id=BOS_ID,
        image_start_id=IMAGE_START_ID,
        image_patch_id=IMAGE_PATCH_ID,
        image_end_id=IMAGE_END_ID,
        eos_id=EOS_ID,
    )
    prompt, _target, _image = split_ocr_row(row, eos_id=EOS_ID)
    return prompt


def load_deploy_model(args, device: torch.device):
    """Build + load the deployment model exactly like ``scripts.eval_vlm_ocr``.

    With ``--allow-random-init`` and no ``--checkpoint``, skips loading and
    prints the loud random-weights banner to stderr (pipeline smoke only).
    Returns ``(model, omvt_cfg)``.
    """
    if args.checkpoint:
        _resolve_ocr_position_contract(args)
    elif getattr(args, "ocr_position_contract", None) is None:
        args.ocr_position_contract = BOUNDARY_V1
    model = _build_model(args, device)
    omvt_cfg = model.vision._omvt_cfg
    if args.checkpoint:
        state, ckpt_path = _load_model_state(args.checkpoint)
        model.load_state_dict(state)
        print(f"[infer] loaded {ckpt_path} on {device}", file=sys.stderr)
    else:
        print(_RANDOM_INIT_BANNER, file=sys.stderr)
    return model, omvt_cfg


def autocast_ctx_for(device: torch.device, precision: str):
    """Autocast context factory, mirroring eval_vlm_ocr's precision handling."""
    if precision == "auto":
        precision = "bf16" if device.type == "cuda" else "fp32"
    if precision == "bf16":
        def ctx():
            return torch.autocast(device.type, dtype=torch.bfloat16)

        return ctx
    return torch.no_grad


def make_decode_runtime(
    tokenizer_bundle: str,
    device: torch.device,
    *,
    checkpoint: str = "",
):
    """Return decode while validating the checkpoint tokenizer identity."""
    if tokenizer_bundle:
        from Tokenizer.unified.bundle import TokenizerBundle

        bundle = TokenizerBundle.from_dir(tokenizer_bundle)
        issues = bundle.validate()
        if issues:
            raise ValueError(
                "invalid tokenizer bundle:\n  - " + "\n  - ".join(issues)
            )
        if checkpoint:
            _validate_checkpoint_tokenizer(
                checkpoint,
                bundle,
                tokenizer_bundle,
                require_terminal=True,
            )
        return bundle.tokenizer.decode, None
    print(
        "[infer] no --tokenizer-bundle: printing raw token ids (smoke only)",
        file=sys.stderr,
    )
    return (lambda ids: " ".join(map(str, ids))), None


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.recurrent_steps is not None and args.recurrent_steps <= 0:
        print("scripts/ocr_infer: --recurrent-steps must be positive", file=sys.stderr)
        return 2
    if bool(args.image) == bool(args.pdf):
        print("scripts/ocr_infer: pass exactly one of --image / --pdf",
              file=sys.stderr)
        return 2
    if bool(args.checkpoint) == bool(args.allow_random_init):
        print(
            "scripts/ocr_infer: pass --checkpoint for real inference, or "
            "--allow-random-init (without --checkpoint) for pipeline smoke",
            file=sys.stderr,
        )
        return 2
    if not args.tokenizer_bundle and not args.allow_random_init:
        print("scripts/ocr_infer: --tokenizer-bundle is required "
              "(omittable only with --allow-random-init)", file=sys.stderr)
        return 2
    # Restore geometry before building the visual prompt or letterboxing the
    # page.  The full model builder repeats this for the RDT side later.
    _restore_omvt_geometry(args)

    if args.pdf:
        page_img = render_pdf_page(args.pdf, args.pdf_page, args.pdf_dpi)
        page_mode = True  # a PDF page is a full page by definition
        source = f"{args.pdf}#page{args.pdf_page}"
    else:
        with Image.open(args.image) as raw:
            raw.load()
            page_img = raw.convert("L")
        page_mode = args.page
        source = args.image

    page = np.asarray(page_img)
    if page_mode:
        records = segment_page(
            page,
            column_order=args.column_order,
            target_line_px=args.target_line_px,
            ink_threshold=args.ink_threshold,
            valley_frac=args.valley_frac,
            min_column_width=args.min_column_width,
            min_column_gap=args.min_column_gap,
            min_line_height=args.min_line_height,
        )
        n_columns = 1 + max((r["column"] for r in records), default=-1)
        print(
            f"[infer] {source}: {n_columns} column(s), {len(records)} line "
            f"strip(s), column order {args.column_order}",
            file=sys.stderr,
        )
    else:
        h, w = page.shape
        records = [{"column": 0, "line": 0, "box": [0, 0, w, h], "strip": page}]
    if not records:
        print("[infer] no text detected on the page; nothing to decode",
              file=sys.stderr)
        return 0

    letterboxed = [strip_to_letterboxed(r["strip"], args.image_size)
                   for r in records]

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    prompt = build_inference_prompt(args.n_image_tokens)
    prompts = [prompt] * len(records)
    args.seq_len = len(prompt) + args.max_new_tokens + 1

    decode, _morphology_track_table = make_decode_runtime(
        args.tokenizer_bundle,
        device,
        checkpoint=args.checkpoint,
    )
    model, omvt_cfg = load_deploy_model(args, device)
    if omvt_cfg.compress_to != args.n_image_tokens:
        raise ValueError(
            f"tower compress_to={omvt_cfg.compress_to} but prompts carry "
            f"{args.n_image_tokens} <image_patch> slots; geometry flags must "
            "mirror training"
        )
    processor = PILImageProcessor(image_size=args.image_size)

    def pixels(start, end):
        return _pixel_batch(letterboxed[start:end], processor, omvt_cfg, device)

    autocast_ctx = autocast_ctx_for(device, args.precision)
    # _decode_batches logs progress to stdout; stdout here is reserved for
    # the transcription itself, so route the progress lines to stderr.
    with contextlib.redirect_stdout(sys.stderr):
        preds_ids = _decode_batches(
            model,
            prompts,
            pixels,
            args,
            device,
            autocast_ctx,
        )

    texts = [decode(ids) for ids in preds_ids]
    for text in texts:
        print(text)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            for rec, ids, text in zip(records, preds_ids, texts):
                fh.write(json.dumps(
                    {
                        "column": rec["column"],
                        "line": rec["line"],
                        "box": rec["box"],
                        "n_tokens": len(ids),
                        "text": text,
                    },
                    ensure_ascii=False,
                ) + "\n")
        print(f"[infer] wrote {len(records)} rows -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
