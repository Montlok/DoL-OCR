# -*- coding: utf-8 -*-

"""Deployment-grade L2 OCR evaluation: segmentation error included BY DESIGN.

:mod:`scripts.eval_vlm_ocr` answers the L1 question — "given a perfectly
cropped line strip, how well does the model read it?". Deployment never gets
perfect crops: a page must be segmented first, and the product's CER includes
every mis-cut. This tool measures that L2 number, mirroring the CRNN OCR
line's L2 concept: it synthesizes whole columns by vertically stacking K
consecutive val line images (configurable white gap), runs the FULL
deployment pipeline on each synthetic column —
:func:`Model.ocr.segment.lines_from_column` -> ``letterbox_to_square`` ->
``model.generate`` — concatenates the per-strip decodes in reading order, and
scores the concatenation against the concatenated ground truth with
:func:`Model.ocr.metrics.ocr_report` (grapheme CER headline).

The segmentation functions and letterbox/generate path are imported from
:mod:`Model.ocr.segment` / :mod:`scripts.ocr_infer` / \
:mod:`scripts.eval_vlm_ocr` — the same single implementation deployment runs,
so a segmentation regression moves this number and cannot hide.

Input is the SAME val row format the repo already produces
(:func:`Model.ocr.data.build_ocr_row` rows with absolute image paths, e.g.
``jsonl/val/shard-*.jsonl`` from :mod:`scripts.build_ocr_data_from_pairs`).
``--tokenizer-bundle`` is required: ground truth is decoded from each row's
supervised target ids. Rows are grouped in file order; a trailing partial
group is dropped (reported).

Reported: headline ``ocr_report`` numbers over concatenated columns,
per-script-bucket grapheme CER (mn/cjk/latin/other, see
:func:`Model.ocr.metrics.script_bucket_cer`), per-column grapheme-CER
distribution, n-lines-recovered stats (the segmenter is not told K), and an
index-0 spot check (decoded text only — never confidence proxies).

Usage::

    PYTHONPATH=. python3 -m scripts.eval_l2_deploy \
        --data /path/to/val.jsonl --tokenizer-bundle /path/to/bundle \
        --checkpoint /path/to/vlm_align_out \
        --lines-per-column 3 --gap-px 24 --limit 64 --out l2_preds.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from Model.config import EOS_ID, IMAGE_PATCH_ID  # noqa: E402
from Model.ocr.data import split_ocr_row  # noqa: E402
from Model.ocr.metrics import (  # noqa: E402
    edit_distance,
    grapheme_clusters,
    ocr_report,
)
from Model.ocr.segment import lines_from_column  # noqa: E402
from Tokenizer.multimodal import PILImageProcessor  # noqa: E402
from scripts.eval_vlm_ocr import (  # noqa: E402
    _decode_batches,
    _load_rows,
    _pixel_batch,
    print_script_cer,
)
from scripts.ocr_infer import (  # noqa: E402
    autocast_ctx_for,
    build_inference_prompt,
    load_deploy_model,
    resolve_device,
    strip_to_letterboxed,
)
from scripts.train_rdt import CONFIG_CHOICES  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="L2 deployment OCR eval: stack val lines into columns, "
        "run the full segmentation+generation pipeline, score concatenations"
    )
    p.add_argument("--data", default="", help="val JSONL of build_ocr_row rows "
                   "with absolute image paths")
    p.add_argument("--tokenizer-bundle", default="",
                   help="unified tokenizer bundle dir (required: decodes the "
                   "ground truth from each row's target ids)")
    p.add_argument("--checkpoint", default="",
                   help="train_vlm_align output root, step dir, or model.pt")
    p.add_argument("--allow-random-init", action="store_true",
                   help="run WITHOUT loading a checkpoint (random weights); "
                   "evaluator smoke only, prints a loud warning")
    p.add_argument("--config", choices=list(CONFIG_CHOICES),
                   default="two_stage_pretrain")
    p.add_argument("--mamba", choices=["auto", "official", "naive"], default="auto")
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--precision", choices=("auto", "fp32", "bf16"), default="auto")
    # Geometry: pinned to the production data build; must mirror training.
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--d-vision", type=int, default=512)
    p.add_argument("--n-image-tokens", type=int, default=256)
    p.add_argument("--patch-preset", choices=("derived", "prod"), default="prod")
    p.add_argument("--max-new-tokens", type=int, default=256,
                   help="per recovered strip, not per column")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    # Column synthesis + segmentation.
    p.add_argument("--lines-per-column", type=int, default=3,
                   help="K consecutive val lines stacked per synthetic column")
    p.add_argument("--gap-px", type=int, default=24,
                   help="white rows between stacked line images")
    p.add_argument("--limit", type=int, default=0,
                   help="evaluate at most N synthetic columns (0 = all)")
    p.add_argument("--target-line-px", type=int, default=0,
                   help="lines_from_column target strip height; 0 = --image-size "
                   "(each stacked unit is one letterboxed square)")
    p.add_argument("--ink-threshold", type=int, default=200)
    p.add_argument("--valley-frac", type=float, default=0.05)
    p.add_argument("--min-line-height", type=int, default=8)
    p.add_argument("--out", default="",
                   help="write per-column {column, n_lines_expected, "
                   "n_lines_found, ref, pred, grapheme_cer} JSONL here")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def group_rows(rows: list, k: int) -> tuple[list[list], int]:
    """Split rows into consecutive full groups of ``k``; drop the partial tail.

    Returns ``(groups, n_dropped)``. Grouping is in file order, so each
    synthetic column stacks lines that were adjacent in the val set.
    """
    if k < 1:
        raise ValueError("lines_per_column must be >= 1")
    n_full = len(rows) // k
    groups = [rows[i * k : (i + 1) * k] for i in range(n_full)]
    return groups, len(rows) - n_full * k


def stack_lines(line_images: list[Image.Image], gap_px: int) -> np.ndarray:
    """Vertically stack line images into one synthetic column array.

    White ``gap_px`` rows separate consecutive images; narrower images are
    centered horizontally on white. Returns a 2-D uint8 grayscale array.
    """
    if not line_images:
        raise ValueError("need at least one line image")
    if gap_px < 0:
        raise ValueError("gap_px must be >= 0")
    arrays = [np.asarray(img.convert("L")) for img in line_images]
    width = max(a.shape[1] for a in arrays)
    height = sum(a.shape[0] for a in arrays) + gap_px * (len(arrays) - 1)
    column = np.full((height, width), 255, dtype=np.uint8)
    y = 0
    for i, a in enumerate(arrays):
        if i:
            y += gap_px
        x0 = (width - a.shape[1]) // 2
        column[y : y + a.shape[0], x0 : x0 + a.shape[1]] = a
        y += a.shape[0]
    return column


def per_sample_grapheme_cer(pred: str, ref: str) -> float:
    gp, gr = grapheme_clusters(pred), grapheme_clusters(ref)
    return edit_distance(gp, gr) / max(len(gr), 1)


def _percentiles(values: list[float]) -> str:
    v = np.asarray(values, dtype=np.float64)
    q = np.percentile(v, [0, 25, 50, 75, 100])
    return (f"min={q[0]:.4f} p25={q[1]:.4f} median={q[2]:.4f} "
            f"p75={q[3]:.4f} max={q[4]:.4f}")


def main(argv=None) -> int:
    args = parse_args(argv)
    for flag in ("data", "tokenizer_bundle"):
        if not getattr(args, flag):
            print(f"scripts/eval_l2_deploy: --{flag.replace('_', '-')} is required",
                  file=sys.stderr)
            return 2
    if bool(args.checkpoint) == bool(args.allow_random_init):
        print(
            "scripts/eval_l2_deploy: pass --checkpoint for a real eval, or "
            "--allow-random-init (without --checkpoint) for evaluator smoke",
            file=sys.stderr,
        )
        return 2

    from Tokenizer.unified.bundle import TokenizerBundle

    bundle = TokenizerBundle.from_dir(args.tokenizer_bundle)
    decode = bundle.tokenizer.decode

    rows = _load_rows(args.data, 0)
    groups, n_dropped = group_rows(rows, args.lines_per_column)
    if args.limit:
        groups = groups[: args.limit]
    if not groups:
        print("scripts/eval_l2_deploy: not enough rows for one column "
              f"(rows={len(rows)}, K={args.lines_per_column})", file=sys.stderr)
        return 2
    # Geometry guard: the val rows must carry exactly the patch count the
    # tower will emit (mirrors eval_vlm_ocr's compress_to check).
    row_prompt, _, _ = split_ocr_row(groups[0][0], eos_id=EOS_ID)
    n_img = row_prompt.count(IMAGE_PATCH_ID)
    if n_img != args.n_image_tokens:
        print(
            f"scripts/eval_l2_deploy: rows carry {n_img} <image_patch> slots "
            f"but --n-image-tokens={args.n_image_tokens}; geometry flags must "
            "mirror the data build / training run",
            file=sys.stderr,
        )
        return 2

    target_line_px = args.target_line_px or args.image_size

    # Segment every synthetic column through the deployment pipeline.
    refs: list[str] = []
    letterboxed: list[Image.Image] = []
    col_slices: list[tuple[int, int]] = []  # strip range per column
    n_lines_found: list[int] = []
    for group in groups:
        targets, images = [], []
        for row in group:
            _, target, image_ref = split_ocr_row(row, eos_id=EOS_ID)
            if image_ref is None:
                raise ValueError("val rows must carry an image reference")
            targets.append(target)
            images.append(image_ref)
        refs.append(" ".join(decode(t) for t in targets))
        pils = []
        for ref in images:
            with Image.open(ref) as raw:
                raw.load()
                pils.append(raw.convert("L"))
        column = stack_lines(pils, args.gap_px)
        # No page_w/page_h: this column is a synthetic reconstruction (K val
        # line images vertically stacked), not a crop from a real page, so
        # there is no meaningful "page width" to check the width-outlier
        # rule against -- the ink-fraction/absolute-width/height checks in
        # Model.ocr.segment.is_plausible_line still apply unconditionally.
        spans, _rejections = lines_from_column(
            column,
            target_line_px,
            ink_threshold=args.ink_threshold,
            valley_frac=args.valley_frac,
            min_height=args.min_line_height,
        )
        start = len(letterboxed)
        for y0, y1 in spans:
            letterboxed.append(strip_to_letterboxed(column[y0:y1, :],
                                                    args.image_size))
        col_slices.append((start, len(letterboxed)))
        n_lines_found.append(len(spans))

    print(
        f"[l2] {len(groups)} synthetic column(s) x K={args.lines_per_column} "
        f"lines (gap {args.gap_px}px, {n_dropped} tail row(s) dropped); "
        f"segmenter recovered {len(letterboxed)} strip(s)",
        flush=True,
    )

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    prompt = build_inference_prompt(args.n_image_tokens)
    args.seq_len = len(prompt) + args.max_new_tokens + 1
    model, omvt_cfg = load_deploy_model(args, device)
    if omvt_cfg.compress_to != args.n_image_tokens:
        raise ValueError(
            f"tower compress_to={omvt_cfg.compress_to} != "
            f"--n-image-tokens {args.n_image_tokens}"
        )
    processor = PILImageProcessor(image_size=args.image_size)

    texts: list[str] = []
    if letterboxed:
        def pixels(start, end):
            return _pixel_batch(letterboxed[start:end], processor, omvt_cfg,
                                device)

        autocast_ctx = autocast_ctx_for(device, args.precision)
        with contextlib.redirect_stdout(sys.stderr):
            preds_ids = _decode_batches(
                model, [prompt] * len(letterboxed), pixels, args, device,
                autocast_ctx,
            )
        texts = [decode(ids) for ids in preds_ids]

    # A column whose segmentation found nothing decodes to "" and takes the
    # full deletion penalty — that is the deployment truth, not an error.
    preds = [" ".join(texts[s:e]) for s, e in col_slices]

    rep = ocr_report(preds, refs)
    print(
        f"[l2] n={rep.n} grapheme_cer={rep.grapheme_cer:.4f} "
        f"norm_cer={rep.norm_cer:.4f} raw_cer={rep.raw_cer:.4f} "
        f"wer={rep.wer:.4f} column_exact={rep.line_exact:.4f} "
        f"(backend={rep.backend})"
    )
    print_script_cer("[l2]", rep)
    per_col = [per_sample_grapheme_cer(p, r) for p, r in zip(preds, refs)]
    print(f"[l2] per-column grapheme CER: {_percentiles(per_col)}")

    found = np.asarray(n_lines_found)
    exact = float((found == args.lines_per_column).mean())
    hist = {int(k): int((found == k).sum()) for k in sorted(set(found.tolist()))}
    print(
        f"[l2] n_lines recovered: expected {args.lines_per_column}/column, "
        f"mean {found.mean():.2f}, exact-count rate {exact:.4f}, "
        f"histogram {hist}"
    )

    print("[l2] index-0 spot check (decoded text only):")
    print(f"  ref : {refs[0]}")
    print(f"  pred: {preds[0]}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            for i, (p, r) in enumerate(zip(preds, refs)):
                fh.write(json.dumps(
                    {
                        "column": i,
                        "n_lines_expected": args.lines_per_column,
                        "n_lines_found": n_lines_found[i],
                        "ref": r,
                        "pred": p,
                        "grapheme_cer": per_col[i],
                    },
                    ensure_ascii=False,
                ) + "\n")
        print(f"[l2] wrote {len(preds)} columns -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
