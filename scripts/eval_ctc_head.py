# -*- coding: utf-8 -*-

"""Evaluate a CTC head checkpoint (frozen OMVT tower + trained head).

The CTC checkpoint (``scripts/train_ctc_head.py``) carries only the trained
head plus the ``omvt_config`` needed to reconstruct tower *geometry* — not
the (frozen, unchanged) tower *weights*, which still live in the original
``--omvt-checkpoint`` this head was trained against. This mirrors
``scripts/eval_omvt_ssl.py`` / ``scripts/eval_vlm_ocr.py``: the tower is
loaded once from its own checkpoint via
:mod:`Model.training.omvt_checkpoint`, then the head is loaded on top.

Rows are scored with :func:`Model.ocr.metrics.ocr_report` — the same
yardstick :mod:`scripts.eval_vlm_ocr` uses for the generative path, so the
two numbers are directly comparable.

Usage::

    PYTHONPATH=. python3 -m scripts.eval_ctc_head \\
        --checkpoint ~/dolocr/runs/ctc_head_v1/latest \\
        --omvt-checkpoint ~/dolocr/runs/omvt_ssl_v1/latest \\
        --val ~/dolocr/data_v1/val.jsonl \\
        --tokenizer-bundle ~/dolocr/bundle_v3b \\
        --out ~/dolocr/runs/ctc_head_v1/eval_preds.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Model.config import OMVTConfig  # noqa: E402
from Model.ocr.metrics import ocr_report  # noqa: E402
from Model.omvt import OMVTVisionTower  # noqa: E402
from Model.training.omvt_checkpoint import (  # noqa: E402
    load_omvt_payload,
    tower_state_from_payload,
)
from Tokenizer.multimodal import PILImageProcessor  # noqa: E402

from scripts.eval_vlm_ocr import print_script_cer  # noqa: E402
from scripts.train_ctc_head import (  # noqa: E402
    CTCHead,
    bytes_to_text,
    greedy_ctc_decode,
    load_ctc_payload,
    row_to_byte_target,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Greedy CTC-head OCR eval (CER)")
    p.add_argument("--checkpoint", required=True, help="train_ctc_head.py output (dir or ctc_head.pt)")
    p.add_argument(
        "--omvt-checkpoint", required=True,
        help="train_omvt_ssl checkpoint the head was trained against (tower weights)",
    )
    p.add_argument(
        "--use-ema-tower",
        type=lambda s: s.lower() not in ("0", "false", "no"),
        default=True,
        help="prefer 'tower_ema' from --omvt-checkpoint when present (default True, "
        "matching the trainer's default)",
    )
    p.add_argument("--val", required=True, help="val JSONL of build_ocr_row rows")
    p.add_argument("--tokenizer-bundle", required=True, help="unified tokenizer bundle dir")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit", type=int, default=0, help="evaluate at most N rows (0 = all)")
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument(
        "--precision", choices=("auto", "fp32", "bf16"), default="auto",
        help="'auto' = bf16 autocast on cuda (matches training), fp32 elsewhere",
    )
    p.add_argument("--out", default="", help="write {image, ref, pred} JSONL here")
    return p.parse_args(argv)


def _load_rows(path: str, limit: int) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


@torch.no_grad()
def _run_eval(
    rows: list[dict],
    decode,
    tower: OMVTVisionTower,
    head: CTCHead,
    processor: PILImageProcessor,
    device: torch.device,
    batch_size: int,
    use_bf16: bool,
) -> tuple[list[str], list[str], list[str]]:
    """Returns ``(images, preds, refs)`` (as text), skipping rows with no image."""

    images: list[str] = []
    refs: list[str] = []
    byte_targets: list[list[int]] = []
    for row in rows:
        image_ref, byte_ids = row_to_byte_target(row, decode)
        if image_ref is None:
            continue
        images.append(image_ref)
        byte_targets.append(byte_ids)
        refs.append(bytes_to_text(byte_ids))

    preds: list[str] = []
    n = len(images)
    t0 = time.time()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        pixels = processor(images[start:end]).to(device)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
        ):
            features = tower(pixels)["compressed"]
        logits = head(features.float())
        for byte_ids in greedy_ctc_decode(logits):
            preds.append(bytes_to_text(byte_ids))
        done = min(end, n)
        rate = (time.time() - t0) / done if done else 0.0
        print(f"[eval] decoded {done}/{n} rows ({rate:.2f}s/row)", flush=True)

    return images, preds, refs


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_bf16 = args.precision == "bf16" or (
        args.precision == "auto" and device.type == "cuda"
    )

    payload = load_ctc_payload(args.checkpoint)
    omvt_cfg = OMVTConfig(**payload["omvt_config"])
    head_cfg = payload.get("head_config", {})
    hidden = int(head_cfg.get("hidden", 384))

    omvt_payload = load_omvt_payload(args.omvt_checkpoint, weights_only=False)
    if isinstance(omvt_payload, dict) and "omvt_config" in omvt_payload:
        tower_cfg = OMVTConfig(**omvt_payload["omvt_config"])
        if tower_cfg != omvt_cfg:
            raise ValueError(
                "omvt_config mismatch between --checkpoint "
                f"({args.checkpoint}) and --omvt-checkpoint ({args.omvt_checkpoint}); "
                "the CTC head must be evaluated against the same tower geometry "
                "it was trained on"
            )
    tower = OMVTVisionTower(omvt_cfg).to(device).eval()
    use_ema = args.use_ema_tower and isinstance(omvt_payload, dict) and bool(
        omvt_payload.get("tower_ema")
    )
    if use_ema:
        print("[eval] using EMA tower weights")
    tower.load_state_dict(tower_state_from_payload(omvt_payload, use_ema=use_ema))
    if payload.get("tower_state") is not None:
        # A jointly fine-tuned run saved its own tower; that supersedes the
        # SSL weights or the eval would silently score the wrong vision.
        tower.load_state_dict(payload["tower_state"])
        print("[eval] using fine-tuned tower_state from CTC checkpoint")
    for p in tower.parameters():
        p.requires_grad_(False)

    head = CTCHead(d_vision=omvt_cfg.d_vision, hidden=hidden).to(device)
    head.load_state_dict(payload["head"])
    head.eval()
    print(f"[eval] loaded CTC checkpoint step={payload.get('step')} from {args.checkpoint}")

    from Tokenizer.unified.bundle import TokenizerBundle

    bundle = TokenizerBundle.from_dir(args.tokenizer_bundle)
    decode = bundle.tokenizer.decode

    rows = _load_rows(args.val, args.limit)
    processor = PILImageProcessor(image_size=omvt_cfg.image_size)
    images, preds, refs = _run_eval(
        rows, decode, tower, head, processor, device, args.batch_size, use_bf16
    )

    rep = ocr_report(preds, refs)
    print(
        f"[eval] n={rep.n} grapheme_cer={rep.grapheme_cer:.4f} "
        f"norm_cer={rep.norm_cer:.4f} raw_cer={rep.raw_cer:.4f} "
        f"wer={rep.wer:.4f} line_exact={rep.line_exact:.4f}"
    )
    print_script_cer("[eval]", rep)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            for image, pred, ref in zip(images, preds, refs):
                fh.write(
                    json.dumps({"image": image, "pred": pred, "ref": ref}, ensure_ascii=False)
                    + "\n"
                )
        print(f"[eval] wrote {len(images)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
