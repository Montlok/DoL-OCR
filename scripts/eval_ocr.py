# -*- coding: utf-8 -*-

"""OCR accuracy evaluation CLI for traditional Mongolian.

Scores a JSONL of OCR predictions against references. Each line is::

    {"pred": "<recognized text>", "ref": "<ground truth>", "confidence": 0.97}

``confidence`` is optional; with ``--reject-below T`` any sample whose confidence
is below ``T`` is withheld (counted toward the rejection rate, not scored) — this
models high-precision corpus ingestion where low-confidence pages are dropped
rather than transcribed.

Reports (see :mod:`Model.ocr.metrics` for the rationale):

- **grapheme CER** (headline): raw text clustered into user-perceived
  graphemes before comparing, so one missed combining mark counts as one
  error, not one per code point.
- **normalized CER**: both sides folded to nominal Mongolian Unicode
  first, so FVS/MVS/joiner rendering differences are not charged as errors.
- **raw CER**: unmodified code points (true encoding gap).
- **WER**: whitespace word error rate.
- **line-exact**: fraction of lines matching exactly after folding.
- **rejection rate**: fraction withheld by the confidence gate.

Usage::

    python -m scripts.eval_ocr --pred ocr.jsonl
    python -m scripts.eval_ocr --pred ocr.jsonl --reject-below 0.9 --backend rust
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Model.ocr.metrics import ocr_report  # noqa: E402


def _load(path: str) -> tuple[list[str], list[str], list[float | None]]:
    preds: list[str] = []
    refs: list[str] = []
    confs: list[float | None] = []
    # Stream line-by-line: OCR eval corpora can be large, so avoid loading the
    # whole file into memory. Wrap parse errors with path:lineno for debugging.
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if "pred" not in obj or "ref" not in obj:
                raise ValueError(
                    f"{path}:{lineno}: each line needs 'pred' and 'ref' keys"
                )
            preds.append(obj["pred"])
            refs.append(obj["ref"])
            c = obj.get("confidence")
            try:
                confs.append(float(c) if c is not None else None)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}:{lineno}: 'confidence' is not a number: {c!r}"
                ) from exc
    return preds, refs, confs


def main() -> int:
    ap = argparse.ArgumentParser(description="Traditional Mongolian OCR eval")
    ap.add_argument("--pred", required=True, help="JSONL of {pred, ref[, confidence]}")
    ap.add_argument(
        "--backend",
        choices=["auto", "rust", "python"],
        default="auto",
        help="nominal-folding backend for normalized CER (default: auto)",
    )
    ap.add_argument(
        "--reject-below",
        type=float,
        default=None,
        help="withhold samples whose confidence is below this threshold",
    )
    args = ap.parse_args()

    preds, refs, confs = _load(args.pred)
    if not preds:
        print("[ocr-eval] no samples found")
        return 1

    rejected = None
    if args.reject_below is not None:
        rejected = [
            (c is None) or (c < args.reject_below) for c in confs
        ]

    rep = ocr_report(preds, refs, backend=args.backend, rejected=rejected)
    print(
        f"[ocr-eval] n={rep.n} backend={rep.backend} "
        f"grapheme_cer={rep.grapheme_cer:.4f} "
        f"norm_cer={rep.norm_cer:.4f} raw_cer={rep.raw_cer:.4f} "
        f"wer={rep.wer:.4f} line_exact={rep.line_exact:.4f} "
        f"rejection={rep.rejection_rate:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
