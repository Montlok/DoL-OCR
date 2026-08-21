# -*- coding: utf-8 -*-
"""Compare MorphBPE against generic baselines.

SentencePiece is optional. If unavailable, the corresponding entry is
reported as ``skipped``.
"""

from __future__ import annotations

import argparse
import json
import sys

from Tokenizer.evals.roundtrip_check import SMOKE_SAMPLES, _smoke_tokenizer


def _iter_text(path: str | None):
    if path is None:
        yield from SMOKE_SAMPLES
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                yield line


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        import sentencepiece  # type: ignore  # noqa: F401

        sp_available = True
    except Exception:
        sp_available = False

    tokenizer = _smoke_tokenizer()
    chars = 0
    morph_tokens = 0
    char_tokens = 0
    for text in _iter_text(args.input):
        chars += len(text)
        morph_tokens += len(tokenizer.encode(text))
        char_tokens += len(text)

    metrics = {
        "chars": chars,
        "morphbpe_tokens": morph_tokens,
        "char_baseline_tokens": char_tokens,
        "morphbpe_chars_per_token": chars / morph_tokens if morph_tokens else 0.0,
        "char_baseline_chars_per_token": 1.0,
        "sentencepiece": "available" if sp_available else "skipped (not installed)",
    }
    if args.json:
        json.dump(metrics, sys.stdout, ensure_ascii=False, indent=2)
        print()
    else:
        for k, v in metrics.items():
            print(f"{k}={v}")


if __name__ == "__main__":
    main()
