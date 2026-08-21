# -*- coding: utf-8 -*-
"""Character-per-token metrics by routed track."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from Tokenizer.evals.roundtrip_check import SMOKE_SAMPLES, _smoke_tokenizer


def _iter_text(path: str | None):
    if path is None:
        yield from SMOKE_SAMPLES
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if path.endswith(".jsonl"):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield str(obj.get("text", ""))
            else:
                yield line


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    tokenizer = _smoke_tokenizer()
    chars = 0
    tokens = 0
    by_track: Counter[str] = Counter()
    chars_by_track: Counter[str] = Counter()
    for text in _iter_text(args.input):
        result = tokenizer.encode_with_spans(text)
        chars += len(text)
        tokens += len(result.input_ids)
        for tok in result.tokens:
            by_track[tok.track] += 1
            chars_by_track[tok.track] += max(0, tok.end - tok.start)
    metrics = {
        "chars": chars,
        "tokens": tokens,
        "chars_per_token": chars / tokens if tokens else 0.0,
        "tokens_by_track": dict(by_track),
        "chars_per_token_by_track": {
            t: (chars_by_track[t] / by_track[t] if by_track[t] else 0.0)
            for t in by_track
        },
    }
    if args.json:
        json.dump(metrics, sys.stdout, ensure_ascii=False, indent=2)
        print()
    else:
        print(f"chars={metrics['chars']}")
        print(f"tokens={metrics['tokens']}")
        print(f"chars_per_token={metrics['chars_per_token']:.4f}")
        for track, count in sorted(by_track.items()):
            cpt = metrics["chars_per_token_by_track"][track]
            print(f"track_{track}_tokens={count} chars_per_token={cpt:.4f}")


if __name__ == "__main__":
    main()
