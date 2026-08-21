# -*- coding: utf-8 -*-
"""Validate token offset monotonicity and bounds."""

from __future__ import annotations

import argparse
import json
import sys

from Tokenizer.evals.chars_per_token import _iter_text
from Tokenizer.evals.roundtrip_check import _smoke_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    tokenizer = _smoke_tokenizer()
    checked = 0
    invalid = 0
    non_monotonic = 0
    for text in _iter_text(args.input):
        prev_start = -1
        for token in tokenizer.encode_with_spans(text).tokens:
            if token.start == -1 and token.end == -1:
                continue
            checked += 1
            if token.start < 0 or token.end < token.start or token.end > len(text):
                invalid += 1
            if token.start < prev_start:
                non_monotonic += 1
            prev_start = token.start
    metrics = {
        "tokens_checked": checked,
        "invalid_offsets": invalid,
        "non_monotonic_spans": non_monotonic,
    }
    if args.json:
        json.dump(metrics, sys.stdout, ensure_ascii=False, indent=2)
        print()
    else:
        for k, v in metrics.items():
            print(f"offset_{k}={v}")


if __name__ == "__main__":
    main()
