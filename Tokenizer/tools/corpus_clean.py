#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Corpus cleaning pre-stage: global dedup + quality filter -> clean JSONL.

Runs after per-source cleaning and before ``build_corpus_mix.py``. Reads one or
more JSONL/TXT inputs, drops (near-)duplicate and low-quality documents across
*all* inputs with a single shared :class:`Deduper`, and writes the survivors to
one JSONL plus a stats report.

Examples:
  # dedup + quality-filter several Mongolian sources into one clean corpus
  python3 corpus_clean.py --in gov.jsonl mc2.jsonl --out mn_clean.jsonl \
      --script mongolian --min-script-ratio 0.6 --min-chars 80

  # just report what would be dropped (no write)
  python3 corpus_clean.py --in zhihu.jsonl --text-key RESPONSE --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus_filters import Deduper, QualityConfig, quality_ok  # noqa: E402


def _iter_docs(path: str, text_key: str):
    fmt = "jsonl" if path.endswith((".jsonl", ".json", ".ndjson")) else "txt"
    with open(path, encoding="utf-8") as fh:
        if fmt == "txt":
            blob = fh.read()
            for chunk in blob.split("\n\n"):
                chunk = chunk.strip()
                if chunk:
                    yield {"text": chunk}
            return
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, str):
                rec = {text_key: rec}
            yield rec


def run(args) -> Counter:
    cfg = QualityConfig(
        min_chars=args.min_chars,
        script=args.script,
        min_script_ratio=args.min_script_ratio,
    )
    deduper = Deduper(thresh=args.thresh)
    out_fh = None
    if args.out and not args.report:
        out_fh = open(args.out, "w", encoding="utf-8")

    stats: Counter = Counter()
    try:
        for path in args.inputs:
            for rec in _iter_docs(path, args.text_key):
                stats["total"] += 1
                text = rec.get(args.text_key, "") or ""
                ok, reason = quality_ok(text, cfg)
                if not ok:
                    stats[f"drop_{reason}"] += 1
                    continue
                if not deduper.seen(text):
                    stats["drop_duplicate"] += 1
                    continue
                stats["kept"] += 1
                if out_fh is not None:
                    out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    finally:
        if out_fh is not None:
            out_fh.close()
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inputs", nargs="+", required=True)
    ap.add_argument("--out", dest="out", default=None)
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--thresh", type=int, default=3)
    ap.add_argument("--min-chars", type=int, default=40)
    ap.add_argument("--script", default="", help="mongolian|zh to enforce purity")
    ap.add_argument("--min-script-ratio", type=float, default=0.5)
    ap.add_argument("--report", action="store_true", help="only print stats")
    args = ap.parse_args()

    stats = run(args)
    total = stats["total"] or 1
    print(f"total={stats['total']} kept={stats['kept']} "
          f"({stats['kept'] / total * 100:.1f}%)")
    for k in sorted(stats):
        if k.startswith("drop_"):
            print(f"  {k}={stats[k]} ({stats[k] / total * 100:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
