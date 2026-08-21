# -*- coding: utf-8 -*-
"""Train a boundary-constrained MorphBPE model.

Usage:
  python -m Tokenizer.tools.build_morphbpe --input data.jsonl --output mb.json --vocab-size 4096
"""

from __future__ import annotations

import argparse
import json
from typing import Iterable

from Tokenizer.morphbpe import MorphBPETrainer
from Tokenizer.traditional_mongolian.stemmer import MongolStemmer


def _iter_texts(path: str) -> Iterable[str]:
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
                text = obj.get("text") if isinstance(obj, dict) else None
                if text:
                    yield str(text)
            else:
                yield line


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="path to .jsonl or .txt")
    parser.add_argument("--output", required=True, help="output .json model file")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--min-pair-freq", type=int, default=2)
    parser.add_argument("--min-boundary-confidence", type=float, default=0.60)
    args = parser.parse_args()

    trainer = MorphBPETrainer(
        stemmer=MongolStemmer(),
        vocab_size=args.vocab_size,
        min_pair_freq=args.min_pair_freq,
        min_boundary_confidence=args.min_boundary_confidence,
    )
    tokenizer = trainer.train(_iter_texts(args.input))
    tokenizer.save(args.output, extra_config={"source": args.input})
    print(f"vocab_size={len(tokenizer.vocab)}")
    print(f"merges={len(tokenizer.merges)}")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
