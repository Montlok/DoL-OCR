# -*- coding: utf-8 -*-
"""Train the general multilingual byte-level BPE track.

  python -m Tokenizer.tools.build_general_bpe \
      --input general_corpus.jsonl --output general.json --vocab-size 40000

Input is ``.jsonl`` (one object per line, ``text`` field) or ``.txt`` (one
document per line). The trained model is a byte-level BPE: it covers every
script losslessly (Chinese, English, Japanese, Cyrillic, ...) and never emits
``<unk>``.
"""

from __future__ import annotations

import argparse
import json
from typing import Iterable


def _iter_texts(paths: list[str]) -> Iterable[str]:
    for path in paths:
        is_jsonl = path.endswith(".jsonl")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                if is_jsonl:
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
    parser.add_argument(
        "--input", required=True, nargs="+", help="one or more .jsonl/.txt files"
    )
    parser.add_argument("--output", required=True, help="output tokenizers JSON")
    parser.add_argument("--vocab-size", type=int, default=40000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--progress", action="store_true", help="show trainer bars")
    args = parser.parse_args()

    # General segment is [24576, 65536) -> at most 40960 tokens.
    max_general = 65536 - 24576
    if args.vocab_size > max_general:
        parser.error(
            f"--vocab-size {args.vocab_size} exceeds the general segment "
            f"capacity ({max_general})"
        )

    try:
        from Tokenizer.generic_bpe import GeneralBPETrainer
    except ImportError as exc:
        raise SystemExit(f"Missing dependency: {exc}") from exc

    trainer = GeneralBPETrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        show_progress=args.progress,
    )
    model = trainer.train(_iter_texts(args.input))
    model.save(args.output)
    print(f"vocab_size={model.vocab_size}")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
