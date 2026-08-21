# -*- coding: utf-8 -*-

"""Vocab-coverage gate: detect under-trained ("glitch") tokens.

The vocab is trained on one corpus snapshot but the model trains on the
*mixed* corpus (``build_corpus_mix``), so a token can exist in the vocab yet
(almost) never occur in what the model actually sees. Such tokens keep their
random-init embeddings — the SolidGoldMagikarp class of anomalies: feed one
at inference time and the model behaves erratically.

This tool encodes a sample of the final training mix, counts per-token hits,
and reports / gates on the fraction of trainable vocab ids that never fire.
Byte-fallback and special ids are excluded from the denominator (byte tokens
are legitimately rare; specials are injected by builders, not text).

Usage::

    python3 -m Tokenizer.evals.vocab_coverage \
        --bundle corpus/outputs/tok_build_v2/tokenizer/bundle \
        --data 'corpus/cleaned/**/*.jsonl' --sample-rows 200000 \
        --max-zero-frac 0.02 --report coverage.json
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import sys

from Tokenizer.unified.bundle import TokenizerBundle
from Tokenizer.unified.vocab import SPECIAL_TOKENS, make_byte_tokens


def iter_texts(paths: list[str], sample_rows: int):
    # Spread the sample budget evenly across files instead of reading the
    # glob in path order: a sorted multi-corpus glob would otherwise spend
    # the whole budget on the alphabetically-first domain (e.g. code/) and
    # report every other language's tokens as zero-hit.
    per_file = max(1, sample_rows // max(1, len(paths))) if sample_rows > 0 else 0
    for path in paths:
        taken = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = row.get("text", "")
                if not text:
                    continue
                yield text
                taken += 1
                if 0 < per_file <= taken:
                    break


def compute_coverage(bundle: TokenizerBundle, texts) -> dict:
    counts: collections.Counter[int] = collections.Counter()
    rows = 0
    for text in texts:
        counts.update(bundle.encode(text))
        rows += 1

    vocab = bundle.tokenizer.vocab
    excluded = set(SPECIAL_TOKENS) | set(make_byte_tokens())
    trainable = {tok: idx for tok, idx in vocab.items() if tok not in excluded}
    zero = sorted(tok for tok, idx in trainable.items() if counts[idx] == 0)
    rare_cut = 5
    rare = sorted(
        (tok for tok, idx in trainable.items() if 0 < counts[idx] < rare_cut),
        key=lambda tok: counts[vocab[tok]],
    )
    return {
        "rows_sampled": rows,
        "tokens_emitted": int(sum(counts.values())),
        "trainable_vocab": len(trainable),
        "zero_hit": len(zero),
        "zero_frac": len(zero) / max(1, len(trainable)),
        "rare_hit_lt5": len(rare),
        "zero_tokens_head": zero[:200],
        "rare_tokens_head": rare[:200],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True, help="tokenizer bundle dir")
    p.add_argument(
        "--data",
        required=True,
        nargs="+",
        help="JSONL globs of the FINAL training mix (not the vocab corpus)",
    )
    p.add_argument("--sample-rows", type=int, default=200_000)
    p.add_argument(
        "--max-zero-frac",
        type=float,
        default=None,
        help="gate: exit 1 if zero-hit fraction of trainable vocab exceeds this",
    )
    p.add_argument("--report", default="", help="optional JSON report path")
    args = p.parse_args()

    paths = sorted({m for pat in args.data for m in glob.glob(pat, recursive=True)})
    if not paths:
        print(f"vocab_coverage: no files match {args.data}", file=sys.stderr)
        return 2

    bundle = TokenizerBundle.from_dir(args.bundle)
    result = compute_coverage(bundle, iter_texts(paths, args.sample_rows))

    print(
        "vocab_coverage: {zero_hit}/{trainable_vocab} trainable tokens "
        "({zero_frac:.2%}) never occur in {rows_sampled} sampled rows; "
        "{rare_hit_lt5} occur fewer than 5 times".format(**result)
    )
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"vocab_coverage: report written to {args.report}")

    if args.max_zero_frac is not None and result["zero_frac"] > args.max_zero_frac:
        print(
            f"vocab_coverage: GATE FAILED zero_frac {result['zero_frac']:.2%} "
            f"> {args.max_zero_frac:.2%} — these ids would train as glitch "
            "tokens; rebuild the vocab or fix the mix",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
