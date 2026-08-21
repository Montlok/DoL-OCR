# -*- coding: utf-8 -*-
"""Mongolian morpheme-boundary recall for MorphBPE."""

from __future__ import annotations

import argparse
import json
import sys

from Tokenizer.morphbpe import MorphBPETokenizer
from Tokenizer.morphbpe.offsets import MorphToken
from Tokenizer.traditional_mongolian.unicode_norm import MVS, NNBSP, strip_all_with_map
from Tokenizer.traditional_mongolian.stemmer import MongolStemmer


SMOKE_WORDS = [
    "ᠮᠣᠩᠭᠣᠯ",
    "ᠪᠢᠴᠢᠭ" + MVS + "ᠦᠨ",
    "ᠨᠡᠷ" + NNBSP + "ᠦᠦ",
]


def _iter_words(path: str | None):
    if path is None:
        yield from SMOKE_WORDS
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for word in line.split():
                yield word


def _crossed_boundaries(
    word: str, boundaries: set[int], tokens: list[MorphToken]
) -> int:
    _, boundary_map = strip_all_with_map(word)
    crossed = 0
    for boundary in boundaries:
        if boundary <= 0 or boundary >= len(boundary_map):
            continue
        original_boundary = boundary_map[boundary]
        if any(tok.start < original_boundary < tok.end for tok in tokens):
            crossed += 1
    return crossed


def compute_metrics(words, tokenizer: MorphBPETokenizer | None = None) -> dict:
    stemmer = MongolStemmer()
    total_words = 0
    morpheme_boundaries = 0
    crossed = 0
    token_boundaries = 0
    for word in words:
        analysis = stemmer.analyze(word)
        total_words += 1
        boundaries = set(analysis.skeleton_boundaries[1:-1])
        morpheme_boundaries += len(boundaries)
        if tokenizer is None:
            _, boundary_map = strip_all_with_map(word)
            tokens = [
                MorphToken("", -1, boundary_map[i], boundary_map[i + 1])
                for i in range(max(0, len(boundary_map) - 1))
            ]
        else:
            tokens = tokenizer.tokenize_word(word)
        token_boundaries += max(0, len(tokens) - 1)
        crossed += _crossed_boundaries(word, boundaries, tokens)
    respected = morpheme_boundaries - crossed
    return {
        "words": total_words,
        "morpheme_boundaries": morpheme_boundaries,
        "token_boundaries": token_boundaries,
        "crossed_boundary_count": crossed,
        "boundary_respecting_rate": (
            respected / morpheme_boundaries if morpheme_boundaries else 1.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input")
    parser.add_argument("--morphbpe", help="optional MorphBPE model JSON")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    tokenizer = (
        MorphBPETokenizer.from_file(args.morphbpe, MongolStemmer())
        if args.morphbpe
        else None
    )
    metrics = compute_metrics(_iter_words(args.input), tokenizer)
    if args.json:
        json.dump(metrics, sys.stdout, ensure_ascii=False, indent=2)
        print()
    else:
        for k, v in metrics.items():
            print(f"boundary_{k}={v}")


if __name__ == "__main__":
    main()
