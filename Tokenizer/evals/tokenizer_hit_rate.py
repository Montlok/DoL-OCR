# -*- coding: utf-8 -*-
"""Tokenizer coverage and hit-rate metrics for mixed Mongolian/general text."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from typing import Any, Iterable

from Tokenizer.evals.mongolian_boundary_recall import compute_metrics
from Tokenizer.generic_bpe import is_byte_token
from Tokenizer.morphbpe import MorphBPETrainer
from Tokenizer.traditional_mongolian.alphabet import is_mongolian_codepoint
from Tokenizer.unified.bundle import TokenizerBundle
from Tokenizer.unified.dual_tokenizer import (
    DualTrackTokenizer,
    build_unified_vocab,
)


def iter_text(path: str) -> list[str]:
    texts = []
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
                text = str(obj.get("text", ""))
            else:
                text = line
            if text:
                texts.append(text)
    return texts


def build_experimental_tokenizer(
    train_texts: list[str],
    vocab_size: int,
    min_pair_freq: int,
    seed_alphabet: bool = True,
    general_vocab_size: int = 8000,
) -> DualTrackTokenizer:
    morphbpe = MorphBPETrainer(
        vocab_size=vocab_size,
        min_pair_freq=min_pair_freq,
        seed_alphabet=seed_alphabet,
    ).train(train_texts)

    from Tokenizer.generic_bpe import GeneralBPETrainer

    general = GeneralBPETrainer(
        vocab_size=general_vocab_size,
        min_frequency=min_pair_freq,
        show_progress=False,
    ).train(train_texts)

    vocab = build_unified_vocab(
        morphbpe_vocab=morphbpe.vocab,
        general_vocab=general.get_vocab(),
    )
    return DualTrackTokenizer(
        unified_vocab=vocab,
        morphbpe=morphbpe,
        general=general,
    )


def mongolian_words(texts: Iterable[str]) -> list[str]:
    words = []
    for text in texts:
        for word in text.split():
            if any(is_mongolian_codepoint(ch) for ch in word):
                words.append(word)
    return words


def compute_hit_rate(texts: list[str], tokenizer: DualTrackTokenizer) -> dict[str, Any]:
    chars = 0
    tokens = 0
    unk_count = 0
    byte_fallback = 0
    mongolian_chars = 0
    mongolian_native_chars = 0
    mongolian_fallback_tokens = 0
    by_track: Counter[str] = Counter()
    chars_by_track: Counter[str] = Counter()
    unk_by_track: Counter[str] = Counter()
    mn_word_total = 0
    mn_word_hit = 0
    failures = []

    for text in texts:
        chars += len(text)
        result = tokenizer.encode_with_spans(text)
        decoded = tokenizer.decode(result.input_ids)
        if decoded != text and len(failures) < 10:
            failures.append({"text": text, "decoded": decoded})

        tokens += len(result.tokens)
        for token in result.tokens:
            by_track[token.track] += 1
            chars_by_track[token.track] += max(0, token.end - token.start)
            if token.id == tokenizer.unk_id:
                unk_count += 1
                unk_by_track[token.track] += 1
            if token.track == "general" and is_byte_token(token.token):
                byte_fallback += 1

        for span in result.spans:
            if span.lang != "mn":
                continue
            span_tokens = [
                token
                for token in result.tokens
                if token.start >= span.start and token.end <= span.end
            ]
            native_positions = set(range(span.start, span.end))
            for token in span_tokens:
                if token.track == "mn" and token.id != tokenizer.unk_id:
                    continue
                mongolian_fallback_tokens += 1
                native_positions.difference_update(
                    range(
                        max(span.start, token.start),
                        min(span.end, token.end),
                    )
                )
            mongolian_chars += span.end - span.start
            mongolian_native_chars += len(native_positions)
            for word in span.text.split():
                mn_word_total += 1
            if span.text and all(token.id != tokenizer.unk_id for token in span_tokens):
                mn_word_hit += len(span.text.split())

    words = mongolian_words(texts)
    boundary = compute_metrics(words, tokenizer.morphbpe) if words else {}
    return {
        "texts": len(texts),
        "chars": chars,
        "tokens": tokens,
        "chars_per_token": chars / tokens if tokens else 0.0,
        "unk_count": unk_count,
        "unk_rate": unk_count / tokens if tokens else 0.0,
        "token_hit_rate": 1.0 - (unk_count / tokens if tokens else 0.0),
        "byte_fallback_tokens": byte_fallback,
        "byte_fallback_rate": byte_fallback / tokens if tokens else 0.0,
        "mongolian_chars": mongolian_chars,
        "mongolian_native_chars": mongolian_native_chars,
        "mongolian_fallback_tokens": mongolian_fallback_tokens,
        "mongolian_native_char_rate": (
            mongolian_native_chars / mongolian_chars
            if mongolian_chars
            else 1.0
        ),
        "mongolian_words": mn_word_total,
        "mongolian_word_hit_rate": mn_word_hit / mn_word_total
        if mn_word_total
        else 1.0,
        "tokens_by_track": dict(by_track),
        "unk_by_track": dict(unk_by_track),
        "chars_per_token_by_track": {
            track: chars_by_track[track] / by_track[track]
            for track in by_track
            if by_track[track]
        },
        "boundary": boundary,
        "roundtrip_failures_sample": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help=".txt or .jsonl evaluation data")
    parser.add_argument("--train-input", help="optional .txt/.jsonl training data")
    parser.add_argument("--tokenizer-bundle", help="load a persisted TokenizerBundle")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--min-pair-freq", type=int, default=2)
    parser.add_argument("--no-seed-alphabet", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    texts = iter_text(args.input)
    if args.tokenizer_bundle:
        tokenizer = TokenizerBundle.from_dir(args.tokenizer_bundle).tokenizer
    else:
        train_texts = iter_text(args.train_input) if args.train_input else texts
        tokenizer = build_experimental_tokenizer(
            train_texts,
            vocab_size=args.vocab_size,
            min_pair_freq=args.min_pair_freq,
            seed_alphabet=not args.no_seed_alphabet,
        )

    metrics = compute_hit_rate(texts, tokenizer)
    if args.json:
        json.dump(metrics, sys.stdout, ensure_ascii=False, indent=2)
        print()
        return

    for key, value in metrics.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
