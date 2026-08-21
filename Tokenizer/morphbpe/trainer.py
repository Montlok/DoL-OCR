# -*- coding: utf-8 -*-
"""Simple boundary-constrained MorphBPE trainer."""

from __future__ import annotations

import heapq
import json
from collections import Counter, defaultdict
from typing import Iterable

from Tokenizer.traditional_mongolian.alphabet import MONGOLIAN_LETTERS
from Tokenizer.traditional_mongolian.stemmer import MongolStemmer
from Tokenizer.traditional_mongolian.unicode_norm import CTRL_ALL, strip_all_with_map

from .offsets import split_on_ascii_space
from .tokenizer import MorphBPETokenizer

MONGOLIAN_RANGES = [
    (0x1800, 0x18AF),
    (0x11660, 0x1167F),
]


def contains_mongolian(text: str) -> bool:
    return any(lo <= ord(ch) <= hi for ch in text for lo, hi in MONGOLIAN_RANGES)


class MorphBPETrainer:
    def __init__(
        self,
        stemmer: MongolStemmer | None = None,
        vocab_size: int = 4096,
        min_pair_freq: int = 2,
        min_boundary_confidence: float = 0.60,
        seed_alphabet: bool = True,
    ):
        self.stemmer = stemmer or MongolStemmer()
        self.vocab_size = vocab_size
        self.min_pair_freq = min_pair_freq
        self.min_boundary_confidence = min_boundary_confidence
        self.seed_alphabet = seed_alphabet

    def train(self, texts: Iterable[str]) -> MorphBPETokenizer:
        # Deduplicate words into a frequency table so each merge iteration costs
        # O(unique symbols) instead of O(total occurrences). Pair statistics are
        # maintained incrementally with a lazy max-heap, which keeps the greedy
        # most-frequent-pair selection (and the morpheme-boundary "forbidden"
        # constraint) identical while making large corpora tractable on CPU.
        words = self._unique_words(texts)

        vocab = {"<unk>": 0}
        if self.seed_alphabet:
            for ch in MONGOLIAN_LETTERS:
                vocab.setdefault(ch, len(vocab))
        # Register MVS/FVS/NIRUGU/NNBSP as standalone tokens: their presence
        # in the vocab is what switches MorphBPETokenizer into control-
        # preserving mode (encode keeps them in the id stream instead of
        # folding them — the GPT-2-Turkish-İ class of loss, 2.65% of mn
        # corpus chars). BPE itself still runs on control-free segments, so
        # merges never cross or include them.
        for ch in sorted(CTRL_ALL):
            vocab.setdefault(ch, len(vocab))
        for syms, _ends, _forbidden, _count in words:
            for piece in syms:
                vocab.setdefault(piece, len(vocab))

        pair_freq: Counter[tuple[str, str]] = Counter()
        pair_words: dict[tuple[str, str], set[int]] = defaultdict(set)
        for idx, (syms, ends, forbidden, count) in enumerate(words):
            for pair in self._word_pairs(syms, ends, forbidden):
                pair_freq[pair] += count
                pair_words[pair].add(idx)

        heap: list[tuple[int, tuple[str, str]]] = [
            (-freq, pair) for pair, freq in pair_freq.items()
        ]
        heapq.heapify(heap)

        merges: dict[tuple[str, str], tuple[str, int]] = {}
        while len(vocab) < self.vocab_size:
            best: tuple[str, str] | None = None
            best_freq = 0
            while heap:
                neg_freq, pair = heapq.heappop(heap)
                freq = pair_freq.get(pair, 0)
                if freq == 0 or -neg_freq != freq:
                    continue  # stale heap entry
                best, best_freq = pair, freq
                break
            if best is None or best_freq < self.min_pair_freq:
                break

            left, right = best
            merged = left + right
            merges[best] = (merged, len(merges))
            vocab.setdefault(merged, len(vocab))

            for idx in list(pair_words.get(best, ())):
                syms, ends, forbidden, count = words[idx]
                for pair in self._word_pairs(syms, ends, forbidden):
                    new_freq = pair_freq.get(pair, 0) - count
                    if new_freq > 0:
                        pair_freq[pair] = new_freq
                        heapq.heappush(heap, (-new_freq, pair))
                    else:
                        pair_freq.pop(pair, None)
                    pair_words[pair].discard(idx)

                new_syms, new_ends = self._apply_merge(
                    syms, ends, forbidden, left, right, merged
                )
                words[idx][0] = new_syms
                words[idx][1] = new_ends

                for pair in self._word_pairs(new_syms, new_ends, forbidden):
                    new_freq = pair_freq.get(pair, 0) + count
                    pair_freq[pair] = new_freq
                    pair_words[pair].add(idx)
                    heapq.heappush(heap, (-new_freq, pair))

            pair_words.pop(best, None)
            pair_freq.pop(best, None)

        return MorphBPETokenizer(
            vocab=vocab,
            merges=merges,
            stemmer=self.stemmer,
            min_boundary_confidence=self.min_boundary_confidence,
            seed_alphabet=self.seed_alphabet,
        )

    def save(self, tokenizer: MorphBPETokenizer, path: str) -> None:
        tokenizer.save(path)

    def train_to_file(self, texts: Iterable[str], path: str) -> MorphBPETokenizer:
        tokenizer = self.train(texts)
        tokenizer.save(path)
        return tokenizer

    def _unique_words(
        self, texts: Iterable[str]
    ) -> list[list]:
        counts: dict[str, int] = {}
        for text in texts:
            for start, end in split_on_ascii_space(text):
                word = text[start:end]
                if not contains_mongolian(word):
                    continue
                counts[word] = counts.get(word, 0) + 1

        words: list[list] = []
        for word, count in counts.items():
            skeleton, boundary_map = strip_all_with_map(word)
            if not skeleton:
                continue
            analysis = self.stemmer.analyze(word)
            forbidden = (
                set(analysis.skeleton_boundaries[1:-1])
                if analysis.confidence >= self.min_boundary_confidence
                else set()
            )
            syms = list(skeleton)
            ends = [boundary_map[i + 1] for i in range(len(skeleton))]
            words.append([syms, ends, forbidden, count])
        return words

    @staticmethod
    def _word_pairs(
        syms: list[str], ends: list[int], forbidden: set[int]
    ) -> Iterable[tuple[str, str]]:
        for i in range(len(syms) - 1):
            if ends[i] in forbidden:
                continue
            yield (syms[i], syms[i + 1])

    @staticmethod
    def _apply_merge(
        syms: list[str],
        ends: list[int],
        forbidden: set[int],
        left: str,
        right: str,
        merged: str,
    ) -> tuple[list[str], list[int]]:
        new_syms: list[str] = []
        new_ends: list[int] = []
        i = 0
        n = len(syms)
        while i < n:
            if (
                i < n - 1
                and syms[i] == left
                and syms[i + 1] == right
                and ends[i] not in forbidden
            ):
                new_syms.append(merged)
                new_ends.append(ends[i + 1])
                i += 2
            else:
                new_syms.append(syms[i])
                new_ends.append(ends[i])
                i += 1
        return new_syms, new_ends


def save_training_json(tokenizer: MorphBPETokenizer, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "vocab": tokenizer.vocab,
                "merges": [
                    {"pair": list(pair), "merged": merged, "rank": rank}
                    for pair, (merged, rank) in tokenizer.merges.items()
                ],
                "specials": {"unk": "<unk>"},
                "config": {
                    "type": "morphbpe",
                    "boundary_constrained": True,
                    "min_boundary_confidence": getattr(
                        tokenizer, "min_boundary_confidence", 0.60
                    ),
                    "seed_alphabet": getattr(tokenizer, "seed_alphabet", False),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
