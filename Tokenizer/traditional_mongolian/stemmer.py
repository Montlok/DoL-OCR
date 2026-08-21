# -*- coding: utf-8 -*-
"""Traditional Mongolian reverse stemmer.

Input is normalized Unicode traditional Mongolian. Matching is performed on a
control-stripped skeleton, so FVS, MVS, NIRUGU, and NNBSP do not block suffix
recognition. ``boundaries`` are offsets into the original input string;
``skeleton_boundaries`` are offsets into the control-stripped skeleton.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

try:
    from .morph_rules import get_harmony, harmony_ok, sequence_sane, stacking_ok
    from .suffixes import ALL_SUFFIXES_BY_ORDER
    from .unicode_norm import control_boundaries, strip_all, strip_all_with_map
except ImportError:  # pragma: no cover - supports direct script execution.
    from morph_rules import (  # type: ignore[no-redef]
        get_harmony,
        harmony_ok,
        sequence_sane,
        stacking_ok,
    )
    from suffixes import ALL_SUFFIXES_BY_ORDER  # type: ignore[no-redef]
    from unicode_norm import (  # type: ignore[no-redef]
        control_boundaries,
        strip_all,
        strip_all_with_map,
    )


def slice_original(text: str, start: int, end: int, boundary_map: list[int]) -> str:
    if start >= end:
        return ""
    return text[boundary_map[start] : boundary_map[end]]


@dataclass(frozen=True)
class SuffixItem:
    skeleton: str
    surface: str
    suffix_id: str
    suffix_type: str
    order: int
    declared_harmony: str
    surface_harmony: str
    attaches_to: str


@dataclass
class StemResult:
    word: str
    root: str
    suffixes: list[str] = field(default_factory=list)
    suffix_ids: list[str] = field(default_factory=list)
    suffix_types: list[str] = field(default_factory=list)
    boundaries: list[int] = field(default_factory=list)
    skeleton_boundaries: list[int] = field(default_factory=list)
    confidence: float = 0.0


INFLECTIONAL_SUFFIX_TYPES = {
    "case",
    "possessive",
    "plural",
    "participle",
    "converb",
    "particle",
    "tense",
    "mood",
    "negation",
}

INNER_FORMATION_TYPES = {
    "derivational",
    "voice",
    "aspect",
}


def build_suffix_lookup(suffix_list: Iterable[dict[str, Any]]) -> list[SuffixItem]:
    lookup: list[SuffixItem] = []
    seen: set[tuple[str, str]] = set()

    for entry in suffix_list:
        for surface in entry.get("surface", []):
            if not surface:
                continue

            skeleton = strip_all(surface)
            if not skeleton:
                continue

            key = (skeleton, entry["id"])
            if key in seen:
                continue

            seen.add(key)
            lookup.append(
                SuffixItem(
                    skeleton=skeleton,
                    surface=surface,
                    suffix_id=entry["id"],
                    suffix_type=entry["type"],
                    order=entry.get("order", 0),
                    declared_harmony=entry.get("harmony", "none"),
                    surface_harmony=get_harmony(surface),
                    attaches_to=entry.get("attaches_to", "any"),
                )
            )

    lookup.sort(
        key=lambda x: (
            -len(x.skeleton),
            -x.order,
            x.suffix_type,
            x.suffix_id,
        )
    )
    return lookup


class MongolStemmer:
    def __init__(
        self,
        min_root_len: int = 2,
        max_depth: int = 6,
        min_suffix_confidence: float = 0.60,
    ):
        self.min_root_len = min_root_len
        self.max_depth = max_depth
        self.min_suffix_confidence = min_suffix_confidence
        self.lookup = build_suffix_lookup(ALL_SUFFIXES_BY_ORDER)

    def analyze(self, word: str) -> StemResult:
        skeleton, boundary_map = strip_all_with_map(word)
        separator_boundaries = control_boundaries(word)

        if len(skeleton) < self.min_root_len:
            return StemResult(
                word=word,
                root=word,
                boundaries=[0, len(word)],
                skeleton_boundaries=[0, len(skeleton)],
                confidence=0.30,
            )

        best = self._strip(
            skeleton=skeleton,
            end=len(skeleton),
            separator_boundaries=separator_boundaries,
            suffix_spans=[],
            suffix_ids=[],
            suffix_types=[],
            depth=0,
        )

        if best is None:
            return StemResult(
                word=word,
                root=word,
                boundaries=[0, len(word)],
                skeleton_boundaries=[0, len(skeleton)],
                confidence=0.50,
            )

        root_end, suffix_spans, suffix_ids, suffix_types = best
        confidence = self._confidence(
            root_len=root_end,
            word_len=len(skeleton),
            suffix_count=len(suffix_spans),
            suffix_types=suffix_types,
            separator_hits=sum(
                1 for start, _end in suffix_spans if start in separator_boundaries
            ),
            suffix_lengths=[end - start for start, end in suffix_spans],
        )

        if suffix_spans and confidence < self.min_suffix_confidence:
            return StemResult(
                word=word,
                root=word,
                boundaries=[0, len(word)],
                skeleton_boundaries=[0, len(skeleton)],
                confidence=self._confidence(
                    root_len=len(skeleton),
                    word_len=len(skeleton),
                    suffix_count=0,
                    suffix_types=[],
                ),
            )

        root = slice_original(word, 0, root_end, boundary_map)
        suffixes = [
            slice_original(word, start, end, boundary_map)
            for start, end in suffix_spans
        ]

        skeleton_boundaries = [0, root_end] + [end for _, end in suffix_spans]
        boundaries = [boundary_map[i] for i in skeleton_boundaries]

        return StemResult(
            word=word,
            root=root,
            suffixes=suffixes,
            suffix_ids=suffix_ids,
            suffix_types=suffix_types,
            boundaries=boundaries,
            skeleton_boundaries=skeleton_boundaries,
            confidence=confidence,
        )

    def analyze_batch(self, words: list[str]) -> list[StemResult]:
        return [self.analyze(word) for word in words]

    def _strip(
        self,
        skeleton: str,
        end: int,
        separator_boundaries: set[int],
        suffix_spans: list[tuple[int, int]],
        suffix_ids: list[str],
        suffix_types: list[str],
        depth: int,
    ) -> tuple[int, list[tuple[int, int]], list[str], list[str]] | None:
        candidates: list[tuple[int, list[tuple[int, int]], list[str], list[str]]] = []

        if end >= self.min_root_len:
            candidates.append((end, suffix_spans, suffix_ids, suffix_types))

        if depth >= self.max_depth:
            return self._best(skeleton, separator_boundaries, candidates)

        for item in self.lookup:
            sfx = item.skeleton
            if len(sfx) > end:
                continue

            start = end - len(sfx)
            if start < self.min_root_len:
                continue
            if skeleton[start:end] != sfx:
                continue

            stem = skeleton[:start]
            stem_harmony = get_harmony(stem)
            if not harmony_ok(stem_harmony, item.surface_harmony, item.declared_harmony):
                continue
            if suffix_types and not stacking_ok(item.suffix_type, suffix_types[0]):
                continue
            if not sequence_sane([item.suffix_type] + suffix_types):
                continue

            result = self._strip(
                skeleton=skeleton,
                end=start,
                separator_boundaries=separator_boundaries,
                suffix_spans=[(start, end)] + suffix_spans,
                suffix_ids=[item.suffix_id] + suffix_ids,
                suffix_types=[item.suffix_type] + suffix_types,
                depth=depth + 1,
            )
            if result is not None:
                candidates.append(result)

        return self._best(skeleton, separator_boundaries, candidates)

    def _best(
        self,
        skeleton: str,
        separator_boundaries: set[int],
        candidates: list[tuple[int, list[tuple[int, int]], list[str], list[str]]],
    ) -> tuple[int, list[tuple[int, int]], list[str], list[str]] | None:
        if not candidates:
            return None

        def score(
            item: tuple[int, list[tuple[int, int]], list[str], list[str]],
        ) -> tuple[float, int, int, int, int]:
            root_end, spans, _ids, types = item
            separator_hits = sum(1 for start, _end in spans if start in separator_boundaries)
            conf = self._confidence(
                root_len=root_end,
                word_len=len(skeleton),
                suffix_count=len(spans),
                suffix_types=types,
                separator_hits=separator_hits,
                suffix_lengths=[end - start for start, end in spans],
            )
            inner_count = sum(1 for suffix_type in types if suffix_type in INNER_FORMATION_TYPES)
            return conf, separator_hits, root_end, -inner_count, -len(spans)

        candidates.sort(key=score, reverse=True)
        return candidates[0]

    def _confidence(
        self,
        root_len: int,
        word_len: int,
        suffix_count: int,
        suffix_types: list[str],
        separator_hits: int = 0,
        suffix_lengths: list[int] | None = None,
    ) -> float:
        if suffix_count == 0:
            return 0.52

        score = 0.56
        if suffix_count == 1:
            score += 0.05
        else:
            score += min(0.10, 0.04 * suffix_count)

        score += min(0.12, 0.08 * separator_hits)

        stripped_ratio = (word_len - root_len) / max(word_len, 1)
        root_ratio = root_len / max(word_len, 1)
        inner_count = sum(1 for suffix_type in suffix_types if suffix_type in INNER_FORMATION_TYPES)
        inflection_count = sum(
            1 for suffix_type in suffix_types if suffix_type in INFLECTIONAL_SUFFIX_TYPES
        )
        suffix_lengths = suffix_lengths or []

        if suffix_types and suffix_types[-1] in INFLECTIONAL_SUFFIX_TYPES:
            score += 0.04
        if inflection_count == suffix_count and suffix_count <= 2:
            score += 0.03

        if separator_hits == 0:
            if suffix_count >= 2:
                score -= min(0.18, 0.12 + 0.03 * (suffix_count - 2))
            if inner_count:
                score -= min(0.16, 0.07 * inner_count)
            if root_ratio < 0.45:
                score -= 0.10
            if (
                suffix_lengths
                and suffix_lengths[-1] <= 1
                and suffix_types[-1] in {"case", "plural", "possessive"}
            ):
                score -= 0.10
        elif inner_count and separator_hits < suffix_count:
            score -= min(0.06, 0.03 * inner_count)

        if stripped_ratio > 0.65:
            score -= 0.18
        elif stripped_ratio > 0.50:
            score -= 0.08
        if root_len <= self.min_root_len and suffix_count:
            score -= 0.12
        if root_len <= 2 and suffix_count >= 2:
            score -= 0.12
        if separator_hits == 0 and root_ratio >= 0.55:
            score += 0.02

        return round(max(0.0, min(0.95, score)), 3)


def discover_roots(
    lines: Iterable[str], stemmer: MongolStemmer | None = None
) -> Counter[str]:
    if stemmer is None:
        stemmer = MongolStemmer()

    counter: Counter[str] = Counter()
    n = 0

    for line in lines:
        for word in line.strip().split():
            result = stemmer.analyze(word)
            if result.confidence >= 0.60:
                counter[result.root] += 1

            n += 1
            if n % 500000 == 0:
                print(f"{n} words, {len(counter)} unique roots")

    return counter


def build_root_dict(counter: Counter[str], min_freq: int = 10) -> list[str]:
    return [root for root, freq in counter.most_common() if freq >= min_freq]


def print_result(result: StemResult) -> None:
    print(f"word:                {result.word}")
    print(f"skeleton:            {strip_all(result.word)}")
    print(f"root:                {result.root}")

    if result.suffixes:
        print(f"suffixes:            {' + '.join(result.suffixes)}")
        print(f"suffix_ids:          {' + '.join(result.suffix_ids)}")
        print(f"suffix_types:        {' + '.join(result.suffix_types)}")
    else:
        print("suffixes:            (none)")

    print(f"boundaries:          {result.boundaries}")
    print(f"skeleton_boundaries: {result.skeleton_boundaries}")
    print(f"confidence:          {result.confidence:.2f}")
    print()


def main() -> None:
    import sys

    stemmer = MongolStemmer()

    if len(sys.argv) > 1 and sys.argv[1] == "corpus":
        if len(sys.argv) < 3:
            print("Usage: python -m Tokenizer.traditional_mongolian.stemmer corpus <corpus_path> [out_path] [min_freq]")
            return

        corpus_path = sys.argv[2]
        out_path = sys.argv[3] if len(sys.argv) > 3 else "roots.txt"
        min_freq = int(sys.argv[4]) if len(sys.argv) > 4 else 10

        print(f"Processing {corpus_path} ...")
        with open(corpus_path, "r", encoding="utf-8") as f:
            counter = discover_roots(f, stemmer)

        roots = build_root_dict(counter, min_freq)
        print(f"Found {len(roots)} roots, min_freq={min_freq}")

        with open(out_path, "w", encoding="utf-8") as f:
            for root in roots:
                f.write(f"{root}\t{counter[root]}\n")

        print(f"Saved to {out_path}")
        return

    print("Traditional Mongolian stemmer")
    print("Input word, q to quit.\n")

    while True:
        try:
            word = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if word in {"q", "quit", "exit"}:
            break
        if not word:
            continue

        print_result(stemmer.analyze(word))


if __name__ == "__main__":
    main()
