# -*- coding: utf-8 -*-
"""Offset helpers for morphology-constrained BPE."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MorphToken:
    token: str
    id: int
    start: int
    end: int


@dataclass
class Piece:
    text: str
    start: int
    end: int


def split_on_ascii_space(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for i, ch in enumerate(text):
        if ch in {" ", "\t", "\n", "\r"}:
            if start is not None:
                spans.append((start, i))
                start = None
        elif start is None:
            start = i
    if start is not None:
        spans.append((start, len(text)))
    return spans
