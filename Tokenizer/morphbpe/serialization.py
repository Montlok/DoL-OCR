# -*- coding: utf-8 -*-
"""MorphBPE JSON v1 serialization."""

from __future__ import annotations

import json
from typing import Any

SCHEMA_VERSION = 1


def dump(tokenizer: Any, path: str, extra_config: dict[str, Any] | None = None) -> None:
    """Write a MorphBPETokenizer to disk using v1 schema."""
    merges_sorted = sorted(tokenizer.merges.items(), key=lambda item: item[1][1])
    merges_arr = [
        {"pair": [left, right], "merged": merged, "rank": rank}
        for (left, right), (merged, rank) in merges_sorted
    ]
    payload = {
        "version": SCHEMA_VERSION,
        "type": "morphbpe",
        "vocab": tokenizer.vocab,
        "merges": merges_arr,
        "special_tokens": {"unk": "<unk>"},
        "config": {
            "boundary_constrained": True,
            "min_boundary_confidence": getattr(
                tokenizer, "min_boundary_confidence", 0.60
            ),
            "seed_alphabet": getattr(tokenizer, "seed_alphabet", False),
            **(extra_config or {}),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    version = payload.get("version")
    if version is not None and int(version) != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported morphbpe schema version: {version}; expected {SCHEMA_VERSION}"
        )
    return payload


def merges_from_payload(
    payload: dict[str, Any],
) -> dict[tuple[str, str], tuple[str, int]]:
    merges: dict[tuple[str, str], tuple[str, int]] = {}
    for i, item in enumerate(payload.get("merges", [])):
        if isinstance(item, dict):
            left, right = item["pair"]
            merged = item.get("merged", left + right)
            rank = int(item.get("rank", i))
        else:
            left, right, merged, rank = item
        merges[(left, right)] = (merged, rank)
    return merges
