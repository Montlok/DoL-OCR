# -*- coding: utf-8 -*-
"""Shared WebDataset pair iteration for OCR corpus producers and readers."""

from __future__ import annotations

import json
import os
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass
class ShardCounters:
    """One shard's routing, skip, and output tallies."""

    shard_index: int
    n_members_seen: int = 0
    n_samples_seen: int = 0
    n_orphans: int = 0
    n_train: int = 0
    n_val: int = 0
    n_test_skipped: int = 0
    n_non_line_skipped: int = 0
    n_over_length_skipped: int = 0
    n_align_written: int = 0
    n_val_written: int = 0
    n_ssl_written: int = 0
    n_val_cap_skipped: int = 0
    max_row_len: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "shard_index": self.shard_index,
            "n_members_seen": self.n_members_seen,
            "n_samples_seen": self.n_samples_seen,
            "n_orphans": self.n_orphans,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test_skipped": self.n_test_skipped,
            "n_non_line_skipped": self.n_non_line_skipped,
            "n_over_length_skipped": self.n_over_length_skipped,
            "n_align_written": self.n_align_written,
            "n_val_written": self.n_val_written,
            "n_ssl_written": self.n_ssl_written,
            "n_val_cap_skipped": self.n_val_cap_skipped,
            "max_row_len": self.max_row_len,
        }


def _stem_and_suffix(name: str) -> tuple[str, str]:
    base = os.path.basename(name)
    stem, suffix = os.path.splitext(base)
    return stem, suffix.lower()


def iter_tar_pairs(
    tar_path: str | Path,
    counters: ShardCounters,
) -> Iterator[tuple[str, bytes, dict[str, Any]]]:
    """Yield adjacent PNG/JSON pairs from a sequential WebDataset tar."""

    pending: dict[str, bytes] | None = None
    pending_stem: str | None = None

    def flush_orphan() -> None:
        nonlocal pending, pending_stem
        if pending is not None:
            counters.n_orphans += 1
        pending = None
        pending_stem = None

    with tarfile.open(tar_path, "r|") as archive:
        for member in archive:
            if not member.isfile():
                continue
            counters.n_members_seen += 1
            stem, suffix = _stem_and_suffix(member.name)
            if suffix not in (".png", ".json"):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            data = handle.read()

            if pending_stem is not None and stem != pending_stem:
                flush_orphan()
            if pending_stem is None:
                pending = {suffix: data}
                pending_stem = stem
                continue
            if suffix in pending:
                flush_orphan()
                pending = {suffix: data}
                pending_stem = stem
                continue

            pending[suffix] = data
            png_bytes = pending.get(".png")
            json_bytes = pending.get(".json")
            pending = None
            pending_stem = None
            if png_bytes is None or json_bytes is None:
                counters.n_orphans += 1
                continue
            try:
                metadata = json.loads(json_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"{tar_path}: key={stem!r}: invalid JSON sidecar: {exc}"
                ) from exc
            counters.n_samples_seen += 1
            yield stem, png_bytes, metadata

        flush_orphan()


__all__ = ["ShardCounters", "iter_tar_pairs"]
