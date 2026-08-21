# -*- coding: utf-8 -*-

"""Deterministic, resumable streaming over the synthetic OCR corpus.

The production corpus has two physical layouts:

* Onon/Noto samples are adjacent ``.png``/``.json`` pairs in sparse WDS tar
  shards.
* Hanshi samples are rows in one large ``meta.jsonl`` with PNG files below a
  bucketed ``pages`` directory.

This module deliberately keeps iteration in the training process.  A small
batch prefetch queue may run ahead, but every yielded batch carries the exact
cursor *after that batch*.  Checkpoints therefore save the consumed cursor,
not the producer thread's potentially-ahead position.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import re
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch

from Tokenizer.multimodal import PILImageProcessor
from Model.ocr.image_preprocess import letterbox_grayscale_to_square
from Model.ocr.pair_shards import (
    ShardCounters,
    iter_tar_pairs,
)

_SHARD_RE = re.compile(r"^shard-(\d+)\.tar$")
CURSOR_VERSION = 1
_COPY_CHUNK_BYTES = 4 * 1024 * 1024


@contextmanager
def _locally_staged_wds_shard(source: str | Path) -> Iterator[Path]:
    """Copy exactly one remote shard to local scratch for tar iteration.

    GVFS SMB files do not reliably implement the file operations used by
    :mod:`tarfile`, even in streaming mode (some reads fail with ``EINVAL``).
    A manual sequential copy avoids those operations on GVFS; tarfile then
    reads an ordinary local file.  ``TemporaryDirectory`` bounds storage to
    one shard and removes it on normal completion, errors, or cancellation.

    ``DOL_OCR_WDS_CACHE_DIR`` may point at a dedicated local scratch volume.
    It must not point back at the NAS mount.
    """

    source_path = Path(source)
    source_stat = source_path.stat()
    expected_size = int(source_stat.st_size)
    cache_root_value = os.environ.get("DOL_OCR_WDS_CACHE_DIR")
    cache_root = Path(cache_root_value).expanduser() if cache_root_value else None
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)

    scratch_root = cache_root if cache_root is not None else Path(tempfile.gettempdir())
    free_bytes = int(shutil.disk_usage(scratch_root).free)
    safety_bytes = max(_COPY_CHUNK_BYTES, expected_size // 100)
    required_bytes = expected_size + safety_bytes
    if free_bytes < required_bytes:
        raise OSError(
            f"insufficient local scratch for WDS shard {source_path}: "
            f"need at least {required_bytes} bytes, have {free_bytes} bytes under "
            f"{scratch_root}"
        )

    with tempfile.TemporaryDirectory(
        prefix=f"dolocr-{source_path.stem}-",
        dir=str(cache_root) if cache_root is not None else None,
    ) as temp_dir:
        cached_path = Path(temp_dir) / source_path.name
        copied = 0
        try:
            # Do not use shutil.copyfile/copy2 here: their fast-copy syscalls
            # can trigger the same unsupported-operation failure on GVFS.
            with (
                source_path.open("rb", buffering=0) as src,
                cached_path.open("wb", buffering=0) as dst,
            ):
                while True:
                    chunk = src.read(_COPY_CHUNK_BYTES)
                    if not chunk:
                        break
                    dst.write(chunk)
                    copied += len(chunk)
        except OSError as exc:
            raise OSError(
                f"failed to stage WDS shard {source_path} to local scratch: {exc}"
            ) from exc
        if copied != expected_size:
            raise OSError(
                f"short copy while staging WDS shard {source_path}: "
                f"copied {copied} of {expected_size} bytes"
            )
        final_source_stat = source_path.stat()
        if int(final_source_stat.st_size) != expected_size or int(
            final_source_stat.st_mtime_ns
        ) != int(source_stat.st_mtime_ns):
            raise OSError(f"WDS shard changed while it was being staged: {source_path}")
        yield cached_path


def discover_wds_shards(
    root: str | Path,
    *,
    exclude_ids: Sequence[int] = (2303,),
    limit: int = 0,
) -> list[Path]:
    """Return real sparse shard paths, sorted by numeric shard id.

    The renderer's ids are not contiguous, so range expansion is unsafe.  The
    known corrupt shard 2303 is excluded by default and remains visible in the
    run manifest's ``excluded_shard_ids`` field.
    """

    root = Path(root)
    excluded = {int(x) for x in exclude_ids}
    found: list[tuple[int, Path]] = []
    for path in root.glob("shard-*.tar"):
        match = _SHARD_RE.match(path.name)
        if match is None:
            continue
        shard_id = int(match.group(1))
        if shard_id not in excluded:
            found.append((shard_id, path))
    found.sort(key=lambda item: item[0])
    paths = [path for _, path in found]
    if limit > 0:
        paths = paths[: int(limit)]
    if not paths:
        raise FileNotFoundError(f"no usable shard-*.tar files under {root}")
    return paths


def corpus_manifest(
    wds_paths: Sequence[str | Path],
    *,
    hanshi_meta: str | Path,
    hanshi_pages: str | Path,
    excluded_shard_ids: Sequence[int],
    val_src_doc_min: int,
    test_src_doc_min: int,
    seed: int,
) -> tuple[dict[str, Any], str]:
    """Build a compact source inventory and its canonical SHA256.

    Hashing two terabytes before every launch is not practical.  The manifest
    instead binds the complete sparse shard inventory (id/name/size), Hanshi
    metadata size, split contract, exclusions, and ordering seed.  Operators
    can additionally pin independently audited corpus hashes in the run notes.
    """

    shards: list[dict[str, Any]] = []
    for value in wds_paths:
        path = Path(value)
        match = _SHARD_RE.match(path.name)
        if match is None:
            raise ValueError(f"not a canonical WDS shard name: {path}")
        shards.append(
            {
                "id": int(match.group(1)),
                "name": path.name,
                "size": int(path.stat().st_size),
            }
        )
    meta_path = Path(hanshi_meta)
    pages_path = Path(hanshi_pages)
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    if not pages_path.is_dir():
        raise FileNotFoundError(pages_path)
    manifest: dict[str, Any] = {
        "version": 1,
        "wds": {"shards": shards},
        "hanshi": {
            "meta_name": meta_path.name,
            "meta_size": int(meta_path.stat().st_size),
            "pages_name": pages_path.name,
        },
        "excluded_shard_ids": sorted({int(x) for x in excluded_shard_ids}),
        "split": {
            "train": [0, int(val_src_doc_min)],
            "val": [int(val_src_doc_min), int(test_src_doc_min)],
            "test": [int(test_src_doc_min), None],
            "group_key": "src_doc",
        },
        "mix_schedule": ["wds", "wds", "hanshi"],
        "seed": int(seed),
    }
    canonical = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return manifest, hashlib.sha256(canonical).hexdigest()


def initial_cursor() -> dict[str, Any]:
    return {
        "version": CURSOR_VERSION,
        "mix_position": 0,
        "wds": {"shard_position": 0, "sample_position": 0},
        "hanshi": {"byte_offset": 0, "line_number": 0},
        "exhausted": {"wds": False, "hanshi": False},
        "counts": {"wds": 0, "hanshi": 0, "total": 0},
    }


def validate_cursor(cursor: dict[str, Any] | None) -> dict[str, Any]:
    if cursor is None:
        return initial_cursor()
    state = copy.deepcopy(cursor)
    if int(state.get("version", -1)) != CURSOR_VERSION:
        raise ValueError(
            f"unsupported corpus cursor version {state.get('version')!r}; "
            f"expected {CURSOR_VERSION}"
        )
    for key in ("mix_position", "wds", "hanshi", "exhausted", "counts"):
        if key not in state:
            raise ValueError(f"corpus cursor missing {key!r}")
    return state


def _valid_train_row(
    row: dict[str, Any], *, val_src_doc_min: int, max_target_len: int
) -> list[int] | None:
    if row.get("kind") != "line":
        return None
    src_doc = row.get("src_doc")
    if not isinstance(src_doc, int):
        raise ValueError(f"invalid src_doc in corpus row: {src_doc!r}")
    if src_doc >= val_src_doc_min:
        return None
    text = row.get("text")
    if not isinstance(text, str) or not text:
        raise ValueError("training row has missing/empty text")
    target = list(text.encode("utf-8"))
    if not target or len(target) > max_target_len:
        return None
    return target


class _WDSRecordStream:
    def __init__(
        self,
        paths: Sequence[Path],
        *,
        seed: int,
        cursor: dict[str, Any],
        val_src_doc_min: int,
        max_target_len: int,
    ) -> None:
        self.paths = list(paths)
        random.Random(int(seed)).shuffle(self.paths)
        self.cursor = copy.deepcopy(cursor)
        self.val_src_doc_min = int(val_src_doc_min)
        self.max_target_len = int(max_target_len)
        self._iterator = self._records()

    def _records(self) -> Iterator[dict[str, Any]]:
        start_shard = int(self.cursor["shard_position"])
        start_sample = int(self.cursor["sample_position"])
        if start_shard < 0 or start_shard > len(self.paths):
            raise ValueError(f"WDS cursor shard position out of range: {start_shard}")
        for shard_position in range(start_shard, len(self.paths)):
            path = self.paths[shard_position]
            counters = ShardCounters(shard_position)
            valid_position = 0
            with _locally_staged_wds_shard(path) as staged_path:
                for key, png_bytes, row in iter_tar_pairs(staged_path, counters):
                    target = _valid_train_row(
                        row,
                        val_src_doc_min=self.val_src_doc_min,
                        max_target_len=self.max_target_len,
                    )
                    if target is None:
                        continue
                    if shard_position == start_shard and valid_position < start_sample:
                        valid_position += 1
                        continue
                    valid_position += 1
                    after = {
                        "shard_position": shard_position,
                        "sample_position": valid_position,
                    }
                    yield {
                        "source": "wds",
                        "key": f"{path.name}:{key}",
                        "image": png_bytes,
                        "target": target,
                        "font": str(row.get("font", "")),
                        "src_doc": int(row["src_doc"]),
                        "cursor_after": after,
                    }
            # A checkpoint taken after the final yielded record still points at
            # that shard.  Resume re-scans only this one tar, then advances.
            start_sample = 0
        self.cursor = {"shard_position": len(self.paths), "sample_position": 0}

    def __iter__(self) -> "_WDSRecordStream":
        return self

    def __next__(self) -> dict[str, Any]:
        record = next(self._iterator)
        self.cursor = copy.deepcopy(record["cursor_after"])
        return record


class _HanshiRecordStream:
    def __init__(
        self,
        meta_path: str | Path,
        pages_root: str | Path,
        *,
        cursor: dict[str, Any],
        val_src_doc_min: int,
        max_target_len: int,
    ) -> None:
        self.meta_path = Path(meta_path)
        self.pages_root = Path(pages_root)
        self.cursor = copy.deepcopy(cursor)
        self.val_src_doc_min = int(val_src_doc_min)
        self.max_target_len = int(max_target_len)
        self._iterator = self._records()

    def _records(self) -> Iterator[dict[str, Any]]:
        offset = int(self.cursor["byte_offset"])
        line_number = int(self.cursor["line_number"])
        with self.meta_path.open("rb") as fh:
            fh.seek(offset)
            while True:
                raw = fh.readline()
                if not raw:
                    break
                line_number += 1
                after_offset = fh.tell()
                try:
                    row = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"{self.meta_path}: line {line_number}: invalid JSON: {exc}"
                    ) from exc
                target = _valid_train_row(
                    row,
                    val_src_doc_min=self.val_src_doc_min,
                    max_target_len=self.max_target_len,
                )
                if target is None:
                    continue
                bucket = str(row.get("bucket", ""))
                doc_id = str(row.get("doc_id", ""))
                if not bucket or not doc_id:
                    raise ValueError(
                        f"{self.meta_path}: line {line_number}: missing bucket/doc_id"
                    )
                image_path = self.pages_root / bucket / f"{doc_id}.png"
                after = {"byte_offset": after_offset, "line_number": line_number}
                yield {
                    "source": "hanshi",
                    "key": f"hanshi:{bucket}/{doc_id}",
                    "image": image_path,
                    "target": target,
                    "font": str(row.get("font", "hanshi")),
                    "src_doc": int(row["src_doc"]),
                    "cursor_after": after,
                }
        self.cursor = {
            "byte_offset": self.meta_path.stat().st_size,
            "line_number": line_number,
        }

    def __iter__(self) -> "_HanshiRecordStream":
        return self

    def __next__(self) -> dict[str, Any]:
        record = next(self._iterator)
        self.cursor = copy.deepcopy(record["cursor_after"])
        return record


class MixedOCRCorpus:
    """Finite one-pass stream with a deterministic 2×WDS + 1×Hanshi mix."""

    schedule = ("wds", "wds", "hanshi")

    def __init__(
        self,
        wds_paths: Sequence[str | Path],
        *,
        hanshi_meta: str | Path,
        hanshi_pages: str | Path,
        image_size: int,
        seed: int,
        val_src_doc_min: int = 434600,
        max_target_len: int = 256,
        cursor: dict[str, Any] | None = None,
    ) -> None:
        self.state = validate_cursor(cursor)
        self.image_size = int(image_size)
        self.processor = PILImageProcessor(image_size=self.image_size)
        self.sources: dict[str, Any] = {
            "wds": _WDSRecordStream(
                [Path(p) for p in wds_paths],
                seed=seed,
                cursor=self.state["wds"],
                val_src_doc_min=val_src_doc_min,
                max_target_len=max_target_len,
            ),
            "hanshi": _HanshiRecordStream(
                hanshi_meta,
                hanshi_pages,
                cursor=self.state["hanshi"],
                val_src_doc_min=val_src_doc_min,
                max_target_len=max_target_len,
            ),
        }

    def state_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.state)

    def _next_record(self) -> dict[str, Any]:
        while not all(bool(v) for v in self.state["exhausted"].values()):
            mix_position = int(self.state["mix_position"])
            source = self.schedule[mix_position % len(self.schedule)]
            self.state["mix_position"] = mix_position + 1
            if self.state["exhausted"][source]:
                continue
            try:
                record = next(self.sources[source])
            except StopIteration:
                self.state["exhausted"][source] = True
                continue
            self.state[source] = copy.deepcopy(record["cursor_after"])
            self.state["counts"][source] += 1
            self.state["counts"]["total"] += 1
            return record
        raise StopIteration

    def __iter__(self) -> "MixedOCRCorpus":
        return self

    def __next__(self) -> tuple[torch.Tensor, list[int], dict[str, Any]]:
        record = self._next_record()
        image = record["image"]
        if record["source"] == "wds":
            square = letterbox_grayscale_to_square(image, self.image_size)
        else:
            try:
                png_bytes = Path(image).read_bytes()
            except OSError as exc:
                raise OSError(f"failed to read Hanshi image {image}: {exc}") from exc
            square = letterbox_grayscale_to_square(png_bytes, self.image_size)
        pixels = self.processor([square])[0]
        metadata = {
            "source": record["source"],
            "key": record["key"],
            "font": record["font"],
            "src_doc": record["src_doc"],
            "cursor": self.state_dict(),
        }
        return pixels, record["target"], metadata

    def batches(self, batch_size: int) -> Iterator[dict[str, Any]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        rows: list[tuple[torch.Tensor, list[int], dict[str, Any]]] = []
        for row in self:
            rows.append(row)
            if len(rows) == batch_size:
                yield _collate_stream_rows(rows)
                rows = []
        if rows:
            yield _collate_stream_rows(rows)


def _collate_stream_rows(
    rows: list[tuple[torch.Tensor, list[int], dict[str, Any]]],
) -> dict[str, Any]:
    pixels = torch.stack([row[0] for row in rows])
    targets = [row[1] for row in rows]
    target_lengths = torch.tensor([len(row) for row in targets], dtype=torch.long)
    flat_targets = torch.tensor(
        [token for row in targets for token in row], dtype=torch.long
    )
    metadata = [row[2] for row in rows]
    return {
        "pixels": pixels,
        "targets": flat_targets,
        "target_lengths": target_lengths,
        "keys": [row["key"] for row in metadata],
        "fonts": [row["font"] for row in metadata],
        "sources": [row["source"] for row in metadata],
        # The final sample's cursor is exactly the post-batch consumed cursor.
        "corpus_cursor": copy.deepcopy(metadata[-1]["cursor"]),
    }


__all__ = [
    "CURSOR_VERSION",
    "MixedOCRCorpus",
    "corpus_manifest",
    "discover_wds_shards",
    "initial_cursor",
    "validate_cursor",
]
