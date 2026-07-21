# -*- coding: utf-8 -*-

"""Extract supervised target tokens from OCR align rows into packed text rows.

Phase 2 (text pretraining) input builder: the align JSONL rows already carry
byte-fallback-verified target token ids in their supervised tail
(``labels != IGNORE_INDEX``), so no tokenizer or decoding is needed here.
Each output row is a packed ``[BOS] t1 [BOS] t2 ...`` sequence of exactly
``--seq-len`` tokens (the terminal EOS of each target is part of the target
itself), with ``labels == input_ids`` and full attention.

Rows whose target contains any image/special-structure id are rejected loudly
instead of silently skipped. Output shards are plain pretraining JSONL
consumable by ``scripts.train_rdt --data``.

Usage:

    python -m scripts.build_text_rows_from_align \
        --align 'data_v1/align_*.jsonl' --output data_v1/text_pretrain \
        --seq-len 1024 --shard-rows 200000
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

from Model.config import (
    BOS_ID,
    EOS_ID,
    IGNORE_INDEX,
    IMAGE_END_ID,
    IMAGE_PATCH_ID,
    IMAGE_START_ID,
    PAD_ID,
    UNK_ID,
)

FORBIDDEN_IDS = {IMAGE_PATCH_ID, IMAGE_START_ID, IMAGE_END_ID, PAD_ID, UNK_ID}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--align", required=True, help="glob of align JSONL shards")
    p.add_argument("--output", required=True, help="output directory")
    p.add_argument("--prefix", default="text", help="output shard name prefix")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--shard-rows", type=int, default=200_000)
    p.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="stop after this many source rows (0 = all); for small-batch QA",
    )
    p.add_argument(
        "--bundle",
        default="",
        help="optional tokenizer bundle dir; when set, decode-verifies the "
        "first packed row's first document against its source target text",
    )
    return p.parse_args(argv)


def extract_target(row: dict) -> list[int]:
    input_ids = row["input_ids"]
    labels = row["labels"]
    if len(input_ids) != len(labels):
        raise ValueError("input_ids and labels must have aligned lengths")
    split = 0
    while split < len(labels) and labels[split] == IGNORE_INDEX:
        split += 1
    if split == 0 or split == len(labels):
        raise ValueError("row has no masked prompt or no supervised tail")
    tail_labels = [int(t) for t in labels[split:]]
    if any(t == IGNORE_INDEX for t in tail_labels):
        raise ValueError("supervised tail is not contiguous (IGNORE after split)")
    target = [int(t) for t in input_ids[split:]]
    if tail_labels != target:
        raise ValueError("labels tail != input_ids tail (shifted rows?)")
    bad = FORBIDDEN_IDS.intersection(target)
    if bad:
        raise ValueError(f"target contains forbidden ids: {sorted(bad)}")
    return target


class ShardWriter:
    def __init__(self, out_dir: Path, prefix: str, shard_rows: int):
        self.out_dir = out_dir
        self.prefix = prefix
        self.shard_rows = shard_rows
        self.shard_idx = 0
        self.rows_in_shard = 0
        self.rows_total = 0
        self._fh = None

    def _open_next(self):
        if self._fh:
            self._fh.close()
        path = self.out_dir / f"{self.prefix}_{self.shard_idx:04d}.jsonl"
        self._fh = path.open("w", encoding="utf-8")
        self.shard_idx += 1
        self.rows_in_shard = 0

    def write(self, row: dict):
        if self._fh is None or self.rows_in_shard >= self.shard_rows:
            self._open_next()
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.rows_in_shard += 1
        self.rows_total += 1

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None


def main(argv=None) -> int:
    args = parse_args(argv)
    shard_paths = sorted(glob.glob(args.align))
    if not shard_paths:
        print(f"no align shards match {args.align!r}", file=sys.stderr)
        return 2
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_len = args.seq_len
    writer = ShardWriter(out_dir, args.prefix, args.shard_rows)
    buf: list[int] = []
    n_src = 0
    n_tokens = 0
    target_len_sum = 0
    first_doc: list[int] | None = None
    first_src_target: list[int] | None = None

    def flush_full():
        nonlocal buf
        while len(buf) >= seq_len:
            chunk = buf[:seq_len]
            buf = buf[seq_len:]
            writer.write(
                {
                    "input_ids": chunk,
                    "attention_mask": [1] * seq_len,
                    "labels": list(chunk),
                }
            )

    stop = False
    for path in shard_paths:
        if stop:
            break
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    target = extract_target(row)
                except (ValueError, KeyError, json.JSONDecodeError) as exc:
                    raise SystemExit(f"{path}:{line_no}: {exc}") from exc
                if first_doc is None:
                    first_doc = [BOS_ID] + target
                    first_src_target = target
                buf.extend([BOS_ID] + target)
                n_src += 1
                n_tokens += 1 + len(target)
                target_len_sum += len(target)
                flush_full()
                if args.max_rows and n_src >= args.max_rows:
                    stop = True
                    break
    # Drop the tail remainder (< seq_len tokens) rather than padding: at this
    # corpus size the loss is negligible and keeps every row full-length.
    dropped_tail = len(buf)
    writer.close()

    print(
        f"[build_text_rows] src_rows={n_src} packed_rows={writer.rows_total} "
        f"tokens={n_tokens} avg_target_len={target_len_sum / max(1, n_src):.1f} "
        f"seq_len={seq_len} shards={writer.shard_idx} dropped_tail_tokens={dropped_tail}"
    )
    if writer.rows_total == 0:
        print("no packed rows produced", file=sys.stderr)
        return 2

    # QA: re-read packed index 0, check the first document round-trips.
    first_shard = out_dir / f"{args.prefix}_0000.jsonl"
    with first_shard.open("r", encoding="utf-8") as f:
        row0 = json.loads(f.readline())
    ids0 = row0["input_ids"]
    assert len(ids0) == seq_len, f"index0 len {len(ids0)} != seq_len {seq_len}"
    assert row0["labels"] == ids0, "index0 labels != input_ids"
    assert ids0[0] == BOS_ID, f"index0 does not start with BOS ({ids0[0]})"
    assert first_doc is not None
    head = first_doc[: min(len(first_doc), seq_len)]
    assert ids0[: len(head)] == head, "index0 head != first source document"
    forbidden_hits = FORBIDDEN_IDS.intersection(ids0)
    assert not forbidden_hits, f"index0 contains forbidden ids {forbidden_hits}"
    print(f"[qa] index0 OK: starts with BOS + first target ({len(head)} ids checked)")

    if args.bundle:
        from Tokenizer.unified import TokenizerBundle

        bundle = TokenizerBundle.from_dir(args.bundle)
        assert first_src_target is not None
        body = [t for t in first_src_target if t != EOS_ID]
        text = bundle.tokenizer.decode(body)
        print(f"[qa] index0 first-doc decode ({len(body)} ids): {text[:80]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
