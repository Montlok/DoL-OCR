# -*- coding: utf-8 -*-

"""Extract supervised target tokens from OCR align rows into packed text rows.

Phase 2 (text pretraining) input builder: the align JSONL rows already carry
strict-native target token ids in their supervised tail
(``labels != IGNORE_INDEX``), so no tokenizer or decoding is needed here.
Each output row is a packed ``[BOS] t1 [BOS] t2 ...`` sequence of exactly
``--seq-len`` tokens (the terminal EOS of each target is part of the target
itself), with ``labels == input_ids``, full attention, and the exact
id-derived ``word_pos``/``morph_depth`` representation used by the LM.

Rows whose target contains any image/special-structure id are rejected loudly
instead of silently skipped. Output shards are plain pretraining JSONL
consumable by ``scripts.train_rdt`` together with the emitted immutable
pretraining receipt.  The source native OCR alignment receipt is mandatory;
this command never retroactively signs unproven token-id shards.

Usage:

    python -m scripts.build_text_rows_from_align \
        --align 'data_v1/align_*.jsonl' --output data_v1/text_pretrain \
        --align-receipt data_v1/ocr_data_contract.json \
        --bundle data_v1/tokenizer --seq-len 1024 --shard-rows 200000
"""

from __future__ import annotations

import argparse
import glob
import json
import os
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
from Model.ocr.alignment_contract import (
    load_and_validate_ocr_alignment_data_contract,
)
from Model.ocr.tokenization import (
    native_tokenization_contract,
    tokenizer_morphology_track_table,
)
from Tokenizer.pretraining.data_contract import (
    PRETRAINING_PRODUCER_OCR_ALIGN_TEXT,
    build_pretraining_data_contract,
    write_pretraining_data_contract,
)
from Tokenizer.pretraining.morphology import derive_morph_info_from_track_ids
from Tokenizer.unified import TokenizerBundle
from Tokenizer.unified.contract import (
    tokenizer_algorithm_contract,
    tokenizer_bundle_contract,
)

FORBIDDEN_IDS = {IMAGE_PATCH_ID, IMAGE_START_ID, IMAGE_END_ID, PAD_ID, UNK_ID}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--align", required=True, help="glob of align JSONL shards")
    p.add_argument(
        "--align-receipt",
        required=True,
        help="native OCR alignment receipt for the exact --align shards",
    )
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
        required=True,
        help="tokenizer bundle bound by --align-receipt and output receipt",
    )
    p.add_argument(
        "--receipt",
        default="",
        help=(
            "output pretraining receipt; defaults to "
            "<output>/pretraining_data_receipt.json"
        ),
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
        self.paths: list[Path] = []

    def _open_next(self):
        self.close()
        path = self.out_dir / f"{self.prefix}_{self.shard_idx:04d}.jsonl"
        self._fh = path.open("w", encoding="utf-8")
        self.paths.append(path)
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
            handle = self._fh
            self._fh = None
            path = Path(handle.name)
            try:
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                handle.close()
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)


def main(argv=None) -> int:
    args = parse_args(argv)
    shard_paths = sorted(glob.glob(args.align))
    if not shard_paths:
        print(f"no align shards match {args.align!r}", file=sys.stderr)
        return 2
    bundle_identity = tokenizer_bundle_contract(args.bundle)
    algorithm_identity = tokenizer_algorithm_contract()
    bundle = TokenizerBundle.from_dir(args.bundle)
    native_contract = native_tokenization_contract(
        bundle.tokenizer,
        args.bundle,
    )
    try:
        source_lineage = load_and_validate_ocr_alignment_data_contract(
            args.align_receipt,
            args.align,
            native_contract,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        print(f"unsafe source alignment data: {exc}", file=sys.stderr)
        return 2
    track_table = tokenizer_morphology_track_table(bundle.tokenizer)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_len = args.seq_len
    writer = ShardWriter(out_dir, args.prefix, args.shard_rows)
    buf: list[int] = []
    track_buf: list[int] = []
    n_src = 0
    n_tokens = 0
    target_len_sum = 0
    first_doc: list[int] | None = None
    first_src_target: list[int] | None = None

    def flush_full():
        nonlocal buf, track_buf
        while len(buf) >= seq_len:
            chunk = buf[:seq_len]
            buf = buf[seq_len:]
            track_chunk = track_buf[:seq_len]
            track_buf = track_buf[seq_len:]
            word_pos, morph_depth = derive_morph_info_from_track_ids(
                track_chunk
            )
            writer.write(
                {
                    "input_ids": chunk,
                    "attention_mask": [1] * seq_len,
                    "labels": list(chunk),
                    "word_pos": word_pos,
                    "morph_depth": morph_depth,
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
                document = [BOS_ID] + target
                invalid_ids = [
                    token_id
                    for token_id in document
                    if token_id < 0 or token_id >= len(track_table)
                ]
                if invalid_ids:
                    raise SystemExit(
                        f"{path}:{line_no}: token ids outside tokenizer "
                        f"vocabulary: {sorted(set(invalid_ids))}"
                    )
                document_tracks = [
                    track_table[token_id] for token_id in document
                ]
                buf.extend(document)
                track_buf.extend(document_tracks)
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
    if len(track_buf) != dropped_tail:
        raise RuntimeError("packed token and morphology-track buffers diverged")
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
    assert len(row0["word_pos"]) == seq_len, "index0 word_pos length mismatch"
    assert len(row0["morph_depth"]) == seq_len, "index0 morph_depth length mismatch"
    assert ids0[0] == BOS_ID, f"index0 does not start with BOS ({ids0[0]})"
    assert first_doc is not None
    head = first_doc[: min(len(first_doc), seq_len)]
    assert ids0[: len(head)] == head, "index0 head != first source document"
    forbidden_hits = FORBIDDEN_IDS.intersection(ids0)
    assert not forbidden_hits, f"index0 contains forbidden ids {forbidden_hits}"
    print(f"[qa] index0 OK: starts with BOS + first target ({len(head)} ids checked)")

    assert first_src_target is not None
    body = [t for t in first_src_target if t != EOS_ID]
    text = bundle.tokenizer.decode(body)
    print(f"[qa] index0 first-doc decode ({len(body)} ids): {text[:80]!r}")

    receipt_path = (
        Path(args.receipt)
        if args.receipt
        else out_dir / "pretraining_data_receipt.json"
    )
    output_receipt = build_pretraining_data_contract(
        writer.paths,
        producer_kind=PRETRAINING_PRODUCER_OCR_ALIGN_TEXT,
        tokenizer_bundle=bundle_identity,
        tokenizer_algorithm=algorithm_identity,
    )
    write_pretraining_data_contract(receipt_path, output_receipt)
    print(
        f"[receipt] {receipt_path} data_sha256="
        f"{output_receipt['data_sha256']} source_alignment_sha256="
        f"{source_lineage['contract_canonical_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
