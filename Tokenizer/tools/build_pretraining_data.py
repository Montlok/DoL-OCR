# -*- coding: utf-8 -*-
"""Build minimum JSONL pretraining data with a tokenizer bundle."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from Tokenizer.pretraining import (
    IGNORE_INDEX,
    EncodedSample,
    PretrainingDataBuilder,
    encoded_sample_to_dict,
    iter_pack_samples,
)
from Tokenizer.pretraining.data_contract import (
    PRETRAINING_PRODUCER_GENERIC_BUILDER,
    build_pretraining_data_contract,
    default_pretraining_data_contract_path,
    write_pretraining_data_contract,
)
from Tokenizer.unified.bundle import TokenizerBundle
from Tokenizer.unified.contract import (
    tokenizer_algorithm_contract,
    tokenizer_bundle_contract,
)


def _iter_samples(
    path: str, builder: PretrainingDataBuilder
) -> Iterable[EncodedSample]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if path.endswith(".jsonl"):
                yield builder.encode_jsonl_line(line)
            else:
                yield builder.encode_text(line, metadata={"type": "text"})


@dataclass
class _BuildSummary:
    unk_id: int
    skipped_empty: int = 0
    num_samples: int = 0
    total_tokens: int = 0
    unk_count: int = 0
    supervised_tokens: int = 0
    max_len: int = 0
    max_morph_depth: int = 0
    shards: list[str] = field(default_factory=list)

    def add_row(self, row: dict) -> None:
        length = len(row["input_ids"])
        self.num_samples += 1
        self.total_tokens += length
        self.max_len = max(self.max_len, length)
        self.unk_count += row["input_ids"].count(self.unk_id)
        self.supervised_tokens += sum(
            1 for label in row["labels"] if int(label) != IGNORE_INDEX
        )
        self.max_morph_depth = max(
            self.max_morph_depth,
            max((int(value) for value in row.get("morph_depth", [])), default=0),
        )

    def to_dict(self) -> dict:
        return {
            "num_samples": self.num_samples,
            "skipped_empty": self.skipped_empty,
            "avg_len": (
                self.total_tokens / self.num_samples if self.num_samples else 0.0
            ),
            "max_len": self.max_len,
            "unk_rate": (
                self.unk_count / self.total_tokens if self.total_tokens else 0.0
            ),
            "supervised_tokens": self.supervised_tokens,
            "supervised_rate": (
                self.supervised_tokens / self.total_tokens
                if self.total_tokens
                else 0.0
            ),
            "max_morph_depth": self.max_morph_depth,
            "shards": self.shards,
        }


class _JsonlShardWriter:
    """Bounded-memory JSONL writer with optional token/sample rotation."""

    def __init__(
        self,
        output: str,
        *,
        token_budget: int = 0,
        sample_budget: int = 0,
    ) -> None:
        if token_budget < 0 or sample_budget < 0:
            raise ValueError("shard budgets must be non-negative")
        self.output = Path(output)
        self.token_budget = int(token_budget)
        self.sample_budget = int(sample_budget)
        self.rotate = bool(self.token_budget or self.sample_budget)
        self._fh = None
        self._shard_idx = 0
        self._cur_tokens = 0
        self._cur_samples = 0
        self.paths: list[str] = []

    def __enter__(self) -> "_JsonlShardWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
        if exc_type is None and self._fh is None and not self.paths:
            path = self._path_for_idx(0)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            self.paths.append(str(path))

    def _path_for_idx(self, idx: int) -> Path:
        if not self.rotate:
            return self.output
        if self.output.suffix:
            return self.output.with_name(
                f"{self.output.stem}-{idx:05d}{self.output.suffix}"
            )
        return self.output / f"shard-{idx:05d}.jsonl"

    def _open_next(self) -> None:
        self.close()
        path = self._path_for_idx(self._shard_idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", encoding="utf-8")
        self.paths.append(str(path))
        self._cur_tokens = 0
        self._cur_samples = 0
        self._shard_idx += 1

    def _would_exceed(self, row_tokens: int) -> bool:
        if self._cur_samples == 0:
            return False
        if self.sample_budget and self._cur_samples >= self.sample_budget:
            return True
        return bool(
            self.token_budget and self._cur_tokens + row_tokens > self.token_budget
        )

    def write(self, row: dict) -> None:
        row_tokens = len(row["input_ids"])
        if self._fh is None or self._would_exceed(row_tokens):
            self._open_next()
        assert self._fh is not None
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._cur_tokens += row_tokens
        self._cur_samples += 1

    def close(self) -> None:
        if self._fh is not None:
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


def _nonempty_samples(
    samples: Iterable[EncodedSample],
    summary: _BuildSummary,
) -> Iterator[EncodedSample]:
    for sample in samples:
        if not sample.input_ids:
            summary.skipped_empty += 1
            continue
        yield sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-bundle", required=True)
    parser.add_argument("--input", required=True, help=".txt or .jsonl")
    parser.add_argument("--output", required=True, help="output JSONL")
    parser.add_argument(
        "--receipt",
        default="",
        help=(
            "immutable output-shard receipt; defaults to "
            "<output>.receipt.json, or <output>/pretraining_data_receipt.json "
            "when --output is a sharded directory"
        ),
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--pack", action="store_true", help="pack text-only samples")
    parser.add_argument(
        "--pad-to-max-length",
        action="store_true",
        help="pad packed samples to the configured packed sequence length",
    )
    parser.add_argument(
        "--pack-max-length",
        type=int,
        default=None,
        help="packed sequence length; defaults to --max-length",
    )
    parser.add_argument(
        "--shard-token-budget",
        type=int,
        default=0,
        help=(
            "rotate output shards after roughly this many tokens; 0 keeps the "
            "single-file --output behaviour"
        ),
    )
    parser.add_argument(
        "--shard-sample-budget",
        type=int,
        default=0,
        help=(
            "rotate output shards after this many emitted rows; 0 disables this "
            "rotation limit"
        ),
    )
    args = parser.parse_args()

    bundle = TokenizerBundle.from_dir(args.tokenizer_bundle)
    builder = PretrainingDataBuilder(bundle, max_length=args.max_length)
    if args.shard_token_budget < 0 or args.shard_sample_budget < 0:
        parser.error("shard budgets must be non-negative")

    summary = _BuildSummary(unk_id=bundle.tokenizer.unk_id)
    samples: Iterable[EncodedSample] = _nonempty_samples(
        _iter_samples(args.input, builder), summary
    )
    if args.pack:
        samples = iter_pack_samples(
            samples,
            max_length=args.pack_max_length or args.max_length,
            pad_id=bundle.tokenizer.vocab["<pad>"],
            eos_id=bundle.tokenizer.vocab["<eos>"],
            pad_to_max_length=args.pad_to_max_length,
        )
    else:
        samples = iter(samples)

    with _JsonlShardWriter(
        args.output,
        token_budget=args.shard_token_budget,
        sample_budget=args.shard_sample_budget,
    ) as writer:
        for sample in samples:
            row = encoded_sample_to_dict(sample)
            writer.write(row)
            summary.add_row(row)
    summary.shards = list(writer.paths)
    bundle_identity = tokenizer_bundle_contract(args.tokenizer_bundle)
    algorithm_identity = tokenizer_algorithm_contract()
    receipt = build_pretraining_data_contract(
        [Path(path) for path in writer.paths],
        producer_kind=PRETRAINING_PRODUCER_GENERIC_BUILDER,
        tokenizer_bundle=bundle_identity,
        tokenizer_algorithm=algorithm_identity,
    )
    sharded_directory = bool(
        (args.shard_token_budget or args.shard_sample_budget)
        and not Path(args.output).suffix
    )
    receipt_path = Path(args.receipt) if args.receipt else (
        default_pretraining_data_contract_path(
            args.output,
            sharded_directory=sharded_directory,
        )
    )
    write_pretraining_data_contract(receipt_path, receipt)

    rendered_summary = summary.to_dict()
    rendered_summary["receipt"] = str(receipt_path)
    rendered_summary["data_sha256"] = receipt["data_sha256"]
    rendered_summary["data_total_size_bytes"] = receipt[
        "data_total_size_bytes"
    ]
    print(
        json.dumps(
            rendered_summary, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
