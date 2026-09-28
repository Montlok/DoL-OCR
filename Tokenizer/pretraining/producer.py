# -*- coding: utf-8 -*-
"""Generic pretraining row producer, including windowing, packing and CLI."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

from .builder import (
    IGNORE_INDEX,
    EncodedSample,
    PretrainingDataBuilder,
    encoded_sample_to_dict,
)
from .data_contract import (
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


def iter_text_windows(
    sample: EncodedSample, max_length: int,
) -> Iterator[EncodedSample]:
    """Preserve forward and reverse adjacent-token targets with one-token overlap."""

    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if any(sample.modality_spans.values()) or sample.images or sample.videos:
        raise ValueError("text windows cannot split multimodal samples")
    if len(sample.attention_mask) != len(sample.input_ids):
        raise ValueError("attention_mask must align with input_ids")
    end = next(
        (i + 1 for i in range(len(sample.attention_mask) - 1, -1, -1)
         if sample.attention_mask[i]), 0,
    )
    if end == 0:
        return
    if not all(sample.attention_mask[:end]):
        raise ValueError("text windows require a contiguous valid prefix (right padding only)")
    if end < len(sample.input_ids):
        metadata = dict(sample.metadata)
        metadata.pop("padded", None)
        sample = replace(
            sample,
            input_ids=sample.input_ids[:end],
            attention_mask=sample.attention_mask[:end],
            labels=sample.labels[:end],
            token_offsets=sample.token_offsets[:end],
            word_pos=sample.word_pos[:end],
            morph_depth=sample.morph_depth[:end],
            metadata=metadata,
        )
    if len(sample.input_ids) <= max_length:
        yield sample
        return
    if max_length < 2:
        raise ValueError("text windows need max_length >= 2 for next-token targets")
    parent = sample.metadata.get("text_window", {})
    base = int(parent.get("token_start", 0))
    total = int(parent.get("total_tokens", len(sample.input_ids)))
    for start in range(0, len(sample.input_ids) - 1, max_length - 1):
        end = min(start + max_length, len(sample.input_ids))
        word_base = sample.word_pos[start]
        yield EncodedSample(
            input_ids=sample.input_ids[start:end],
            attention_mask=sample.attention_mask[start:end],
            labels=sample.labels[start:end],
            token_offsets=sample.token_offsets[start:end],
            word_pos=[max(0, p - word_base) for p in sample.word_pos[start:end]],
            morph_depth=sample.morph_depth[start:end],
            modality_spans={key: [] for key in sample.modality_spans},
            metadata={**sample.metadata, "text_window": {
                "token_start": base + start,
                "token_end": base + end,
                "total_tokens": total,
            }},
        )


class WindowedPretrainingDataBuilder(PretrainingDataBuilder):
    """Producer-specific overflow policy; the shared tokenizer stays unchanged."""

    def _truncate(self, sample: EncodedSample) -> EncodedSample:
        if (
            any(sample.modality_spans.values()) or sample.images or sample.videos
            or sample.metadata.get("type") in {"image_text", "ocr", "video_text"}
        ):
            return super()._truncate(sample)
        return sample

    def encode_text(self, text: str, metadata: dict | None = None) -> EncodedSample:
        sample = super().encode_text(text, metadata)
        if len(sample.input_ids) > self.max_length:
            raise ValueError(
                "text exceeds max_length; use iter_encode_text to retain all windows"
            )
        return sample

    def iter_encode_text(
        self, text: str, metadata: dict | None = None,
    ) -> Iterator[EncodedSample]:
        yield from iter_text_windows(
            super().encode_text(text, metadata), self.max_length,
        )

    def iter_encode_json_obj(self, obj: dict) -> Iterator[EncodedSample]:
        if (
            obj.get("type", "text") in {"image_text", "ocr", "video_text"}
            or obj.get("images") or obj.get("videos")
        ):
            yield self.encode_json_obj(obj)
        else:
            yield from self.iter_encode_text(
                str(obj.get("text", "")), metadata=self._metadata(obj),
            )


def pack_samples(
    samples: list[EncodedSample],
    max_length: int,
    pad_id: int,
    eos_id: int,
    pad_to_max_length: bool = False,
) -> list[EncodedSample]:
    return list(
        iter_pack_samples(
            samples,
            max_length=max_length,
            pad_id=pad_id,
            eos_id=eos_id,
            pad_to_max_length=pad_to_max_length,
        )
    )


def iter_pack_samples(
    samples: Iterable[EncodedSample],
    max_length: int,
    pad_id: int,
    eos_id: int,
    pad_to_max_length: bool = False,
) -> Iterator[EncodedSample]:
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    current = _empty_text_pack()
    count = 0

    def flush() -> EncodedSample | None:
        nonlocal current, count
        emitted = None
        if current.input_ids:
            current.metadata = {"type": "packed_text", "num_samples": count}
            if pad_to_max_length:
                current = _pad(current, max_length, pad_id)
            emitted = current
        current = _empty_text_pack()
        count = 0
        return emitted

    for sample in samples:
        is_multimodal = _has_modality(sample)
        if not is_multimodal and (
            len(sample.input_ids) > max_length or "text_window" in sample.metadata
        ):
            flushed = flush()
            if flushed is not None:
                yield flushed
            for window in iter_text_windows(sample, max_length):
                yield _pad(window, max_length, pad_id) if pad_to_max_length else window
            continue
        sample = _trim(sample, max_length)
        if not sample.input_ids:
            continue
        if is_multimodal:
            flushed = flush()
            if flushed is not None:
                yield flushed
            if pad_to_max_length:
                sample = _pad(sample, max_length, pad_id)
            yield sample
            continue

        extra_ids = list(sample.input_ids)
        extra_mask = list(sample.attention_mask)
        extra_labels = list(sample.labels)
        extra_offsets = list(sample.token_offsets)
        extra_word_pos = list(sample.word_pos)
        extra_morph_depth = list(sample.morph_depth)
        # Documents are packed back-to-back with one EOS separator and NO
        # cross-document attention isolation: attention and Mamba state may
        # flow across the boundary. Deliberate — the official Mamba kernel has
        # no affordable mid-sequence state reset, so block-diagonal masking
        # could only ever isolate the attention half of the hybrid; EOS is the
        # learned boundary signal instead (GPT-2/3-style packing).
        if current.input_ids:
            word_base = _next_word_pos(current.word_pos)
            extra_ids = [eos_id] + extra_ids
            extra_mask = [1] + extra_mask
            extra_labels = [eos_id] + extra_labels
            extra_offsets = [(-1, -1)] + extra_offsets
            extra_word_pos = [word_base] + [
                pos + word_base + 1 for pos in extra_word_pos
            ]
            extra_morph_depth = [0] + extra_morph_depth

        if len(current.input_ids) + len(extra_ids) > max_length:
            flushed = flush()
            if flushed is not None:
                yield flushed
            extra_ids = list(sample.input_ids)
            extra_mask = list(sample.attention_mask)
            extra_labels = list(sample.labels)
            extra_offsets = list(sample.token_offsets)
            extra_word_pos = list(sample.word_pos)
            extra_morph_depth = list(sample.morph_depth)

        current.input_ids.extend(extra_ids[:max_length])
        current.attention_mask.extend(extra_mask[:max_length])
        current.labels.extend(extra_labels[:max_length])
        current.token_offsets.extend(extra_offsets[:max_length])
        current.word_pos.extend(extra_word_pos[:max_length])
        current.morph_depth.extend(extra_morph_depth[:max_length])
        count += 1

    flushed = flush()
    if flushed is not None:
        yield flushed


def _empty_text_pack() -> EncodedSample:
    return EncodedSample(
        input_ids=[],
        attention_mask=[],
        labels=[],
        token_offsets=[],
        word_pos=[],
        morph_depth=[],
        modality_spans={"image_token_spans": [], "video_token_spans": []},
        metadata={"type": "packed_text", "num_samples": 0},
        images=[],
        image_sizes=[],
        videos=[],
        video_sizes=[],
        ocr_labels=[],
        reading_order=[],
    )


def _has_modality(sample: EncodedSample) -> bool:
    return bool(
        sample.modality_spans.get("image_token_spans")
        or sample.modality_spans.get("video_token_spans")
    )


def _trim(sample: EncodedSample, max_length: int) -> EncodedSample:
    if len(sample.input_ids) <= max_length:
        return sample
    cutoff = max_length
    for spans in sample.modality_spans.values():
        for start, end in spans:
            start_i = int(start)
            end_i = int(end)
            if start_i < cutoff < end_i:
                cutoff = start_i
    modality_spans = {
        key: [span for span in spans if int(span[1]) <= cutoff]
        for key, spans in sample.modality_spans.items()
    }
    # Drop media payloads whose <image_patch>/<video_patch> spans were
    # trimmed away — counts of surviving spans tell us how many of the
    # leading entries in ``images`` / ``videos`` remain valid. This keeps
    # ``len(images) == len(image_token_spans)`` after trim, which is the
    # invariant the downstream collator relies on.
    n_image_spans = len(modality_spans.get("image_token_spans", []))
    n_video_spans = len(modality_spans.get("video_token_spans", []))
    return EncodedSample(
        input_ids=sample.input_ids[:cutoff],
        attention_mask=sample.attention_mask[:cutoff],
        labels=sample.labels[:cutoff],
        token_offsets=sample.token_offsets[:cutoff],
        word_pos=sample.word_pos[:cutoff],
        morph_depth=sample.morph_depth[:cutoff],
        modality_spans=modality_spans,
        metadata={**sample.metadata, "truncated": True},
        images=list(sample.images[:n_image_spans]),
        image_sizes=list(sample.image_sizes[:n_image_spans]),
        videos=list(sample.videos[:n_video_spans]),
        video_sizes=list(sample.video_sizes[:n_video_spans]),
        ocr_labels=list(sample.ocr_labels[:n_image_spans]),
        reading_order=list(sample.reading_order[:n_image_spans]),
    )


def _pad(sample: EncodedSample, max_length: int, pad_id: int) -> EncodedSample:
    pad_count = max_length - len(sample.input_ids)
    if pad_count <= 0:
        return sample
    return EncodedSample(
        input_ids=sample.input_ids + [pad_id] * pad_count,
        attention_mask=sample.attention_mask + [0] * pad_count,
        labels=sample.labels + [IGNORE_INDEX] * pad_count,
        token_offsets=sample.token_offsets + [(-1, -1)] * pad_count,
        word_pos=sample.word_pos + [0] * pad_count,
        morph_depth=sample.morph_depth + [0] * pad_count,
        modality_spans={
            key: list(spans) for key, spans in sample.modality_spans.items()
        },
        metadata={**sample.metadata, "padded": True},
        images=list(sample.images),
        image_sizes=list(sample.image_sizes),
        videos=list(sample.videos),
        video_sizes=list(sample.video_sizes),
        ocr_labels=list(sample.ocr_labels),
        reading_order=list(sample.reading_order),
    )


def _next_word_pos(word_pos: list[int]) -> int:
    return (max(word_pos) + 1) if word_pos else 0


def _iter_samples(
    path: str, builder: WindowedPretrainingDataBuilder
) -> Iterable[EncodedSample]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if path.endswith(".jsonl"):
                yield from builder.iter_encode_json_obj(json.loads(line))
            else:
                yield from builder.iter_encode_text(line, metadata={"type": "text"})


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
            1 for label, active in zip(row["labels"][1:], row["attention_mask"][1:])
            if active and int(label) != IGNORE_INDEX
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

    if args.max_length < 2 or (
        args.pack_max_length is not None and args.pack_max_length < 2
    ):
        parser.error("sequence lengths must be >= 2 for next-token supervision")
    bundle = TokenizerBundle.from_dir(args.tokenizer_bundle)
    window_length = min(args.max_length, args.pack_max_length) if (
        args.pack and args.pack_max_length is not None
    ) else args.max_length
    builder = WindowedPretrainingDataBuilder(bundle, max_length=window_length)
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
