# -*- coding: utf-8 -*-
"""Gate a tokenizer bundle and input file for pretraining data construction."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from Tokenizer.pretraining import (
    IGNORE_INDEX,
    EncodedSample,
    PretrainingDataBuilder,
    derive_morph_info_from_offsets,
    nested_int_lists,
)
from Tokenizer.unified.bundle import TokenizerBundle


def run_gate(
    bundle_dir: str,
    input_path: str,
    max_length: int,
    max_unk_rate: float = 0.01,
    min_supervised_rate: float = 0.01,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    bundle = TokenizerBundle.from_dir(bundle_dir)
    for issue in bundle.validate():
        failures.append({"scope": "bundle", "message": issue})

    builder = PretrainingDataBuilder(bundle, max_length=max_length)
    num_samples = 0
    total_tokens = 0
    unk_count = 0
    supervised_tokens = 0
    max_len_seen = 0
    max_morph_depth = 0
    for idx, obj in enumerate(_iter_input(input_path)):
        if _is_encoded_row(obj):
            try:
                sample = _sample_from_encoded_row(obj)
                text = _source_text_from_encoded_row(obj)
            except Exception as exc:
                failures.append(
                    {"sample": idx, "message": f"encoded row parse failed: {exc}"}
                )
                continue
        else:
            text = str(obj.get("text", ""))
            try:
                sample = builder.encode_json_obj(obj)
            except Exception as exc:
                failures.append({"sample": idx, "message": f"encode failed: {exc}"})
                continue
        num_samples += 1
        max_len_seen = max(max_len_seen, len(sample.input_ids))
        total_tokens += len(sample.input_ids)
        unk_count += sample.input_ids.count(bundle.tokenizer.unk_id)
        supervised_tokens += sum(1 for label in sample.labels if label != IGNORE_INDEX)
        max_morph_depth = max(max_morph_depth, max(sample.morph_depth, default=0))
        failures.extend(_validate_sample(bundle, sample, text, idx))

    metrics = {
        "total_tokens": total_tokens,
        "supervised_tokens": supervised_tokens,
        "supervised_rate": (supervised_tokens / total_tokens) if total_tokens else 0.0,
        "unk_rate": (unk_count / total_tokens) if total_tokens else 0.0,
        "max_len": max_len_seen,
        "max_morph_depth": max_morph_depth,
    }
    if not num_samples:
        failures.append({"scope": "dataset", "message": "no valid samples"})
    if metrics["unk_rate"] > max_unk_rate:
        failures.append(
            {
                "scope": "dataset",
                "message": f"unk_rate {metrics['unk_rate']:.6f} exceeds {max_unk_rate:.6f}",
            }
        )
    if total_tokens and metrics["supervised_rate"] < min_supervised_rate:
        failures.append(
            {
                "scope": "dataset",
                "message": (
                    f"supervised_rate {metrics['supervised_rate']:.6f} "
                    f"is below {min_supervised_rate:.6f}"
                ),
            }
        )
    return {
        "passed": not failures,
        "num_samples": num_samples,
        "metrics": metrics,
        "failures": failures,
    }


def _iter_input(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if path.endswith(".jsonl"):
                yield json.loads(line)
            else:
                yield {"type": "text", "text": line}


def _is_encoded_row(obj: dict[str, Any]) -> bool:
    required = {
        "input_ids",
        "attention_mask",
        "labels",
        "token_offsets",
        "modality_spans",
    }
    return required.issubset(obj)


def _sample_from_encoded_row(obj: dict[str, Any]) -> EncodedSample:
    modality_spans = obj.get("modality_spans") or {}
    token_offsets = [tuple(item) for item in obj["token_offsets"]]
    word_pos = obj.get("word_pos")
    morph_depth = obj.get("morph_depth")
    if word_pos is None or morph_depth is None:
        word_pos, morph_depth = derive_morph_info_from_offsets(token_offsets)
    return EncodedSample(
        input_ids=[int(item) for item in obj["input_ids"]],
        attention_mask=[int(item) for item in obj["attention_mask"]],
        labels=[int(item) for item in obj["labels"]],
        token_offsets=token_offsets,
        word_pos=[int(item) for item in word_pos],
        morph_depth=[int(item) for item in morph_depth],
        modality_spans={
            "image_token_spans": [
                tuple(span) for span in modality_spans.get("image_token_spans", [])
            ],
            "video_token_spans": [
                tuple(span) for span in modality_spans.get("video_token_spans", [])
            ],
        },
        metadata=dict(obj.get("metadata") or {}),
        images=list(obj.get("images") or []),
        image_sizes=list(obj.get("image_sizes") or []),
        videos=list(obj.get("videos") or []),
        video_sizes=list(obj.get("video_sizes") or []),
        ocr_labels=nested_int_lists(obj.get("ocr_labels")),
        reading_order=nested_int_lists(obj.get("reading_order")),
    )


def _source_text_from_encoded_row(obj: dict[str, Any]) -> str | None:
    if "text" in obj:
        return str(obj["text"])
    metadata = obj.get("metadata")
    if isinstance(metadata, dict) and "text" in metadata:
        return str(metadata["text"])
    return None


def _validate_sample(
    bundle: TokenizerBundle, sample: EncodedSample, text: str | None, idx: int
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    n = len(sample.input_ids)
    if len(sample.labels) != n or len(sample.attention_mask) != n:
        failures.append(
            {
                "sample": idx,
                "message": "input_ids/labels/attention_mask length mismatch",
            }
        )
    if len(sample.token_offsets) != n:
        failures.append({"sample": idx, "message": "token_offsets length mismatch"})
    if len(sample.word_pos) != n:
        failures.append({"sample": idx, "message": "word_pos length mismatch"})
    if len(sample.morph_depth) != n:
        failures.append({"sample": idx, "message": "morph_depth length mismatch"})
    vocab_size = bundle.tokenizer.vocab_size
    for pos, token_id in enumerate(sample.input_ids):
        if token_id < 0 or token_id >= vocab_size:
            failures.append(
                {
                    "sample": idx,
                    "token": pos,
                    "message": f"token id {token_id} out of range",
                }
            )
    for pos, label in enumerate(sample.labels):
        if label == IGNORE_INDEX:
            continue
        if label < 0 or label >= vocab_size:
            failures.append(
                {
                    "sample": idx,
                    "token": pos,
                    "message": f"label id {label} out of range",
                }
            )
    for pos, offset in enumerate(sample.token_offsets):
        start, end = int(offset[0]), int(offset[1])
        if (start, end) == (-1, -1):
            continue
        if start < 0 or end < start:
            failures.append(
                {
                    "sample": idx,
                    "token": pos,
                    "message": f"invalid offset {(start, end)}",
                }
            )
            continue
        if text is not None and end > len(text):
            failures.append(
                {
                    "sample": idx,
                    "token": pos,
                    "message": f"invalid offset {(start, end)}",
                }
            )
    for pos, value in enumerate(sample.word_pos):
        if int(value) < 0:
            failures.append(
                {"sample": idx, "token": pos, "message": "negative word_pos"}
            )
    for pos, value in enumerate(sample.morph_depth):
        if int(value) < 0:
            failures.append(
                {"sample": idx, "token": pos, "message": "negative morph_depth"}
            )
    failures.extend(
        _validate_spans(
            bundle,
            sample,
            idx,
            "image_token_spans",
            "<image_start>",
            "<image_end>",
        )
    )
    failures.extend(
        _validate_spans(
            bundle,
            sample,
            idx,
            "video_token_spans",
            "<video_start>",
            "<video_end>",
        )
    )
    failures.extend(_validate_media_payloads(sample, idx))
    return failures


def _validate_media_payloads(
    sample: EncodedSample, sample_idx: int
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    n_image_spans = len(sample.modality_spans.get("image_token_spans", []))
    n_video_spans = len(sample.modality_spans.get("video_token_spans", []))

    checks = [
        ("images", sample.images, n_image_spans),
        ("videos", sample.videos, n_video_spans),
    ]
    for field, values, expected in checks:
        if len(values) != expected:
            failures.append(
                {
                    "sample": sample_idx,
                    "message": f"{field} count {len(values)} does not match span count {expected}",
                }
            )

    optional_checks = [
        ("image_sizes", sample.image_sizes, n_image_spans),
        ("video_sizes", sample.video_sizes, n_video_spans),
        ("ocr_labels", sample.ocr_labels, n_image_spans),
        ("reading_order", sample.reading_order, n_image_spans),
    ]
    for field, values, expected in optional_checks:
        if values and len(values) != expected:
            failures.append(
                {
                    "sample": sample_idx,
                    "message": f"{field} count {len(values)} does not match span count {expected}",
                }
            )
    return failures


def _validate_spans(
    bundle: TokenizerBundle,
    sample: EncodedSample,
    sample_idx: int,
    key: str,
    start_token: str,
    end_token: str,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    start_id = bundle.tokenizer.vocab[start_token]
    end_id = bundle.tokenizer.vocab[end_token]
    for span_idx, span in enumerate(sample.modality_spans.get(key, [])):
        start, end = int(span[0]), int(span[1])
        if start < 0 or end > len(sample.input_ids) or end <= start:
            failures.append(
                {
                    "sample": sample_idx,
                    "span": span_idx,
                    "message": f"{key} out of bounds",
                }
            )
            continue
        if end - start < 2:
            failures.append(
                {
                    "sample": sample_idx,
                    "span": span_idx,
                    "message": f"{key} missing start/end",
                }
            )
            continue
        if sample.input_ids[start] != start_id or sample.input_ids[end - 1] != end_id:
            failures.append(
                {
                    "sample": sample_idx,
                    "span": span_idx,
                    "message": f"{key} markers mismatch",
                }
            )
        label_end = min(end, len(sample.labels))
        for token_pos in range(start, label_end):
            if sample.labels[token_pos] != IGNORE_INDEX:
                failures.append(
                    {
                        "sample": sample_idx,
                        "span": span_idx,
                        "token": token_pos,
                        "message": f"{key} label is not ignored",
                    }
                )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-bundle", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-unk-rate", type=float, default=0.01)
    parser.add_argument("--min-supervised-rate", type=float, default=0.01)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = run_gate(
        args.tokenizer_bundle,
        args.input,
        args.max_length,
        max_unk_rate=args.max_unk_rate,
        min_supervised_rate=args.min_supervised_rate,
    )
    if args.json:
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
        print()
    else:
        print(f"passed={str(result['passed']).lower()}")
        print(f"num_samples={result['num_samples']}")
        print(f"unk_rate={result['metrics']['unk_rate']:.6f}")
        print(f"supervised_rate={result['metrics']['supervised_rate']:.6f}")
        for failure in result["failures"]:
            print(f"failure={failure}")
    if not result["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
