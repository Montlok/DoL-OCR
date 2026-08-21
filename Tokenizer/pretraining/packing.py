# -*- coding: utf-8 -*-
"""Simple sample packing for minimum pretraining data builds."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from .builder import IGNORE_INDEX, EncodedSample


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
