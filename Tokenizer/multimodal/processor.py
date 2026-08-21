# -*- coding: utf-8 -*-
"""Multimodal tokenizer processor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from Tokenizer.unified.encoded import EncodedToken

from .image_placeholders import (
    image_patch_count,
    image_placeholder_tokens,
)
from .video_placeholders import (
    video_patch_count,
    video_placeholder_tokens,
)
from .tokens import IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER


@dataclass
class MultimodalEncoding:
    input_ids: list[int]
    attention_mask: list[int]
    tokens: list[EncodedToken]
    image_token_spans: list[tuple[int, int]]
    video_token_spans: list[tuple[int, int]] = field(default_factory=list)
    images: list[Any] = field(default_factory=list)
    videos: list[Any] = field(default_factory=list)
    pixel_values: Any = None


@dataclass(frozen=True)
class _PlaceholderRecord:
    start: int
    end: int
    patch_count: int


class MultimodalProcessor:
    def __init__(
        self,
        tokenizer,
        image_processor=None,
        patch_size: int = 14,
        merge_size: int = 2,
        temporal_patch_size: int = 2,
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.temporal_patch_size = temporal_patch_size

    def __call__(
        self,
        text: str,
        images: list[Any] | None = None,
        image_sizes: list[Any] | None = None,
        videos: list[Any] | None = None,
        video_sizes: list[Any] | None = None,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> MultimodalEncoding:
        image_list = list(images or [])
        sizes = list(image_sizes or [])
        video_list = list(videos or [])
        vsizes = list(video_sizes or [])

        n_img_holes = text.count(IMAGE_PLACEHOLDER)
        n_vid_holes = text.count(VIDEO_PLACEHOLDER)

        if len(image_list) != n_img_holes:
            raise ValueError(
                f"<image> placeholder count ({n_img_holes}) does not match "
                f"images length ({len(image_list)})"
            )
        if len(video_list) != n_vid_holes:
            raise ValueError(
                f"<video> placeholder count ({n_vid_holes}) does not match "
                f"videos length ({len(video_list)})"
            )
        if sizes and len(sizes) != n_img_holes:
            raise ValueError(
                f"<image> placeholder count ({n_img_holes}) does not match "
                f"image_sizes length ({len(sizes)})"
            )
        if vsizes and len(vsizes) != n_vid_holes:
            raise ValueError(
                f"<video> placeholder count ({n_vid_holes}) does not match "
                f"video_sizes length ({len(vsizes)})"
            )

        # Default sizes for missing entries: assume single-patch image/video.
        if not sizes and n_img_holes:
            sizes = [None] * n_img_holes
        if not vsizes and n_vid_holes:
            vsizes = [None] * n_vid_holes

        expanded, image_records, video_records, boundary_map = self._expand_placeholders(
            text, sizes, vsizes
        )

        result = self.tokenizer.encode_with_spans(
            expanded, add_bos=add_bos, add_eos=add_eos
        )
        tokens = self._annotate_patches(
            result.tokens, image_records, video_records, boundary_map
        )
        img_spans = self._collect_spans(
            tokens, "<image_start>", "<image_patch>", "<image_end>", image_records
        )
        vid_spans = self._collect_spans(
            tokens, "<video_start>", "<video_patch>", "<video_end>", video_records
        )

        # Invariant: attention_mask aligns with input_ids
        assert len(result.attention_mask) == len(result.input_ids)

        processed_images = image_list
        if self.image_processor is not None and image_list:
            processed_images = self.image_processor(image_list)

        return MultimodalEncoding(
            input_ids=result.input_ids,
            attention_mask=result.attention_mask,
            tokens=tokens,
            image_token_spans=img_spans,
            video_token_spans=vid_spans,
            images=processed_images,
            videos=video_list,
        )

    # ---- expansion ----

    def _expand_placeholders(
        self, text: str, image_sizes: list[Any], video_sizes: list[Any]
    ) -> tuple[str, list[_PlaceholderRecord], list[_PlaceholderRecord], list[int]]:
        out: list[str] = []
        boundary_map = [0]
        image_records: list[_PlaceholderRecord] = []
        video_records: list[_PlaceholderRecord] = []
        img_idx = 0
        vid_idx = 0
        i = 0
        while i < len(text):
            if text.startswith(IMAGE_PLACEHOLDER, i):
                n = self._patches_image(image_sizes[img_idx])
                end = i + len(IMAGE_PLACEHOLDER)
                image_records.append(_PlaceholderRecord(i, end, n))
                expanded = "".join(image_placeholder_tokens(n))
                out.append(expanded)
                boundary_map.extend([end] * len(expanded))
                img_idx += 1
                i = end
                continue
            if text.startswith(VIDEO_PLACEHOLDER, i):
                n = self._patches_video(video_sizes[vid_idx])
                end = i + len(VIDEO_PLACEHOLDER)
                video_records.append(_PlaceholderRecord(i, end, n))
                expanded = "".join(video_placeholder_tokens(n))
                out.append(expanded)
                boundary_map.extend([end] * len(expanded))
                vid_idx += 1
                i = end
                continue
            out.append(text[i])
            boundary_map.append(i + 1)
            i += 1
        return "".join(out), image_records, video_records, boundary_map

    def _expand_images(self, text: str, sizes: list[Any]) -> str:
        parts = text.split(IMAGE_PLACEHOLDER)
        if len(parts) == 1:
            return text
        out = [parts[0]]
        for i, size in enumerate(sizes):
            n = self._patches_image(size)
            out.append("".join(image_placeholder_tokens(n)))
            out.append(parts[i + 1])
        return "".join(out)

    def _expand_videos(self, text: str, sizes: list[Any]) -> str:
        parts = text.split(VIDEO_PLACEHOLDER)
        if len(parts) == 1:
            return text
        out = [parts[0]]
        for i, size in enumerate(sizes):
            n = self._patches_video(size)
            out.append("".join(video_placeholder_tokens(n)))
            out.append(parts[i + 1])
        return "".join(out)

    def _patches_image(self, size: Any) -> int:
        if size is None:
            return 1
        if isinstance(size, dict):
            w = int(size.get("width") or size.get("w"))
            h = int(size.get("height") or size.get("h"))
        else:
            w, h = int(size[0]), int(size[1])
        return image_patch_count(w, h, self.patch_size, self.merge_size)

    def _patches_video(self, size: Any) -> int:
        if size is None:
            return 1
        if isinstance(size, dict):
            f = int(size.get("frames") or size.get("num_frames"))
            w = int(size.get("width") or size.get("w"))
            h = int(size.get("height") or size.get("h"))
        else:
            f, w, h = int(size[0]), int(size[1]), int(size[2])
        return video_patch_count(
            f, w, h, self.patch_size, self.temporal_patch_size, self.merge_size
        )

    # ---- annotation & spans ----

    def _annotate_patches(
        self,
        tokens: list[EncodedToken],
        image_records: list[_PlaceholderRecord],
        video_records: list[_PlaceholderRecord],
        boundary_map: list[int],
    ) -> list[EncodedToken]:
        """Attach image_index/video_index metadata to patch tokens."""
        out: list[EncodedToken] = []
        img_idx = -1
        vid_idx = -1
        in_img = False
        in_vid = False
        for t in tokens:
            meta = t.metadata
            start, end = self._remap_offsets(t.start, t.end, boundary_map)
            if t.token == "<image_start>":
                img_idx += 1
                in_img = True
                meta = self._placeholder_metadata(meta, image_records, img_idx, "image_index")
            elif t.token == "<image_end>":
                meta = self._placeholder_metadata(meta, image_records, img_idx, "image_index")
                in_img = False
            elif t.token == "<image_patch>" and in_img:
                meta = self._placeholder_metadata(meta, image_records, img_idx, "image_index")
            elif t.token == "<video_start>":
                vid_idx += 1
                in_vid = True
                meta = self._placeholder_metadata(meta, video_records, vid_idx, "video_index")
            elif t.token == "<video_end>":
                meta = self._placeholder_metadata(meta, video_records, vid_idx, "video_index")
                in_vid = False
            elif t.token == "<video_patch>" and in_vid:
                meta = self._placeholder_metadata(meta, video_records, vid_idx, "video_index")
            if meta is not t.metadata:
                source_span = meta.get("source_span") if meta else None
                if source_span is not None:
                    start, end = int(source_span[0]), int(source_span[1])
                out.append(
                    EncodedToken(
                        t.id, t.token, t.track, start, end, t.surface, meta
                    )
                )
            elif start != t.start or end != t.end:
                out.append(
                    EncodedToken(
                        t.id, t.token, t.track, start, end, t.surface, t.metadata
                    )
                )
            else:
                out.append(t)
        return out

    def _remap_offsets(
        self, start: int, end: int, boundary_map: list[int]
    ) -> tuple[int, int]:
        if start == -1 and end == -1:
            return start, end
        if not boundary_map:
            return start, end
        start_idx = min(max(start, 0), len(boundary_map) - 1)
        end_idx = min(max(end, 0), len(boundary_map) - 1)
        return boundary_map[start_idx], boundary_map[end_idx]

    def _placeholder_metadata(
        self,
        meta: dict | None,
        records: list[_PlaceholderRecord],
        index: int,
        index_key: str,
    ) -> dict:
        merged = {**(meta or {}), index_key: index}
        if 0 <= index < len(records):
            record = records[index]
            merged["source_span"] = [record.start, record.end]
        return merged

    def _collect_spans(
        self,
        tokens: list[EncodedToken],
        start_tok: str,
        patch_tok: str,
        end_tok: str,
        records: list[_PlaceholderRecord],
    ) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        start: int | None = None
        for idx, t in enumerate(tokens):
            if t.token == start_tok:
                start = idx
            elif t.token == end_tok and start is not None:
                spans.append((start, idx + 1))
                start = None
        if len(spans) != len(records):
            raise ValueError(
                f"{start_tok}/{end_tok} span count ({len(spans)}) does not match "
                f"placeholder count ({len(records)})"
            )
        for i, (span_start, span_end) in enumerate(spans):
            patch_count = sum(1 for token in tokens[span_start:span_end] if token.token == patch_tok)
            if patch_count != records[i].patch_count:
                raise ValueError(
                    f"{patch_tok} count in span {i} is {patch_count}, "
                    f"expected {records[i].patch_count}"
                )
            if span_end - span_start != records[i].patch_count + 2:
                raise ValueError(
                    f"{start_tok}/{end_tok} span length for placeholder {i} is "
                    f"{span_end - span_start}, expected {records[i].patch_count + 2}"
                )
        return spans
