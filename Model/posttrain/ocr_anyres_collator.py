# -*- coding: utf-8 -*-

"""Joint global/detail SFT collation for admitted DoL OCR anyres v2 items."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from Model.config import (
    BOS_ID,
    EOS_ID,
    IGNORE_INDEX,
    IMAGE_END_ID,
    IMAGE_PATCH_ID,
    IMAGE_START_ID,
    PAD_ID,
    OMVTConfig,
)
from Model.ocr.data import build_ocr_row
from Model.ocr.image_preprocess import letterbox_grayscale_to_square
from Model.ocr.position_contract import BOUNDARY_V1
from Model.omvt.native_patcher import pack_native_omvt_batch
from Model.omvt.patcher import collate_omvt_batch
from Model.posttrain.ocr_anyres_manifest import (
    ANYRES_VISUAL_CONTRACT,
    quota_bucket,
)
from Tokenizer.multimodal import NativeImageProcessorV2, PILImageProcessor


ANYRES_GLOBAL_IMAGE_TOKENS = 256


def _image_spec(payload: object, where: str) -> bytes | str:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{where} must be an AnyresOCRDataset image payload")
    delivery = payload.get("delivery")
    if delivery == "bytes" and set(payload).issuperset({"bytes", "metadata"}):
        value = payload.get("bytes")
        if not isinstance(value, (bytes, bytearray, memoryview)):
            raise ValueError(f"{where}.bytes must contain encoded image bytes")
        if "path" in payload:
            raise ValueError(f"{where} must not mix bytes and path delivery")
        return bytes(value)
    if delivery == "path" and set(payload).issuperset({"path", "metadata"}):
        value = payload.get("path")
        if not isinstance(value, str) or not value:
            raise ValueError(f"{where}.path must be a non-empty path")
        if "bytes" in payload:
            raise ValueError(f"{where} must not mix bytes and path delivery")
        return value
    raise ValueError(f"{where}.delivery must be bytes or path with metadata")


def _sample_from_item(item: object, index: int) -> Mapping[str, Any]:
    if not isinstance(item, Mapping):
        raise ValueError(f"items[{index}] must be an AnyresOCRDataset item")
    sample = item.get("sample")
    if not isinstance(sample, Mapping):
        raise ValueError(f"items[{index}].sample must be an object")
    if sample.get("visual_contract") != ANYRES_VISUAL_CONTRACT:
        raise ValueError(
            f"items[{index}] is not admitted under {ANYRES_VISUAL_CONTRACT!r}"
        )
    if sample.get("qa_state") != "accepted":
        raise ValueError(f"items[{index}] must be an accepted anyres sample")
    return sample


def _pad_rows(
    rows: Sequence[Mapping[str, Sequence[int]]],
    *,
    pad_id: int,
    ignore_index: int,
) -> dict[str, torch.Tensor]:
    max_length = max(len(row["input_ids"]) for row in rows)
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    labels: list[list[int]] = []
    for row in rows:
        length = len(row["input_ids"])
        padding = max_length - length
        input_ids.append([int(value) for value in row["input_ids"]] + [pad_id] * padding)
        attention_mask.append(
            [int(value) for value in row["attention_mask"]] + [0] * padding
        )
        labels.append(
            [int(value) for value in row["labels"]]
            + [ignore_index] * padding
        )
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class AnyresOCRSFTCollator:
    """Build one boundary-v1 text batch plus global and native detail inputs."""

    def __init__(
        self,
        *,
        encode_reference: Callable[[str], Sequence[int]],
        omvt_cfg: OMVTConfig,
        global_processor: PILImageProcessor,
        native_processor: NativeImageProcessorV2,
        max_raw_patch_tokens_per_view: int,
        max_seq_len: int = 512,
        bos_id: int = BOS_ID,
        image_start_id: int = IMAGE_START_ID,
        image_patch_id: int = IMAGE_PATCH_ID,
        image_end_id: int = IMAGE_END_ID,
        eos_id: int = EOS_ID,
        pad_id: int = PAD_ID,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        if getattr(encode_reference, "mode", None) != "native":
            raise ValueError("encode_reference must be the reviewed native OCR encoder")
        if not isinstance(omvt_cfg, OMVTConfig):
            raise TypeError("omvt_cfg must be an OMVTConfig")
        if omvt_cfg.compress_to != ANYRES_GLOBAL_IMAGE_TOKENS:
            raise ValueError("anyres global OMVT must emit exactly 256 visual tokens")
        if not isinstance(global_processor, PILImageProcessor):
            raise TypeError("global_processor must be PILImageProcessor")
        if not isinstance(native_processor, NativeImageProcessorV2):
            raise TypeError("native_processor must be NativeImageProcessorV2")
        if global_processor.image_size != omvt_cfg.image_size:
            raise ValueError("global processor image_size differs from OMVTConfig")
        if global_processor.in_channels != omvt_cfg.in_channels:
            raise ValueError("global processor channels differ from OMVTConfig")
        if native_processor.in_channels != omvt_cfg.in_channels:
            raise ValueError("native processor channels differ from OMVTConfig")
        if (
            isinstance(max_raw_patch_tokens_per_view, bool)
            or not isinstance(max_raw_patch_tokens_per_view, int)
            or max_raw_patch_tokens_per_view <= 0
        ):
            raise ValueError("max_raw_patch_tokens_per_view must be positive")
        if isinstance(max_seq_len, bool) or not isinstance(max_seq_len, int) or max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        self.encode_reference = encode_reference
        self.omvt_cfg = omvt_cfg
        self.global_processor = global_processor
        self.native_processor = native_processor
        self.max_raw_patch_tokens_per_view = max_raw_patch_tokens_per_view
        self.max_seq_len = max_seq_len
        self.bos_id = int(bos_id)
        self.image_start_id = int(image_start_id)
        self.image_patch_id = int(image_patch_id)
        self.image_end_id = int(image_end_id)
        self.eos_id = int(eos_id)
        self.pad_id = int(pad_id)
        self.ignore_index = int(ignore_index)

    def __call__(self, items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not items:
            raise ValueError("anyres SFT items must not be empty")

        rows: list[dict[str, Any]] = []
        samples: list[Mapping[str, Any]] = []
        global_specs: list[Any] = []
        detail_specs: list[Any] = []
        view_to_sample: list[int] = []
        view_ids: list[str] = []
        quota_buckets: list[str] = []
        dataset_contracts: list[str] = []

        for sample_index, item in enumerate(items):
            dataset_contract = item.get("dataset_contract_sha256")
            if (
                not isinstance(dataset_contract, str)
                or len(dataset_contract) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in dataset_contract
                )
            ):
                raise ValueError(
                    f"items[{sample_index}].dataset_contract_sha256 is invalid"
                )
            dataset_contracts.append(dataset_contract)
            sample = _sample_from_item(item, sample_index)
            reference = sample.get("reference_model")
            if not isinstance(reference, Mapping) or not isinstance(
                reference.get("text"), str
            ):
                raise ValueError(
                    f"items[{sample_index}].sample.reference_model.text is required"
                )
            target_ids = [int(value) for value in self.encode_reference(reference["text"])]
            if not target_ids:
                raise ValueError(f"items[{sample_index}] native reference is empty")
            if self.eos_id in target_ids:
                raise ValueError(
                    f"items[{sample_index}] native reference must not contain EOS"
                )
            expected_count = sample.get("reference_token_count")
            if expected_count != len(target_ids):
                raise ValueError(
                    f"items[{sample_index}] reference_token_count mismatch: "
                    f"manifest={expected_count!r} native={len(target_ids)}; "
                    "the manifest count excludes the one appended EOS"
                )
            row = build_ocr_row(
                target_ids,
                ANYRES_GLOBAL_IMAGE_TOKENS,
                None,
                bos_id=self.bos_id,
                image_start_id=self.image_start_id,
                image_patch_id=self.image_patch_id,
                image_end_id=self.image_end_id,
                eos_id=self.eos_id,
                add_eos=True,
                ignore_index=self.ignore_index,
            )
            if "word_pos" in row or "morph_depth" in row:
                raise RuntimeError("boundary_v1 rows must not materialize positions")
            if len(row["input_ids"]) > self.max_seq_len:
                raise ValueError(
                    f"items[{sample_index}] OCR row length {len(row['input_ids'])} "
                    f"exceeds max_seq_len={self.max_seq_len}"
                )
            supervised = [value for value in row["labels"] if value != self.ignore_index]
            if supervised != target_ids + [self.eos_id]:
                raise RuntimeError("OCR row supervision does not match target plus EOS")
            rows.append(row)
            samples.append(sample)

            canonical = item.get("canonical_image")
            global_spec = _image_spec(
                canonical, f"items[{sample_index}].canonical_image"
            )
            global_specs.append(
                letterbox_grayscale_to_square(global_spec, self.omvt_cfg.image_size)
            )

            derived = item.get("derived_images")
            if not isinstance(derived, Sequence) or isinstance(
                derived, (str, bytes, bytearray)
            ):
                raise ValueError(f"items[{sample_index}].derived_images must be an array")
            reading_order = sample.get("reading_order")
            if not isinstance(reading_order, Sequence) or isinstance(
                reading_order, (str, bytes, bytearray)
            ):
                raise ValueError(f"items[{sample_index}].reading_order must be an array")
            observed_ids: list[str] = []
            for detail_index, payload in enumerate(derived):
                if not isinstance(payload, Mapping):
                    raise ValueError(
                        f"items[{sample_index}].derived_images[{detail_index}] "
                        "must be an image payload"
                    )
                metadata = payload.get("metadata")
                if not isinstance(metadata, Mapping) or not isinstance(
                    metadata.get("view_id"), str
                ):
                    raise ValueError(
                        f"items[{sample_index}].derived_images[{detail_index}] "
                        "must carry view metadata"
                    )
                observed_ids.append(str(metadata["view_id"]))
                detail_specs.append(
                    _image_spec(
                        payload,
                        f"items[{sample_index}].derived_images[{detail_index}]",
                    )
                )
                view_to_sample.append(sample_index)
                view_ids.append(str(metadata["view_id"]))
            if not observed_ids or observed_ids != list(reading_order):
                raise ValueError(
                    f"items[{sample_index}] derived_images must follow reading_order"
                )
            expected_bucket = quota_bucket(sample)
            if item.get("quota_bucket") != expected_bucket:
                raise ValueError(f"items[{sample_index}] quota_bucket differs from sample")
            quota_buckets.append(expected_bucket)

        if len(set(dataset_contracts)) != 1:
            raise ValueError("one OCR batch cannot mix dataset contracts")
        text_batch = _pad_rows(
            rows,
            pad_id=self.pad_id,
            ignore_index=self.ignore_index,
        )
        global_images = self.global_processor(global_specs)
        global_pixel_values = dict(
            collate_omvt_batch(global_images, self.omvt_cfg)
        )
        native_samples = self.native_processor(detail_specs)
        native_packed = pack_native_omvt_batch(
            native_samples,
            self.omvt_cfg,
            normalized_white=self.native_processor.normalized_white,
            max_raw_patch_tokens_per_sample=(
                self.max_raw_patch_tokens_per_view
            ),
        )
        return {
            **text_batch,
            "position_contract": BOUNDARY_V1,
            "dataset_contract_sha256": dataset_contracts[0],
            "global_pixel_values": global_pixel_values,
            "native_packed": native_packed,
            "view_to_sample": torch.tensor(view_to_sample, dtype=torch.long),
            "view_ids": view_ids,
            "sample_metadata": [copy.deepcopy(dict(sample)) for sample in samples],
            "quota_buckets": quota_buckets,
        }


def merge_packed_detail_views(
    detail_memory: torch.Tensor,
    detail_cu: torch.Tensor,
    view_to_sample: torch.Tensor,
    B: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable-group per-view packed memory into isolated logical samples."""

    if detail_memory.ndim < 2:
        raise ValueError("detail_memory must have shape [tokens, ...]")
    if detail_cu.ndim != 1 or view_to_sample.ndim != 1:
        raise ValueError("detail_cu and view_to_sample must be one-dimensional")
    if isinstance(B, bool) or not isinstance(B, int) or B <= 0:
        raise ValueError("B must be a positive integer")
    view_count = int(view_to_sample.numel())
    if detail_cu.numel() != view_count + 1:
        raise ValueError("detail_cu must contain one boundary per view plus one")
    boundaries = [int(value) for value in detail_cu.detach().cpu().tolist()]
    if not boundaries or boundaries[0] != 0:
        raise ValueError("detail_cu must start at zero")
    if any(right < left for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("detail_cu must be monotonic")
    if boundaries[-1] != detail_memory.shape[0]:
        raise ValueError("detail_cu terminal boundary differs from detail_memory")
    owners = [int(value) for value in view_to_sample.detach().cpu().tolist()]
    if any(owner < 0 or owner >= B for owner in owners):
        raise ValueError("view_to_sample contains an out-of-range sample id")

    segments: list[torch.Tensor] = []
    cumulative = [0]
    for sample_index in range(B):
        sample_tokens = 0
        for view_index, owner in enumerate(owners):
            if owner != sample_index:
                continue
            start, end = boundaries[view_index], boundaries[view_index + 1]
            segments.append(detail_memory[start:end])
            sample_tokens += end - start
        if sample_tokens == 0:
            raise ValueError(f"logical sample {sample_index} has no detail views")
        cumulative.append(cumulative[-1] + sample_tokens)
    merged = torch.cat(segments, dim=0)
    cu = torch.tensor(cumulative, dtype=torch.int32, device="cpu")
    return merged, cu


__all__ = [
    "ANYRES_GLOBAL_IMAGE_TOKENS",
    "AnyresOCRSFTCollator",
    "merge_packed_detail_views",
]
