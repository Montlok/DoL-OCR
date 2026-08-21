# -*- coding: utf-8 -*-

from __future__ import annotations

import io
import unittest

import torch
from PIL import Image

from Model.config import EOS_ID, IMAGE_PATCH_ID, OMVTConfig
from Model.posttrain.ocr_anyres_collator import (
    AnyresOCRSFTCollator,
    merge_packed_detail_views,
)
from Tokenizer.multimodal import NativeImageProcessorV2, PILImageProcessor


class _NativeEncoder:
    mode = "native"

    def __call__(self, text: str) -> list[int]:
        return [300 + index for index, _character in enumerate(text)]


def _png(width: int, height: int, color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _cfg() -> OMVTConfig:
    return OMVTConfig(
        image_size=16,
        vertical_patch=(8, 4),
        horizontal_patch=(4, 8),
        square_patch=(4, 4),
        layout_patch=(16, 16),
        d_vision=16,
        vision_n_heads=4,
        vision_ffn_hidden=32,
        compress_to=256,
    )


def _payload(data: bytes, view_id: str | None = None) -> dict:
    metadata = {"view_id": view_id} if view_id is not None else {"asset_id": "asset"}
    return {"delivery": "bytes", "bytes": data, "metadata": metadata}


def _item(
    sample_id: str,
    text: str,
    view_ids: list[str],
    *,
    style: str = "print",
    difficulty: str | None = None,
) -> dict:
    details = [
        _payload(_png(5 + index, 9 - index, (20 + index, 30, 40)), view_id)
        for index, view_id in enumerate(view_ids)
    ]
    writer = "writer-1" if style == "handwritten" else None
    sample = {
        "schema_version": 2,
        "sample_id": sample_id,
        "split": "train",
        "visual_contract": "dol_ocr_anyres_v2",
        "asset_id": f"asset-{sample_id}",
        "view_ids": list(view_ids),
        "preprocess_contract_sha256": "a" * 64,
        "reference_token_count": len(text),
        "writing_mode": "vertical-lr",
        "reading_order": list(view_ids),
        "leakage_cluster": f"leak-{sample_id}",
        "group_ids": {
            "document_id": f"doc-{sample_id}",
            "capture_id": f"capture-{sample_id}",
            "writer_id": writer,
            "near_duplicate_cluster": f"near-{sample_id}",
        },
        "style": style,
        "font_id": "Noto" if style == "print" else None,
        "writer_id": writer,
        "difficulty": difficulty,
        "reference_raw": {"text": text},
        "reference_model": {"text": text},
        "qa_state": "accepted",
    }
    quota = "print" if style == "print" else f"handwritten_{difficulty}"
    return {
        "dataset_contract_sha256": "d" * 64,
        "sample": sample,
        "asset": {"asset_id": sample["asset_id"]},
        "canonical_image": _payload(_png(4, 8, (0, 0, 0))),
        "derived_images": details,
        "quota_bucket": quota,
    }


def _collator(*, budget: int = 200) -> AnyresOCRSFTCollator:
    return AnyresOCRSFTCollator(
        encode_reference=_NativeEncoder(),
        omvt_cfg=_cfg(),
        global_processor=PILImageProcessor(
            image_size=16, in_channels=3, mean=None, std=None
        ),
        native_processor=NativeImageProcessorV2(
            in_channels=3, mean=None, std=None, max_decode_pixels=10_000
        ),
        max_raw_patch_tokens_per_view=budget,
    )


class AnyresOCRSFTCollatorTest(unittest.TestCase):
    def test_joint_batch_has_global_detail_text_and_boundary_contract(self):
        first = _item("one", "abc", ["one-b", "one-a"])
        second = _item(
            "two", "de", ["two-a"], style="handwritten", difficulty="good"
        )

        batch = _collator()([first, second])

        self.assertEqual(tuple(batch["input_ids"].shape), (2, 263))
        self.assertEqual(batch["position_contract"], "boundary_v1")
        self.assertEqual(batch["dataset_contract_sha256"], "d" * 64)
        self.assertNotIn("word_pos", batch)
        self.assertNotIn("morph_depth", batch)
        self.assertEqual(
            (batch["input_ids"] == IMAGE_PATCH_ID).sum(dim=1).tolist(),
            [256, 256],
        )
        supervised = (batch["labels"] != -100).sum(dim=1).tolist()
        self.assertEqual(supervised, [4, 3])
        self.assertEqual(int(batch["labels"][0, 262]), EOS_ID)
        self.assertEqual(batch["view_ids"], ["one-b", "one-a", "two-a"])
        self.assertEqual(batch["view_to_sample"].tolist(), [0, 0, 1])
        self.assertEqual(batch["native_packed"].original_hw.tolist(), [[9, 5], [8, 6], [9, 5]])
        self.assertEqual(batch["quota_buckets"], ["print", "handwritten_good"])

        global_images = batch["global_pixel_values"]["images"]
        self.assertEqual(tuple(global_images.shape), (2, 3, 16, 16))
        self.assertTrue(bool((global_images[:, :, 0, 0] > 0.9).all()))
        self.assertTrue(bool((global_images[:, :, 8, 8] < 0.1).all()))

    def test_reference_token_count_and_eos_convention_fail_closed(self):
        mismatch = _item("one", "abc", ["view"])
        mismatch["sample"]["reference_token_count"] = 4
        with self.assertRaisesRegex(ValueError, "count excludes the one appended EOS"):
            _collator()([mismatch])

        class _EncoderWithEOS(_NativeEncoder):
            def __call__(self, _text: str) -> list[int]:
                return [300, EOS_ID]

        with self.assertRaisesRegex(ValueError, "must not contain EOS"):
            AnyresOCRSFTCollator(
                encode_reference=_EncoderWithEOS(),
                omvt_cfg=_cfg(),
                global_processor=PILImageProcessor(16),
                native_processor=NativeImageProcessorV2(),
                max_raw_patch_tokens_per_view=200,
            )([_item("one", "ab", ["view"])])

    def test_non_native_encoder_and_non_256_global_contract_are_rejected(self):
        encoder = _NativeEncoder()
        encoder.mode = "byte_fallback"
        with self.assertRaisesRegex(ValueError, "native OCR encoder"):
            AnyresOCRSFTCollator(
                encode_reference=encoder,
                omvt_cfg=_cfg(),
                global_processor=PILImageProcessor(16),
                native_processor=NativeImageProcessorV2(),
                max_raw_patch_tokens_per_view=200,
            )

        cfg = _cfg()
        cfg.compress_to = 255
        with self.assertRaisesRegex(ValueError, "exactly 256"):
            AnyresOCRSFTCollator(
                encode_reference=_NativeEncoder(),
                omvt_cfg=cfg,
                global_processor=PILImageProcessor(16),
                native_processor=NativeImageProcessorV2(),
                max_raw_patch_tokens_per_view=200,
            )

    def test_detail_order_and_raw_patch_budget_fail_closed(self):
        item = _item("one", "a", ["view-a", "view-b"])
        item["derived_images"].reverse()
        with self.assertRaisesRegex(ValueError, "must follow reading_order"):
            _collator()([item])

        with self.assertRaisesRegex(ValueError, "streamed macro-window"):
            _collator(budget=1)([_item("one", "a", ["view-a"])])

        first = _item("one", "a", ["view-a"])
        second = _item("two", "b", ["view-b"])
        second["dataset_contract_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "mix dataset contracts"):
            _collator()([first, second])

    def test_padding_masks_shorter_target_without_position_tensors(self):
        batch = _collator()(
            [_item("one", "abcd", ["v1"]), _item("two", "x", ["v2"])]
        )
        self.assertEqual(batch["attention_mask"][1, -5:].tolist(), [1, 1, 0, 0, 0])
        self.assertEqual(batch["labels"][1, -1].item(), -100)
        self.assertNotIn("word_pos", batch)

    def test_merge_packed_detail_views_is_stable_and_sample_isolated(self):
        memory = torch.tensor(
            [[10.0], [11.0], [20.0], [30.0], [31.0]], requires_grad=True
        )
        cu = torch.tensor([0, 2, 3, 5], dtype=torch.int32)
        owners = torch.tensor([1, 0, 1], dtype=torch.long)

        merged, merged_cu = merge_packed_detail_views(memory, cu, owners, 2)

        self.assertEqual(merged[:, 0].tolist(), [20.0, 10.0, 11.0, 30.0, 31.0])
        self.assertEqual(merged_cu.tolist(), [0, 1, 5])
        merged.sum().backward()
        self.assertEqual(memory.grad[:, 0].tolist(), [1.0] * 5)

        with self.assertRaisesRegex(ValueError, "has no detail views"):
            merge_packed_detail_views(memory.detach(), cu, owners, 3)


if __name__ == "__main__":
    unittest.main()
