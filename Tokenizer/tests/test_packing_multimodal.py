# -*- coding: utf-8 -*-

"""Regression: pack_samples must preserve EncodedSample multimodal fields.

Before PR #10 review fix the trim/pad/empty-pack helpers silently dropped
the new ``images / ocr_labels / reading_order`` payload, so a packed
multimodal sample would still contain ``<image_patch>`` token spans but
no media references downstream — leading to inject-time crashes that are
hard to attribute.
"""

from __future__ import annotations

import unittest

from Tokenizer.pretraining import EncodedSample, pack_samples
from Tokenizer.pretraining.packing import _empty_text_pack, _pad, _trim


def _make_mm_sample(n: int = 6) -> EncodedSample:
    return EncodedSample(
        input_ids=list(range(n)),
        attention_mask=[1] * n,
        labels=list(range(n)),
        token_offsets=[(i, i + 1) for i in range(n)],
        word_pos=[0] * n,
        morph_depth=[0] * n,
        modality_spans={
            "image_token_spans": [(1, 4)],
            "video_token_spans": [],
        },
        metadata={"type": "vlm"},
        images=["/tmp/a.png"],
        image_sizes=[[16, 16]],
        videos=[],
        video_sizes=[],
        ocr_labels=[[1, 2, 3]],
        reading_order=[[0, 1, 2]],
    )


class PackingMultimodalFieldsTest(unittest.TestCase):
    def test_pack_samples_preserves_multimodal_payload(self) -> None:
        sample = _make_mm_sample()
        packed = pack_samples([sample], max_length=32, pad_id=0, eos_id=99)
        self.assertEqual(len(packed), 1)
        self.assertEqual(packed[0].images, ["/tmp/a.png"])
        self.assertEqual(packed[0].ocr_labels, [[1, 2, 3]])
        self.assertEqual(packed[0].reading_order, [[0, 1, 2]])

    def test_pad_preserves_multimodal_payload(self) -> None:
        sample = _make_mm_sample()
        padded = _pad(sample, max_length=12, pad_id=0)
        self.assertEqual(padded.images, sample.images)
        self.assertEqual(padded.image_sizes, sample.image_sizes)
        self.assertEqual(padded.ocr_labels, sample.ocr_labels)
        self.assertEqual(padded.reading_order, sample.reading_order)

    def test_trim_drops_image_when_span_falls_off(self) -> None:
        # Image span lives at [1,4); trim cutoff lands before the span end
        # so the offending image must be dropped from the payload too,
        # keeping len(images) == len(image_token_spans).
        sample = _make_mm_sample(n=10)
        sample.modality_spans["image_token_spans"] = [(1, 4), (6, 9)]
        sample.images = ["/tmp/a.png", "/tmp/b.png"]
        sample.image_sizes = [[16, 16], [16, 16]]
        sample.ocr_labels = [[1], [2]]
        sample.reading_order = [[0], [0]]
        trimmed = _trim(sample, max_length=8)
        # Only the first span survives → only the first image payload
        self.assertEqual(len(trimmed.modality_spans["image_token_spans"]), 1)
        self.assertEqual(trimmed.images, ["/tmp/a.png"])
        self.assertEqual(trimmed.image_sizes, [[16, 16]])
        self.assertEqual(trimmed.ocr_labels, [[1]])
        self.assertEqual(trimmed.reading_order, [[0]])

    def test_empty_text_pack_default_fields(self) -> None:
        empty = _empty_text_pack()
        self.assertEqual(empty.images, [])
        self.assertEqual(empty.image_sizes, [])
        self.assertEqual(empty.ocr_labels, [])
        self.assertEqual(empty.reading_order, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
