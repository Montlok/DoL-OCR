# -*- coding: utf-8 -*-

import unittest

from Tokenizer.multimodal import (
    IMAGE_END,
    IMAGE_PATCH,
    IMAGE_PLACEHOLDER,
    IMAGE_START,
    MultimodalProcessor,
    expand_image_placeholders,
    image_patch_count,
)
from Tokenizer.tests.test_dual_tokenizer import build_fake_tokenizer


class MultimodalProcessorTest(unittest.TestCase):
    def test_single_image_placeholder_expansion(self):
        expanded = expand_image_placeholders(IMAGE_PLACEHOLDER, 2)
        self.assertEqual(expanded, IMAGE_START + IMAGE_PATCH + IMAGE_PATCH + IMAGE_END)

    def test_multi_image_placeholder_expansion(self):
        processor = MultimodalProcessor(build_fake_tokenizer())
        out = processor(
            IMAGE_PLACEHOLDER + " test " + IMAGE_PLACEHOLDER,
            images=["img1", "img2"],
            image_sizes=[(14, 14), (28, 28)],
        )
        self.assertEqual(len(out.image_token_spans), 2)
        first_start, first_end = out.image_token_spans[0]
        second_start, second_end = out.image_token_spans[1]
        self.assertEqual(out.tokens[first_start].token, IMAGE_START)
        self.assertEqual(out.tokens[first_end - 1].token, IMAGE_END)
        self.assertEqual(out.tokens[second_start].token, IMAGE_START)
        self.assertEqual(out.tokens[second_end - 1].token, IMAGE_END)

    def test_image_token_spans_are_input_id_ranges(self):
        processor = MultimodalProcessor(build_fake_tokenizer())
        out = processor(IMAGE_PLACEHOLDER, images=["img"], image_sizes=[(28, 28)])
        start, end = out.image_token_spans[0]
        span_tokens = out.tokens[start:end]
        self.assertEqual([tok.token for tok in span_tokens], [IMAGE_START, IMAGE_PATCH, IMAGE_END])
        self.assertEqual(out.input_ids[start:end], [tok.id for tok in span_tokens])

    def test_patch_count_stable(self):
        self.assertEqual(image_patch_count(28, 28, patch_size=14, merge_size=2), 1)
        self.assertEqual(image_patch_count(29, 29, patch_size=14, merge_size=2), 4)

    def test_image_size_mismatch_raises(self):
        processor = MultimodalProcessor(build_fake_tokenizer())
        with self.assertRaises(ValueError):
            processor(IMAGE_PLACEHOLDER + " " + IMAGE_PLACEHOLDER, image_sizes=[(28, 28)])

    def test_image_count_mismatch_raises(self):
        processor = MultimodalProcessor(build_fake_tokenizer())
        with self.assertRaises(ValueError):
            processor(IMAGE_PLACEHOLDER, images=[])
        with self.assertRaises(ValueError):
            processor(IMAGE_PLACEHOLDER, images=["a", "b"])
        with self.assertRaises(ValueError):
            processor("test", images=["orphan"])

    def test_attention_mask_matches_input_ids(self):
        processor = MultimodalProcessor(build_fake_tokenizer())
        out = processor("hello " + IMAGE_PLACEHOLDER, images=["img"], image_sizes=[(28, 28)])
        self.assertEqual(len(out.attention_mask), len(out.input_ids))
        self.assertTrue(all(m == 1 for m in out.attention_mask))

    def test_image_index_metadata(self):
        processor = MultimodalProcessor(build_fake_tokenizer())
        out = processor(
            IMAGE_PLACEHOLDER + " test " + IMAGE_PLACEHOLDER,
            images=["img1", "img2"],
            image_sizes=[(14, 14), (14, 14)],
        )
        # Collect image_index values from start markers
        starts = [t for t in out.tokens if t.token == IMAGE_START]
        self.assertEqual(len(starts), 2)
        self.assertEqual(starts[0].metadata["image_index"], 0)
        self.assertEqual(starts[1].metadata["image_index"], 1)

    def test_video_placeholder_expansion(self):
        from Tokenizer.multimodal import VIDEO_END, VIDEO_PATCH, VIDEO_PLACEHOLDER, VIDEO_START

        processor = MultimodalProcessor(build_fake_tokenizer())
        out = processor(VIDEO_PLACEHOLDER, videos=["vid"], video_sizes=[(2, 28, 28)])
        spans = out.video_token_spans
        self.assertEqual(len(spans), 1)
        start, end = spans[0]
        seq = [t.token for t in out.tokens[start:end]]
        self.assertEqual(seq[0], VIDEO_START)
        self.assertEqual(seq[-1], VIDEO_END)
        self.assertTrue(all(t == VIDEO_PATCH for t in seq[1:-1]))

    def test_video_count_mismatch_raises(self):
        from Tokenizer.multimodal import VIDEO_PLACEHOLDER

        processor = MultimodalProcessor(build_fake_tokenizer())
        with self.assertRaises(ValueError):
            processor(VIDEO_PLACEHOLDER, videos=[])
        with self.assertRaises(ValueError):
            processor(VIDEO_PLACEHOLDER, videos=["a", "b"])
        with self.assertRaises(ValueError):
            processor("test", videos=["orphan"])

    def test_bbox_normalize_roundtrip(self):
        from Tokenizer.multimodal import decode_bbox_tokens, encode_bbox_tokens, normalize_bbox

        bbox = (10.0, 20.0, 100.0, 200.0)
        tok = encode_bbox_tokens(bbox, width=200, height=400)
        self.assertTrue(tok.startswith("<bbox_") and tok.endswith(">"))
        coords = normalize_bbox(bbox, 200, 400)
        self.assertEqual(len(coords), 4)
        # Decode is approximate (quantization), but ordering preserved
        decoded = decode_bbox_tokens(tok, width=200, height=400)
        self.assertLess(abs(decoded[0] - 10.0), 1.0)
        self.assertLess(abs(decoded[3] - 200.0), 1.0)


if __name__ == "__main__":
    unittest.main()
