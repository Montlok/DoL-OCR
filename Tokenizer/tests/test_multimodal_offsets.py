# -*- coding: utf-8 -*-

import unittest

from Tokenizer.multimodal import (
    IMAGE_END,
    IMAGE_PATCH,
    IMAGE_PLACEHOLDER,
    IMAGE_START,
    VIDEO_END,
    VIDEO_PATCH,
    VIDEO_PLACEHOLDER,
    VIDEO_START,
    MultimodalProcessor,
)
from Tokenizer.tests.test_dual_tokenizer import build_fake_tokenizer


class MultimodalOffsetTest(unittest.TestCase):
    def test_single_image_tokens_use_original_placeholder_offset(self):
        processor = MultimodalProcessor(build_fake_tokenizer())
        text = "abc " + IMAGE_PLACEHOLDER + " def"
        start = text.index(IMAGE_PLACEHOLDER)
        end = start + len(IMAGE_PLACEHOLDER)
        out = processor(text, images=["img"], image_sizes=[(14, 14)])

        span_start, span_end = out.image_token_spans[0]
        tokens = out.tokens[span_start:span_end]
        self.assertEqual([t.token for t in tokens], [IMAGE_START, IMAGE_PATCH, IMAGE_END])
        for token in tokens:
            self.assertEqual((token.start, token.end), (start, end))
            self.assertEqual(token.metadata["source_span"], [start, end])
            self.assertEqual(token.metadata["image_index"], 0)
        for token in out.tokens:
            if token.start != -1:
                self.assertLessEqual(token.end, len(text))

    def test_multiple_images_keep_distinct_source_spans(self):
        processor = MultimodalProcessor(build_fake_tokenizer())
        text = IMAGE_PLACEHOLDER + " x " + IMAGE_PLACEHOLDER
        first = (0, len(IMAGE_PLACEHOLDER))
        second_start = text.rindex(IMAGE_PLACEHOLDER)
        second = (second_start, second_start + len(IMAGE_PLACEHOLDER))
        out = processor(
            text,
            images=["img1", "img2"],
            image_sizes=[(14, 14), (14, 14)],
        )

        first_tokens = out.tokens[out.image_token_spans[0][0]:out.image_token_spans[0][1]]
        second_tokens = out.tokens[out.image_token_spans[1][0]:out.image_token_spans[1][1]]
        self.assertTrue(all((t.start, t.end) == first for t in first_tokens))
        self.assertTrue(all((t.start, t.end) == second for t in second_tokens))
        self.assertEqual(first_tokens[0].metadata["image_index"], 0)
        self.assertEqual(second_tokens[0].metadata["image_index"], 1)
        self.assertEqual(first_tokens[0].metadata["source_span"], list(first))
        self.assertEqual(second_tokens[0].metadata["source_span"], list(second))

    def test_video_tokens_use_original_placeholder_offset(self):
        processor = MultimodalProcessor(build_fake_tokenizer())
        text = "a " + VIDEO_PLACEHOLDER + " z"
        start = text.index(VIDEO_PLACEHOLDER)
        end = start + len(VIDEO_PLACEHOLDER)
        out = processor(text, videos=["vid"], video_sizes=[(2, 14, 14)])

        span_start, span_end = out.video_token_spans[0]
        tokens = out.tokens[span_start:span_end]
        self.assertEqual([t.token for t in tokens], [VIDEO_START, VIDEO_PATCH, VIDEO_END])
        for token in tokens:
            self.assertEqual((token.start, token.end), (start, end))
            self.assertEqual(token.metadata["source_span"], [start, end])
            self.assertEqual(token.metadata["video_index"], 0)

    def test_image_token_spans_are_half_open_and_count_checked(self):
        processor = MultimodalProcessor(build_fake_tokenizer())
        out = processor(IMAGE_PLACEHOLDER, images=["img"], image_sizes=[(29, 29)])
        start, end = out.image_token_spans[0]
        span_tokens = out.tokens[start:end]
        self.assertEqual(span_tokens[0].token, IMAGE_START)
        self.assertEqual(span_tokens[-1].token, IMAGE_END)
        self.assertEqual(sum(1 for t in span_tokens if t.token == IMAGE_PATCH), 4)
        self.assertEqual(end - start, 6)
        self.assertEqual(out.input_ids[start:end], [token.id for token in span_tokens])


if __name__ == "__main__":
    unittest.main()
