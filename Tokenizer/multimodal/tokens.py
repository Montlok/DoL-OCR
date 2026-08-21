# -*- coding: utf-8 -*-
"""Multimodal placeholder and patch tokens."""

IMAGE_PLACEHOLDER = "<image>"
IMAGE_START = "<image_start>"
IMAGE_PATCH = "<image_patch>"
IMAGE_END = "<image_end>"
VIDEO_PLACEHOLDER = "<video>"
VIDEO_START = "<video_start>"
VIDEO_PATCH = "<video_patch>"
VIDEO_END = "<video_end>"
BBOX_TOKEN = "<bbox>"
OCR_TOKEN = "<ocr>"
OCR_START = "<ocr_start>"
OCR_END = "<ocr_end>"
DOC_TOKEN = "<doc>"
TABLE_TOKEN = "<table>"
LAYOUT_TOKEN = "<layout>"
AUDIO_PLACEHOLDER = "<audio>"
AUDIO_START = "<audio_start>"
AUDIO_PATCH = "<audio_patch>"
AUDIO_END = "<audio_end>"

MULTIMODAL_SPECIAL_TOKENS = {
    IMAGE_PLACEHOLDER: 5,
    IMAGE_START: 6,
    IMAGE_PATCH: 7,
    IMAGE_END: 8,
    VIDEO_PLACEHOLDER: 9,
    VIDEO_START: 10,
    VIDEO_PATCH: 11,
    VIDEO_END: 12,
    BBOX_TOKEN: 13,
    OCR_TOKEN: 14,
    OCR_START: 15,
    OCR_END: 16,
    DOC_TOKEN: 19,
    TABLE_TOKEN: 20,
    LAYOUT_TOKEN: 21,
    AUDIO_PLACEHOLDER: 22,
    AUDIO_START: 23,
    AUDIO_PATCH: 24,
    AUDIO_END: 25,
}


def expand_image_placeholders(text: str, patches_per_image: int) -> str:
    from .image_placeholders import expand_image_placeholders as expand

    return expand(text, patches_per_image)
