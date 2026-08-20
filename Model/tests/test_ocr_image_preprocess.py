# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from Model.config import OMVTConfig
from Model.ocr.image_preprocess import letterbox_grayscale_to_square
from Model.ocr.streaming_corpus import MixedOCRCorpus
from Model.posttrain.ocr_eval import build_ocr_pixel_batch
from Tokenizer.multimodal import PILImageProcessor
from scripts.build_ocr_data_from_pairs import letterbox_to_square
from scripts.ocr_infer import strip_to_letterboxed


def _pattern_png(width: int = 11, height: int = 173) -> bytes:
    image = Image.new("L", (width, height), 255)
    for y in range(7, height - 5):
        image.putpixel((width // 2, y), (y * 13) % 180)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _write_pair_tar(path: Path, image_bytes: bytes) -> None:
    metadata = json.dumps(
        {
            "kind": "line",
            "text": "ᠠ",
            "src_doc": 1,
            "font": "OnonSoninSans",
        }
    ).encode("utf-8")
    with tarfile.open(path, "w") as archive:
        for name, payload in (("line.png", image_bytes), ("line.json", metadata)):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


class OCRImagePreprocessTests(unittest.TestCase):
    def test_builder_streaming_rl_batch_and_infer_are_pixel_identical(self) -> None:
        image_bytes = _pattern_png()
        processor = PILImageProcessor(image_size=224, in_channels=3)
        expected_image = letterbox_to_square(image_bytes, 224)
        expected = processor([expected_image])[0]

        direct = processor(
            [letterbox_grayscale_to_square(image_bytes, 224)]
        )[0]
        with Image.open(io.BytesIO(image_bytes)) as raw:
            inferred_image = strip_to_letterboxed(np.asarray(raw.convert("L")), 224)
        inferred = processor([inferred_image])[0]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "raw.png"
            image_path.write_bytes(image_bytes)
            digest = hashlib.sha256(image_bytes).hexdigest()
            config = OMVTConfig(image_size=224)
            evaluated = build_ocr_pixel_batch(
                [{"id": "sample", "image": image_path, "sha256": digest}],
                processor,
                config,
                torch.device("cpu"),
            )["images"][0]

            shard = root / "shard-00000.tar"
            _write_pair_tar(shard, image_bytes)
            hanshi_meta = root / "hanshi.jsonl"
            hanshi_meta.write_text("", encoding="utf-8")
            hanshi_pages = root / "pages"
            hanshi_pages.mkdir()
            streamed, _, _ = next(
                MixedOCRCorpus(
                    [shard],
                    hanshi_meta=hanshi_meta,
                    hanshi_pages=hanshi_pages,
                    image_size=224,
                    seed=0,
                    max_target_len=32,
                )
            )

        self.assertEqual(expected.dtype, torch.float32)
        for actual in (direct, inferred, evaluated, streamed):
            self.assertEqual(actual.dtype, torch.float32)
            self.assertTrue(torch.equal(actual, expected))

    def test_extreme_aspect_ratio_is_not_stretched(self) -> None:
        image = Image.new("L", (2, 400), 0)
        output = letterbox_grayscale_to_square(image, 224)
        dark = np.argwhere(np.asarray(output) < 128)
        height = int(dark[:, 0].max() - dark[:, 0].min() + 1)
        width = int(dark[:, 1].max() - dark[:, 1].min() + 1)
        self.assertGreaterEqual(height, 220)
        self.assertLessEqual(width, 4)

    def test_exif_orientation_is_applied_before_letterbox(self) -> None:
        image = Image.new("L", (80, 20), 0)
        exif = Image.Exif()
        exif[274] = 6
        encoded = io.BytesIO()
        image.save(encoded, format="JPEG", quality=100, exif=exif)

        output = letterbox_grayscale_to_square(encoded.getvalue(), 224)
        dark = np.argwhere(np.asarray(output) < 128)
        height = int(dark[:, 0].max() - dark[:, 0].min() + 1)
        width = int(dark[:, 1].max() - dark[:, 1].min() + 1)
        self.assertGreater(height, width * 3)


if __name__ == "__main__":
    unittest.main()
