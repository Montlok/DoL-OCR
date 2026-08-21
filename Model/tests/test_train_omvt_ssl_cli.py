# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts import train_omvt_ssl
from Tokenizer.multimodal import PILImageProcessor

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore


class TrainOMVTSSLCliTest(unittest.TestCase):
    def test_lr_decay_steps_defaults_to_stop_steps(self) -> None:
        args = train_omvt_ssl.parse_args(["--steps", "8"])
        self.assertIsNone(args.lr_decay_steps)

        cfg = train_omvt_ssl._omvt_cfg(args)
        train_cfg = train_omvt_ssl._training_cfg(args, cfg, use_bf16=False)
        self.assertEqual(train_cfg.max_steps, 8)
        self.assertEqual(train_cfg.lr_decay_steps, 8)

    def test_lr_decay_steps_can_exceed_stop_steps(self) -> None:
        args = train_omvt_ssl.parse_args(
            [
                "--steps", "8000",
                "--lr-decay-steps", "80000",
                "--warmup-steps", "2000",
                "--batch-size", "256",
                "--image-size", "224",
                "--d-vision", "512",
                "--compress-to", "256",
                "--ocr-vocab", "65536",
                "--patch-preset", "prod",
                "--lr", "3e-4",
            ]
        )

        cfg = train_omvt_ssl._omvt_cfg(args)
        train_cfg = train_omvt_ssl._training_cfg(args, cfg, use_bf16=True)

        self.assertEqual(train_cfg.max_steps, 8000)
        self.assertEqual(train_cfg.lr_decay_steps, 80000)
        self.assertEqual(train_cfg.warmup_steps, 2000)
        self.assertEqual(train_cfg.micro_batch_size, 256)
        self.assertEqual(train_cfg.learning_rate, 3e-4)
        self.assertEqual(train_cfg.seq_len, 256)
        self.assertEqual(train_cfg.precision, "bf16")
        self.assertEqual(cfg.image_size, 224)
        self.assertEqual(cfg.d_vision, 512)
        self.assertEqual(cfg.compress_to, 256)

    @unittest.skipIf(Image is None, "Pillow not installed")
    def test_pair_shards_stream_letterboxed_train_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shards = root / "shards"
            shards.mkdir()
            tar_path = shards / "shard-00000.tar"
            rows = {
                "line_00000000_v0": {"kind": "line", "text": "abc", "src_doc": 0},
                "line_val_v0": {"kind": "line", "text": "skip", "src_doc": 434600},
                "line_00000001_v0": {"kind": "line", "text": "de", "src_doc": 1},
            }
            with tarfile.open(tar_path, "w") as tf:
                for stem, row in rows.items():
                    img_buf = io.BytesIO()
                    Image.new("L", (16, 48), color=180).save(img_buf, format="PNG")
                    img_data = img_buf.getvalue()
                    img_info = tarfile.TarInfo(f"{stem}.png")
                    img_info.size = len(img_data)
                    tf.addfile(img_info, io.BytesIO(img_data))

                    meta_data = json.dumps(row).encode("utf-8")
                    meta_info = tarfile.TarInfo(f"{stem}.json")
                    meta_info.size = len(meta_data)
                    tf.addfile(meta_info, io.BytesIO(meta_data))

            it = train_omvt_ssl._iter_pair_shards(
                str(shards),
                [0],
                2,
                PILImageProcessor(image_size=32),
                lambda text: [ord(ch) for ch in text],
                seed=0,
                n_image_tokens=8,
                max_seq_len=32,
            )
            batch = next(it)

            self.assertEqual(tuple(batch["images"].shape), (2, 3, 32, 32))
            self.assertEqual(batch["ocr_labels"], [[97, 98, 99], [100, 101]])
            self.assertEqual(batch["reading_order"], [None, None])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
