# -*- coding: utf-8 -*-

from __future__ import annotations

import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from PIL import Image

from Model.ocr.streaming_corpus import (
    MixedOCRCorpus,
    _locally_staged_wds_shard,
    corpus_manifest,
)
from Model.omvt import OMVTVisionTower
from scripts.train_ctc_head import (
    BLANK_ID,
    NUM_CLASSES,
    CTCHead,
    bytes_to_text,
    greedy_ctc_decode,
    load_ctc_payload,
    main as train_ctc_main,
    save_checkpoint,
)


def _png_bytes(value: int = 180) -> bytes:
    out = io.BytesIO()
    Image.new("L", (12, 28), color=value).save(out, format="PNG")
    return out.getvalue()


def _write_wds(path: Path, n: int) -> None:
    with tarfile.open(path, "w") as tf:
        for index in range(n):
            key = f"line_{index:08d}_v0"
            image = _png_bytes(120 + index)
            image_info = tarfile.TarInfo(f"{key}.png")
            image_info.size = len(image)
            tf.addfile(image_info, io.BytesIO(image))
            row = json.dumps(
                {
                    "kind": "line",
                    "text": f"w{index}",
                    "src_doc": index,
                    "font": "OnonSoninSans" if index % 2 == 0 else "NotoSansMongolian",
                }
            ).encode("utf-8")
            row_info = tarfile.TarInfo(f"{key}.json")
            row_info.size = len(row)
            tf.addfile(row_info, io.BytesIO(row))


class StreamingCorpusTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[list[Path], Path, Path]:
        shards = root / "shards"
        shards.mkdir()
        shard = shards / "shard-00000.tar"
        _write_wds(shard, 4)

        hanshi = root / "hanshi"
        pages = hanshi / "pages"
        bucket = pages / "00000"
        bucket.mkdir(parents=True)
        meta = hanshi / "meta.jsonl"
        with meta.open("w", encoding="utf-8") as fh:
            for index in range(2):
                doc_id = f"h{index}"
                (bucket / f"{doc_id}.png").write_bytes(_png_bytes(200 + index))
                fh.write(
                    json.dumps(
                        {
                            "doc_id": doc_id,
                            "kind": "line",
                            "text": f"h{index}",
                            "src_doc": index,
                            "bucket": "00000",
                            "font": "hanshi",
                        }
                    )
                    + "\n"
                )
        return [shard], meta, pages

    def test_three_font_mix_and_exact_cursor_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths, meta, pages = self._fixture(Path(tmp))
            corpus = MixedOCRCorpus(
                paths,
                hanshi_meta=meta,
                hanshi_pages=pages,
                image_size=32,
                seed=7,
                max_target_len=32,
            )
            batches = corpus.batches(3)
            first = next(batches)
            self.assertEqual(first["sources"], ["wds", "wds", "hanshi"])
            self.assertEqual(tuple(first["pixels"].shape), (3, 3, 32, 32))
            cursor = first["corpus_cursor"]
            expected = next(batches)

            resumed = MixedOCRCorpus(
                paths,
                hanshi_meta=meta,
                hanshi_pages=pages,
                image_size=32,
                seed=7,
                max_target_len=32,
                cursor=cursor,
            )
            actual = next(resumed.batches(3))
            self.assertEqual(actual["keys"], expected["keys"])
            self.assertTrue(torch.equal(actual["targets"], expected["targets"]))
            self.assertEqual(actual["corpus_cursor"], expected["corpus_cursor"])

    def test_manifest_hash_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths, meta, pages = self._fixture(Path(tmp))
            first, first_sha = corpus_manifest(
                paths,
                hanshi_meta=meta,
                hanshi_pages=pages,
                excluded_shard_ids=[2303],
                val_src_doc_min=434600,
                test_src_doc_min=435200,
                seed=42,
            )
            second, second_sha = corpus_manifest(
                paths,
                hanshi_meta=meta,
                hanshi_pages=pages,
                excluded_shard_ids=[2303],
                val_src_doc_min=434600,
                test_src_doc_min=435200,
                seed=42,
            )
            self.assertEqual(first, second)
            self.assertEqual(first_sha, second_sha)

    def test_wds_shard_is_staged_and_cleaned_up(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            tempfile.TemporaryDirectory() as cache,
        ):
            source = Path(tmp) / "shard-00000.tar"
            _write_wds(source, 1)
            with mock.patch.dict(
                "os.environ", {"DOL_OCR_WDS_CACHE_DIR": cache}, clear=False
            ):
                with _locally_staged_wds_shard(source) as staged:
                    self.assertNotEqual(staged, source)
                    self.assertEqual(staged.parent.parent, Path(cache))
                    self.assertTrue(staged.is_file())
                    self.assertEqual(staged.read_bytes(), source.read_bytes())
            self.assertFalse(staged.exists())

    def test_wds_staging_fails_before_copy_when_scratch_is_too_small(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "shard-00000.tar"
            _write_wds(source, 1)
            with mock.patch(
                "Model.ocr.streaming_corpus.shutil.disk_usage",
                return_value=SimpleNamespace(free=0),
            ):
                with self.assertRaisesRegex(OSError, "insufficient local scratch"):
                    with _locally_staged_wds_shard(source):
                        self.fail("staging unexpectedly entered the context")

    def test_streaming_cli_saves_resume_contract(self) -> None:
        from Model.tests.test_ctc_head import _tiny_omvt_config

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, meta, pages = self._fixture(root)
            cfg = _tiny_omvt_config()
            tower = OMVTVisionTower(cfg)
            head = CTCHead(d_vision=cfg.d_vision, hidden=8)
            optimizer = torch.optim.AdamW(
                list(tower.parameters()) + list(head.parameters()), lr=1e-3
            )
            init_dir = root / "init"
            save_checkpoint(
                init_dir,
                0,
                head,
                optimizer,
                cfg,
                d_vision=cfg.d_vision,
                hidden=8,
                tower=tower,
            )
            output = root / "run"
            rc = train_ctc_main(
                [
                    "--init-ctc-checkpoint",
                    str(init_dir),
                    "--stream-wds-dir",
                    str(paths[0].parent),
                    "--stream-hanshi-meta",
                    str(meta),
                    "--stream-hanshi-pages",
                    str(pages),
                    "--output",
                    str(output),
                    "--unfreeze-tower",
                    "--steps",
                    "2",
                    "--batch-size",
                    "3",
                    "--warmup-steps",
                    "1",
                    "--save-every",
                    "0",
                    "--probe-every",
                    "0",
                    "--log-every",
                    "1",
                    "--device",
                    "cpu",
                    "--precision",
                    "fp32",
                    "--tokenizer-fingerprint",
                    "fixture-tokenizer-sha",
                ]
            )
            self.assertEqual(rc, 0)
            payload = load_ctc_payload(output)
            self.assertEqual(payload["step"], 2)
            self.assertIn("scheduler", payload)
            self.assertEqual(payload["corpus_cursor"]["counts"]["total"], 6)
            self.assertTrue(payload["run_metadata"]["corpus_manifest_sha256"])
            self.assertEqual(
                payload["run_metadata"]["tokenizer_fingerprint"],
                "fixture-tokenizer-sha",
            )


class CTCMongolianRoundTripTest(unittest.TestCase):
    def test_fvs_mvs_nnbsp_roundtrip(self) -> None:
        text = "ᠠ\u180b\u180e\u202fᠪ"
        byte_ids = list(text.encode("utf-8"))
        frames: list[int] = []
        for token in byte_ids:
            frames.extend((token, BLANK_ID))
        logits = torch.full((1, len(frames), NUM_CLASSES), -100.0)
        for index, token in enumerate(frames):
            logits[0, index, token] = 100.0
        decoded = greedy_ctc_decode(logits)[0]
        self.assertEqual(bytes_to_text(decoded), text)


if __name__ == "__main__":
    unittest.main()
