# -*- coding: utf-8 -*-

"""Tests for scripts.build_ocr_data_from_pairs (CPU, no real tokenizer bundle).

Builds a synthetic WebDataset-style mini-tar (5 samples spanning the
train/val/test bands, one non-"line" kind, one over-length text) and drives
the builder's factored core (:func:`process_shard` / :func:`build_one_shard`)
directly with the in-memory tokenizer fixture from
``test_ocr_target_encoding.py`` — the same pattern that test uses to avoid
depending on a real :class:`TokenizerBundle` on disk. The CLI
(``main``/argument parsing/``TokenizerBundle.from_dir``/multiprocessing) is
intentionally left thin and untested here; it is exercised (against the same
fixture, bypassing ``from_dir``) by the harness's own dry run, documented in
the delivery report rather than as a unittest.
"""

from __future__ import annotations

import io
import json
import os
import unittest
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw

from Model.config import IGNORE_INDEX, IMAGE_PATCH_ID
from Tokenizer.generic_bpe import GeneralBPEModel
from Tokenizer.unified.dual_tokenizer import DualTrackTokenizer, build_unified_vocab

from scripts.build_ocr_data import make_ocr_target_encoder
from scripts.build_ocr_data_from_pairs import (
    ShardCounters,
    build_one_shard,
    iter_tar_pairs,
    letterbox_to_square,
    parse_shard_indices,
    process_shard,
    route_band,
)

# Seeded single-char Mongolian letters, same fixture shape as
# test_ocr_target_encoding.py's (kept small and self-contained here rather
# than imported, since that module's fixture is private to its own file).
LETTER_A = "ᠠ"  # MONGOLIAN LETTER A
LETTER_NA = "ᠨ"  # MONGOLIAN LETTER NA
LETTER_GA = "ᠭ"  # MONGOLIAN LETTER GA
LETTER_RA = "ᠲ"  # MONGOLIAN LETTER TA
SEEDED_LETTERS = (LETTER_A, LETTER_NA, LETTER_GA, LETTER_RA)

VAL_MIN = 434600
TEST_MIN = 435200

N_IMAGE_TOKENS = 256
IMAGE_SIZE = 224
MAX_SEQ_LEN = 512


class _FakeMorphBPE:
    """Same shape as the fixture in test_ocr_target_encoding.py / test_stream_decode.py."""

    vocab: dict[str, int] = {"<unk>": 0}
    vocab.update({f"<0x{i:02X}>": i + 1 for i in range(256)})
    for _i, _ch in enumerate(SEEDED_LETTERS):
        vocab[_ch] = 300 + _i
    del _i, _ch

    def encode(self, text: str) -> list[int]:
        return [self.vocab[text]]


def make_fixture_tokenizer() -> DualTrackTokenizer:
    """Build an in-memory DualTrackTokenizer, mirroring test_ocr_target_encoding.py.

    ``make_ocr_target_encoder`` (and therefore the builder) only needs an
    object shaped like ``DualTrackTokenizer`` (``.vocab``,
    ``.general_global_to_local``, ``.unk_id``, ``.decode``) — not a full
    on-disk :class:`TokenizerBundle` — so the CLI's ``TokenizerBundle`` layer
    is not required for these tests.
    """

    general = GeneralBPEModel.minimal()
    vocab = build_unified_vocab(
        morphbpe_vocab=dict(_FakeMorphBPE.vocab), general_vocab=general.get_vocab()
    )
    return DualTrackTokenizer(vocab, _FakeMorphBPE(), general)


def _make_strip_png_bytes(width: int = 64, height: int = 600) -> bytes:
    """A small L-mode vertical strip with a few black rectangles (not blank)."""

    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)
    draw.rectangle([8, 10, 28, 40], fill=0)
    draw.rectangle([12, 120, 40, 170], fill=0)
    draw.rectangle([5, 300, 35, 340], fill=0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _write_pair(
    tf: tarfile.TarFile, key: str, meta: dict, *, order: str = "png_first"
) -> None:
    """Append a <key>.png + <key>.json pair to an open tar, in either order."""

    png_bytes = _make_strip_png_bytes()
    meta_bytes = json.dumps(meta, ensure_ascii=False).encode("utf-8")

    def _add(name: str, data: bytes) -> None:
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))

    if order == "png_first":
        _add(f"{key}.png", png_bytes)
        _add(f"{key}.json", meta_bytes)
    else:
        _add(f"{key}.json", meta_bytes)
        _add(f"{key}.png", png_bytes)


# The five canonical samples the task spec calls for: one clean train row,
# one clean val row, one test-band row (must be skipped whole), one
# kind != "line" row, and one over-length-text row. The over-length text
# must be long enough that its byte-fallback target alone busts
# MAX_SEQ_LEN=512 (target_ids are ~1 id/byte via <0xNN> fallback for text
# outside the seeded 4-letter alphabet, so plain ASCII padding is enough).
SAMPLES = [
    {
        "key": "000001",
        "meta": {
            "kind": "line",
            "text": LETTER_A + LETTER_NA,
            "src_doc": 100,
            "bucket": "b-train",
            "font": "Onon",
            "font_px": 28,
        },
    },
    {
        "key": "000002",
        "meta": {
            "kind": "line",
            "text": LETTER_GA + LETTER_RA,
            "src_doc": VAL_MIN,
            "bucket": "b-val",
            "font": "Noto",
            "font_px": 30,
        },
    },
    {
        "key": "000003",
        "meta": {
            "kind": "line",
            "text": LETTER_A,
            "src_doc": TEST_MIN,
            "bucket": "b-test",
            "font": "Hanshi",
            "font_px": 24,
        },
    },
    {
        "key": "000004",
        "meta": {
            "kind": "page",  # not "line" -> must be skipped, counted
            "text": LETTER_NA,
            "src_doc": 200,
            "bucket": "b-nonline",
            "font": "Onon",
            "font_px": 28,
        },
    },
    {
        "key": "000005",
        "meta": {
            "kind": "line",
            "text": "x" * 600,  # busts MAX_SEQ_LEN via <0xNN> byte fallback
            "src_doc": 300,
            "bucket": "b-overlong",
            "font": "Onon",
            "font_px": 28,
        },
    },
]


def _write_mini_tar(path: Path) -> None:
    with tarfile.open(path, "w") as tf:
        for i, sample in enumerate(SAMPLES):
            order = "png_first" if i % 2 == 0 else "json_first"
            _write_pair(tf, sample["key"], sample["meta"], order=order)


class RouteBandTest(unittest.TestCase):
    def test_below_val_min_is_train(self):
        self.assertEqual(route_band(VAL_MIN - 1, VAL_MIN, TEST_MIN), "train")

    def test_val_min_boundary_is_val(self):
        self.assertEqual(route_band(VAL_MIN, VAL_MIN, TEST_MIN), "val")

    def test_below_test_min_is_val(self):
        self.assertEqual(route_band(TEST_MIN - 1, VAL_MIN, TEST_MIN), "val")

    def test_test_min_boundary_is_test(self):
        self.assertEqual(route_band(TEST_MIN, VAL_MIN, TEST_MIN), "test")


class ParseShardIndicesTest(unittest.TestCase):
    def test_range_spec(self):
        self.assertEqual(parse_shard_indices("0:10:3"), [0, 3, 6, 9])

    def test_comma_list(self):
        self.assertEqual(parse_shard_indices("5,1,9"), [5, 1, 9])

    def test_comma_list_rejects_duplicates(self):
        with self.assertRaises(ValueError):
            parse_shard_indices("1,2,1")

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            parse_shard_indices("")

    def test_zero_step_raises(self):
        with self.assertRaises(ValueError):
            parse_shard_indices("0:10:0")


class OrphanPairingTest(unittest.TestCase):
    def test_unpaired_png_is_counted_as_orphan(self):
        with TemporaryDirectory() as td:
            tar_path = Path(td) / "orphan.tar"
            with tarfile.open(tar_path, "w") as tf:
                png_bytes = _make_strip_png_bytes()
                info = tarfile.TarInfo(name="lonely.png")
                info.size = len(png_bytes)
                tf.addfile(info, io.BytesIO(png_bytes))
                # deliberately no lonely.json

            counters = ShardCounters(shard_index=99)
            pairs = list(iter_tar_pairs(tar_path, counters))
            self.assertEqual(pairs, [])
            self.assertEqual(counters.n_orphans, 1)
            self.assertEqual(counters.n_samples_seen, 0)

    def test_reversed_order_pair_still_pairs_up(self):
        with TemporaryDirectory() as td:
            tar_path = Path(td) / "reversed.tar"
            with tarfile.open(tar_path, "w") as tf:
                _write_pair(
                    tf,
                    "k1",
                    {"kind": "line", "text": LETTER_A, "src_doc": 1},
                    order="json_first",
                )
            counters = ShardCounters(shard_index=0)
            pairs = list(iter_tar_pairs(tar_path, counters))
            self.assertEqual(len(pairs), 1)
            self.assertEqual(counters.n_orphans, 0)
            key, png_bytes, meta = pairs[0]
            self.assertEqual(key, "k1")
            self.assertEqual(meta["text"], LETTER_A)


class LetterboxTest(unittest.TestCase):
    def test_letterboxed_output_is_square_l_mode(self):
        png_bytes = _make_strip_png_bytes(width=64, height=600)
        out = letterbox_to_square(png_bytes, IMAGE_SIZE)
        self.assertEqual(out.size, (IMAGE_SIZE, IMAGE_SIZE))
        self.assertEqual(out.mode, "L")

    def test_already_square_input_still_resizes(self):
        png_bytes = _make_strip_png_bytes(width=300, height=300)
        out = letterbox_to_square(png_bytes, IMAGE_SIZE)
        self.assertEqual(out.size, (IMAGE_SIZE, IMAGE_SIZE))


class ProcessShardTest(unittest.TestCase):
    """Full pipeline against the synthetic 5-sample mini-tar."""

    def setUp(self):
        self.tokenizer = make_fixture_tokenizer()
        self.encode_target = make_ocr_target_encoder(self.tokenizer)
        self._td = TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.tmp_dir = Path(self._td.name)
        self.tar_path = self.tmp_dir / "shard-00000.tar"
        _write_mini_tar(self.tar_path)
        self.out_dir = self.tmp_dir / "out"

    def _run(self, **overrides):
        kwargs = dict(
            n_image_tokens=N_IMAGE_TOKENS,
            image_size=IMAGE_SIZE,
            max_seq_len=MAX_SEQ_LEN,
            val_src_doc_min=VAL_MIN,
            test_src_doc_min=TEST_MIN,
            val_cap_per_shard=1000,
            ssl_quota_per_shard=1000,
            instruction_ids=[],
        )
        kwargs.update(overrides)
        return process_shard(
            0, self.tar_path, self.out_dir, self.encode_target, **kwargs
        )

    # -- routing counts -----------------------------------------------

    def test_routing_counts(self):
        counters = self._run()
        self.assertEqual(counters.n_samples_seen, 5)
        self.assertEqual(counters.n_orphans, 0)
        self.assertEqual(counters.n_train, 1)
        self.assertEqual(counters.n_val, 1)
        self.assertEqual(counters.n_test_skipped, 1)
        self.assertEqual(counters.n_non_line_skipped, 1)
        self.assertEqual(counters.n_over_length_skipped, 1)
        self.assertEqual(counters.n_align_written, 1)
        self.assertEqual(counters.n_val_written, 1)
        self.assertEqual(counters.n_ssl_written, 1)

    # -- align row shape ------------------------------------------------

    def test_align_row_has_256_image_patch_slots(self):
        self._run()
        rows = self._read_jsonl(self.out_dir / "jsonl" / "align" / "shard-00000.jsonl")
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["input_ids"].count(IMAGE_PATCH_ID), N_IMAGE_TOKENS)

    def test_align_row_prompt_labels_all_ignored(self):
        self._run()
        rows = self._read_jsonl(self.out_dir / "jsonl" / "align" / "shard-00000.jsonl")
        row = rows[0]
        # prompt = BOS + <image_start> + 256 patches + <image_end> (no
        # instruction in this test) = 259 masked positions, per
        # Model/ocr/data.py:build_ocr_row's layout.
        expected_prompt_len = 3 + N_IMAGE_TOKENS
        self.assertEqual(
            row["labels"][:expected_prompt_len], [IGNORE_INDEX] * expected_prompt_len
        )
        self.assertTrue(all(lab != IGNORE_INDEX for lab in row["labels"][expected_prompt_len:]))

    def test_align_row_decode_target_section_is_byte_exact(self):
        self._run()
        rows = self._read_jsonl(self.out_dir / "jsonl" / "align" / "shard-00000.jsonl")
        row = rows[0]
        n_masked = sum(1 for lab in row["labels"] if lab == IGNORE_INDEX)
        # Supervised tail includes EOS as the final id; strip it to compare
        # against the source transcription text.
        target_and_eos = row["input_ids"][n_masked:]
        decoded = self.tokenizer.decode(target_and_eos[:-1])
        expected_text = SAMPLES[0]["meta"]["text"]
        self.assertEqual(decoded, expected_text)
        self.assertEqual(
            decoded.encode("utf-8", "surrogatepass"),
            expected_text.encode("utf-8", "surrogatepass"),
        )

    def test_align_row_image_path_is_absolute(self):
        self._run()
        rows = self._read_jsonl(self.out_dir / "jsonl" / "align" / "shard-00000.jsonl")
        row = rows[0]
        self.assertEqual(len(row["images"]), 1)
        self.assertTrue(os.path.isabs(row["images"][0]))
        self.assertTrue(Path(row["images"][0]).is_file())

    def test_align_image_is_letterboxed_square(self):
        self._run()
        rows = self._read_jsonl(self.out_dir / "jsonl" / "align" / "shard-00000.jsonl")
        row = rows[0]
        with Image.open(row["images"][0]) as img:
            self.assertEqual(img.size, (IMAGE_SIZE, IMAGE_SIZE))
            self.assertEqual(img.mode, "L")

    # -- val row ----------------------------------------------------------

    def test_val_row_written_with_correct_text(self):
        self._run()
        rows = self._read_jsonl(self.out_dir / "jsonl" / "val" / "shard-00000.jsonl")
        self.assertEqual(len(rows), 1)
        row = rows[0]
        n_masked = sum(1 for lab in row["labels"] if lab == IGNORE_INDEX)
        decoded = self.tokenizer.decode(row["input_ids"][n_masked:-1])
        self.assertEqual(decoded, SAMPLES[1]["meta"]["text"])

    # -- SSL row schema -----------------------------------------------

    def test_ssl_row_schema_and_labels_match_target(self):
        self._run()
        align_rows = self._read_jsonl(
            self.out_dir / "jsonl" / "align" / "shard-00000.jsonl"
        )
        ssl_rows = self._read_jsonl(self.out_dir / "jsonl" / "ssl" / "shard-00000.jsonl")
        self.assertEqual(len(ssl_rows), 1)
        ssl_row = ssl_rows[0]
        self.assertEqual(set(ssl_row.keys()), {"images", "image_sizes", "ocr_labels"})
        self.assertEqual(ssl_row["images"], align_rows[0]["images"])
        self.assertEqual(ssl_row["image_sizes"], [[IMAGE_SIZE, IMAGE_SIZE]])

        align_row = align_rows[0]
        n_masked = sum(1 for lab in align_row["labels"] if lab == IGNORE_INDEX)
        target_ids = align_row["input_ids"][n_masked:-1]  # exclude EOS
        self.assertEqual(ssl_row["ocr_labels"], [target_ids])

    # -- sentinel + idempotent resume -----------------------------------

    def test_sentinel_written_with_counters(self):
        counters = self._run()
        sentinel_path = self.out_dir / "done" / "shard-00000.json"
        self.assertTrue(sentinel_path.is_file())
        with sentinel_path.open() as fh:
            payload = json.load(fh)
        self.assertEqual(payload["n_align_written"], counters.n_align_written)
        self.assertEqual(payload["n_samples_seen"], 5)

    def test_rerun_with_sentinel_skips_and_counters_unchanged(self):
        first = self._run_via_build_one_shard()
        align_path = self.out_dir / "jsonl" / "align" / "shard-00000.jsonl"
        mtime_before = align_path.stat().st_mtime_ns

        second = self._run_via_build_one_shard()

        self.assertEqual(first.as_dict(), second.as_dict())
        self.assertEqual(align_path.stat().st_mtime_ns, mtime_before)

    def _run_via_build_one_shard(self):
        return build_one_shard(
            0,
            self.tmp_dir,
            self.out_dir,
            self.encode_target,
            n_image_tokens=N_IMAGE_TOKENS,
            image_size=IMAGE_SIZE,
            max_seq_len=MAX_SEQ_LEN,
            val_src_doc_min=VAL_MIN,
            test_src_doc_min=TEST_MIN,
            val_cap_per_shard=1000,
            ssl_quota_per_shard=1000,
            instruction_ids=[],
        )

    def test_crashed_shard_partial_outputs_are_rebuilt_not_left_stale(self):
        # Simulate a crash: write a stale align jsonl with no sentinel, then
        # rebuild. build_one_shard must reset (delete) the stale file before
        # rebuilding rather than appending to or trusting it.
        align_dir = self.out_dir / "jsonl" / "align"
        align_dir.mkdir(parents=True, exist_ok=True)
        stale_path = align_dir / "shard-00000.jsonl"
        stale_path.write_text('{"stale": true}\n', encoding="utf-8")

        counters = self._run_via_build_one_shard()
        self.assertEqual(counters.n_align_written, 1)
        rows = self._read_jsonl(stale_path)
        self.assertEqual(len(rows), 1)
        self.assertNotIn("stale", rows[0])

    # -- helpers ------------------------------------------------------

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        rows = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows


class WorkerLoopContinuationTest(unittest.TestCase):
    """A failing shard must not abandon a worker's remaining shards.

    ``_worker_main`` itself needs a real on-disk TokenizerBundle (spawn
    context + real bundle loading; left thin and untested here per this
    module's docstring), but its per-shard try/except body is exactly the
    loop below: one shard raising must be reported and the loop must move
    on to the next shard_index, not abort the whole worker. Regression test
    for a real bug caught during review: an earlier version called
    ``return`` on the first exception, silently dropping every subsequent
    shard this worker owned and hanging the parent's result-collection loop
    (it waits for exactly one message per shard in the full job).
    """

    def test_failing_shard_does_not_abandon_remaining_shards(self):
        tokenizer = make_fixture_tokenizer()
        encode_target = make_ocr_target_encoder(tokenizer)
        with TemporaryDirectory() as td:
            tmp_dir = Path(td)
            # shard 0 and shard 2 are real tars; shard 1's tar is missing,
            # which build_one_shard turns into a FileNotFoundError.
            _write_mini_tar(tmp_dir / "shard-00000.tar")
            _write_mini_tar(tmp_dir / "shard-00002.tar")
            out_dir = tmp_dir / "out"

            kwargs = dict(
                n_image_tokens=N_IMAGE_TOKENS,
                image_size=IMAGE_SIZE,
                max_seq_len=MAX_SEQ_LEN,
                val_src_doc_min=VAL_MIN,
                test_src_doc_min=TEST_MIN,
                val_cap_per_shard=1000,
                ssl_quota_per_shard=1000,
                instruction_ids=[],
            )

            # Mirrors _worker_main's per-shard loop body exactly.
            results = []
            for shard_index in (0, 1, 2):
                try:
                    counters = build_one_shard(
                        shard_index, tmp_dir, out_dir, encode_target, **kwargs
                    )
                    results.append(("ok", shard_index, counters.as_dict()))
                except Exception as exc:  # noqa: BLE001
                    results.append(
                        ("error", shard_index, f"{type(exc).__name__}: {exc}")
                    )

            # All three shards produced a result -- the loop did not stop
            # after shard 1's failure.
            self.assertEqual([r[1] for r in results], [0, 1, 2])
            self.assertEqual(results[0][0], "ok")
            self.assertEqual(results[1][0], "error")
            self.assertIn("FileNotFoundError", results[1][2])
            self.assertEqual(results[2][0], "ok")
            # The failed shard wrote no sentinel, so a rerun would retry it.
            self.assertFalse((out_dir / "done" / "shard-00001.json").exists())
            self.assertTrue((out_dir / "done" / "shard-00000.json").exists())
            self.assertTrue((out_dir / "done" / "shard-00002.json").exists())


class ValCapTest(unittest.TestCase):
    """A tighter val_cap_per_shard must be respected (rows still routed val)."""

    def test_val_cap_of_zero_skips_writing_but_still_counts_val(self):
        tokenizer = make_fixture_tokenizer()
        encode_target = make_ocr_target_encoder(tokenizer)
        with TemporaryDirectory() as td:
            tmp_dir = Path(td)
            tar_path = tmp_dir / "shard-00000.tar"
            _write_mini_tar(tar_path)
            out_dir = tmp_dir / "out"
            counters = process_shard(
                0,
                tar_path,
                out_dir,
                encode_target,
                n_image_tokens=N_IMAGE_TOKENS,
                image_size=IMAGE_SIZE,
                max_seq_len=MAX_SEQ_LEN,
                val_src_doc_min=VAL_MIN,
                test_src_doc_min=TEST_MIN,
                val_cap_per_shard=0,
                ssl_quota_per_shard=1000,
                instruction_ids=[],
            )
            self.assertEqual(counters.n_val, 1)
            self.assertEqual(counters.n_val_written, 0)
            self.assertEqual(counters.n_val_cap_skipped, 1)


class SSLQuotaTest(unittest.TestCase):
    """Zero SSL quota must skip SSL rows without affecting the align row."""

    def test_zero_ssl_quota_skips_ssl_row(self):
        tokenizer = make_fixture_tokenizer()
        encode_target = make_ocr_target_encoder(tokenizer)
        with TemporaryDirectory() as td:
            tmp_dir = Path(td)
            tar_path = tmp_dir / "shard-00000.tar"
            _write_mini_tar(tar_path)
            out_dir = tmp_dir / "out"
            counters = process_shard(
                0,
                tar_path,
                out_dir,
                encode_target,
                n_image_tokens=N_IMAGE_TOKENS,
                image_size=IMAGE_SIZE,
                max_seq_len=MAX_SEQ_LEN,
                val_src_doc_min=VAL_MIN,
                test_src_doc_min=TEST_MIN,
                val_cap_per_shard=1000,
                ssl_quota_per_shard=0,
                instruction_ids=[],
            )
            self.assertEqual(counters.n_align_written, 1)
            self.assertEqual(counters.n_ssl_written, 0)
            ssl_path = out_dir / "jsonl" / "ssl" / "shard-00000.jsonl"
            self.assertEqual(ssl_path.read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
