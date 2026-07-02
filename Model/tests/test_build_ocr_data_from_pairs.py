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

A second suite below (``Hanshi*Test``) covers the hanshi-corpus input mode
(:func:`iter_hanshi_meta_rows` / :func:`process_hanshi_virtual_shard` /
:func:`build_one_hanshi_virtual_shard`) with a synthetic ~30-row meta.jsonl +
``pages/<bucket>/<doc_id>.png`` tree, built the same way. Per that module's
own docstring, its multiprocess reader/worker pool
(``_hanshi_reader_main``/``_hanshi_worker_main``/``_run_hanshi_mode``) needs
a real on-disk :class:`TokenizerBundle` and ``spawn``-context processes and
is therefore left thin and untested at the unittest level here too; the
reader's queue protocol (batching, done-sentinel count, sentinel-skip on
resume) was verified manually with an in-process ``threading.Thread`` stand-in
against ``queue.Queue`` (safe here since neither function touches torch/CUDA
state) and is documented in the delivery report rather than as a unittest.
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
    build_one_hanshi_virtual_shard,
    build_one_shard,
    hanshi_image_path,
    hanshi_output_shard_number,
    hanshi_virtual_shard_index,
    iter_hanshi_meta_rows,
    iter_tar_pairs,
    letterbox_to_square,
    parse_args,
    parse_shard_indices,
    process_hanshi_virtual_shard,
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


# ===========================================================================
# Hanshi mode: synthetic ~30-row meta.jsonl + pages/<bucket>/<doc_id>.png
# ===========================================================================

HANSHI_SHARD_SIZE = 6
HANSHI_SHARD_OFFSET = 10_000


def _make_hanshi_corpus(tmp_dir: Path) -> tuple[Path, Path, list[dict]]:
    """Write a synthetic hanshi meta.jsonl + pages tree; return (meta_path, pages_root, rows).

    30 rows total, laid out so the default ``HANSHI_SHARD_SIZE=6`` groups
    them into exactly 5 virtual shards (0..4), each spanning a mix of
    bands/edge-cases:

    - rows 0..17 (virtual shards 0, 1, 2): clean train-band "line" rows,
      alternating between two of the four seeded letters -- these carry the
      routing-count and byte-exact-decode assertions.
    - row 18 (virtual shard 3, index 0): val-band (src_doc == VAL_MIN).
    - row 19 (virtual shard 3, index 1): test-band (src_doc == TEST_MIN) --
      must be skipped whole, never even orphan/length-checked.
    - row 20 (virtual shard 3, index 2): "kind": "page", not "line" -- must
      be skipped and counted, not fatal.
    - row 21 (virtual shard 3, index 3): over-length text (busts
      MAX_SEQ_LEN via byte-fallback, same trick as SAMPLES[4] above).
    - row 22 (virtual shard 3, index 4): clean train row whose PNG is
      deliberately never written to disk -- the missing-image-is-an-orphan
      (not fatal) case unique to hanshi mode.
    - row 23 (virtual shard 3, index 5): clean train row, closes out shard 3.
    - rows 24..29 (virtual shard 4): clean train rows.

    All bucket dirs are ``"00000"`` (single bucket is enough to exercise the
    ``<pages_root>/<bucket>/<doc_id>.png`` path; multi-bucket routing is not
    special-cased by the builder beyond string-joining ``meta["bucket"]``).
    """

    meta_path = tmp_dir / "meta.jsonl"
    pages_root = tmp_dir / "pages"
    bucket = "00000"
    bucket_dir = pages_root / bucket
    bucket_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    def _add(doc_id: str, text: str, src_doc: int, kind: str = "line", write_image: bool = True):
        row = {
            "doc_id": doc_id,
            "kind": kind,
            "text": text,
            "src_doc": src_doc,
            "bucket": bucket,
            "font": "hanshi",
            "font_px": 32,
        }
        rows.append(row)
        if write_image:
            (bucket_dir / f"{doc_id}.png").write_bytes(_make_strip_png_bytes())

    for i in range(18):
        letters = SEEDED_LETTERS[i % 2] + SEEDED_LETTERS[(i + 1) % 2]
        _add(f"line_{i:08d}_v0", letters, src_doc=i)

    _add("line_00000018_v0", LETTER_GA, src_doc=VAL_MIN)  # val band
    _add("line_00000019_v0", LETTER_RA, src_doc=TEST_MIN)  # test band, skipped whole
    _add("line_00000020_v0", LETTER_A, src_doc=20, kind="page")  # non-"line", skipped
    _add("line_00000021_v0", "x" * 600, src_doc=21)  # over-length
    _add("line_00000022_v0", LETTER_NA, src_doc=22, write_image=False)  # missing file
    _add("line_00000023_v0", LETTER_A + LETTER_GA, src_doc=23)

    for i in range(24, 30):
        _add(f"line_{i:08d}_v0", LETTER_A + LETTER_RA, src_doc=i)

    with meta_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    return meta_path, pages_root, rows


class HanshiMetaStreamingTest(unittest.TestCase):
    def setUp(self):
        self._td = TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.tmp_dir = Path(self._td.name)
        self.meta_path, self.pages_root, self.rows = _make_hanshi_corpus(self.tmp_dir)

    def test_stride_one_keeps_every_row(self):
        kept = list(iter_hanshi_meta_rows(self.meta_path, stride=1))
        self.assertEqual(len(kept), 30)
        self.assertEqual([k for k, _, _ in kept], list(range(30)))
        self.assertEqual(kept[0][1], "line_00000000_v0")

    def test_stride_keeps_every_kth_raw_line_and_reindexes_kept_idx(self):
        kept = list(iter_hanshi_meta_rows(self.meta_path, stride=3))
        # raw line indices 0, 3, 6, ... -> doc_ids line_00000000, 00000003, ...
        self.assertEqual(len(kept), 10)  # ceil(30 / 3)
        doc_ids = [doc_id for _, doc_id, _ in kept]
        self.assertEqual(
            doc_ids,
            [f"line_{i:08d}_v0" for i in range(0, 30, 3)],
        )
        # kept_idx is dense (0..9), NOT the raw line index (0,3,6,...).
        self.assertEqual([k for k, _, _ in kept], list(range(10)))

    def test_stride_rejects_zero_and_negative_via_generator_contract(self):
        # iter_hanshi_meta_rows itself does not validate stride (that is
        # parse_args' job via --hanshi-stride); a stride of 0 must not be
        # silently passed through by the CLI (covered in
        # HanshiArgparseTest.test_stride_below_one_rejected below). Guard
        # here only that stride=1 truly is a no-op / identity subset.
        kept_all = [doc_id for _, doc_id, _ in iter_hanshi_meta_rows(self.meta_path, stride=1)]
        self.assertEqual(len(kept_all), len(self.rows))

    def test_missing_doc_id_raises(self):
        bad_path = self.tmp_dir / "bad_meta.jsonl"
        bad_path.write_text(
            json.dumps({"kind": "line", "text": "x", "src_doc": 0, "bucket": "00000"})
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            list(iter_hanshi_meta_rows(bad_path, stride=1))

    def test_invalid_json_line_raises(self):
        bad_path = self.tmp_dir / "bad_meta2.jsonl"
        bad_path.write_text("{not valid json\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            list(iter_hanshi_meta_rows(bad_path, stride=1))

    def test_blank_lines_are_skipped_without_consuming_kept_idx_or_stride(self):
        path_with_blanks = self.tmp_dir / "with_blanks.jsonl"
        rows = [
            {"doc_id": "a", "kind": "line", "text": "x", "src_doc": 0, "bucket": "00000"},
            {"doc_id": "b", "kind": "line", "text": "y", "src_doc": 1, "bucket": "00000"},
        ]
        with path_with_blanks.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(rows[0]) + "\n")
            fh.write("\n")
            fh.write("   \n")
            fh.write(json.dumps(rows[1]) + "\n")
        kept = list(iter_hanshi_meta_rows(path_with_blanks, stride=1))
        self.assertEqual([doc_id for _, doc_id, _ in kept], ["a", "b"])
        self.assertEqual([k for k, _, _ in kept], [0, 1])


class HanshiVirtualShardNamingTest(unittest.TestCase):
    def test_virtual_shard_index_groups_kept_idx_into_fixed_windows(self):
        self.assertEqual(hanshi_virtual_shard_index(0, 6), 0)
        self.assertEqual(hanshi_virtual_shard_index(5, 6), 0)
        self.assertEqual(hanshi_virtual_shard_index(6, 6), 1)
        self.assertEqual(hanshi_virtual_shard_index(29, 6), 4)

    def test_output_shard_number_uses_offset_and_never_collides_with_tar_range(self):
        self.assertEqual(hanshi_output_shard_number(0, 10_000), 10_000)
        self.assertEqual(hanshi_output_shard_number(4, 10_000), 10_004)
        # Tar-mode shards run 0..4053; default offset must stay clear of it.
        self.assertGreater(hanshi_output_shard_number(0, 10_000), 4053)

    def test_image_path_joins_pages_root_bucket_and_doc_id(self):
        p = hanshi_image_path("/nas/hanshi/pages", "00042", "line_00000000_v0")
        self.assertEqual(p, Path("/nas/hanshi/pages/00042/line_00000000_v0.png"))


class HanshiProcessVirtualShardTest(unittest.TestCase):
    """Full pipeline against the synthetic 30-row hanshi corpus, virtual shard 3."""

    def setUp(self):
        self.tokenizer = make_fixture_tokenizer()
        self.encode_target = make_ocr_target_encoder(self.tokenizer)
        self._td = TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.tmp_dir = Path(self._td.name)
        self.meta_path, self.pages_root, self.rows = _make_hanshi_corpus(self.tmp_dir)
        self.out_dir = self.tmp_dir / "out"

    def _kept_rows_for_virtual_shard(self, virtual_index: int) -> list[tuple[str, dict]]:
        all_kept = list(iter_hanshi_meta_rows(self.meta_path, stride=1))
        return [
            (doc_id, meta)
            for kept_idx, doc_id, meta in all_kept
            if hanshi_virtual_shard_index(kept_idx, HANSHI_SHARD_SIZE) == virtual_index
        ]

    def _run(self, virtual_index: int, **overrides):
        rows = self._kept_rows_for_virtual_shard(virtual_index)
        output_shard_number = hanshi_output_shard_number(virtual_index, HANSHI_SHARD_OFFSET)
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
        return output_shard_number, process_hanshi_virtual_shard(
            output_shard_number,
            rows,
            self.pages_root,
            self.out_dir,
            self.encode_target,
            **kwargs,
        )

    # -- routing + orphan counts (virtual shard 3: val/test/non-line/over-length/missing-image) --

    def test_routing_and_orphan_counts_on_edge_case_shard(self):
        _, counters = self._run(3)
        # virtual shard 3 = rows 18..23: val, test, page(non-line),
        # over-length, missing-image, clean.
        self.assertEqual(counters.n_samples_seen, 6)
        self.assertEqual(counters.n_test_skipped, 1)
        self.assertEqual(counters.n_non_line_skipped, 1)
        self.assertEqual(counters.n_over_length_skipped, 1)
        self.assertEqual(counters.n_orphans, 1)  # the missing-image row
        self.assertEqual(counters.n_val, 1)
        self.assertEqual(counters.n_train, 1)  # only the final clean row
        self.assertEqual(counters.n_align_written, 1)
        self.assertEqual(counters.n_val_written, 1)

    def test_missing_image_does_not_abort_the_shard(self):
        # The whole point of the orphan path: a missing file must not raise
        # -- the shard must still complete and write a sentinel.
        output_shard_number, counters = self._run(3)
        sentinel_path = self.out_dir / "done" / f"shard-{output_shard_number:05d}.json"
        self.assertTrue(sentinel_path.is_file())
        self.assertEqual(counters.n_orphans, 1)

    def test_clean_shard_routing_counts(self):
        # virtual shard 0 = rows 0..5, all clean train-band "line" rows.
        _, counters = self._run(0)
        self.assertEqual(counters.n_samples_seen, 6)
        self.assertEqual(counters.n_orphans, 0)
        self.assertEqual(counters.n_train, 6)
        self.assertEqual(counters.n_val, 0)
        self.assertEqual(counters.n_test_skipped, 0)
        self.assertEqual(counters.n_align_written, 6)
        self.assertEqual(counters.n_ssl_written, 6)

    # -- output naming uses the offset ------------------------------------

    def test_outputs_are_named_with_shard_offset(self):
        output_shard_number, _ = self._run(0)
        self.assertEqual(output_shard_number, 10_000)
        self.assertTrue(
            (self.out_dir / "jsonl" / "align" / "shard-10000.jsonl").is_file()
        )
        self.assertTrue((self.out_dir / "images" / "shard-10000").is_dir())
        self.assertTrue((self.out_dir / "done" / "shard-10000.json").is_file())

    # -- byte-exact decode, same contract as tar mode ----------------------

    def test_align_row_decode_is_byte_exact(self):
        output_shard_number, _ = self._run(0)
        rows = self._read_jsonl(
            self.out_dir / "jsonl" / "align" / f"shard-{output_shard_number:05d}.jsonl"
        )
        self.assertEqual(len(rows), 6)
        expected_texts = [
            SEEDED_LETTERS[i % 2] + SEEDED_LETTERS[(i + 1) % 2] for i in range(6)
        ]
        for row, expected_text in zip(rows, expected_texts):
            n_masked = sum(1 for lab in row["labels"] if lab == IGNORE_INDEX)
            target_and_eos = row["input_ids"][n_masked:]
            decoded = self.tokenizer.decode(target_and_eos[:-1])
            self.assertEqual(decoded, expected_text)
            self.assertEqual(
                decoded.encode("utf-8", "surrogatepass"),
                expected_text.encode("utf-8", "surrogatepass"),
            )

    def test_align_image_is_letterboxed_square_and_file_exists(self):
        output_shard_number, _ = self._run(0)
        rows = self._read_jsonl(
            self.out_dir / "jsonl" / "align" / f"shard-{output_shard_number:05d}.jsonl"
        )
        row = rows[0]
        self.assertTrue(os.path.isabs(row["images"][0]))
        with Image.open(row["images"][0]) as img:
            self.assertEqual(img.size, (IMAGE_SIZE, IMAGE_SIZE))
            self.assertEqual(img.mode, "L")

    # -- ssl row schema, same contract as tar mode -------------------------

    def test_ssl_row_schema(self):
        output_shard_number, _ = self._run(0)
        ssl_rows = self._read_jsonl(
            self.out_dir / "jsonl" / "ssl" / f"shard-{output_shard_number:05d}.jsonl"
        )
        self.assertEqual(len(ssl_rows), 6)
        self.assertEqual(set(ssl_rows[0].keys()), {"images", "image_sizes", "ocr_labels"})
        self.assertEqual(ssl_rows[0]["image_sizes"], [[IMAGE_SIZE, IMAGE_SIZE]])

    # -- val-cap / ssl-quota are per-virtual-shard, not divided ------------

    def test_val_cap_and_ssl_quota_are_used_directly_per_virtual_shard(self):
        # virtual shard 3 has exactly one val row; a cap of 0 must skip
        # writing it but still count it, identically to tar mode's
        # val_cap_per_shard=0 semantics (ValCapTest above) -- just fed the
        # raw --val-cap value directly instead of a divided one.
        _, counters = self._run(3, val_cap_per_shard=0)
        self.assertEqual(counters.n_val, 1)
        self.assertEqual(counters.n_val_written, 0)
        self.assertEqual(counters.n_val_cap_skipped, 1)

    # -- sentinel + idempotent resume, same contract as tar mode -----------

    def test_sentinel_written_with_counters(self):
        output_shard_number, counters = self._run(0)
        sentinel_path = self.out_dir / "done" / f"shard-{output_shard_number:05d}.json"
        with sentinel_path.open() as fh:
            payload = json.load(fh)
        self.assertEqual(payload["n_align_written"], counters.n_align_written)
        self.assertEqual(payload["n_samples_seen"], 6)

    def test_rerun_via_build_one_hanshi_virtual_shard_skips_and_counters_unchanged(self):
        rows = self._kept_rows_for_virtual_shard(0)
        output_shard_number = hanshi_output_shard_number(0, HANSHI_SHARD_OFFSET)
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
        first = build_one_hanshi_virtual_shard(
            output_shard_number, rows, self.pages_root, self.out_dir,
            self.encode_target, **kwargs,
        )
        align_path = self.out_dir / "jsonl" / "align" / f"shard-{output_shard_number:05d}.jsonl"
        mtime_before = align_path.stat().st_mtime_ns

        # Second call passes an EMPTY rows list to prove the sentinel short-
        # circuits actual (re)processing rather than happening to reprocess
        # identical rows and land on the same counters by coincidence.
        second = build_one_hanshi_virtual_shard(
            output_shard_number, [], self.pages_root, self.out_dir,
            self.encode_target, **kwargs,
        )

        self.assertEqual(first.as_dict(), second.as_dict())
        self.assertEqual(align_path.stat().st_mtime_ns, mtime_before)

    def test_crashed_virtual_shard_partial_outputs_are_rebuilt_not_left_stale(self):
        align_dir = self.out_dir / "jsonl" / "align"
        align_dir.mkdir(parents=True, exist_ok=True)
        output_shard_number = hanshi_output_shard_number(0, HANSHI_SHARD_OFFSET)
        stale_path = align_dir / f"shard-{output_shard_number:05d}.jsonl"
        stale_path.write_text('{"stale": true}\n', encoding="utf-8")

        rows = self._kept_rows_for_virtual_shard(0)
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
        counters = build_one_hanshi_virtual_shard(
            output_shard_number, rows, self.pages_root, self.out_dir,
            self.encode_target, **kwargs,
        )
        self.assertEqual(counters.n_align_written, 6)
        rows_out = self._read_jsonl(stale_path)
        self.assertEqual(len(rows_out), 6)
        self.assertNotIn("stale", rows_out[0])

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


class HanshiArgparseTest(unittest.TestCase):
    """--hanshi-* and --shards-dir/--shard-indices are mutually exclusive."""

    def test_both_modes_given_is_an_argparse_error(self):
        with self.assertRaises(SystemExit) as ctx:
            parse_args(
                [
                    "--shards-dir", "/tmp/x",
                    "--shard-indices", "0:1",
                    "--hanshi-meta", "/tmp/meta.jsonl",
                    "--hanshi-pages", "/tmp/pages",
                    "--out", "/tmp/out",
                    "--tokenizer-bundle", "/tmp/bundle",
                ]
            )
        self.assertEqual(ctx.exception.code, 2)

    def test_neither_mode_given_is_an_argparse_error(self):
        with self.assertRaises(SystemExit) as ctx:
            parse_args(["--out", "/tmp/out", "--tokenizer-bundle", "/tmp/bundle"])
        self.assertEqual(ctx.exception.code, 2)

    def test_hanshi_mode_alone_parses(self):
        ns = parse_args(
            [
                "--hanshi-meta", "/tmp/meta.jsonl",
                "--hanshi-pages", "/tmp/pages",
                "--out", "/tmp/out",
                "--tokenizer-bundle", "/tmp/bundle",
            ]
        )
        self.assertEqual(ns.hanshi_meta, "/tmp/meta.jsonl")
        self.assertEqual(ns.hanshi_pages, "/tmp/pages")
        self.assertIsNone(ns.shards_dir)
        self.assertIsNone(ns.shard_indices)
        # defaults from the task spec
        self.assertEqual(ns.hanshi_stride, 1)
        self.assertEqual(ns.hanshi_shard_size, 100_000)
        self.assertEqual(ns.shard_offset, 10_000)

    def test_tar_mode_alone_still_parses(self):
        ns = parse_args(
            [
                "--shards-dir", "/tmp/x",
                "--shard-indices", "0:1",
                "--out", "/tmp/out",
                "--tokenizer-bundle", "/tmp/bundle",
            ]
        )
        self.assertEqual(ns.shards_dir, "/tmp/x")
        self.assertIsNone(ns.hanshi_meta)

    def test_hanshi_meta_without_hanshi_pages_is_an_error(self):
        with self.assertRaises(SystemExit) as ctx:
            parse_args(
                [
                    "--hanshi-meta", "/tmp/meta.jsonl",
                    "--out", "/tmp/out",
                    "--tokenizer-bundle", "/tmp/bundle",
                ]
            )
        self.assertEqual(ctx.exception.code, 2)

    def test_shards_dir_without_shard_indices_is_an_error(self):
        with self.assertRaises(SystemExit) as ctx:
            parse_args(
                [
                    "--shards-dir", "/tmp/x",
                    "--out", "/tmp/out",
                    "--tokenizer-bundle", "/tmp/bundle",
                ]
            )
        self.assertEqual(ctx.exception.code, 2)

    def test_stride_below_one_rejected(self):
        with self.assertRaises(SystemExit) as ctx:
            parse_args(
                [
                    "--hanshi-meta", "/tmp/meta.jsonl",
                    "--hanshi-pages", "/tmp/pages",
                    "--hanshi-stride", "0",
                    "--out", "/tmp/out",
                    "--tokenizer-bundle", "/tmp/bundle",
                ]
            )
        self.assertEqual(ctx.exception.code, 2)

    def test_shard_size_below_one_rejected(self):
        with self.assertRaises(SystemExit) as ctx:
            parse_args(
                [
                    "--hanshi-meta", "/tmp/meta.jsonl",
                    "--hanshi-pages", "/tmp/pages",
                    "--hanshi-shard-size", "0",
                    "--out", "/tmp/out",
                    "--tokenizer-bundle", "/tmp/bundle",
                ]
            )
        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
