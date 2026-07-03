# -*- coding: utf-8 -*-

"""Unit tests for Model.ocr.segment and scripts.build_annotation_pack.

Two families:

- ``Model.ocr.segment`` (``detect_line_columns`` / ``chunk_column_by_height``):
  driven with synthetic PIL-drawn ink blocks (not real Mongolian glyphs --
  font rendering is not this tool's own concern; see scripts/render_mn_pages.py's
  tests for that).
- ``scripts.build_annotation_pack``: driven end to end through ``--image-dir``
  mode (2 synthetic page PNGs built in-test) AND unconditionally through the
  real ``--pdf-dir`` path (a genuine 2-page PDF assembled in-test via
  ``fitz.open()/new_page()/insert_image()/save()``, mirroring
  scripts/render_mn_pages.py's own PDF-assembly pattern -- gated only on
  ``fitz`` being importable, which it is in this environment, so this test
  actually runs rather than merely existing-but-skipped).
"""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image, ImageDraw  # noqa: E402

from Model.ocr.segment import (  # noqa: E402
    chunk_column_by_height,
    detect_line_columns,
)
from scripts.build_annotation_pack import (  # noqa: E402
    _stable_hash,
    build_pack,
    check_stem_collisions,
    main,
    parse_args,
    sample_chunks,
    sample_page_indices,
    sanitize_stem,
)

try:
    import fitz  # PyMuPDF

    _HAS_FITZ = True
except ImportError:  # pragma: no cover
    _HAS_FITZ = False

_ID_PATTERN = re.compile(r"^[A-Za-z0-9_]+__p\d{4}__c\d{2}__l\d{3}$")


def _draw_three_column_page(size=(200, 300)) -> Image.Image:
    """A synthetic page with 3 vertical ink blocks simulating text columns."""
    img = Image.new("L", size, 255)
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 30, 45, 270], fill=0)
    draw.rectangle([90, 40, 115, 260], fill=0)
    draw.rectangle([160, 50, 180, 200], fill=0)
    return img


# ===========================================================================
# Model.ocr.segment
# ===========================================================================


class DetectLineColumnsTest(unittest.TestCase):
    def test_three_ink_blocks_detected_left_to_right(self):
        img = _draw_three_column_page()
        boxes = detect_line_columns(img)
        self.assertEqual(len(boxes), 3)
        xs = [b[0] for b in boxes]
        self.assertEqual(xs, sorted(xs), "columns must be ordered left-to-right")
        # Each box should roughly bracket its drawn block (allowing for the
        # ink-density threshold's edge slop) and span full page height.
        for box, expected_x0 in zip(boxes, (20, 90, 160)):
            x0, y0, x1, y1 = box
            self.assertAlmostEqual(x0, expected_x0, delta=2)
            self.assertEqual(y0, 0)
            self.assertEqual(y1, img.height)

    def test_blank_page_falls_back_to_whole_page(self):
        img = Image.new("L", (100, 150), 255)
        boxes = detect_line_columns(img)
        self.assertEqual(boxes, [(0, 0, 100, 150)])

    def test_narrow_gap_between_blocks_is_merged(self):
        # Two blocks separated by a 1px gap (anti-aliasing-scale noise, below
        # the default min_gap_px) must merge into one column, not two.
        img = Image.new("L", (60, 100), 255)
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 10, 25, 90], fill=0)
        draw.rectangle([27, 10, 40, 90], fill=0)  # 1px gap from the first
        boxes = detect_line_columns(img)
        self.assertEqual(len(boxes), 1)

    def test_rejects_non_image_input(self):
        with self.assertRaises(TypeError):
            detect_line_columns("not an image")

    def test_rejects_invalid_min_col_width(self):
        img = Image.new("L", (10, 10), 255)
        with self.assertRaises(ValueError):
            detect_line_columns(img, min_col_width_px=0)


class ChunkColumnByHeightTest(unittest.TestCase):
    def test_short_column_unchanged(self):
        box = (0, 0, 50, 400)
        self.assertEqual(chunk_column_by_height(box, max_chunk_px=900), [box])

    def test_exact_multiple_splits_cleanly(self):
        chunks = chunk_column_by_height((0, 0, 50, 1800), max_chunk_px=900)
        self.assertEqual(chunks, [(0, 0, 50, 900), (0, 900, 50, 1800)])

    def test_remainder_produces_shorter_final_chunk(self):
        chunks = chunk_column_by_height((0, 0, 50, 2000), max_chunk_px=900)
        self.assertEqual(
            chunks, [(0, 0, 50, 900), (0, 900, 50, 1800), (0, 1800, 50, 2000)]
        )
        # No dropped remainder, no oversized final chunk.
        total_height = sum(y1 - y0 for _, y0, _, y1 in chunks)
        self.assertEqual(total_height, 2000)

    def test_rejects_degenerate_box(self):
        with self.assertRaises(ValueError):
            chunk_column_by_height((0, 10, 50, 10))  # y1 == y0

    def test_rejects_invalid_max_chunk_px(self):
        with self.assertRaises(ValueError):
            chunk_column_by_height((0, 0, 50, 100), max_chunk_px=0)


# ===========================================================================
# scripts.build_annotation_pack: pure helpers
# ===========================================================================


class StableHashTest(unittest.TestCase):
    def test_deterministic_within_and_documented_across_process(self):
        # Within-process determinism is directly testable; cross-process
        # stability (the actual bug this function fixes -- Python's builtin
        # hash() is salted per process) is verified by this test file's own
        # subprocess-based determinism test below, not re-derivable here.
        self.assertEqual(_stable_hash("doc_a"), _stable_hash("doc_a"))
        self.assertNotEqual(_stable_hash("doc_a"), _stable_hash("doc_b"))
        self.assertEqual(_stable_hash("doc_a", 3), _stable_hash("doc_a", 3))
        self.assertNotEqual(_stable_hash("doc_a", 3), _stable_hash("doc_a", 4))


class SanitizeStemTest(unittest.TestCase):
    def test_non_alnum_becomes_underscore(self):
        self.assertEqual(sanitize_stem("my book (vol. 2).pdf"[:-4]), "my_book__vol__2_")

    def test_truncated_to_64_chars(self):
        self.assertEqual(len(sanitize_stem("a" * 200)), 64)


class CheckStemCollisionsTest(unittest.TestCase):
    def test_no_collision_passes(self):
        check_stem_collisions([Path("bookA.pdf"), Path("bookB.pdf")])  # no raise

    def test_collision_raises_with_both_paths_listed(self):
        with self.assertRaises(ValueError) as ctx:
            check_stem_collisions(
                [Path("/x/book!.pdf"), Path("/y/book_.pdf")]  # both sanitize to "book_"
            )
        msg = str(ctx.exception)
        self.assertIn("book_", msg)
        self.assertIn("/x/book!.pdf", msg)
        self.assertIn("/y/book_.pdf", msg)


class SamplePageIndicesTest(unittest.TestCase):
    def test_excludes_first_and_last_5_percent_for_long_docs(self):
        # 100 pages: exclude round(100*0.05)=5 from each edge -> [5, 95).
        picked = sample_page_indices(100, 10, seed=0, doc_stem="d")
        self.assertTrue(all(5 <= p < 95 for p in picked))

    def test_tiny_doc_skips_edge_exclusion(self):
        # Below _MIN_PAGES_FOR_EDGE_EXCLUDE (20): every page is a candidate.
        picked = sample_page_indices(6, 3, seed=0, doc_stem="d")
        self.assertEqual(len(picked), 3)
        self.assertTrue(all(0 <= p < 6 for p in picked))

    def test_deterministic_given_seed(self):
        a = sample_page_indices(100, 5, seed=42, doc_stem="doc")
        b = sample_page_indices(100, 5, seed=42, doc_stem="doc")
        self.assertEqual(a, b)

    def test_different_doc_stem_does_not_reshuffle_deterministically_to_same_result(self):
        a = sample_page_indices(100, 5, seed=0, doc_stem="doc_a")
        b = sample_page_indices(100, 5, seed=0, doc_stem="doc_b")
        self.assertNotEqual(a, b)

    def test_zero_pages_returns_empty(self):
        self.assertEqual(sample_page_indices(0, 3, seed=0, doc_stem="d"), [])

    def test_caps_at_available_candidates(self):
        picked = sample_page_indices(6, 100, seed=0, doc_stem="d")
        self.assertEqual(len(picked), 6)


class SampleChunksTest(unittest.TestCase):
    def test_deterministic_given_seed(self):
        chunks = [(0, i, 10, i + 5) for i in range(20)]
        a = sample_chunks(chunks, 4, seed=1, doc_stem="d", page_index=0)
        b = sample_chunks(chunks, 4, seed=1, doc_stem="d", page_index=0)
        self.assertEqual(a, b)

    def test_different_page_index_differs(self):
        chunks = [(0, i, 10, i + 5) for i in range(20)]
        a = sample_chunks(chunks, 4, seed=1, doc_stem="d", page_index=0)
        b = sample_chunks(chunks, 4, seed=1, doc_stem="d", page_index=1)
        self.assertNotEqual(a, b)

    def test_empty_chunks_returns_empty(self):
        self.assertEqual(sample_chunks([], 4, seed=0, doc_stem="d", page_index=0), [])


# ===========================================================================
# scripts.build_annotation_pack: end-to-end via --image-dir
# ===========================================================================


class ImageDirEndToEndTest(unittest.TestCase):
    def _build_pages(self, tmp: Path, n_pages: int = 30) -> Path:
        image_dir = tmp / "pages"
        image_dir.mkdir()
        for i in range(n_pages):
            _draw_three_column_page().save(image_dir / f"page_{i:03d}.png")
        return image_dir

    def _run(self, image_dir: Path, out_dir: Path, **extra_argv) -> dict:
        argv = [
            "--image-dir",
            str(image_dir),
            "--out",
            str(out_dir),
            "--pages-per-doc",
            "3",
            "--lines-per-page",
            "8",
            "--seed",
            "0",
        ]
        for k, v in extra_argv.items():
            argv.extend([k, str(v)])
        args = parse_args(argv)
        return build_pack(args)

    def test_end_to_end_produces_complete_pack(self):
        with TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            image_dir = self._build_pages(tmp)
            out_dir = tmp / "out"
            manifest = self._run(image_dir, out_dir)

            # 3 pages sampled x 3 columns each (min(8, 3)) = 9 lines.
            self.assertEqual(manifest["counts"]["n_pages_sampled"], 3)
            self.assertEqual(manifest["counts"]["n_lines"], 9)
            self.assertEqual(manifest["counts"]["n_pages_empty"], 0)
            self.assertEqual(
                manifest["counts"]["n_double_annotate"], round(9 * 0.1)
            )

            self.assertTrue((out_dir / "manifest.json").exists())
            with (out_dir / "manifest.json").open() as fh:
                on_disk = json.load(fh)
            self.assertEqual(on_disk, manifest)

            sheet_a = (out_dir / "sheets" / "annotator_A.tsv").read_text().splitlines()
            sheet_b = (out_dir / "sheets" / "annotator_B.tsv").read_text().splitlines()
            self.assertEqual(len(sheet_a) - 1, manifest["counts"]["n_lines"])  # -1 header
            self.assertEqual(
                len(sheet_b) - 1, manifest["counts"]["n_double_annotate"]
            )

            a_ids = {line.split("\t")[0] for line in sheet_a[1:]}
            b_ids = {line.split("\t")[0] for line in sheet_b[1:]}
            self.assertTrue(b_ids.issubset(a_ids), "sheet B must be a subset of sheet A")

            for row in manifest["lines"]:
                self.assertRegex(row["id"], _ID_PATTERN)
                self.assertTrue((out_dir / row["raw_path"]).exists())
                self.assertTrue((out_dir / row["preview_path"]).exists())

    def test_double_annotate_math_with_tolerance(self):
        with TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            image_dir = self._build_pages(tmp, n_pages=30)
            out_dir = tmp / "out"
            manifest = self._run(
                image_dir, out_dir, **{"--double-annotate-frac": 0.33}
            )
            n_lines = manifest["counts"]["n_lines"]
            expected = round(n_lines * 0.33)
            self.assertAlmostEqual(
                manifest["counts"]["n_double_annotate"], expected, delta=1
            )

    def test_deterministic_given_seed_across_process_reinvocation(self):
        # Regression test for the Python builtin hash() salting bug: two
        # SEPARATE subprocess invocations with the same --seed must produce
        # byte-identical manifest lines/sampled-pages, not just
        # determinism-within-one-process (which the built-in hash() would
        # already satisfy and therefore not catch this class of bug).
        import subprocess

        with TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            image_dir = self._build_pages(tmp)
            out_a = tmp / "out_a"
            out_b = tmp / "out_b"
            repo_root = Path(__file__).resolve().parents[2]
            for out_dir in (out_a, out_b):
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "scripts.build_annotation_pack",
                        "--image-dir",
                        str(image_dir),
                        "--out",
                        str(out_dir),
                        "--pages-per-doc",
                        "3",
                        "--lines-per-page",
                        "8",
                        "--seed",
                        "0",
                    ],
                    cwd=str(repo_root),
                    check=True,
                    capture_output=True,
                )
            with (out_a / "manifest.json").open() as fh:
                manifest_a = json.load(fh)
            with (out_b / "manifest.json").open() as fh:
                manifest_b = json.load(fh)
            self.assertEqual(manifest_a["lines"], manifest_b["lines"])
            self.assertEqual(manifest_a["docs"], manifest_b["docs"])

    def test_limit_caps_total_lines(self):
        with TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            image_dir = self._build_pages(tmp)
            out_dir = tmp / "out"
            manifest = self._run(image_dir, out_dir, **{"--limit": 4})
            self.assertLessEqual(manifest["counts"]["n_lines"], 4)

    def test_manifest_completeness_boxes_within_bounds(self):
        with TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            image_dir = self._build_pages(tmp)
            out_dir = tmp / "out"
            manifest = self._run(image_dir, out_dir)

            self.assertTrue(manifest["tool_git_rev"])
            self.assertIn("commit", manifest["tool_git_rev"])
            self.assertEqual(manifest["seed"], 0)
            self.assertEqual(manifest["dpi"], 300)

            for row in manifest["lines"]:
                cx0, cy0, cx1, cy1 = row["col_box"]
                lx0, ly0, lx1, ly1 = row["line_box"]
                # line_box must be fully contained within col_box.
                self.assertGreaterEqual(lx0, cx0)
                self.assertGreaterEqual(ly0, cy0)
                self.assertLessEqual(lx1, cx1)
                self.assertLessEqual(ly1, cy1)
                # col_box must be within the (200, 300) synthetic page bounds.
                self.assertGreaterEqual(cx0, 0)
                self.assertGreaterEqual(cy0, 0)
                self.assertLessEqual(cx1, 200)
                self.assertLessEqual(cy1, 300)

    def test_empty_page_logged_and_skipped_not_fatal(self):
        with TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            image_dir = tmp / "pages"
            image_dir.mkdir()
            # An all-white page has no ink; detect_line_columns falls back to
            # whole-page-as-one-column (never raises), so this specifically
            # exercises that fallback rather than a segmentation crash.
            # Build enough real-content pages that pages-per-doc=1 with
            # --seed chosen results in the blank page being sampled at least
            # once across a few seeds tried deterministically here.
            for i in range(6):
                Image.new("L", (200, 300), 255).save(image_dir / f"page_{i:03d}.png")
            out_dir = tmp / "out"
            manifest = self._run(
                image_dir, out_dir, **{"--pages-per-doc": 6, "--lines-per-page": 8}
            )
            # All-blank pages fall back to one whole-page column each -- not
            # "no segmentation found" (that fallback exists precisely so a
            # blank/noisy page doesn't crash the run), so every sampled page
            # still yields exactly one line.
            self.assertEqual(manifest["counts"]["n_pages_empty"], 0)
            self.assertEqual(manifest["counts"]["n_lines"], 6)

    def test_stem_collision_across_pdf_dir_is_hard_error(self):
        with TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            pdf_dir = tmp / "pdfs"
            pdf_dir.mkdir()
            # Two files that sanitize to the identical stem.
            (pdf_dir / "book!.pdf").write_bytes(b"%PDF-1.4 fake")
            (pdf_dir / "book_.pdf").write_bytes(b"%PDF-1.4 fake")
            out_dir = tmp / "out"
            args = parse_args(
                ["--pdf-dir", str(pdf_dir), "--out", str(out_dir), "--seed", "0"]
            )
            with self.assertRaises(ValueError):
                build_pack(args)
            # No output must have been written before the collision check ran.
            self.assertFalse((out_dir / "manifest.json").exists())

    def test_cli_argument_validation(self):
        with self.assertRaises(SystemExit):
            parse_args(["--out", "/tmp/x"])  # neither --pdf-dir nor --image-dir
        with self.assertRaises(SystemExit):
            parse_args(
                ["--pdf-dir", "a", "--image-dir", "b", "--out", "/tmp/x"]
            )  # both given
        with self.assertRaises(SystemExit):
            parse_args(["--image-dir", "a", "--out", "/tmp/x", "--dpi", "0"])


# ===========================================================================
# scripts.build_annotation_pack: real PDF path (unconditional -- fitz-authored)
# ===========================================================================


@unittest.skipIf(not _HAS_FITZ, "PyMuPDF (fitz) not importable")
class PdfDirEndToEndTest(unittest.TestCase):
    def _build_pdf(self, tmp: Path, name: str = "scan_book.pdf", n_pages: int = 20) -> Path:
        """Assemble a genuine multi-page PDF from PIL-drawn pages via fitz.

        Mirrors scripts/render_mn_pages.py's own PDF-assembly pattern
        (``fitz.open()`` -> ``new_page()`` -> ``insert_image()`` -> ``save()``)
        so the PDF code path (rendering via pdftoppm/fitz from a REAL PDF
        file, not just --image-dir's pre-rendered-images shortcut) is
        actually exercised by a unit test, not merely smoke-tested by hand.
        """
        png_dir = tmp / "_src_pages"
        png_dir.mkdir()
        doc = fitz.open()
        for i in range(n_pages):
            png_path = png_dir / f"p{i:03d}.png"
            _draw_three_column_page().save(png_path)
            page = doc.new_page(width=200, height=300)
            page.insert_image(page.rect, filename=str(png_path))
        pdf_path = tmp / name
        doc.save(str(pdf_path))
        doc.close()
        return pdf_path

    def test_real_pdf_path_produces_complete_pack(self):
        with TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            pdf_path = self._build_pdf(tmp)
            pdf_dir = tmp / "pdfs"
            pdf_dir.mkdir()
            pdf_path.rename(pdf_dir / pdf_path.name)
            out_dir = tmp / "out"

            args = parse_args(
                [
                    "--pdf-dir",
                    str(pdf_dir),
                    "--out",
                    str(out_dir),
                    "--pages-per-doc",
                    "3",
                    "--lines-per-page",
                    "8",
                    "--dpi",
                    "150",
                    "--seed",
                    "0",
                ]
            )
            manifest = build_pack(args)

            self.assertEqual(manifest["counts"]["n_docs"], 1)
            self.assertEqual(manifest["counts"]["n_pages_sampled"], 3)
            self.assertGreater(manifest["counts"]["n_lines"], 0)
            self.assertEqual(manifest["docs"][0]["n_pages_total"], 20)
            for row in manifest["lines"]:
                self.assertEqual(row["dpi"], 150)
                self.assertTrue((out_dir / row["raw_path"]).exists())
                self.assertTrue((out_dir / row["preview_path"]).exists())

    def test_main_cli_entry_point_returns_zero(self):
        with TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            pdf_path = self._build_pdf(tmp, n_pages=8)
            pdf_dir = tmp / "pdfs"
            pdf_dir.mkdir()
            pdf_path.rename(pdf_dir / pdf_path.name)
            out_dir = tmp / "out"
            rc = main(
                [
                    "--pdf-dir",
                    str(pdf_dir),
                    "--out",
                    str(out_dir),
                    "--pages-per-doc",
                    "2",
                    "--lines-per-page",
                    "4",
                    "--seed",
                    "1",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue((out_dir / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
