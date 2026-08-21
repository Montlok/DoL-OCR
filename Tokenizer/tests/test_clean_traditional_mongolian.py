import json
import tempfile
import unittest
from pathlib import Path

from Tokenizer.tools import clean_traditional_mongolian as cleaner


MONG = "ᠮᠣᠩᠭᠣᠯ ᠬᠡᠯᠡ ᠪᠢᠴᠢᠭ ᠰᠤᠳᠤᠯᠤᠯ ᠲᠡᠦᠬᠡ ᠰᠣᠶᠣᠯ ᠨᠣᠮ ᠤᠩᠰᠢᠯᠭ᠎ᠠ"


class CleanTraditionalMongolianTests(unittest.TestCase):
    def test_normalize_removes_page_markers_and_decorations(self):
        text = "||page1#   " + MONG + "\n★★★\npage 2   " + MONG
        cleaned = cleaner.normalize_text(text)
        self.assertNotIn("page1", cleaned.lower())
        self.assertNotIn("★★★", cleaned)
        self.assertEqual(cleaned.count(MONG), 2)

    def test_quality_rejects_low_mongolian_ratio(self):
        reason = cleaner.quality_reason(
            "mostly latin text " * 20,
            min_chars=80,
            min_mong_ratio=0.55,
            max_symbol_ratio=0.35,
            max_repeated_line_fraction=0.25,
            max_latin_ratio=0.20,
        )
        self.assertIn(reason, {"script_impurity", "latin_noise"})

    def test_quality_rejects_cyrillic_mongolian(self):
        cyrillic = "Монгол кирилл өгүүлбэр давамгай байна " * 4
        reason = cleaner.quality_reason(
            (MONG + " ") * 2 + cyrillic,
            min_chars=80,
            min_mong_ratio=0.10,
            max_symbol_ratio=0.35,
            max_repeated_line_fraction=0.25,
            max_latin_ratio=0.80,
            max_cyrillic_ratio=0.05,
        )
        self.assertEqual(reason, "cyrillic_noise")

    def test_normalize_removes_mengguyu_boilerplate(self):
        boilerplate = (
            "1᠂ ᠠᠭᠤᠯᠭ᠎ᠠ ᠨᠢ ᠰᠦᠯᠵᠢᠶ᠎ᠡ ᠡᠴᠡ ᠢᠷᠡᠯᠲᠡ ᠲᠡᠶ ᠂ "
            "ᠬᠡᠷᠪᠡ ᠡᠷᠬᠡ ᠳᠦ ᠬᠠᠯᠳᠠᠭᠰᠠᠨ ᠬᠠᠷᠢᠴᠠᠭ᠎ᠠ ᠪᠠᠢᠪᠠᠯ "
            "ᠪᠢᠳᠡ ᠲᠡᠷᠡ ᠳᠠᠷᠤᠢ ᠬᠠᠰᠤᠨ᠎ᠠ᠃"
        )
        cleaned = cleaner.normalize_text(MONG + "\n" + boilerplate + "\n" + MONG)
        self.assertNotIn("ᠰᠦᠯᠵᠢᠶ᠎ᠡ ᠡᠴᠡ ᠢᠷᠡᠯᠲᠡ", cleaned)
        self.assertEqual(cleaned.count(MONG), 2)

    def test_split_long_records(self):
        text = "\n\n".join([MONG * 20 for _ in range(10)])
        chunks = list(cleaner.split_text(text, max_chars=500))
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 800 for chunk in chunks))

    def test_run_deduplicates_after_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inp = root / "in.jsonl"
            out = root / "out.jsonl"
            report = root / "report.json"
            duplicated = "page1 " + (MONG + " ") * 8
            inp.write_text(
                json.dumps({"text": duplicated}, ensure_ascii=False)
                + "\n"
                + json.dumps({"text": duplicated.replace("page1", "page2")}, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            stats = cleaner.main(
                [
                    "--in",
                    str(inp),
                    "--out",
                    str(out),
                    "--report",
                    str(report),
                    "--max-chars",
                    "20000",
                ]
            )
            self.assertEqual(stats, 0)
            self.assertEqual(len(out.read_text(encoding="utf-8").splitlines()), 1)

    def test_run_parallel_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inp = root / "in.jsonl"
            out = root / "out.jsonl"
            report = root / "report.json"
            inp.write_text(
                "".join(
                    json.dumps({"text": f"{MONG} {i} " * 4}, ensure_ascii=False) + "\n"
                    for i in range(12)
                ),
                encoding="utf-8",
            )
            rc = cleaner.main(
                [
                    "--in",
                    str(inp),
                    "--out",
                    str(out),
                    "--report",
                    str(report),
                    "--workers",
                    "2",
                    "--batch-size",
                    "3",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertEqual(len(out.read_text(encoding="utf-8").splitlines()), 12)
            self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["settings"]["workers"], 2)


if __name__ == "__main__":
    unittest.main()
