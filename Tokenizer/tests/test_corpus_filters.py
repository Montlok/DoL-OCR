# -*- coding: utf-8 -*-

"""Tests for corpus-wide dedup (Deduper) and quality filtering (quality_ok)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from Tokenizer.tools.corpus_filters import (
    Deduper,
    QualityConfig,
    exact_key,
    quality_ok,
    repeated_line_fraction,
    script_ratio,
    simhash,
)

_MN = "ᠮᠣᠩᠭᠣᠯ ᠤᠯᠤᠰ ᠤᠨ ᠵᠠᠰᠠᠭ ᠤᠨ ᠭᠠᠵᠠᠷ ᠮᠡᠳᠡᠭᠡ "


class DeduperTest(unittest.TestCase):
    def test_exact_duplicate_dropped(self):
        d = Deduper()
        doc = _MN * 40
        self.assertTrue(d.seen(doc))
        self.assertFalse(d.seen(doc))  # verbatim repost

    def test_whitespace_only_change_is_exact_dup(self):
        d = Deduper()
        doc = _MN * 40
        self.assertTrue(d.seen(doc))
        self.assertFalse(d.seen("  " + doc.replace(" ", "  ") + "\n"))

    def test_near_duplicate_long_doc_dropped(self):
        d = Deduper(thresh=3)
        base = _MN * 60
        self.assertTrue(d.seen(base))
        self.assertFalse(d.seen(base + "ᠨᠡᠮᠡᠯᠲᠡ ᠮᠡᠳᠡᠭᠡ ᠨᠢᠭᠡ ᠬᠣᠶᠠᠷ"))

    def test_near_duplicate_non_mongolian_long_doc_dropped(self):
        d = Deduper(thresh=3)
        base = "the quick brown fox jumps over the lazy dog " * 30
        self.assertTrue(d.seen(base))
        self.assertFalse(d.seen(base + "small appendix"))

    def test_distinct_long_docs_kept(self):
        d = Deduper(thresh=3)
        a = _MN * 60
        b = "ᠦᠪᠦᠷ ᠮᠣᠩᠭᠣᠯ ᠰᠤᠷᠭᠠᠭᠤᠯᠢ ᠰᠤᠷᠤᠭᠴᠢ ᠪᠠᠭᠰᠢ " * 60
        self.assertTrue(d.seen(a))
        self.assertTrue(d.seen(b))

    def test_short_boilerplate_stubs_not_merged(self):
        # distinct short stubs that share a long masthead must survive
        d = Deduper(thresh=3, min_simhash_words=120)
        masthead = "ᠦᠪᠦᠷ ᠮᠣᠩᠭᠤᠯ ᠦᠪᠡᠷᠳᠡᠭᠡᠨ ᠵᠠᠰᠠᠬᠤ ᠣᠷᠣᠨ " * 8
        s1 = "ᠤᠷᠤᠭ ᠭᠡᠷ ᠪᠦᠯᠢ " + masthead
        s2 = "ᠦᠢᠯᠡᠳᠪᠦᠷᠢ ᠬᠤᠳᠠᠯᠳᠤᠭ᠎ᠠ " + masthead
        self.assertTrue(d.seen(s1))
        self.assertTrue(d.seen(s2))

    def test_persistence_round_trip(self):
        d = Deduper(thresh=2, min_simhash_words=50)
        doc = _MN * 60
        d.seen(doc)
        d2 = Deduper.from_dict(json.loads(json.dumps(d.to_dict())))
        self.assertEqual(d2.thresh, 2)
        self.assertEqual(d2.min_simhash_words, 50)
        self.assertFalse(d2.seen(doc))  # remembered across reload
        self.assertTrue(d2.seen("ᠰᠢᠨ᠎ᠡ ᠮᠡᠳᠡᠭᠡ " * 60))

    def test_helpers(self):
        self.assertEqual(exact_key(" a  b "), exact_key("a b"))
        self.assertIsInstance(simhash(_MN * 30), int)


class QualityTest(unittest.TestCase):
    def test_too_short_rejected(self):
        ok, reason = quality_ok("ᠪᠠ", QualityConfig(min_chars=40))
        self.assertFalse(ok)
        self.assertEqual(reason, "too_short")

    def test_script_purity_enforced(self):
        cfg = QualityConfig(min_chars=1, script="mongolian", min_script_ratio=0.6)
        ok, _ = quality_ok(_MN * 10, cfg)
        self.assertTrue(ok)
        bad, reason = quality_ok("the quick brown fox " * 10, cfg)
        self.assertFalse(bad)
        self.assertEqual(reason, "script_impurity")

    def test_repeated_lines_rejected(self):
        text = "\n".join([_MN] * 10)
        cfg = QualityConfig(min_chars=1, max_repeated_line_fraction=0.4)
        ok, reason = quality_ok(text, cfg)
        self.assertFalse(ok)
        self.assertEqual(reason, "repeated_lines")

    def test_symbol_spam_rejected(self):
        cfg = QualityConfig(min_chars=1, max_symbol_to_word_ratio=0.5)
        ok, reason = quality_ok("# # # # … … • • | |", cfg)
        self.assertFalse(ok)
        self.assertEqual(reason, "symbol_spam")

    def test_boilerplate_blocklist(self):
        cfg = QualityConfig(min_chars=1, script="")
        ok, reason = quality_ok(_MN * 20 + " 蒙ICP备12345号", cfg)
        self.assertFalse(ok)
        self.assertEqual(reason, "boilerplate")

    def test_chinese_doc_not_dropped_by_word_len(self):
        # CJK has no word spaces; the word-length heuristic must not fire.
        cfg = QualityConfig(min_chars=10, script="zh", min_script_ratio=0.5)
        zh = "今天天气很好我们一起去公园散步看花喝茶聊天非常开心。" * 5
        ok, reason = quality_ok(zh, cfg)
        self.assertTrue(ok, msg=f"unexpected drop: {reason}")

    def test_english_word_len_still_enforced(self):
        cfg = QualityConfig(min_chars=1, max_mean_word_len=10.0)
        ok, reason = quality_ok("supercalifragilistic " * 10, cfg)
        self.assertFalse(ok)
        self.assertEqual(reason, "word_len")

    def test_clean_doc_passes(self):
        cfg = QualityConfig(min_chars=10, script="mongolian", min_script_ratio=0.5)
        ok, reason = quality_ok(_MN * 30, cfg)
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_metric_helpers(self):
        self.assertAlmostEqual(script_ratio("ᠠᠠabc", "mongolian"), 2 / 5)
        self.assertAlmostEqual(repeated_line_fraction("a\na\nb"), 1 / 3)


class CorpusCleanCliTest(unittest.TestCase):
    def test_end_to_end_dedup_and_filter(self):
        from Tokenizer.tools.corpus_clean import run

        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "src.jsonl")
            out = os.path.join(d, "out.jsonl")
            doc = _MN * 40
            rows = [
                {"text": doc},            # kept
                {"text": doc},            # exact dup -> dropped
                {"text": "ᠪᠠ"},          # too short -> dropped
                {"text": "ᠨᠢᠭᠡ ᠬᠣᠶᠠᠷ ᠭᠤᠷᠪᠠ ᠳᠦᠷᠪᠡ " * 40},  # distinct -> kept
            ]
            with open(src, "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")

            from types import SimpleNamespace

            args = SimpleNamespace(
                inputs=[src],
                out=out,
                text_key="text",
                thresh=3,
                min_chars=40,
                script="mongolian",
                min_script_ratio=0.5,
                report=False,
            )
            stats = run(args)
            self.assertEqual(stats["total"], 4)
            self.assertEqual(stats["kept"], 2)
            self.assertEqual(stats["drop_duplicate"], 1)
            self.assertEqual(stats["drop_too_short"], 1)
            with open(out, encoding="utf-8") as fh:
                self.assertEqual(sum(1 for _ in fh), 2)


if __name__ == "__main__":
    unittest.main()
