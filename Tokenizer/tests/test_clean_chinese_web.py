# -*- coding: utf-8 -*-

"""Tests for the strict Chinese web cleaner (clean_chinese_web)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from Tokenizer.tools.clean_chinese_web import (
    CleanConfig,
    char_ngram_dup_fraction,
    chinese_ratio,
    clean_doc,
    digit_ratio,
    discover_units,
    normalize_text,
    repeated_line_fraction,
    run,
    sentence_punct_ratio,
    strip_boilerplate_lines,
    top_char_ngram_fraction,
)

# A clean, natural Chinese paragraph (>200 chars) that should always pass.
_GOOD = (
    "今天上午，市政府召开新闻发布会，介绍了今年以来全市经济社会发展的总体情况。"
    "有关负责人表示，下一步将继续推动产业升级，改善民生福祉，让群众有更多获得感。"
    "会议还就环境保护、教育医疗等方面的工作作了具体部署。"
    "与会代表围绕乡村振兴、科技创新和城市治理等议题展开了深入讨论，并提出了许多建设性意见。"
    "据介绍，相关部门将抓紧研究制定配套政策，确保各项任务落到实处，努力推动高质量发展不断取得新成效。"
    "此外，发布会还通报了近期民生实事项目的进展情况，回应了社会各界普遍关心的热点问题。"
)


class NormalizeTest(unittest.TestCase):
    def test_nfkc_fullwidth_folded(self):
        self.assertEqual(normalize_text("ＡＢＣ１２３"), "ABC123")

    def test_zero_width_and_control_removed(self):
        self.assertEqual(normalize_text("中\u200b文\x07字"), "中文字")

    def test_whitespace_canonicalized(self):
        self.assertEqual(normalize_text("a   b\r\nc \n\n\n d"), "a b\nc\n\nd")

    def test_empty(self):
        self.assertEqual(normalize_text(""), "")
        self.assertEqual(normalize_text(None), "")


class BoilerplateTest(unittest.TestCase):
    def test_footer_lines_stripped(self):
        text = (
            "正文第一段内容在这里。\n"
            "免责声明：本站转载不代表观点。\n"
            "责任编辑：张三\n"
            "京ICP备12345678号"
        )
        out = strip_boilerplate_lines(text)
        self.assertEqual(out, "正文第一段内容在这里。")

    def test_nav_and_player_junk_stripped(self):
        text = "真正的文章内容。\nplay stop mute max volume\n扫一扫关注公众号"
        out = strip_boilerplate_lines(text)
        self.assertEqual(out, "真正的文章内容。")

    def test_body_sentence_not_overstripped(self):
        # mentions 版权 inside a real sentence body but not as a footer line
        text = "他在文章中讨论了知识产权与版权保护的重要意义和深远影响。"
        self.assertEqual(strip_boilerplate_lines(text), text)


class MetricTest(unittest.TestCase):
    def test_chinese_ratio(self):
        self.assertAlmostEqual(chinese_ratio("中文abc"), 2 / 5)
        self.assertEqual(chinese_ratio("   "), 0.0)

    def test_digit_ratio(self):
        self.assertAlmostEqual(digit_ratio("中12"), 2 / 3)

    def test_repeated_line_fraction(self):
        self.assertAlmostEqual(repeated_line_fraction("a\na\nb"), 1 / 3)

    def test_char_ngram_dup_fraction(self):
        # "abab" 2-grams: ab,ba,ab -> one dup of three
        self.assertAlmostEqual(char_ngram_dup_fraction("abab", 2), 1 / 3)

    def test_top_char_ngram_fraction(self):
        self.assertGreater(top_char_ngram_fraction("啦啦啦啦啦啦", 2), 0.8)

    def test_sentence_punct_ratio(self):
        self.assertGreater(sentence_punct_ratio("你好。再见。"), 0.0)
        self.assertEqual(sentence_punct_ratio("无标点纯汉字串"), 0.0)


class CleanDocTest(unittest.TestCase):
    def setUp(self):
        self.cfg = CleanConfig(min_chars=80)

    def test_good_doc_kept(self):
        cleaned, reason = clean_doc(_GOOD, self.cfg)
        self.assertIsNotNone(cleaned, msg=reason)
        self.assertEqual(reason, "")

    def test_too_short_dropped(self):
        cleaned, reason = clean_doc("很短。", self.cfg)
        self.assertIsNone(cleaned)
        self.assertEqual(reason, "too_short")

    def test_low_chinese_dropped(self):
        cleaned, reason = clean_doc("hello world " * 40, self.cfg)
        self.assertIsNone(cleaned)
        self.assertEqual(reason, "low_chinese")

    def test_spam_dropped(self):
        spam = _GOOD + "\n代办各类发票，加微信：abc12345 办理"
        cleaned, reason = clean_doc(spam, self.cfg)
        self.assertIsNone(cleaned)
        self.assertEqual(reason, "spam")

    def test_digit_heavy_dropped(self):
        # >55% Han but >25% digits, so it passes the Chinese gate and trips digit.
        nums = "公司今年利润增长12和34与56共计78。" * 12
        cleaned, reason = clean_doc(nums, self.cfg)
        self.assertIsNone(cleaned)
        self.assertEqual(reason, "digit_heavy")

    def test_no_sentences_dropped(self):
        nav = "新闻体育财经科技娱乐军事汽车房产教育旅游健康历史美食时尚游戏" * 8
        cleaned, reason = clean_doc(nav, self.cfg)
        self.assertIsNone(cleaned)
        self.assertEqual(reason, "no_sentences")

    def test_repetitive_dropped(self):
        rep = "买买买买买买买买。" * 60
        cleaned, reason = clean_doc(rep, self.cfg)
        self.assertIsNone(cleaned)
        self.assertIn(reason, {"repetitive", "ngram_spam", "repeated_lines"})

    def test_garbled_dropped(self):
        garbled = _GOOD + "\ufffd\ufffd\ufffd\ufffd\ufffd"
        cleaned, reason = clean_doc(garbled, self.cfg)
        self.assertIsNone(cleaned)
        self.assertEqual(reason, "garbled")

    def test_boilerplate_stripped_but_body_kept(self):
        doc = _GOOD + "\n责任编辑：李四\n版权所有 未经授权不得转载"
        cleaned, reason = clean_doc(doc, self.cfg)
        self.assertIsNotNone(cleaned, msg=reason)
        self.assertNotIn("责任编辑", cleaned)
        self.assertIn("市政府", cleaned)


class RunCliTest(unittest.TestCase):
    def test_discover_units_jsonl(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.jsonl")
            open(p, "w").close()
            units = discover_units([p], "Content")
            self.assertEqual(len(units), 1)
            self.assertEqual(units[0].kind, "jsonl")

    def test_end_to_end_filter_and_dedup(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "src.jsonl")
            out_dir = os.path.join(d, "out")
            rows = [
                {"Content": _GOOD},                 # kept
                {"Content": _GOOD},                 # exact dup -> dropped
                {"Content": "太短"},                # too_short
                {"Content": "hello world " * 40},   # low_chinese
                {"Content": "春天来了，公园里的樱花竞相开放，吸引了众多市民和游客前来观赏拍照。"
                 "孩子们在草坪上奔跑嬉戏，老人们在长椅上悠闲地晒着太阳聊着家常。"
                 "湖面上荡漾着几只小船，岸边的柳枝随风轻轻摆动，构成一幅生机盎然的画面。"
                 "这样的好天气让人心情格外舒畅，也让整座城市充满了温暖与活力的气息。"},  # distinct -> kept
            ]
            with open(src, "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")

            args = SimpleNamespace(
                inputs=[src], out_dir=out_dir, text_key="Content",
                workers=1, batch_size=10, thresh=3, no_near_dup=False,
                min_chars=80, min_chinese_ratio=0.55, keep_spam=False,
                limit_units=0, report=False,
            )
            stats = run(args)
            self.assertEqual(stats["kept"], 2)
            self.assertEqual(stats["drop_duplicate"], 1)
            self.assertEqual(stats["drop_too_short"], 1)
            self.assertEqual(stats["drop_low_chinese"], 1)

            shards = [f for f in os.listdir(out_dir) if f.endswith(".jsonl")]
            kept = 0
            for s in shards:
                with open(os.path.join(out_dir, s), encoding="utf-8") as fh:
                    kept += sum(1 for _ in fh)
            self.assertEqual(kept, 2)

    def test_resume_skips_done_units(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "src.jsonl")
            out_dir = os.path.join(d, "out")
            with open(src, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"Content": _GOOD}, ensure_ascii=False) + "\n")
            base = dict(
                inputs=[src], out_dir=out_dir, text_key="Content",
                workers=1, batch_size=10, thresh=3, no_near_dup=False,
                min_chars=80, min_chinese_ratio=0.55, keep_spam=False,
                limit_units=0, report=False,
            )
            run(SimpleNamespace(**base))
            stats2 = run(SimpleNamespace(**base))  # second pass: unit already done
            self.assertEqual(stats2["kept"], 1)  # not double-counted


if __name__ == "__main__":
    unittest.main()
