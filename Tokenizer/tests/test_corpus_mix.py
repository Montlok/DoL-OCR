# -*- coding: utf-8 -*-

"""Regression tests for the token-weighted corpus mixer."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from Tokenizer.tools.build_corpus_mix import (
    MixConfig,
    SourceSpec,
    _emit_counts,
    compute_plan,
    emit_mix,
    measure_source,
)


# Char-proxy encoder: ~1 token per 4 chars (matches the smoke fallback).
def _proxy_encode(text: str):
    return [0] * max(len(text) // 4, 1)


def _write_jsonl(path: str, texts: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for t in texts:
            fh.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")


class ManifestTest(unittest.TestCase):
    def test_weight_normalization_not_required(self) -> None:
        cfg = MixConfig.from_raw(
            {
                "sources": [
                    {"path": "a.jsonl", "lang": "mn", "weight": 5},
                    {"path": "b.jsonl", "lang": "en", "weight": 5},
                ]
            }
        )
        self.assertEqual(len(cfg.sources), 2)
        self.assertEqual(cfg.max_epochs, 4)

    def test_missing_path_raises(self) -> None:
        with self.assertRaises(ValueError):
            SourceSpec.from_raw({"lang": "mn"})

    def test_empty_sources_raises(self) -> None:
        with self.assertRaises(ValueError):
            MixConfig.from_raw({"sources": []})

    def test_format_inference(self) -> None:
        self.assertEqual(SourceSpec.from_raw({"path": "x.parquet"}).fmt, "parquet")
        self.assertEqual(SourceSpec.from_raw({"path": "x.jsonl"}).fmt, "jsonl")
        self.assertEqual(SourceSpec.from_raw({"path": "x.txt"}).fmt, "txt")


class GlobSourceTest(unittest.TestCase):
    def test_jsonl_glob_expands(self) -> None:
        from Tokenizer.tools.build_corpus_mix import SourceSpec, _iter_source_texts

        with tempfile.TemporaryDirectory() as tmp:
            _write_jsonl(os.path.join(tmp, "a.jsonl"), ["alpha"])
            _write_jsonl(os.path.join(tmp, "b.jsonl"), ["beta"])
            texts = sorted(
                _iter_source_texts(
                    SourceSpec(path=os.path.join(tmp, "*.jsonl"), fmt="jsonl")
                )
            )
            self.assertEqual(texts, ["alpha", "beta"])

    def test_jsonl_directory_expands(self) -> None:
        from Tokenizer.tools.build_corpus_mix import SourceSpec, _iter_source_texts

        with tempfile.TemporaryDirectory() as tmp:
            _write_jsonl(os.path.join(tmp, "a.jsonl"), ["one", "two"])
            texts = sorted(_iter_source_texts(SourceSpec(path=tmp, fmt="jsonl")))
            self.assertEqual(texts, ["one", "two"])

    def test_txt_glob_expands(self) -> None:
        from Tokenizer.tools.build_corpus_mix import SourceSpec, _iter_source_texts

        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "a.txt"), "w", encoding="utf-8") as fh:
                fh.write("line-a\n")
            with open(os.path.join(tmp, "b.txt"), "w", encoding="utf-8") as fh:
                fh.write("line-b\n")
            texts = sorted(
                _iter_source_texts(
                    SourceSpec(path=os.path.join(tmp, "*.txt"), fmt="txt")
                )
            )
            self.assertEqual(texts, ["line-a", "line-b"])

    def test_txt_utf16le_decoded(self) -> None:
        """UTF-16LE txt drops (e.g. the 1000-traditional Mongolian corpus)
        must decode to real text, not utf-8 replacement garbage."""
        from Tokenizer.tools.build_corpus_mix import SourceSpec, _iter_source_texts

        mong = "\u182d\u1824\u1837\u182d\u1824"  # real Mongolian codepoints
        with tempfile.TemporaryDirectory() as tmp:
            # Repeated BOM + content, exactly the shape of the real corpus files.
            payload = ("\ufeff" * 3) + mong + "\r\n" + mong
            with open(os.path.join(tmp, "m.txt"), "wb") as fh:
                fh.write(payload.encode("utf-16-le"))
            texts = list(
                _iter_source_texts(
                    SourceSpec(path=os.path.join(tmp, "m.txt"), fmt="txt")
                )
            )
            self.assertEqual(texts, [mong, mong])
            self.assertNotIn("\ufffd", "".join(texts))
            self.assertNotIn("\ufeff", "".join(texts))

    def test_txt_gb18030_decoded(self) -> None:
        from Tokenizer.tools.build_corpus_mix import SourceSpec, _iter_source_texts

        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "g.txt"), "wb") as fh:
                fh.write("中文测试内容一二三四".encode("gb18030"))
            texts = list(
                _iter_source_texts(
                    SourceSpec(path=os.path.join(tmp, "g.txt"), fmt="txt")
                )
            )
            self.assertEqual(texts, ["中文测试内容一二三四"])

    def test_detect_encoding_tiny_sample(self) -> None:
        from Tokenizer.tools.build_corpus_mix import _detect_text_encoding

        # 1-3 byte samples must not be mis-detected as utf-8 via boundary trim.
        self.assertEqual(_detect_text_encoding(b"\xff"), "gb18030")
        self.assertEqual(_detect_text_encoding(b"\xc3"), "gb18030")
        self.assertEqual(_detect_text_encoding(b"hi"), "utf-8")

    def test_missing_glob_raises(self) -> None:
        from Tokenizer.tools.build_corpus_mix import SourceSpec, _iter_source_texts

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                list(
                    _iter_source_texts(
                        SourceSpec(
                            path=os.path.join(tmp, "nope-*.jsonl"), fmt="jsonl"
                        )
                    )
                )


class CleaningTest(unittest.TestCase):
    def test_strip_url_lines(self) -> None:
        from Tokenizer.tools.build_corpus_mix import _iter_source_texts

        with tempfile.TemporaryDirectory() as tmp:
            doc = "real content here\nhttps://example.com/page\nmore content"
            _write_jsonl(os.path.join(tmp, "a.jsonl"), [doc])
            spec = SourceSpec(
                path=os.path.join(tmp, "a.jsonl"),
                fmt="jsonl",
                strip_url_lines=True,
            )
            out = list(_iter_source_texts(spec))
            self.assertEqual(out, ["real content here\nmore content"])

    def test_keep_zh_lines_drops_english(self) -> None:
        from Tokenizer.tools.build_corpus_mix import _iter_source_texts

        with tempfile.TemporaryDirectory() as tmp:
            doc = "跳到主要内容\nSkip to main content\n香港政府一站通"
            _write_jsonl(os.path.join(tmp, "a.jsonl"), [doc])
            spec = SourceSpec(
                path=os.path.join(tmp, "a.jsonl"),
                fmt="jsonl",
                keep_lines="zh",
            )
            out = list(_iter_source_texts(spec))
            self.assertEqual(out, ["跳到主要内容\n香港政府一站通"])

    def test_min_chars_drops_short_docs(self) -> None:
        from Tokenizer.tools.build_corpus_mix import _iter_source_texts

        with tempfile.TemporaryDirectory() as tmp:
            _write_jsonl(os.path.join(tmp, "a.jsonl"), ["short", "a much longer doc"])
            spec = SourceSpec(
                path=os.path.join(tmp, "a.jsonl"), fmt="jsonl", min_chars=10
            )
            out = list(_iter_source_texts(spec))
            self.assertEqual(out, ["a much longer doc"])

    def test_no_cleaning_by_default(self) -> None:
        from Tokenizer.tools.build_corpus_mix import _iter_source_texts

        with tempfile.TemporaryDirectory() as tmp:
            doc = "keep\nhttps://x.com\nSkip to main content"
            _write_jsonl(os.path.join(tmp, "a.jsonl"), [doc])
            out = list(_iter_source_texts(SourceSpec(path=os.path.join(tmp, "a.jsonl"), fmt="jsonl")))
            self.assertEqual(out, [doc])

    def test_invalid_keep_lines_raises(self) -> None:
        with self.assertRaises(ValueError):
            SourceSpec.from_raw({"path": "x.jsonl", "keep_lines": "klingon"})


class EmitCountsTest(unittest.TestCase):
    def test_subsample_expectation(self) -> None:
        import random

        rng = random.Random(0)
        total = sum(_emit_counts(0.3, rng) for _ in range(10000))
        self.assertAlmostEqual(total / 10000, 0.3, delta=0.03)

    def test_upsample_expectation(self) -> None:
        import random

        rng = random.Random(0)
        total = sum(_emit_counts(2.5, rng) for _ in range(10000))
        self.assertAlmostEqual(total / 10000, 2.5, delta=0.05)

    def test_zero(self) -> None:
        import random

        self.assertEqual(_emit_counts(0.0, random.Random(0)), 0)


class PlanTest(unittest.TestCase):
    def _stats(self, tmp):
        mn = os.path.join(tmp, "mn.jsonl")
        en = os.path.join(tmp, "en.jsonl")
        # Mongolian: small (low-resource). English: large.
        _write_jsonl(mn, ["A" * 40 for _ in range(10)])  # 100 tokens total
        _write_jsonl(en, ["B" * 40 for _ in range(100)])  # 1000 tokens total
        specs = [
            SourceSpec.from_raw({"path": mn, "lang": "mn", "weight": 0.7}),
            SourceSpec.from_raw({"path": en, "lang": "en", "weight": 0.3}),
        ]
        return [measure_source(s, _proxy_encode) for s in specs]

    def test_low_resource_upsampled_with_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stats = self._stats(tmp)
            stats = compute_plan(stats, total_tokens=None, max_epochs=4)
            mn, en = stats
            # mn weight dominates so it should up-sample, capped at 4 epochs.
            self.assertGreater(mn.repeat_factor, 1.0)
            self.assertLessEqual(mn.repeat_factor, 4.0)
            # en is plentiful so it should sub-sample (<1).
            self.assertLess(en.repeat_factor, 1.0)

    def test_cap_blocks_runaway_upsampling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stats = self._stats(tmp)
            # Huge budget would demand >>4 epochs of mn; cap must hold.
            stats = compute_plan(stats, total_tokens=10_000_000, max_epochs=4)
            self.assertLessEqual(stats[0].repeat_factor, 4.0)


class EndToEndTest(unittest.TestCase):
    def test_emit_produces_text_jsonl_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mn = os.path.join(tmp, "mn.jsonl")
            en = os.path.join(tmp, "en.jsonl")
            _write_jsonl(mn, ["ᠮᠣᠩᠭᠣᠯ" * 8 for _ in range(20)])
            _write_jsonl(en, ["hello world " * 8 for _ in range(200)])
            specs = [
                SourceSpec.from_raw({"path": mn, "lang": "mn", "weight": 0.6}),
                SourceSpec.from_raw({"path": en, "lang": "en", "weight": 0.4}),
            ]
            stats = [measure_source(s, _proxy_encode) for s in specs]
            # Budget chosen so Mongolian's target stays under its 4-epoch cap,
            # letting the higher weight translate into a higher realized share.
            stats = compute_plan(stats, total_tokens=1500, max_epochs=4)
            out = os.path.join(tmp, "mix.jsonl")
            report = emit_mix(stats, out, seed=0)

            self.assertTrue(os.path.exists(out))
            with open(out, "r", encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
            self.assertTrue(rows)
            # Every emitted row is a text-only record.
            for r in rows:
                self.assertEqual(list(r.keys()), ["text"])
                self.assertTrue(r["text"].strip())

            # Report aggregates a realized language mixture summing to ~1.
            self.assertIn("mn", report["by_lang"])
            self.assertIn("en", report["by_lang"])
            self.assertAlmostEqual(sum(report["by_lang"].values()), 1.0, delta=0.02)
            # Mongolian's realized share should track its higher weight.
            self.assertGreater(report["by_lang"]["mn"], report["by_lang"]["en"])

    def test_new_source_rebalances(self) -> None:
        # Dropping in a second Mongolian source must shift realized shares
        # without any manual reconfiguration (weights are re-solved each run).
        with tempfile.TemporaryDirectory() as tmp:
            mn1 = os.path.join(tmp, "mn1.jsonl")
            en = os.path.join(tmp, "en.jsonl")
            _write_jsonl(mn1, ["A" * 40 for _ in range(10)])
            _write_jsonl(en, ["B" * 40 for _ in range(100)])

            def shares(specs):
                stats = [measure_source(s, _proxy_encode) for s in specs]
                stats = compute_plan(stats, total_tokens=None, max_epochs=4)
                out = os.path.join(tmp, "o.jsonl")
                rep = emit_mix(stats, out, seed=1)
                return rep["by_lang"]

            base = shares(
                [
                    SourceSpec.from_raw({"path": mn1, "lang": "mn", "weight": 0.5}),
                    SourceSpec.from_raw({"path": en, "lang": "en", "weight": 0.5}),
                ]
            )
            mn2 = os.path.join(tmp, "mn2.jsonl")
            _write_jsonl(mn2, ["C" * 40 for _ in range(50)])
            grown = shares(
                [
                    SourceSpec.from_raw({"path": mn1, "lang": "mn", "weight": 0.25}),
                    SourceSpec.from_raw({"path": mn2, "lang": "mn", "weight": 0.25}),
                    SourceSpec.from_raw({"path": en, "lang": "en", "weight": 0.5}),
                ]
            )
            # Both runs target mn~=en, realized shares stay balanced.
            self.assertAlmostEqual(grown["mn"], base["mn"], delta=0.15)

    def test_sources_interleaved_not_blocked(self) -> None:
        # Two sources both larger than a small shuffle buffer must interleave in
        # the output, not appear as two solid contiguous runs.
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.jsonl")
            b = os.path.join(tmp, "b.jsonl")
            _write_jsonl(a, [f"AAA-{i}" for i in range(500)])
            _write_jsonl(b, [f"BBB-{i}" for i in range(500)])
            specs = [
                SourceSpec.from_raw({"path": a, "lang": "a", "weight": 0.5}),
                SourceSpec.from_raw({"path": b, "lang": "b", "weight": 0.5}),
            ]
            stats = [measure_source(s, _proxy_encode) for s in specs]
            stats = compute_plan(stats, total_tokens=None, max_epochs=4)
            out = os.path.join(tmp, "mix.jsonl")
            # Tiny buffer forces several mid-stream flushes.
            emit_mix(stats, out, seed=0, buffer_size=50)
            with open(out, "r", encoding="utf-8") as fh:
                tags = [json.loads(line)["text"][:3] for line in fh if line.strip()]
            # The first 100 emitted docs should contain BOTH sources, which is
            # only true if sources interleave rather than flush one at a time.
            head = set(tags[:100])
            self.assertEqual(head, {"AAA", "BBB"})


class SignatureTest(unittest.TestCase):
    def _specs(self, a, b):
        return MixConfig.from_raw(
            {
                "sources": [
                    {"path": a, "lang": "mn", "weight": 0.5},
                    {"path": b, "lang": "en", "weight": 0.5},
                ]
            }
        )

    def test_signature_changes_when_source_changes(self) -> None:
        from Tokenizer.tools.build_corpus_mix import _compute_signature

        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.jsonl")
            b = os.path.join(tmp, "b.jsonl")
            _write_jsonl(a, ["one"])
            _write_jsonl(b, ["two"])
            mbytes = b'{"x":1}'
            cfg = self._specs(a, b)
            sig1 = _compute_signature(mbytes, cfg, None)
            sig2 = _compute_signature(mbytes, cfg, None)
            self.assertEqual(sig1, sig2)  # stable when nothing changes
            # Rewriting a source (new size/mtime) flips the signature.
            _write_jsonl(a, ["one", "three", "four"])
            sig3 = _compute_signature(mbytes, cfg, None)
            self.assertNotEqual(sig1, sig3)

    def test_signature_changes_with_manifest(self) -> None:
        from Tokenizer.tools.build_corpus_mix import _compute_signature

        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.jsonl")
            b = os.path.join(tmp, "b.jsonl")
            _write_jsonl(a, ["one"])
            _write_jsonl(b, ["two"])
            cfg = self._specs(a, b)
            self.assertNotEqual(
                _compute_signature(b'{"v":1}', cfg, None),
                _compute_signature(b'{"v":2}', cfg, None),
            )

    def test_skip_if_fresh_roundtrip(self) -> None:
        # Full CLI-style round trip: first build writes a signed report; a second
        # build with --skip-if-fresh is a no-op (output mtime unchanged); editing
        # a source invalidates the signature and forces a rebuild.
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.jsonl")
            b = os.path.join(tmp, "b.jsonl")
            _write_jsonl(a, [f"alpha-{i}" for i in range(50)])
            _write_jsonl(b, [f"beta-{i}" for i in range(50)])
            manifest = os.path.join(tmp, "m.json")
            with open(manifest, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "sources": [
                            {"path": a, "lang": "mn", "weight": 0.5},
                            {"path": b, "lang": "en", "weight": 0.5},
                        ]
                    },
                    f,
                )
            out = os.path.join(tmp, "mix.jsonl")
            report = os.path.join(tmp, "report.json")
            root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            env = {**os.environ, "PYTHONPATH": root}
            cmd = [
                sys.executable, "-m", "Tokenizer.tools.build_corpus_mix",
                "--manifest", manifest, "--output", out, "--report", report,
                "--skip-if-fresh",
            ]
            subprocess.run(cmd, check=True, cwd=root, env=env)
            mtime1 = os.stat(out).st_mtime_ns

            # Fresh rerun: must skip and leave the output untouched.
            res = subprocess.run(
                cmd, check=True, cwd=root, env=env, capture_output=True, text=True
            )
            self.assertIn("skipped", res.stdout)
            self.assertEqual(os.stat(out).st_mtime_ns, mtime1)

            # Edit a source -> signature mismatch -> rebuild (output rewritten).
            _write_jsonl(a, [f"alpha-{i}" for i in range(80)])
            res2 = subprocess.run(
                cmd, check=True, cwd=root, env=env, capture_output=True, text=True
            )
            self.assertNotIn("skipped", res2.stdout)


if __name__ == "__main__":
    unittest.main()
