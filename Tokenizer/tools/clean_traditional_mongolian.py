#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Second-pass cleaner for Unicode Traditional Mongolian JSONL corpora.

The generic corpus_clean.py stage removes obvious low-quality documents and
global duplicates. This pass is stricter and Mongolian-specific: it removes
PDF/OCR page markers, decorative table-of-contents noise, control/PUA residue,
boilerplate lines, repeated blank/space runs, then splits very large book-sized
records into training-friendly chunks before running exact + near dedup again.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_filters import Deduper, repeated_line_fraction, script_ratio, symbol_to_word_ratio


MONG_RE = re.compile(r"[\u1800-\u18AF\U00011660-\U0001167F]")
CYR_RE = re.compile(r"[\u0400-\u04FF]")
PUA_RE = re.compile(r"[\ue000-\uf8ff]")
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SPACE_RE = re.compile(r"[\t\r\f\v\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+")
RUN_SPACE_RE = re.compile(r" {2,}")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
PAGE_TOKEN_RE = re.compile(r"(?i)(?:^|\s|\|)#?\s*page\s*[-_#:]?\s*\d+\s*#?(?=\s|$)")
LEADING_PAGE_RE = re.compile(r"(?i)^\s*(?:\|+\s*)?#?\s*page\s*[-_#:]?\s*\d+\s*#?\s*")
DECORATIVE_RE = re.compile(r"^[\s★☆*#=_\-—–·•|<>《》［\]\[\]（）()【】{}:：;；,.，。、!?！？~～]+$")
HEADING_ONLY_RE = re.compile(r"^(?:ᠭᠠᠷᠴᠠᠭ|ᠲᠣᠳᠣᠷᠬᠠᠶᠢᠯᠠᠯᠲᠠ)\s*(?:\(\)|（\）)?$")
BOILERPLATE_SNIPPETS = (
    "Copyright",
    "All Rights Reserved",
    "版权所有",
    "蒙ICP备",
    "京ICP备",
    "ICP备",
    "未经授权",
    "ᠠᠭᠤᠯᠭ᠎ᠠ ᠨᠢ ᠰᠦᠯᠵᠢᠶ᠎ᠡ ᠡᠴᠡ ᠢᠷᠡᠯᠲᠡ",
    "ᠡᠷᠬᠡ ᠳᠦ ᠬᠠᠯᠳᠠᠭᠰᠠᠨ",
    "ᠪᠢᠳᠡ ᠲᠡᠷᠡ ᠳᠠᠷᠤᠢ ᠬᠠᠰᠤᠨ᠎ᠠ",
)


@dataclass(frozen=True)
class CleanConfig:
    text_key: str
    min_chars: int
    max_chars: int
    min_mong_ratio: float
    max_symbol_ratio: float
    max_repeated_line_fraction: float
    max_latin_ratio: float
    max_cyrillic_ratio: float


def mongolian_count(text: str) -> int:
    return len(MONG_RE.findall(text))


def cyrillic_count(text: str) -> int:
    return len(CYR_RE.findall(text))


def non_space_len(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())


def normalize_line(line: str) -> str:
    line = CTRL_RE.sub("", line)
    line = PUA_RE.sub("", line)
    line = URL_RE.sub(" ", line)
    line = EMAIL_RE.sub(" ", line)
    line = SPACE_RE.sub(" ", line)
    line = LEADING_PAGE_RE.sub("", line)
    line = PAGE_TOKEN_RE.sub(" ", line)
    line = re.sub(r"([★☆*#=_\-—–·•|])\1{2,}", r"\1", line)
    line = RUN_SPACE_RE.sub(" ", line)
    return line.strip()


def keep_line(line: str) -> bool:
    if not line:
        return False
    if any(bp in line for bp in BOILERPLATE_SNIPPETS):
        return False
    if DECORATIVE_RE.fullmatch(line):
        return False
    if HEADING_ONLY_RE.fullmatch(line):
        return False
    if mongolian_count(line) < 4 and len(line) < 40:
        return False
    return True


def normalize_text(text: str) -> str:
    raw_lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines: list[str] = []
    blank = False
    for raw in raw_lines:
        line = normalize_line(raw)
        if not keep_line(line):
            if lines and not blank:
                lines.append("")
                blank = True
            continue
        lines.append(line)
        blank = False
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def split_text(text: str, max_chars: int, overlap_lines: int = 0) -> Iterable[str]:
    if max_chars <= 0 or len(text) <= max_chars:
        yield text
        return
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0

    def split_long_line(line: str) -> list[str]:
        if len(line) <= max_chars:
            return [line]
        parts: list[str] = []
        words = line.split(" ")
        buf: list[str] = []
        buf_len = 0
        for word in words:
            if len(word) > max_chars:
                if buf:
                    parts.append(" ".join(buf))
                    buf = []
                    buf_len = 0
                parts.extend(word[i : i + max_chars] for i in range(0, len(word), max_chars))
                continue
            add_len = len(word) + (1 if buf else 0)
            if buf and buf_len + add_len > max_chars:
                parts.append(" ".join(buf))
                buf = []
                buf_len = 0
            buf.append(word)
            buf_len += add_len
        if buf:
            parts.append(" ".join(buf))
        return parts

    for para in paragraphs:
        pieces = [para]
        if len(para) > max_chars:
            lines = [ln for ln in para.splitlines() if ln.strip()]
            pieces = []
            buf: list[str] = []
            buf_len = 0
            for line in lines or [para]:
                for part in split_long_line(line):
                    if buf and buf_len + len(part) + 1 > max_chars:
                        pieces.append("\n".join(buf))
                        buf = buf[-overlap_lines:] if overlap_lines else []
                        buf_len = sum(len(x) + 1 for x in buf)
                    buf.append(part)
                    buf_len += len(part) + 1
            if buf:
                pieces.append("\n".join(buf))
        for piece in pieces:
            add_len = len(piece) + (2 if cur else 0)
            if cur and cur_len + add_len > max_chars:
                chunks.append("\n\n".join(cur))
                cur = cur[-overlap_lines:] if overlap_lines else []
                cur_len = sum(len(x) + 2 for x in cur)
            if len(piece) > max_chars:
                for part in split_long_line(piece):
                    if cur and cur_len + len(part) + 2 > max_chars:
                        chunks.append("\n\n".join(cur))
                        cur = cur[-overlap_lines:] if overlap_lines else []
                        cur_len = sum(len(x) + 2 for x in cur)
                    cur.append(part)
                    cur_len += len(part) + (2 if cur else 0)
                continue
            cur.append(piece)
            cur_len += add_len
    if cur:
        chunks.append("\n\n".join(cur))
    for chunk in chunks:
        if chunk.strip():
            yield chunk.strip()


def quality_reason(
    text: str,
    *,
    min_chars: int,
    min_mong_ratio: float,
    max_symbol_ratio: float,
    max_repeated_line_fraction: float,
    max_latin_ratio: float,
    max_cyrillic_ratio: float = 0.0,
) -> str:
    if len(text.strip()) < min_chars:
        return "too_short"
    if PUA_RE.search(text):
        return "pua"
    if CTRL_RE.search(text):
        return "control"
    ns = non_space_len(text)
    if ns == 0:
        return "empty"
    if script_ratio(text, "mongolian") < min_mong_ratio:
        return "script_impurity"
    latin = sum(1 for ch in text if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
    if latin / ns > max_latin_ratio:
        return "latin_noise"
    if cyrillic_count(text) / ns > max_cyrillic_ratio:
        return "cyrillic_noise"
    if symbol_to_word_ratio(text) > max_symbol_ratio:
        return "symbol_spam"
    if repeated_line_fraction(text) > max_repeated_line_fraction:
        return "repeated_lines"
    return ""


def iter_jsonl(path: Path, text_key: str) -> Iterable[dict]:
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                yield {"_bad_json_line": line_no, text_key: ""}
                continue
            if isinstance(rec, str):
                rec = {text_key: rec}
            yield rec


def batched(records: Iterable[dict], batch_size: int) -> Iterator[list[dict]]:
    batch: list[dict] = []
    for rec in records:
        batch.append(rec)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def process_record(rec: dict, cfg: CleanConfig) -> tuple[Counter, list[dict]]:
    stats: Counter = Counter()
    candidates: list[dict] = []
    stats["input_records"] += 1
    if rec.get("_bad_json_line"):
        stats["drop_bad_json"] += 1
        return stats, candidates

    text = rec.get(cfg.text_key, "") or ""
    normalized = normalize_text(text)
    if normalized != text:
        stats["normalized_records"] += 1

    for chunk_index, chunk in enumerate(split_text(normalized, cfg.max_chars)):
        stats["candidate_chunks"] += 1
        reason = quality_reason(
            chunk,
            min_chars=cfg.min_chars,
            min_mong_ratio=cfg.min_mong_ratio,
            max_symbol_ratio=cfg.max_symbol_ratio,
            max_repeated_line_fraction=cfg.max_repeated_line_fraction,
            max_latin_ratio=cfg.max_latin_ratio,
            max_cyrillic_ratio=cfg.max_cyrillic_ratio,
        )
        if reason:
            stats[f"drop_{reason}"] += 1
            continue
        out_rec = {"text": chunk}
        for key in ("source", "url", "title", "kind"):
            if key in rec:
                out_rec[key] = rec[key]
        if len(normalized) > cfg.max_chars:
            out_rec["chunk_index"] = chunk_index
        candidates.append(out_rec)
    return stats, candidates


def process_batch(payload: tuple[list[dict], CleanConfig]) -> tuple[Counter, list[dict]]:
    batch, cfg = payload
    stats: Counter = Counter()
    candidates: list[dict] = []
    for rec in batch:
        rec_stats, rec_candidates = process_record(rec, cfg)
        stats.update(rec_stats)
        candidates.extend(rec_candidates)
    return stats, candidates


def iter_input_batches(inputs: list[str], text_key: str, batch_size: int) -> Iterator[list[dict]]:
    for input_path in map(Path, inputs):
        yield from batched(iter_jsonl(input_path, text_key), batch_size)


def run(args: argparse.Namespace) -> Counter:
    out_path = Path(args.out)
    report_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    deduper = Deduper(thresh=args.thresh)
    stats: Counter = Counter()
    cfg = CleanConfig(
        text_key=args.text_key,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        min_mong_ratio=args.min_mong_ratio,
        max_symbol_ratio=args.max_symbol_ratio,
        max_repeated_line_fraction=args.max_repeated_line_fraction,
        max_latin_ratio=args.max_latin_ratio,
        max_cyrillic_ratio=args.max_cyrillic_ratio,
    )
    workers = max(1, args.workers)
    batch_size = max(1, args.batch_size)

    def consume_batch_result(batch_stats: Counter, candidates: list[dict], out_fh) -> None:
        stats.update(batch_stats)
        for out_rec in candidates:
            chunk = out_rec["text"]
            if not deduper.seen(chunk):
                stats["drop_duplicate"] += 1
                continue
            out_fh.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            stats["kept"] += 1
            stats["kept_chars"] += len(chunk)
            stats["kept_mongolian_chars"] += mongolian_count(chunk)

    with out_path.open("w", encoding="utf-8") as out_fh:
        batches = iter_input_batches(args.inputs, args.text_key, batch_size)
        if workers == 1:
            for batch in batches:
                consume_batch_result(*process_batch((batch, cfg)), out_fh)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
                payloads = ((batch, cfg) for batch in batches)
                for batch_stats, candidates in pool.map(process_batch, payloads, chunksize=args.map_chunksize):
                    consume_batch_result(batch_stats, candidates, out_fh)

    report = {
        "inputs": args.inputs,
        "output": args.out,
        "settings": {
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "min_mong_ratio": args.min_mong_ratio,
            "max_symbol_ratio": args.max_symbol_ratio,
            "max_repeated_line_fraction": args.max_repeated_line_fraction,
            "max_latin_ratio": args.max_latin_ratio,
            "max_cyrillic_ratio": args.max_cyrillic_ratio,
            "thresh": args.thresh,
            "workers": workers,
            "batch_size": batch_size,
            "map_chunksize": args.map_chunksize,
        },
        "stats": dict(stats),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--min-chars", type=int, default=80)
    ap.add_argument("--max-chars", type=int, default=20_000)
    ap.add_argument("--min-mong-ratio", type=float, default=0.55)
    ap.add_argument("--max-symbol-ratio", type=float, default=0.35)
    ap.add_argument("--max-repeated-line-fraction", type=float, default=0.25)
    ap.add_argument("--max-latin-ratio", type=float, default=0.20)
    ap.add_argument("--max-cyrillic-ratio", type=float, default=0.0)
    ap.add_argument("--thresh", type=int, default=3)
    ap.add_argument("--workers", type=int, default=1, help="parallel worker processes for parse/clean/split/filter")
    ap.add_argument("--batch-size", type=int, default=256, help="records per worker batch")
    ap.add_argument("--map-chunksize", type=int, default=1, help="ProcessPoolExecutor map chunksize")
    args = ap.parse_args(argv)
    stats = run(args)
    total = stats["candidate_chunks"] or 1
    print(
        f"input_records={stats['input_records']} candidate_chunks={stats['candidate_chunks']} "
        f"kept={stats['kept']} ({stats['kept'] / total * 100:.1f}%)",
        flush=True,
    )
    for key in sorted(stats):
        if key.startswith("drop_"):
            print(f"  {key}={stats[key]}", flush=True)
    print(f"kept_chars={stats['kept_chars']} kept_mongolian_chars={stats['kept_mongolian_chars']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
