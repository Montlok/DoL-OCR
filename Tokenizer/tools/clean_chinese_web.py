# -*- coding: utf-8 -*-

"""Strict cleaning pipeline for the large Chinese web corpus (CHINESE_WEB).

The ``CHINESE_WEB`` drop is tens of GB of ``{"ID", "Content"}`` JSON-lines news /
government / forum text (plus Zhihu parquet). It is mostly well-formed but
carries the usual web cruft: navigation menus, video-player UI tokens, footer
disclaimers / copyright / contact blocks, verbatim reposts, and the occasional
spam page. This module turns that raw drop into a clean, de-duplicated,
text-only ``{"text": ...}`` JSONL ready for :mod:`build_corpus_mix`.

Design goals
------------
* **Strict but not destructive.** The corpus is largely formal news/gov text, so
  ad-keyword doc-dropping is conservative (legitimate words like 发票 / 澳门 are
  not spam signals on their own); the heavy lifting is normalization, line-level
  boilerplate stripping, repetition / quality gates, and global dedup.
* **Scalable.** Documents are cleaned in parallel worker processes; only the
  survivors (post quality-filter) flow through a single in-memory
  :class:`~Tokenizer.tools.corpus_filters.Deduper` for global exact + near-dup
  removal. ~44M docs / ~64GB RAM is comfortably in budget.
* **Resumable.** Each input *unit* (one jsonl, one zip member, one parquet shard)
  writes its own output shard; on restart, completed units are skipped and the
  deduper state is rebuilt by re-scanning the already-written (small) output
  shards before cleaning resumes.

The per-document logic (:func:`clean_doc`) is pure, deterministic and unit
tested; the CLI is the only stateful part.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
import zipfile
from collections import Counter
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Iterable, Iterator, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus_filters import (  # noqa: E402
    Deduper,
    exact_key,
    simhash,
)

# ===========================================================================
# Normalization
# ===========================================================================
# Zero-width / BOM / soft-hyphen and other invisible formatting characters that
# carry no information but defeat exact dedup and inflate length.
_ZERO_WIDTH = "".join(
    [
        "\u200b",  # zero-width space
        "\u200c",  # zero-width non-joiner
        "\u200d",  # zero-width joiner
        "\ufeff",  # BOM / zero-width no-break space
        "\u00ad",  # soft hyphen
        "\u2060",  # word joiner
    ]
)
_ZERO_WIDTH_RE = re.compile("[" + re.escape(_ZERO_WIDTH) + "]")
# C0/C1 control chars except tab/newline.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
# Runs of spaces/tabs (not newlines) -> single space.
_INLINE_WS_RE = re.compile(r"[^\S\n]+")
# 3+ blank lines -> a single blank line.
_MULTI_NL_RE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """NFKC-normalize and strip invisible/control junk; canonicalize whitespace.

    NFKC folds full-width Latin/digits and compatibility forms to their canonical
    shapes so dedup and ratio metrics see one representation. Newlines are kept
    (paragraph structure matters for repetition metrics) but trailing spaces and
    excess blank lines are collapsed.
    """

    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_RE.sub("", text)
    # collapse inline whitespace, strip per-line trailing spaces
    lines = [_INLINE_WS_RE.sub(" ", ln).strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = _MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


# ===========================================================================
# Boilerplate line stripping
# ===========================================================================
# Lines matching these are footer/nav/disclaimer cruft and are dropped wholesale.
# Patterns are intentionally anchored to recognizable web-furniture phrasing so
# real article sentences that merely mention a word are not removed.
_BOILERPLATE_LINE_PATTERNS = [
    r"^.{0,12}声明[:：]",
    r"免责声明",
    r"版权所有",
    r"All Rights Reserved",
    r"未经.{0,6}(授权|许可|书面同意)",
    r"如遇作品内容.{0,4}版权",
    r"本(文|站|网|公众号).{0,8}(转载|来源|版权)",
    r"^责任编辑[:：]",
    r"^编辑[:：]",
    r"^来源[:：].{0,40}$",
    r"^记者[:：].{0,30}$",
    r"^(上一篇|下一篇|相关(阅读|推荐|新闻|链接))",
    r"(我要|发表|查看更多)?评论\s*\(?\d*\)?$",
    r"(扫一扫|扫码).{0,12}(关注|下载|分享|二维码)",
    r"关注.{0,6}(公众号|微信|视频号)",
    r"微信(公众号|号)[:：]",
    r"阅读原文",
    r"分享到[:：]?",
    r"^(打印|收藏|点赞|转发|返回(顶部|列表))$",
    r"网站地图",
    r"^字号[:：]",
    r"^\s*play\s+stop\s+mute",  # embedded video-player UI
    r"^(客服|投诉|举报|咨询|联系)?(热线|电话)[:：]",
    r"京ICP备|蒙ICP备|粤ICP备|沪ICP备|ICP备\d",
    r"公网安备",
]
_BOILERPLATE_LINE_RE = re.compile("|".join(_BOILERPLATE_LINE_PATTERNS))

# Whole-document spam signals. Formal news/gov text frequently contains words
# like 发票 / 身份证 / 办理 in legitimate contexts, so a single keyword is never
# enough: we require an explicit *contact solicitation* (a messaging handle with
# digits) next to a commercial verb, or an unambiguous gambling-site promo. This
# keeps precision high and avoids nuking real articles.
_SPAM_SOLICIT_RE = re.compile(
    r"(代开|代办|出售|批发|收购|转让|办证|代孕|刷单|套现|网赚|私聊|微商|薇商|包邮购)"
)
_SPAM_CONTACT_RE = re.compile(
    r"(微信|薇信|徽信|威信|QQ|扣扣|vx|VX|Vx|whatsapp|telegram|纸飞机)"
    r"\s*[:：]?\s*[0-9A-Za-z][0-9A-Za-z\- ]{4,}"
)
_SPAM_GAMBLE_RE = re.compile(
    r"(六合彩|时时彩|私彩|百家乐|博彩|赌场|赌博|娱乐城|彩金|外围|菠菜平台)"
    r".{0,12}(网址|平台|注册|开户|代理|官网|送|app|APP)"
)


def is_spam(text: str) -> bool:
    """True for high-confidence spam: solicitation+contact, or gambling promo."""

    if _SPAM_GAMBLE_RE.search(text):
        return True
    return bool(_SPAM_CONTACT_RE.search(text) and _SPAM_SOLICIT_RE.search(text))


def strip_boilerplate_lines(text: str) -> str:
    """Remove footer/nav/disclaimer lines; keep the article body."""

    kept = [
        ln
        for ln in text.split("\n")
        if not (ln.strip() and _BOILERPLATE_LINE_RE.search(ln))
    ]
    out = "\n".join(kept)
    return _MULTI_NL_RE.sub("\n\n", out).strip()


# ===========================================================================
# Quality metrics
# ===========================================================================
_CJK_LO, _CJK_HI = 0x4E00, 0x9FFF
_CJK_EXT = ((0x3400, 0x4DBF), (0xF900, 0xFAFF))
# Chinese + ASCII sentence-ending punctuation.
_SENT_END = set("。！？!?…")
_DIGIT_RE = re.compile(r"\d")
_GARBLE_RE = re.compile(r"[\ufffd\ue000-\uf8ff]")
_HAN_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")


def chinese_ratio(text: str) -> float:
    """Fraction of non-space characters that are Han ideographs (0..1)."""

    ns = [ch for ch in text if not ch.isspace()]
    if not ns:
        return 0.0
    hit = 0
    for ch in ns:
        o = ord(ch)
        if _CJK_LO <= o <= _CJK_HI or any(lo <= o <= hi for lo, hi in _CJK_EXT):
            hit += 1
    return hit / len(ns)


def digit_ratio(text: str) -> float:
    ns = [ch for ch in text if not ch.isspace()]
    if not ns:
        return 0.0
    return len(_DIGIT_RE.findall(text)) / len(ns)


def garble_ratio(text: str) -> float:
    ns = [ch for ch in text if not ch.isspace()]
    if not ns:
        return 0.0
    return len(_GARBLE_RE.findall(text)) / len(ns)


def repeated_line_fraction(text: str) -> float:
    """Fraction of non-empty lines that duplicate an earlier line."""

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return 0.0
    seen: set[str] = set()
    dup = 0
    for ln in lines:
        if ln in seen:
            dup += 1
        else:
            seen.add(ln)
    return dup / len(lines)


def char_ngram_dup_fraction(text: str, n: int) -> float:
    """Fraction of character ``n``-grams that are repeats of an earlier one.

    A Gopher-style repetition signal: machine-generated / templated / scraped
    junk repeats character n-grams far more than natural prose.
    """

    s = _INLINE_WS_RE.sub("", text.replace("\n", ""))
    if len(s) < n + 1:
        return 0.0
    total = len(s) - n + 1
    seen: set[str] = set()
    dup = 0
    for i in range(total):
        g = s[i : i + n]
        if g in seen:
            dup += 1
        else:
            seen.add(g)
    return dup / total


def top_char_ngram_fraction(text: str, n: int) -> float:
    """Character coverage of the single most frequent ``n``-gram (0..1)."""

    s = _INLINE_WS_RE.sub("", text.replace("\n", ""))
    if len(s) < n + 1:
        return 0.0
    counts: Counter[str] = Counter(s[i : i + n] for i in range(len(s) - n + 1))
    top = counts.most_common(1)[0][1]
    return top * n / len(s)


def sentence_punct_ratio(text: str) -> float:
    """Ratio of sentence-ending punctuation to Han characters.

    Navigation menus / keyword stuffing are long runs of Han with almost no
    sentence punctuation; natural prose has a healthy ending-punctuation rate.
    """

    han = len(_HAN_RE.findall(text))
    if han == 0:
        return 0.0
    ends = sum(text.count(p) for p in _SENT_END)
    return ends / han


# ===========================================================================
# Config + per-document cleaning
# ===========================================================================
@dataclass
class CleanConfig:
    """Strict-by-default thresholds. All gates are configurable per run."""

    min_chars: int = 200
    min_chinese_ratio: float = 0.55
    max_digit_ratio: float = 0.25
    max_garble_ratio: float = 0.002
    max_repeated_line_fraction: float = 0.30
    max_dup_3gram_fraction: float = 0.50
    max_dup_4gram_fraction: float = 0.42
    max_top_2gram_fraction: float = 0.20
    min_sentence_punct_ratio: float = 0.003
    drop_spam: bool = True
    strip_boilerplate: bool = True


def clean_doc(text: str, cfg: CleanConfig) -> tuple[Optional[str], str]:
    """Normalize + filter one document.

    Returns ``(cleaned_text, "")`` when the document is kept, or ``(None, reason)``
    when it is dropped. ``reason`` is a short stable tag for stats accounting.
    """

    t = normalize_text(text)
    if not t:
        return None, "empty"
    if cfg.strip_boilerplate:
        stripped = strip_boilerplate_lines(t)
        # Most docs are a single line: line-level stripping would otherwise nuke
        # an entire real article that merely ends with a footer phrase. Only
        # adopt the stripped form when something survives; else keep the original
        # and let the quality gates (repetition / sentences) catch true junk.
        if stripped:
            t = stripped
    if len(t) < cfg.min_chars:
        return None, "too_short"
    if cfg.drop_spam and is_spam(t):
        return None, "spam"
    if garble_ratio(t) > cfg.max_garble_ratio:
        return None, "garbled"
    if chinese_ratio(t) < cfg.min_chinese_ratio:
        return None, "low_chinese"
    if digit_ratio(t) > cfg.max_digit_ratio:
        return None, "digit_heavy"
    if sentence_punct_ratio(t) < cfg.min_sentence_punct_ratio:
        return None, "no_sentences"
    if repeated_line_fraction(t) > cfg.max_repeated_line_fraction:
        return None, "repeated_lines"
    if top_char_ngram_fraction(t, 2) > cfg.max_top_2gram_fraction:
        return None, "ngram_spam"
    if char_ngram_dup_fraction(t, 3) > cfg.max_dup_3gram_fraction:
        return None, "repetitive"
    if char_ngram_dup_fraction(t, 4) > cfg.max_dup_4gram_fraction:
        return None, "repetitive"
    return t, ""


# ===========================================================================
# Input sources (streamed; nothing fully extracted to disk)
# ===========================================================================
@dataclass
class Unit:
    """One resumable input unit and how to read it."""

    uid: str
    kind: str  # "zipmember" | "jsonl" | "parquet"
    path: str
    member: str = ""
    text_key: str = "Content"


def discover_units(inputs: Iterable[str], text_key: str) -> list[Unit]:
    """Expand files/dirs/zips into a deterministic list of input units."""

    units: list[Unit] = []
    for path in sorted(inputs):
        if os.path.isdir(path):
            for name in sorted(os.listdir(path)):
                units.extend(discover_units([os.path.join(path, name)], text_key))
        elif path.endswith(".zip"):
            with zipfile.ZipFile(path) as zf:
                for member in sorted(zf.namelist()):
                    if member.endswith((".jsonl", ".json", ".ndjson")):
                        uid = f"{os.path.basename(path)}::{member}"
                        units.append(Unit(uid, "zipmember", path, member, text_key))
        elif path.endswith((".jsonl", ".json", ".ndjson")):
            units.append(Unit(os.path.basename(path), "jsonl", path, "", text_key))
        elif path.endswith(".parquet"):
            units.append(Unit(os.path.basename(path), "parquet", path, "", text_key))
    return units


def _iter_jsonl_lines(fh, text_key: str) -> Iterator[str]:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(rec, str):
            yield rec
        elif isinstance(rec, dict):
            val = rec.get(text_key) or rec.get("text") or rec.get("Content") or ""
            if val:
                yield val


def iter_unit_texts(unit: Unit) -> Iterator[str]:
    """Yield raw document texts from one unit (streaming, low memory)."""

    if unit.kind == "zipmember":
        with zipfile.ZipFile(unit.path) as zf, zf.open(unit.member) as raw:
            yield from _iter_jsonl_lines(
                (ln.decode("utf-8", "replace") for ln in raw), unit.text_key
            )
    elif unit.kind == "jsonl":
        with open(unit.path, encoding="utf-8", errors="replace") as fh:
            yield from _iter_jsonl_lines(fh, unit.text_key)
    elif unit.kind == "parquet":
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(unit.path)
        keys = [unit.text_key, "text", "RESPONSE", "Content"]
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg)
            cols = set(tbl.schema.names)
            key = next((k for k in keys if k in cols), None)
            if key is None:
                continue
            for val in tbl.column(key).to_pylist():
                if val:
                    yield str(val)


# ===========================================================================
# Parallel cleaning worker
# ===========================================================================
_WORKER_CFG: Optional[CleanConfig] = None


def _worker_init(cfg: CleanConfig) -> None:
    global _WORKER_CFG
    _WORKER_CFG = cfg


def _worker_clean(batch: list[str]) -> tuple[list[tuple[str, str, int]], dict]:
    """Clean a batch; return kept ``(text, exact_key, simhash)`` + drop counts.

    Quality filtering and the (expensive) SimHash are done here, in parallel.
    Cross-document dedup decisions stay in the single parent process.
    """

    cfg = _WORKER_CFG
    kept: list[tuple[str, str, int]] = []
    drops: Counter = Counter()
    for raw in batch:
        cleaned, reason = clean_doc(raw, cfg)
        if cleaned is None:
            drops[reason] += 1
            continue
        kept.append((cleaned, exact_key(cleaned), simhash(cleaned)))
    return kept, dict(drops)


def _batched(it: Iterator[str], size: int) -> Iterator[list[str]]:
    batch: list[str] = []
    for x in it:
        batch.append(x)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


# ===========================================================================
# Deduper helpers (near-dup via injected simhash, so workers do the hashing)
# ===========================================================================
class FedDeduper(Deduper):
    """Deduper variant fed precomputed ``(exact_key, simhash)`` pairs.

    The base class recomputes the hashes inside ``seen(text)``; here the worker
    pool already produced them, so we skip recomputation in the hot single-thread
    parent loop.
    """

    def seen_pre(self, key: str, sig: int, n_terms: int) -> bool:
        if key in self._exact:
            return False
        self._exact.add(key)
        if n_terms < self.min_simhash_words:
            return True
        if self._sigs and self._is_near_dup(sig):
            return False
        self._index(sig)
        return True


# ===========================================================================
# CLI orchestration
# ===========================================================================
def _state_path(out_dir: str) -> str:
    return os.path.join(out_dir, "_state.json")


def _load_state(out_dir: str) -> dict:
    p = _state_path(out_dir)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    return {"done_units": [], "stats": {}}


def _save_state(out_dir: str, state: dict) -> None:
    tmp = _state_path(out_dir) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, _state_path(out_dir))


def _shard_path(out_dir: str, uid: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z._-]", "_", uid)
    return os.path.join(out_dir, f"clean__{safe}.jsonl")


def _rebuild_deduper_from_shards(
    out_dir: str, done_units: list[str], near_dup: bool, thresh: int
) -> FedDeduper:
    """Repopulate dedup state from already-written output shards (resume)."""

    dd = FedDeduper(thresh=thresh)
    if not near_dup:
        dd.min_simhash_words = 1 << 30  # disable near-dup, exact only
    for uid in done_units:
        path = _shard_path(out_dir, uid)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    txt = json.loads(line).get("text", "")
                except (json.JSONDecodeError, ValueError):
                    continue
                if txt:
                    dd.seen(txt)  # base impl recomputes hashes; fine on resume
    return dd


def run(args) -> Counter:
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = CleanConfig(
        min_chars=args.min_chars,
        min_chinese_ratio=args.min_chinese_ratio,
        drop_spam=not args.keep_spam,
    )
    units = discover_units(args.inputs, args.text_key)
    if args.limit_units:
        units = units[: args.limit_units]

    state = _load_state(args.out_dir)
    done = set(state.get("done_units", []))
    stats: Counter = Counter(state.get("stats", {}))

    thresh = args.thresh
    near_dup = not args.no_near_dup
    if done and not args.report:
        print(f"[resume] rebuilding deduper from {len(done)} done shards...",
              flush=True)
        deduper: FedDeduper = _rebuild_deduper_from_shards(
            args.out_dir, sorted(done), near_dup, thresh
        )
    else:
        deduper = FedDeduper(thresh=thresh)
        if not near_dup:
            deduper.min_simhash_words = 1 << 30

    pool = Pool(processes=args.workers, initializer=_worker_init, initargs=(cfg,))
    t0 = time.time()
    try:
        for unit in units:
            if unit.uid in done:
                continue
            shard = _shard_path(args.out_dir, unit.uid)
            out_fh = None if args.report else open(shard + ".tmp", "w",
                                                   encoding="utf-8")
            u_total = u_kept = 0
            try:
                batches = _batched(iter_unit_texts(unit), args.batch_size)
                for kept, drops in pool.imap_unordered(_worker_clean, batches):
                    for reason, c in drops.items():
                        stats[f"drop_{reason}"] += c
                        u_total += c
                    for text, key, sig in kept:
                        u_total += 1
                        n_terms = len(text)  # char-count proxy for CJK
                        if not deduper.seen_pre(key, sig, n_terms):
                            stats["drop_duplicate"] += 1
                            continue
                        stats["kept"] += 1
                        u_kept += 1
                        if out_fh is not None:
                            out_fh.write(
                                json.dumps({"text": text}, ensure_ascii=False) + "\n"
                            )
            finally:
                if out_fh is not None:
                    out_fh.close()
                    os.replace(shard + ".tmp", shard)
            stats["total"] += u_total
            done.add(unit.uid)
            if not args.report:
                state["done_units"] = sorted(done)
                state["stats"] = dict(stats)
                _save_state(args.out_dir, state)
            rate = stats["total"] / max(time.time() - t0, 1e-6)
            print(
                f"[{len(done)}/{len(units)}] {unit.uid}: "
                f"+{u_kept}/{u_total} kept | total kept={stats['kept']:,} "
                f"dropped={stats['total'] - stats['kept']:,} | {rate:,.0f} docs/s",
                flush=True,
            )
    finally:
        pool.close()
        pool.join()
    return stats


def _print_stats(stats: Counter) -> None:
    total = stats["total"] or 1
    print(f"\ntotal={stats['total']:,} kept={stats['kept']:,} "
          f"({stats['kept'] / total * 100:.1f}%)")
    for k in sorted(stats):
        if k.startswith("drop_"):
            print(f"  {k}={stats[k]:,} ({stats[k] / total * 100:.1f}%)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inputs", nargs="+", required=True,
                    help="jsonl / parquet / .zip / directory inputs")
    ap.add_argument("--out-dir", required=True, help="output dir for clean shards")
    ap.add_argument("--text-key", default="Content")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--thresh", type=int, default=3, help="near-dup hamming thresh")
    ap.add_argument("--no-near-dup", action="store_true",
                    help="exact dedup only (lower memory)")
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--min-chinese-ratio", type=float, default=0.55)
    ap.add_argument("--keep-spam", action="store_true")
    ap.add_argument("--limit-units", type=int, default=0,
                    help="process only the first N units (validation)")
    ap.add_argument("--report", action="store_true",
                    help="compute stats without writing output shards")
    args = ap.parse_args(argv)

    stats = run(args)
    _print_stats(stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
