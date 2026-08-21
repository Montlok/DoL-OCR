# -*- coding: utf-8 -*-
"""Extract normalized UTF-8 plaintext from reviewed pretraining sources.

Each invocation handles one source and appends JSONL (``{"text": ...}``) lines
to ``--output``:

  # Traditional Mongolian (UTF-16 .txt, recursive)
  python -m Tokenizer.tools.prepare_corpus --source mongolian \
      --input /data/traditional_mongolian --output mn.jsonl

  # CHINESE bundle (人民日报 / 问答 / Journal / Tsinghua)
  python -m Tokenizer.tools.prepare_corpus --source chinese \
      --input /data/multilingual --output zh_en.jsonl

  # Wikipedia from a local parquet dump (offline, e.g. fetched via hf-mirror)
  python -m Tokenizer.tools.prepare_corpus --source wiki --lang ja \
      --input /data/wikipedia --output ja.jsonl

  # Wikipedia via HF datasets (streaming, sampled) when no local dump exists
  python -m Tokenizer.tools.prepare_corpus --source wiki \
      --lang ja --limit 20000 --output ja.jsonl

  # Any local parquet dataset (FineMath, OpenWebMath, ...) by text column
  python -m Tokenizer.tools.prepare_corpus --source parquet \
      --input /data/math --text-column text --output math.jsonl

  # Any local .jsonl dataset (e.g. MC2 Mongolian) by text field
  python -m Tokenizer.tools.prepare_corpus --source jsonl \
      --input mc2_mn.jsonl --text-column text --output mn_extra.jsonl

The output feeds ``build_morphbpe`` (Mongolian) and ``build_general_bpe``
(everything else), and later ``build_pretraining_data``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import unicodedata
from typing import Iterable, Iterator

_MIN_CHARS = 1
_NULLISH = {"", "none", "null", "nan", "n/a"}


def _clean(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\ufeff", "")
    text = unicodedata.normalize("NFC", text)
    # Collapse stray control chars (keep tab/newline) so JSONL stays one-per-line.
    text = "".join(
        ch if (ch in "\t\n" or unicodedata.category(ch)[0] != "C") else " "
        for ch in text
    )
    text = text.replace("\n", " ").replace("\r", " ").strip()
    # Heterogeneous dumps store missing fields as the literal string "None"/
    # "null"; drop them so they do not pollute the BPE corpus.
    if text.lower() in _NULLISH:
        return ""
    return text


def _read_text_any(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ("utf-16", "utf-16-le", "utf-8", "utf-8-sig"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _iter_mongolian(root: str) -> Iterator[str]:
    for dirpath, _dirs, files in os.walk(root):
        for name in sorted(files):
            if not name.lower().endswith(".txt"):
                continue
            text = _clean(_read_text_any(os.path.join(dirpath, name)))
            if len(text) >= _MIN_CHARS:
                yield text


def _iter_jsonl_fields(path: str, fields: tuple[str, ...]) -> Iterator[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            for field in fields:
                val = obj.get(field)
                if val:
                    text = _clean(str(val))
                    if len(text) >= _MIN_CHARS:
                        yield text


def _iter_json_array(path: str, fields: tuple[str, ...]) -> Iterator[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return
    if isinstance(data, dict):
        data = data.get("data", []) or data.get("records", []) or []
    if not isinstance(data, list):
        return
    for obj in data:
        if not isinstance(obj, dict):
            continue
        for field in fields:
            val = obj.get(field)
            if val:
                text = _clean(str(val))
                if len(text) >= _MIN_CHARS:
                    yield text


def _sniff_json_layout(path: str) -> str:
    """Return ``"array"`` if the file's first non-whitespace byte is ``[`` else
    ``"jsonl"``. Lets ``.json`` files that are real JSON arrays and line-
    delimited JSONL both be read, instead of silently yielding nothing when the
    assumed layout is wrong.
    """

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            while True:
                ch = f.read(1)
                if not ch:
                    return "jsonl"
                if not ch.isspace():
                    return "array" if ch == "[" else "jsonl"
    except OSError:
        return "jsonl"


def _iter_json_fields(path: str, fields: tuple[str, ...]) -> Iterator[str]:
    """Read ``fields`` from a ``.json``/``.jsonl`` file, auto-detecting whether
    it is a JSON array or line-delimited JSONL."""

    if _sniff_json_layout(path) == "array":
        yield from _iter_json_array(path, fields)
    else:
        yield from _iter_jsonl_fields(path, fields)


def _iter_chinese(root: str) -> Iterator[str]:
    renmin = os.path.join(root, "人民日报2023.json")
    if os.path.exists(renmin):
        yield from _iter_json_fields(renmin, ("content", "title"))
    qa = os.path.join(root, "问答语料300.json")
    if os.path.exists(qa):
        yield from _iter_json_fields(qa, ("question", "answer"))
    journal = os.path.join(root, "JournalArticle2013_2023")
    if os.path.isdir(journal):
        fields = (
            "remark_c", "title_c", "keyword_c",
            "remark_e", "title_e", "keyword_e",
        )
        for name in sorted(os.listdir(journal)):
            if name.lower().endswith(".json"):
                yield from _iter_json_array(os.path.join(journal, name), fields)
    tsinghua = os.path.join(root, "TsinghuaBilingualCorpus")
    if os.path.isdir(tsinghua):
        for name in sorted(os.listdir(tsinghua)):
            if not name.lower().endswith(".txt"):
                continue
            with open(
                os.path.join(tsinghua, name), "r", encoding="utf-8", errors="replace"
            ) as f:
                for line in f:
                    text = _clean(line)
                    if len(text) >= _MIN_CHARS:
                        yield text


def _iter_wiki(lang: str, limit: int, date: str) -> Iterator[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(f"Missing dependency for --source wiki: {exc}") from exc
    config = f"{date}.{lang}"
    ds = load_dataset(
        "wikimedia/wikipedia", config, split="train", streaming=True
    )
    count = 0
    for row in ds:
        text = _clean(str(row.get("text", "")))
        if len(text) < _MIN_CHARS:
            continue
        yield text
        count += 1
        if limit and count >= limit:
            break


def _iter_wiki_local(root: str, lang: str, date: str, limit: int) -> Iterator[str]:
    """Stream a locally-downloaded ``wikimedia/wikipedia`` parquet dump.

    ``root`` is the directory that holds the ``{date}.{lang}/*.parquet`` shards
    (e.g. a local snapshot fetched with ``hf-mirror``). Reading the parquet
    directly avoids the (often throttled) ``datasets`` streaming path.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(f"Missing dependency for --source wiki: {exc}") from exc

    shard_dir = os.path.join(root, f"{date}.{lang}")
    if not os.path.isdir(shard_dir):
        shard_dir = root
    shards = sorted(glob.glob(os.path.join(shard_dir, "*.parquet")))
    if not shards:
        raise SystemExit(f"No parquet shards found under {shard_dir!r}")

    count = 0
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=1000, columns=["text"]):
            for value in batch.column("text"):
                text = _clean(str(value.as_py() or ""))
                if len(text) < _MIN_CHARS:
                    continue
                yield text
                count += 1
                if limit and count >= limit:
                    return


def _iter_parquet(root: str, text_column: str, limit: int) -> Iterator[str]:
    """Stream the ``text_column`` from arbitrary local parquet shards.

    ``root`` is a parquet file or a directory of ``*.parquet`` shards (searched
    recursively). Reading parquet directly keeps generic HF datasets (FineMath,
    OpenWebMath, MC2, ...) on the offline path that avoids throttled streaming.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency for --source parquet: {exc}"
        ) from exc

    if os.path.isdir(root):
        shards = sorted(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True))
    elif root.endswith(".parquet"):
        shards = [root]
    else:
        shards = sorted(glob.glob(root))
    if not shards:
        raise SystemExit(f"No parquet shards found under {root!r}")

    count = 0
    for shard in shards:
        pf = pq.ParquetFile(shard)
        if text_column not in pf.schema_arrow.names:
            raise SystemExit(
                f"Column {text_column!r} not in {shard!r}; "
                f"available: {pf.schema_arrow.names}"
            )
        for batch in pf.iter_batches(batch_size=1000, columns=[text_column]):
            for value in batch.column(text_column):
                text = _clean(str(value.as_py() or ""))
                if len(text) < _MIN_CHARS:
                    continue
                yield text
                count += 1
                if limit and count >= limit:
                    return


def _iter_jsonl_text(path: str, text_column: str, limit: int) -> Iterator[str]:
    """Stream ``text_column`` from a local ``.jsonl`` file (one object/line)."""
    count = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = _clean(str(obj.get(text_column, "") or ""))
            if len(text) < _MIN_CHARS:
                continue
            yield text
            count += 1
            if limit and count >= limit:
                return


def _write(out_path: str, texts: Iterable[str], append: bool) -> int:
    mode = "a" if append else "w"
    written = 0
    with open(out_path, mode, encoding="utf-8") as f:
        for text in texts:
            f.write(json.dumps({"text": text}, ensure_ascii=False))
            f.write("\n")
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        choices=["mongolian", "chinese", "wiki", "parquet", "jsonl"],
    )
    parser.add_argument("--input", help="root dir/file for the chosen source")
    parser.add_argument("--output", required=True, help="output JSONL")
    parser.add_argument("--lang", help="wiki language code (en/ja/zh/mn/...)")
    parser.add_argument("--limit", type=int, default=20000, help="wiki doc cap")
    parser.add_argument("--wiki-date", default="20231101", help="wiki snapshot")
    parser.add_argument(
        "--text-column",
        default="text",
        help="column/field holding the text for --source parquet|jsonl",
    )
    parser.add_argument(
        "--append", action="store_true", help="append instead of overwrite"
    )
    args = parser.parse_args()

    if args.source == "mongolian":
        if not args.input:
            parser.error("--source mongolian requires --input")
        texts = _iter_mongolian(args.input)
    elif args.source == "chinese":
        if not args.input:
            parser.error("--source chinese requires --input")
        texts = _iter_chinese(args.input)
    elif args.source == "parquet":
        if not args.input:
            parser.error("--source parquet requires --input")
        texts = _iter_parquet(args.input, args.text_column, args.limit)
    elif args.source == "jsonl":
        if not args.input:
            parser.error("--source jsonl requires --input")
        texts = _iter_jsonl_text(args.input, args.text_column, args.limit)
    else:
        if not args.lang:
            parser.error("--source wiki requires --lang")
        if args.input:
            texts = _iter_wiki_local(
                args.input, args.lang, args.wiki_date, args.limit
            )
        else:
            texts = _iter_wiki(args.lang, args.limit, args.wiki_date)

    written = _write(args.output, texts, args.append)
    print(f"source={args.source} written={written} output={args.output}")


if __name__ == "__main__":
    main()
