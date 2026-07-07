# -*- coding: utf-8 -*-
"""Validate a filled annotation TSV before it enters RL reward / golden sets.

Pipeline per row: nominal-Unicode normalization -> bundle encode with a
zero-<unk> assertion -> stats. With --compare (annotator_B sheet), reports
double-annotation agreement as grapheme CER on the overlapping ids.

Usage:
  PYTHONPATH=. python3 scripts/validate_annotations.py \
      --tsv annotator_A_filled.tsv --bundle ~/dolocr/bundle_v3b \
      --output annotations_clean.tsv [--compare annotator_B_filled.tsv]
"""
import argparse
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Tokenizer.tools.normalize_mongolian import normalize  # noqa: E402
from Tokenizer.unified import TokenizerBundle  # noqa: E402


def read_tsv(path: str) -> dict[str, dict]:
    rows = {}
    with open(path, encoding="utf-8") as f:
        header = f.readline()
        if not header.startswith("id\t"):
            raise ValueError(f"{path}: unexpected header {header!r}")
        for ln, line in enumerate(f, start=2):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                raise ValueError(f"{path}:{ln}: expected >=3 columns")
            id_, rel = parts[0], parts[1]
            text = parts[2] if len(parts) > 2 else ""
            notes = parts[3] if len(parts) > 3 else ""
            rows[id_] = {"rel": rel, "text": text.strip(), "notes": notes.strip()}
    return rows


def grapheme_cer(ref: str, hyp: str) -> float:
    """Levenshtein over NFC codepoints (proxy for grapheme CER on Mongolian)."""
    a = unicodedata.normalize("NFC", ref)
    b = unicodedata.normalize("NFC", hyp)
    if not a:
        return 0.0 if not b else 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1] / len(a)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tsv", required=True, help="filled annotator sheet")
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--output", required=True, help="normalized clean TSV")
    ap.add_argument("--compare", default="", help="second annotator sheet (agreement)")
    args = ap.parse_args()

    bundle = TokenizerBundle.from_dir(args.bundle)
    unk = bundle.tokenizer.unk_id
    rows = read_tsv(args.tsv)

    n_filled = n_bad = n_empty = n_unk_rows = 0
    out_lines = ["id\timage_relpath\ttranscription\tnotes"]
    for id_, r in rows.items():
        if not r["text"]:
            if "bad" in r["notes"].lower():
                n_bad += 1
            else:
                n_empty += 1
            continue
        norm = normalize(r["text"], nominal=True)
        ids = bundle.encode(norm)
        n_unk = sum(1 for t in ids if t == unk)
        if n_unk:
            n_unk_rows += 1
            print(f"[unk] {id_}: {n_unk} unk tokens after normalization -- fix input")
            continue
        n_filled += 1
        out_lines.append(f"{id_}\t{r['rel']}\t{norm}\t{r['notes']}")

    Path(args.output).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(
        f"rows={len(rows)} accepted={n_filled} empty={n_empty} "
        f"bad={n_bad} unk_rejected={n_unk_rows} -> {args.output}"
    )

    if args.compare:
        other = read_tsv(args.compare)
        cers = []
        for id_, r in rows.items():
            o = other.get(id_)
            if not o or not r["text"] or not o["text"]:
                continue
            cers.append(
                grapheme_cer(
                    normalize(r["text"], nominal=True),
                    normalize(o["text"], nominal=True),
                )
            )
        if cers:
            mean = sum(cers) / len(cers)
            print(
                f"double-annotation agreement: n={len(cers)} mean_cer={mean:.4f} "
                f"(>0.05 means the two annotators disagree substantially -- "
                f"review the guideline before scaling up)"
            )
        else:
            print("no overlapping filled ids for agreement check")

    return 0 if n_unk_rows == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
