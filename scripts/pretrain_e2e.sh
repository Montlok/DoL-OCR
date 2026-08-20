#!/usr/bin/env bash
# End-to-end "first pretraining" pipeline for the Daffodils Mongolian RDT model.
#
# Run on the server and it goes corpus -> tokenizers -> packed shards -> data
# gate -> two-stage mHC pretraining. Every stage is idempotent (skip-if-exists)
# so a crashed run resumes cheaply.
#
# Quick local self-test (seconds, CPU, synthetic corpus, NaiveSSM fallback):
#   SMOKE=1 scripts/pretrain_e2e.sh
#
# Real run (set the corpus paths to your local data):
#   MN_CORPUS="1000 traditional_mongolian_corpus(DO NOT GIT IT)" \
#   CN_CORPUS="CHINESE(DO NOT GIT IT)" \
#   scripts/pretrain_e2e.sh
#
# Key environment knobs (all optional, sensible defaults below):
#   WORK            working directory for all intermediate artifacts
#   MN_CORPUS       traditional Mongolian corpus dir (UTF-16 .txt, recursive)
#   CN_CORPUS       CHINESE bundle dir (人民日报/问答/Journal/Tsinghua)
#   WIKI_LANGS      space-separated wiki languages (default "en ja zh mn")
#   WIKI_LIMIT      wiki docs per language for tokenizer coverage (default 20000)
#   WIKI_DUMP       local wikimedia parquet dump dir ({DATE}.{lang}/*.parquet);
#                   used per-language when present, else HF streaming fallback
#   WIKI_DATE       wiki snapshot date for the local dump (default 20231101)
#   MORPHBPE_VOCAB  MorphBPE vocab size (default 24000)
#   GENERAL_VOCAB   general byte-level BPE vocab size (default 40000)
#   MAX_LENGTH      packed sequence length for shards (default 2048)
#   CONFIG          train_rdt config (default two_stage_pretrain)
#   MAX_STEPS       training steps (default 100000)
#   MAMBA           Mamba backend for train_rdt (default official; smoke naive)
#   TRAIN_ARGS      extra args forwarded verbatim to scripts/train_rdt; an
#                   added --eval-data must include its --eval-data-receipt
#   MIX_MANIFEST    optional token-weighted mix manifest (JSON). When set, the
#                   packed shards are built from a weighted mixture of the
#                   sources it lists (hitting target language/domain token
#                   proportions) instead of a plain concatenation. See
#                   Tokenizer/tools/build_corpus_mix.py for the schema.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-$ROOT}"

SMOKE="${SMOKE:-0}"
WORK_DEFAULT="$ROOT/outputs/pretrain_e2e"
WIKI_LANGS="${WIKI_LANGS:-en ja zh mn}"
WIKI_LIMIT="${WIKI_LIMIT:-20000}"
WIKI_DUMP="${WIKI_DUMP:-$ROOT/WIKIPEDIA(DO NOT GIT IT)}"
WIKI_DATE="${WIKI_DATE:-20231101}"
MORPHBPE_VOCAB="${MORPHBPE_VOCAB:-24000}"
GENERAL_VOCAB="${GENERAL_VOCAB:-40000}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
CONFIG="${CONFIG:-two_stage_pretrain}"
MAX_STEPS="${MAX_STEPS:-100000}"
PRECISION="${PRECISION:-bf16}"
MAMBA="${MAMBA:-official}"
TRAIN_ARGS="${TRAIN_ARGS:-}"
MIX_MANIFEST="${MIX_MANIFEST:-}"

if [ "$SMOKE" = "1" ]; then
    # Tiny, fast, dependency-light defaults for the local self-test.
    WORK_DEFAULT="$ROOT/outputs/pretrain_e2e_smoke"
    MORPHBPE_VOCAB=400
    GENERAL_VOCAB=500
    MAX_LENGTH=128
    CONFIG="two_stage_tiny"
    MAX_STEPS=2
    WIKI_LANGS=""
    PRECISION="fp32"
    MAMBA="naive"
fi

# Honor an externally provided WORK; otherwise use the mode-specific default.
WORK="${WORK:-$WORK_DEFAULT}"

CORPUS_DIR="$WORK/corpus"
TOK_DIR="$WORK/tokenizer"
BUNDLE_DIR="$TOK_DIR/bundle"
DATA_DIR="$WORK/data"
RUN_DIR="$WORK/run"
mkdir -p "$CORPUS_DIR" "$TOK_DIR" "$DATA_DIR" "$RUN_DIR"

MN_JSONL="$CORPUS_DIR/mn.jsonl"
GENERAL_JSONL="$CORPUS_DIR/general.jsonl"
ALL_JSONL="$CORPUS_DIR/all.jsonl"
MORPHBPE_JSON="$TOK_DIR/morphbpe.json"
GENERAL_JSON="$TOK_DIR/general.json"
SHARD_JSONL="$DATA_DIR/shard-00.jsonl"
SHARD_RECEIPT_JSON="$DATA_DIR/shard-00.receipt.json"
SHARD_BUILD_SUMMARY="$DATA_DIR/shard-00.build-summary.json"
MIX_REPORT="$DATA_DIR/mix-report.json"

log() { printf '\n==> %s\n' "$*"; }

# ---------------------------------------------------------------------------
# Stage 0: dependency check
# ---------------------------------------------------------------------------
log "[0/5] checking build dependencies"
WIKI_LANGS="$WIKI_LANGS" WIKI_DUMP="$WIKI_DUMP" WIKI_DATE="$WIKI_DATE" python3 - <<'PY'
import importlib.util, glob, os, sys

required = ["tokenizers"]
# Wikipedia sampling needs ``pyarrow`` for a local parquet dump and ``datasets``
# for the HF streaming fallback. Require each only when that path is actually
# taken, so local-only / SMOKE runs stay dependency-light. A language counts as
# "local" only when its shard dir actually holds ``*.parquet`` files; an empty
# or metadata-only dir falls back to streaming (mirrors the sampling loop).
langs = os.environ.get("WIKI_LANGS", "").split()
dump = os.environ.get("WIKI_DUMP", "")
date = os.environ.get("WIKI_DATE", "20231101")
def _has_local(lang):
    return bool(glob.glob(os.path.join(dump, f"{date}.{lang}", "*.parquet")))
need_local = any(_has_local(lang) for lang in langs)
need_stream = any(not _has_local(lang) for lang in langs)
if need_local:
    required.append("pyarrow")
if need_stream:
    required.append("datasets")

missing = [m for m in required if importlib.util.find_spec(m) is None]
if missing:
    sys.stderr.write(
        "Missing tokenizer-build deps: %s\n"
        "Install with:  pip install -e '.[tokenizer-build]'\n" % ", ".join(missing)
    )
    sys.exit(1)
print("build deps available: " + ", ".join(required))
PY

# ---------------------------------------------------------------------------
# Stage 1: prepare corpus -> normalized UTF-8 JSONL
# ---------------------------------------------------------------------------
log "[1/5] preparing corpus"
if [ -s "$MN_JSONL" ] && [ -s "$GENERAL_JSONL" ]; then
    echo "corpus already prepared, skipping"
elif [ "$SMOKE" = "1" ]; then
    cat > "$MN_JSONL" <<'JSONL'
{"text": "ᠮᠣᠩᠭᠣᠯ ᠪᠢᠴᠢᠭ ᠰᠠᠢᠨ ᠪᠠᠢᠨᠠ ᠤᠤ"}
{"text": "ᠡᠨᠡ ᠪᠣᠯ ᠨᠢᠭᠡ ᠳᠠᠷᠠᠭᠠ ᠶᠢᠨ ᠰᠣᠷᠢᠯᠲᠠ"}
JSONL
    cat > "$GENERAL_JSONL" <<'JSONL'
{"text": "你好世界 这是一个测试 中文语料"}
{"text": "hello world this is an english smoke sample"}
{"text": "日本語のテキスト サンプル です"}
{"text": "Сайн байна уу кирилл монгол"}
JSONL
else
    : > "$MN_JSONL"
    : > "$GENERAL_JSONL"
    if [ -n "${MN_CORPUS:-}" ] && [ -d "$MN_CORPUS" ]; then
        echo "extracting Mongolian corpus from $MN_CORPUS"
        python3 -m Tokenizer.tools.prepare_corpus --source mongolian \
            --input "$MN_CORPUS" --output "$MN_JSONL"
    else
        echo "WARNING: MN_CORPUS unset or missing; Mongolian corpus will be empty"
    fi
    if [ -n "${CN_CORPUS:-}" ] && [ -d "$CN_CORPUS" ]; then
        echo "extracting CHINESE bundle from $CN_CORPUS"
        python3 -m Tokenizer.tools.prepare_corpus --source chinese \
            --input "$CN_CORPUS" --output "$GENERAL_JSONL" --append
    else
        echo "WARNING: CN_CORPUS unset or missing; skipping CHINESE bundle"
    fi
    for lang in $WIKI_LANGS; do
        if ls "$WIKI_DUMP/$WIKI_DATE.$lang"/*.parquet >/dev/null 2>&1; then
            echo "sampling wikipedia ($lang, up to $WIKI_LIMIT docs) from local dump"
            python3 -m Tokenizer.tools.prepare_corpus --source wiki \
                --lang "$lang" --limit "$WIKI_LIMIT" --wiki-date "$WIKI_DATE" \
                --input "$WIKI_DUMP" --output "$GENERAL_JSONL" --append
        else
            echo "sampling wikipedia ($lang, up to $WIKI_LIMIT docs) via HF streaming"
            python3 -m Tokenizer.tools.prepare_corpus --source wiki \
                --lang "$lang" --limit "$WIKI_LIMIT" --wiki-date "$WIKI_DATE" \
                --output "$GENERAL_JSONL" --append
        fi
    done
fi
if [ -z "$MIX_MANIFEST" ]; then
    # Default: plain concatenation (byte-proportional mixture).
    cat "$MN_JSONL" "$GENERAL_JSONL" > "$ALL_JSONL"
fi
echo "mn lines:      $(wc -l < "$MN_JSONL")"
echo "general lines: $(wc -l < "$GENERAL_JSONL")"

# ---------------------------------------------------------------------------
# Stage 2: train tokenizers + assemble bundle
# ---------------------------------------------------------------------------
log "[2/5] training tokenizers"
if [ -s "$MORPHBPE_JSON" ]; then
    echo "MorphBPE already trained, skipping"
else
    python3 -m Tokenizer.tools.build_morphbpe \
        --input "$MN_JSONL" --output "$MORPHBPE_JSON" \
        --vocab-size "$MORPHBPE_VOCAB"
fi
if [ -s "$GENERAL_JSON" ]; then
    echo "general BPE already trained, skipping"
else
    python3 -m Tokenizer.tools.build_general_bpe \
        --input "$GENERAL_JSONL" --output "$GENERAL_JSON" \
        --vocab-size "$GENERAL_VOCAB"
fi
if [ -s "$BUNDLE_DIR/config.json" ]; then
    echo "tokenizer bundle already assembled, skipping"
else
    python3 -m Tokenizer.tools.build_unified_tokenizer \
        --morphbpe "$MORPHBPE_JSON" --general "$GENERAL_JSON" --output "$BUNDLE_DIR"
fi

# ---------------------------------------------------------------------------
# Stage 2b: token-weighted corpus mix (optional)
# ---------------------------------------------------------------------------
if [ -n "$MIX_MANIFEST" ]; then
    log "[2b] building token-weighted corpus mix"
    # Delegate staleness to the mixer: --skip-if-fresh rebuilds whenever the
    # manifest, any source shard, or the tokenizer changes (signature in the
    # report), and is a no-op otherwise. all.jsonl's mtime therefore only moves
    # when the mixture actually changed.
    python3 -m Tokenizer.tools.build_corpus_mix \
        --manifest "$MIX_MANIFEST" --output "$ALL_JSONL" \
        --tokenizer-bundle "$BUNDLE_DIR" --report "$MIX_REPORT" --skip-if-fresh
    echo "mix report: $MIX_REPORT"
    # If the mix was (re)built, all.jsonl is now newer than any packed shard, so
    # drop the shard to force Stage 3 to repack from the new mixture. When the
    # mix was fresh (untouched), the shard stays newer and is preserved.
    if [ -f "$ALL_JSONL" ] && [ "$ALL_JSONL" -nt "$SHARD_JSONL" ]; then
        rm -f "$SHARD_JSONL" "$SHARD_RECEIPT_JSON" "$SHARD_BUILD_SUMMARY"
    fi
fi

# ---------------------------------------------------------------------------
# Stage 3: build packed pretraining shards
# ---------------------------------------------------------------------------
log "[3/5] building packed shards"
if [ -s "$SHARD_JSONL" ] \
    && [ -s "$SHARD_RECEIPT_JSON" ] \
    && [ -s "$SHARD_BUILD_SUMMARY" ]; then
    echo "shards already built, skipping"
else
    BUILD_SUMMARY="$(
        python3 -m Tokenizer.tools.build_pretraining_data \
            --tokenizer-bundle "$BUNDLE_DIR" \
            --input "$ALL_JSONL" --output "$SHARD_JSONL" \
            --receipt "$SHARD_RECEIPT_JSON" \
            --max-length "$MAX_LENGTH" --pack
    )"
    printf '%s\n' "$BUILD_SUMMARY"
    SHARD_BUILD_SUMMARY_TMP="${SHARD_BUILD_SUMMARY}.tmp.$$"
    printf '%s\n' "$BUILD_SUMMARY" > "$SHARD_BUILD_SUMMARY_TMP"
    mv -f "$SHARD_BUILD_SUMMARY_TMP" "$SHARD_BUILD_SUMMARY"
fi
DATA_RECEIPT="$(
    python3 - "$SHARD_BUILD_SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
try:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"invalid pretraining build summary {summary_path}: {exc}")
receipt = summary.get("receipt")
if not isinstance(receipt, str) or not receipt:
    raise SystemExit(f"pretraining build summary has no receipt: {summary_path}")
print(receipt)
PY
)"
if [ ! -s "$DATA_RECEIPT" ]; then
    echo "pretraining data receipt is missing or empty: $DATA_RECEIPT" >&2
    exit 1
fi
echo "data receipt: $DATA_RECEIPT"

# ---------------------------------------------------------------------------
# Stage 4: data gate
# ---------------------------------------------------------------------------
log "[4/5] gating data"
python3 -m Tokenizer.evals.pretraining_gate \
    --tokenizer-bundle "$BUNDLE_DIR" --input "$SHARD_JSONL" \
    --max-length "$MAX_LENGTH"

# ---------------------------------------------------------------------------
# Stage 5: pretraining (two-stage mHC core)
# ---------------------------------------------------------------------------
log "[5/5] launching pretraining (config=$CONFIG)"
# The e2e "SMOKE=1" mode is a two-step run over the shards built above, not
# train_rdt's in-memory --smoke mode, so it is receipt-backed as well.
RDT_LINEAGE_ARGS=(
    --tokenizer-bundle "$BUNDLE_DIR"
    --data-receipt "$DATA_RECEIPT"
)
# shellcheck disable=SC2086
python3 -m scripts.train_rdt \
    --config "$CONFIG" \
    --data "$SHARD_JSONL" \
    "${RDT_LINEAGE_ARGS[@]}" \
    --seq-len "$MAX_LENGTH" \
    --max-steps "$MAX_STEPS" \
    --precision "$PRECISION" \
    --mamba "$MAMBA" \
    --output "$RUN_DIR" \
    $TRAIN_ARGS

log "done. artifacts under $WORK"
