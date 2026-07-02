#!/usr/bin/env bash
# DoL-OCR Phase 2: OMVT vision-tower SSL pretraining.
#
# Runs a GPU smoke check first (cheap, catches a broken environment before
# committing to the full run), then the full SSL run over the built
# ssl_all.jsonl. Idempotent: a "done" marker per stage means a rerun after a
# clean finish is a no-op; a partially-completed full run resumes from its
# own "latest" checkpoint automatically. On any failure this script does
# NOT retry — it leaves a FAILED marker and exits non-zero so the operator
# investigates (NaN / OOM / data corruption should not be silently retried).
#
# Usage:
#   DATA=/nvme/dolocr/data RUNS=/nvme/dolocr/runs scripts/run_dol_ocr_phase2.sh
#
# Env:
#   PY       python executable (default: python3)
#   DATA     dir holding ssl_all.jsonl (required)
#   RUNS     output root for run dirs (required)
#   EVICT    page-cache evict command, run before every GPU stage
#            (default: "true", i.e. a no-op; the box sets this to its
#            evict_cache.py invocation)
#   STEPS    full SSL run step count (default: 80000)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-python3}"

if [ -z "${DATA:-}" ]; then
    echo "run_dol_ocr_phase2: DATA env var must be set (dir with ssl_all.jsonl)" >&2
    exit 2
fi
if [ -z "${RUNS:-}" ]; then
    echo "run_dol_ocr_phase2: RUNS env var must be set (output root)" >&2
    exit 2
fi
EVICT="${EVICT:-true}"
STEPS="${STEPS:-80000}"

# ---------------------------------------------------------------------------
# Geometry: single source of truth. Every invocation below in this script
# derives its flags from this block; no literal image/vision/patch/seq-len
# number is repeated anywhere else in this file.
# ---------------------------------------------------------------------------
readonly IMAGE_SIZE=224
readonly D_VISION=512
readonly N_IMAGE_TOKENS=256
readonly PATCH_PRESET=prod
readonly SEQ_LEN=512

RUN_DIR="$RUNS/omvt_ssl_v1"
SMOKE_DONE="$RUN_DIR/.smoke_done"
FULL_DONE="$RUN_DIR/.full_done"
FAILED_MARKER="$RUN_DIR/.FAILED"
mkdir -p "$RUN_DIR"

if [ -f "$FAILED_MARKER" ]; then
    echo "run_dol_ocr_phase2: $FAILED_MARKER exists from a previous failed " \
         "run; remove it manually after investigating before rerunning." >&2
    exit 1
fi

on_error() {
    touch "$FAILED_MARKER"
    echo "run_dol_ocr_phase2: FAILED (see above); marker written to $FAILED_MARKER. " \
         "This script will not auto-retry." >&2
}
trap on_error ERR

SSL_DATA="$DATA/ssl_all.jsonl"
if [ ! -s "$SSL_DATA" ]; then
    echo "run_dol_ocr_phase2: missing or empty $SSL_DATA" >&2
    exit 2
fi

echo "==> [1/2] GPU smoke (image_size=$IMAGE_SIZE d_vision=$D_VISION n_image_tokens=$N_IMAGE_TOKENS patch_preset=$PATCH_PRESET)"
if [ -f "$SMOKE_DONE" ]; then
    echo "smoke already done, skipping"
else
    bash -c "$EVICT"
    "$PY" -m scripts.train_omvt_ssl \
        --smoke \
        --steps 4 --batch-size 2 \
        --image-size "$IMAGE_SIZE" --d-vision "$D_VISION" \
        --compress-to "$N_IMAGE_TOKENS" --patch-preset "$PATCH_PRESET" \
        --device cuda --precision bf16 \
        --output "$RUN_DIR/smoke"
    touch "$SMOKE_DONE"
fi

echo "==> [2/2] full SSL run (steps=$STEPS)"
if [ -f "$FULL_DONE" ]; then
    echo "full run already done, skipping"
else
    bash -c "$EVICT"
    RESUME_ARGS=()
    if [ -e "$RUN_DIR/latest" ]; then
        echo "found existing $RUN_DIR/latest, resuming"
        RESUME_ARGS=(--resume "$RUN_DIR/latest")
    fi
    "$PY" -m scripts.train_omvt_ssl \
        --data "$DATA/ssl_all.jsonl" \
        --image-size "$IMAGE_SIZE" --d-vision "$D_VISION" \
        --compress-to "$N_IMAGE_TOKENS" --patch-preset "$PATCH_PRESET" \
        --ocr-vocab 65536 \
        --device cuda --precision bf16 \
        --ema-decay 0.999 --crop-prob 0.3 --prefetch 4 \
        --batch-size 256 --lr 3e-4 --warmup-steps 2000 \
        --steps "$STEPS" --save-every 2000 \
        --output "$RUN_DIR" \
        "${RESUME_ARGS[@]}"
    touch "$FULL_DONE"
fi

trap - ERR
echo "run_dol_ocr_phase2: OK -> $RUN_DIR"
