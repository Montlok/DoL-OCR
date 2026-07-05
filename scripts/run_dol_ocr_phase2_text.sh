#!/usr/bin/env bash
# DoL-OCR Phase 2 (text): RDT text pretraining on packed OCR-target rows.
#
# This is the LM cold-start stage the alignment recipe assumes: Phase 3a
# freezes the RDT and trains projector+tower against it, which is only
# meaningful when the frozen LM already models the target text distribution.
#
# Sub-stages, gated by $1:
#   build   extract packed text rows from align shards (CPU only)
#   probe   100-step throughput probe at micro 4/8/12 (no checkpoint)
#   full    the pretraining run; WSD schedule so an early stop in the
#           stable phase still yields a usable latest checkpoint
#
# Idempotent: build skips when the output dir already has shards; full
# resumes from its own latest checkpoint. Failures leave a FAILED marker
# and are not retried.
#
# Usage:
#   DATA=~/dolocr/data_v1 RUNS=~/dolocr/runs scripts/run_dol_ocr_phase2_text.sh build
#
# Env:
#   PY           python executable (default: python3)
#   DATA         dir holding align shards (required)
#   ALIGN_GLOB   align shard glob relative to DATA (default: 'align_*.jsonl')
#   RUNS         output root for run dirs (required for probe/full)
#   EVICT        page-cache evict command before GPU stages (default: true)
#   SEQ_LEN      packed row length (default: 1024)
#   MICRO        micro batch size for full (default: 8)
#   ACCUM        grad accum steps for full (default: 2)
#   MAX_STEPS    LR-schedule horizon; the run may be stopped earlier inside
#                the WSD stable phase (default: 60000)
#   BUNDLE       tokenizer bundle dir for build-stage decode QA (optional)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-python3}"
EVICT="${EVICT:-true}"
SEQ_LEN="${SEQ_LEN:-1024}"
MICRO="${MICRO:-8}"
ACCUM="${ACCUM:-2}"
MAX_STEPS="${MAX_STEPS:-60000}"
ALIGN_GLOB="${ALIGN_GLOB:-align_*.jsonl}"

STAGE="${1:-}"
if [ -z "$STAGE" ]; then
    echo "usage: $0 {build|probe|full}" >&2
    exit 2
fi
if [ -z "${DATA:-}" ]; then
    echo "run_dol_ocr_phase2_text: DATA env var must be set" >&2
    exit 2
fi

TEXT_DIR="$DATA/text_pretrain"

_fail_marker_for() {
    echo "$1/.FAILED"
}

_check_no_prior_failure() {
    local marker
    marker="$(_fail_marker_for "$1")"
    if [ -f "$marker" ]; then
        echo "run_dol_ocr_phase2_text: $marker exists from a previous failed " \
             "run; remove it manually after investigating." >&2
        exit 1
    fi
}

_stage_build() {
    _check_no_prior_failure "$TEXT_DIR"
    if ls "$TEXT_DIR"/text_*.jsonl >/dev/null 2>&1; then
        echo "build already done ($TEXT_DIR has shards), skipping"
        return 0
    fi
    mkdir -p "$TEXT_DIR"
    trap 'touch "$(_fail_marker_for "$TEXT_DIR")"; echo "run_dol_ocr_phase2_text: build FAILED" >&2' ERR
    BUNDLE_ARGS=()
    if [ -n "${BUNDLE:-}" ]; then
        BUNDLE_ARGS=(--bundle "$BUNDLE")
    fi
    "$PY" -m scripts.build_text_rows_from_align \
        --align "$DATA/$ALIGN_GLOB" \
        --output "$TEXT_DIR" \
        --seq-len "$SEQ_LEN" --shard-rows 200000 \
        "${BUNDLE_ARGS[@]}"
    trap - ERR
    echo "run_dol_ocr_phase2_text: build OK -> $TEXT_DIR"
}

_require_runs() {
    if [ -z "${RUNS:-}" ]; then
        echo "run_dol_ocr_phase2_text: RUNS env var must be set" >&2
        exit 2
    fi
    mkdir -p "$RUNS"
}

_stage_probe() {
    _require_runs
    local probe_dir="$RUNS/text_pretrain_probe"
    _check_no_prior_failure "$probe_dir"
    mkdir -p "$probe_dir"
    trap 'touch "$(_fail_marker_for "$probe_dir")"; echo "run_dol_ocr_phase2_text: probe FAILED" >&2' ERR
    bash -c "$EVICT"
    for micro in 4 8 12; do
        echo "==> probe micro-batch-size=$micro"
        local t0 t1
        t0=$(date +%s.%N)
        "$PY" -m scripts.train_rdt \
            --config two_stage_pretrain --mamba official \
            --data "$TEXT_DIR/text_*.jsonl" \
            --seq-len "$SEQ_LEN" \
            --micro-batch-size "$micro" --grad-accum-steps 1 \
            --max-steps 100 --warmup-steps 10 \
            --save-every 1000000 --eval-every 1000000 \
            --precision bf16 \
            --output "$probe_dir/micro_$micro"
        t1=$(date +%s.%N)
        "$PY" -c "dt=($t1-$t0)/100.0; print(f'micro=$micro s/step={dt:.3f} tok/s={$micro*$SEQ_LEN/dt:.0f}')"
    done
    trap - ERR
    echo "run_dol_ocr_phase2_text: probe OK (pick MICRO/ACCUM from the numbers above)"
}

_stage_full() {
    _require_runs
    local run_dir="$RUNS/text_pretrain_v1"
    _check_no_prior_failure "$run_dir"
    mkdir -p "$run_dir"
    local done_marker="$run_dir/.done"
    if [ -f "$done_marker" ]; then
        echo "full stage already done, skipping"
        return 0
    fi
    trap 'touch "$(_fail_marker_for "$run_dir")"; echo "run_dol_ocr_phase2_text: full FAILED" >&2' ERR
    bash -c "$EVICT"
    RESUME_ARGS=()
    if [ -e "$run_dir/latest" ]; then
        echo "found existing $run_dir/latest, resuming"
        RESUME_ARGS=(--resume "$run_dir/latest")
    fi
    "$PY" -m scripts.train_rdt \
        --config two_stage_pretrain --mamba official \
        --data "$TEXT_DIR/text_*.jsonl" \
        --seq-len "$SEQ_LEN" \
        --micro-batch-size "$MICRO" --grad-accum-steps "$ACCUM" \
        --learning-rate 3e-4 --warmup-steps 1500 \
        --lr-schedule wsd --wsd-stable-ratio 0.9 \
        --max-steps "$MAX_STEPS" \
        --save-every 2000 --eval-every 1000000 --log-every 20 \
        --precision bf16 \
        --output "$run_dir" \
        "${RESUME_ARGS[@]}"
    touch "$done_marker"
    trap - ERR
    echo "run_dol_ocr_phase2_text: full OK -> $run_dir"
}

case "$STAGE" in
    build) _stage_build ;;
    probe) _stage_probe ;;
    full) _stage_full ;;
    *)
        echo "usage: $0 {build|probe|full}" >&2
        exit 2
        ;;
esac
