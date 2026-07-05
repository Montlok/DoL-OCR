#!/usr/bin/env bash
# DoL-OCR joint run: RDT text pretraining + OCR alignment in one process.
#
# Interleaves two streams at optimizer-step granularity (see train_rdt
# --mix-data/--mix-every): packed text rows keep the language side fed so
# the OCR steps' loss reduction has to come from the visual pathway, and
# the OMVT tower starts from the Phase 1 SSL checkpoint. Any checkpoint of
# this run doubles as a text-pretraining init for the two-stage fallback
# (run_dol_ocr_phase3.sh frozen with RDT_CHECKPOINT=<ckpt>) if the visual
# contribution collapses.
#
# Sub-stages, gated by $1:
#   build   extract packed text rows from align shards (CPU only; identical
#           to run_dol_ocr_phase2_text.sh build)
#   probe   100-step joint throughput probe (no checkpoint)
#   full    the joint run; WSD schedule so an early stop in the stable
#           phase still yields a usable latest checkpoint
#
# Idempotent; failures leave a FAILED marker and are not retried.
#
# Usage:
#   DATA=~/dolocr/data_v1 RUNS=~/dolocr/runs SSL_CHECKPOINT=~/dolocr/runs/omvt_ssl_v1/latest \
#       scripts/run_dol_ocr_joint.sh full
#
# Env:
#   PY              python executable (default: python3)
#   DATA            dir holding align shards (required)
#   ALIGN_GLOB      align shard glob under DATA (default: 'align_*.jsonl')
#   RUNS            output root (required for probe/full)
#   SSL_CHECKPOINT  Phase 1 SSL tower checkpoint (required for probe/full)
#   EVICT           page-cache evict command before GPU stages (default: true)
#   SEQ_LEN         packed text row length (default: 1024)
#   MICRO           micro batch size (default: 8)
#   ACCUM           grad accum steps (default: 2)
#   MIX_EVERY       every Nth optimizer step is an OCR step (default: 3)
#   MAX_STEPS       LR-schedule horizon (default: 60000)
#   BUNDLE          tokenizer bundle dir for build-stage decode QA (optional)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-python3}"
EVICT="${EVICT:-true}"
SEQ_LEN="${SEQ_LEN:-1024}"
MICRO="${MICRO:-8}"
ACCUM="${ACCUM:-2}"
MIX_EVERY="${MIX_EVERY:-3}"
MAX_STEPS="${MAX_STEPS:-60000}"
ALIGN_GLOB="${ALIGN_GLOB:-align_*.jsonl}"

readonly IMAGE_SIZE=224
readonly D_VISION=512
readonly N_IMAGE_TOKENS=256
readonly PATCH_PRESET=prod

STAGE="${1:-}"
if [ -z "$STAGE" ]; then
    echo "usage: $0 {build|probe|full}" >&2
    exit 2
fi
if [ -z "${DATA:-}" ]; then
    echo "run_dol_ocr_joint: DATA env var must be set" >&2
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
        echo "run_dol_ocr_joint: $marker exists from a previous failed run; " \
             "remove it manually after investigating." >&2
        exit 1
    fi
}

_require_gpu_env() {
    if [ -z "${RUNS:-}" ]; then
        echo "run_dol_ocr_joint: RUNS env var must be set" >&2
        exit 2
    fi
    if [ -z "${SSL_CHECKPOINT:-}" ]; then
        echo "run_dol_ocr_joint: SSL_CHECKPOINT env var must be set" >&2
        exit 2
    fi
    mkdir -p "$RUNS"
}

_stage_build() {
    _check_no_prior_failure "$TEXT_DIR"
    if ls "$TEXT_DIR"/text_*.jsonl >/dev/null 2>&1; then
        echo "build already done ($TEXT_DIR has shards), skipping"
        return 0
    fi
    mkdir -p "$TEXT_DIR"
    trap 'touch "$(_fail_marker_for "$TEXT_DIR")"; echo "run_dol_ocr_joint: build FAILED" >&2' ERR
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
    echo "run_dol_ocr_joint: build OK -> $TEXT_DIR"
}

_train_args() {
    # shared train_rdt argument list; caller appends step counts / output
    echo --config two_stage_pretrain --mamba official \
        --data "$TEXT_DIR/text_*.jsonl" \
        --mix-data "$DATA/$ALIGN_GLOB" --mix-every "$MIX_EVERY" \
        --multimodal --image-size "$IMAGE_SIZE" --d-vision "$D_VISION" \
        --n-image-tokens "$N_IMAGE_TOKENS" --patch-preset "$PATCH_PRESET" \
        --init-omvt-checkpoint "$SSL_CHECKPOINT" --use-ema-tower \
        --seq-len "$SEQ_LEN" \
        --micro-batch-size "$MICRO" --grad-accum-steps "$ACCUM" \
        --precision bf16
}

_stage_probe() {
    _require_gpu_env
    local probe_dir="$RUNS/joint_probe"
    _check_no_prior_failure "$probe_dir"
    mkdir -p "$probe_dir"
    trap 'touch "$(_fail_marker_for "$probe_dir")"; echo "run_dol_ocr_joint: probe FAILED" >&2' ERR
    bash -c "$EVICT"
    echo "==> joint probe micro=$MICRO accum=$ACCUM mix_every=$MIX_EVERY"
    local t0 t1
    t0=$(date +%s.%N)
    # shellcheck disable=SC2046
    "$PY" -m scripts.train_rdt $(_train_args) \
        --max-steps 100 --warmup-steps 10 \
        --save-every 1000000 --eval-every 1000000 \
        --output "$probe_dir/run"
    t1=$(date +%s.%N)
    "$PY" -c "dt=($t1-$t0)/100.0; print(f's/step={dt:.3f} (mixed); steps/hour={3600/dt:.0f}')"
    trap - ERR
    echo "run_dol_ocr_joint: probe OK"
}

_stage_full() {
    _require_gpu_env
    local run_dir="$RUNS/joint_v1"
    _check_no_prior_failure "$run_dir"
    mkdir -p "$run_dir"
    local done_marker="$run_dir/.done"
    if [ -f "$done_marker" ]; then
        echo "full stage already done, skipping"
        return 0
    fi
    trap 'touch "$(_fail_marker_for "$run_dir")"; echo "run_dol_ocr_joint: full FAILED" >&2' ERR
    bash -c "$EVICT"
    RESUME_ARGS=()
    if [ -e "$run_dir/latest" ]; then
        echo "found existing $run_dir/latest, resuming"
        RESUME_ARGS=(--resume "$run_dir/latest")
    fi
    # shellcheck disable=SC2046
    "$PY" -m scripts.train_rdt $(_train_args) \
        --learning-rate 3e-4 --warmup-steps 1500 \
        --lr-schedule wsd --wsd-stable-ratio 0.9 \
        --max-steps "$MAX_STEPS" \
        --save-every 2000 --eval-every 1000000 --log-every 20 \
        --output "$run_dir" \
        "${RESUME_ARGS[@]}"
    touch "$done_marker"
    trap - ERR
    echo "run_dol_ocr_joint: full OK -> $run_dir"
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
