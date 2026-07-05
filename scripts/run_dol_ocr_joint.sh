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
#   build   extract packed text rows from align shards (CPU only)
#   probe   100-step joint throughput probe (no checkpoint)
#   full    the joint run; WSD schedule so an early stop in the stable
#           phase still yields a usable latest checkpoint
#   stop    graceful stop via the run's control plane (control.json);
#           the trainer finishes the current step, saves a final
#           checkpoint, and exits 0. Use this for planned stops
#           (data-manifest swap, deadline) -- never pkill.
#
# The full stage only writes its .done marker when the run reached
# MAX_STEPS; a graceful early stop leaves the run resumable. A FAILED
# marker means an unplanned non-zero exit (NaN/OOM) OR a sentinel kill --
# check for $RUNS/joint_v1/SENTINEL_STOP before investigating.
#
# Usage:
#   DATA=~/dolocr/data_v1 RUNS=~/dolocr/runs SSL_CHECKPOINT=~/dolocr/runs/omvt_ssl_v1/latest \
#       scripts/run_dol_ocr_joint.sh full
#
# Env:
#   PY              python executable (default: python3)
#   DATA            dir holding align shards (required)
#   ALIGN_GLOB      align shard glob under DATA (default: 'align_*.jsonl');
#                   must NOT match val/test files -- verify before launch
#   RUNS            output root (required for probe/full/stop)
#   SSL_CHECKPOINT  Phase 1 SSL tower checkpoint (required for probe/full)
#   EVICT           page-cache evict command before GPU stages. Default is
#                   "true" (no-op); on the GB10 box this MUST be set to the
#                   evict_cache.py invocation (page cache competes with CUDA
#                   allocations there), and must NEVER run while another GPU
#                   process is alive.
#   SEQ_LEN         packed text row length (default: 1024)
#   MICRO           micro batch size (default: 8)
#   ACCUM           grad accum steps (default: 2)
#   MIX_EVERY       every Nth optimizer step is an OCR step (default: 3)
#   MAX_STEPS       LR-schedule horizon (default: 60000)
#   KEEP_LAST_N     periodic-checkpoint retention (default: 8; ~11-15GB per
#                   checkpoint -- size against free disk). The sentinel
#                   additionally copies the best-contribution checkpoint to
#                   CKPT_KEEP, so rotation cannot eat the peak.
#   BUNDLE          tokenizer bundle dir for build-stage decode QA (optional
#                   locally, REQUIRED on the box per QA discipline)

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
KEEP_LAST_N="${KEEP_LAST_N:-8}"
ALIGN_GLOB="${ALIGN_GLOB:-align_*.jsonl}"

readonly IMAGE_SIZE=224
readonly D_VISION=512
readonly N_IMAGE_TOKENS=256
readonly PATCH_PRESET=prod

STAGE="${1:-}"
if [ -z "$STAGE" ]; then
    echo "usage: $0 {build|probe|full|stop}" >&2
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
             "remove it manually after investigating (if SENTINEL_STOP is " \
             "present the sentinel killed the run on purpose)." >&2
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

# Shared train_rdt argument list as a bash array: glob patterns stay quoted
# single arguments (argparse-side globbing via _resolve_shards), never
# shell-expanded into multiple words.
_set_train_args() {
    TRAIN_ARGS=(
        --config two_stage_pretrain --mamba official
        --data "$TEXT_DIR/text_*.jsonl"
        --mix-data "$DATA/$ALIGN_GLOB" --mix-every "$MIX_EVERY"
        --multimodal --image-size "$IMAGE_SIZE" --d-vision "$D_VISION"
        --n-image-tokens "$N_IMAGE_TOKENS" --patch-preset "$PATCH_PRESET"
        --init-omvt-checkpoint "$SSL_CHECKPOINT" --use-ema-tower
        --seq-len "$SEQ_LEN"
        --micro-batch-size "$MICRO" --grad-accum-steps "$ACCUM"
        --precision bf16
    )
}

_stage_probe() {
    _require_gpu_env
    local probe_dir="$RUNS/joint_probe"
    _check_no_prior_failure "$probe_dir"
    mkdir -p "$probe_dir"
    trap 'touch "$(_fail_marker_for "$probe_dir")"; echo "run_dol_ocr_joint: probe FAILED" >&2' ERR
    bash -c "$EVICT"
    echo "==> joint probe micro=$MICRO accum=$ACCUM mix_every=$MIX_EVERY"
    _set_train_args
    local t0 t1
    t0=$(date +%s.%N)
    "$PY" -m scripts.train_rdt "${TRAIN_ARGS[@]}" \
        --max-steps 100 --warmup-steps 10 \
        --save-every 1000000 --eval-every 1000000 \
        --output "$probe_dir/run"
    t1=$(date +%s.%N)
    "$PY" -c "dt=($t1-$t0)/100.0; print(f's/step={dt:.3f} (mixed); steps/hour={3600/dt:.0f}')"
    trap - ERR
    echo "run_dol_ocr_joint: probe OK"
}

_latest_step() {
    # numeric step of a run dir's latest checkpoint (0 when none)
    local run_dir="$1" tgt
    if [ -e "$run_dir/latest" ]; then
        tgt="$(basename "$(readlink -f "$run_dir/latest")")"
        echo "${tgt##*_}" | sed 's/^0*//' | grep -E '^[0-9]+$' || echo 0
    else
        echo 0
    fi
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
    trap 'touch "$(_fail_marker_for "$run_dir")"; echo "run_dol_ocr_joint: full FAILED (check $run_dir/SENTINEL_STOP)" >&2' ERR
    bash -c "$EVICT"
    RESUME_ARGS=()
    if [ -e "$run_dir/latest" ]; then
        echo "found existing $run_dir/latest, resuming"
        RESUME_ARGS=(--resume "$run_dir/latest")
    fi
    _set_train_args
    "$PY" -m scripts.train_rdt "${TRAIN_ARGS[@]}" \
        --learning-rate 3e-4 --warmup-steps 1500 \
        --lr-schedule wsd --wsd-stable-ratio 0.9 \
        --max-steps "$MAX_STEPS" \
        --save-every 2000 --keep-last-n "$KEEP_LAST_N" \
        --eval-every 1000000 --log-every 20 \
        --output "$run_dir" \
        "${RESUME_ARGS[@]}"
    trap - ERR
    local reached
    reached="$(_latest_step "$run_dir")"
    if [ "${reached:-0}" -ge "$MAX_STEPS" ]; then
        touch "$done_marker"
        echo "run_dol_ocr_joint: full OK (reached $reached/$MAX_STEPS) -> $run_dir"
    else
        echo "run_dol_ocr_joint: full stopped early at step $reached (resumable) -> $run_dir"
    fi
}

_stage_stop() {
    if [ -z "${RUNS:-}" ]; then
        echo "run_dol_ocr_joint: RUNS env var must be set" >&2
        exit 2
    fi
    "$PY" -c "
from Model.training.status import StatusReporter
StatusReporter.request('$RUNS/joint_v1', 'save', 'stop')
print('queued save+stop for $RUNS/joint_v1 (trainer consumes it within one step)')
"
}

case "$STAGE" in
    build) _stage_build ;;
    probe) _stage_probe ;;
    full) _stage_full ;;
    stop) _stage_stop ;;
    *)
        echo "usage: $0 {build|probe|full|stop}" >&2
        exit 2
        ;;
esac
