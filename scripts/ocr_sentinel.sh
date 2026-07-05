#!/usr/bin/env bash
# Visual-contribution sentinel for OCR training runs.
#
# Watches a run dir; whenever a new step checkpoint appears, runs ONE
# generative eval with --blank-baseline (real and blank predictions come
# from the same invocation) and logs
#   contribution = blank_grapheme_cer - real_grapheme_cer   (in CER points)
# A model that reads the image has contribution >> 0; one that collapsed
# onto its text prior has ~0 (v1 signature: 121.9 real vs 120.0 blank).
#
# The best-contribution checkpoint is copied to $CKPT_KEEP/${BEST_NAME} so
# trainer-side checkpoint rotation (keep_last_n) can never eat the peak.
#
# With STOP_BELOW set, two consecutive checkpoints under the threshold stop
# the training run: first gracefully via the run's control plane
# (control.json "stop" -- trainer saves and exits 0), with pkill as the
# fallback if the process is still alive after STOP_GRACE_SEC. Leaves a
# SENTINEL_STOP marker either way.
#
# Usage (observe-only):
#   RUN_DIR=~/dolocr/runs/joint_v1 VAL=~/dolocr/data_v1/val.jsonl \
#   BUNDLE=~/dolocr/bundle_v3b CKPT_KEEP=~/dolocr/ckpt_keep scripts/ocr_sentinel.sh
#
# Env:
#   RUN_DIR       training output dir to watch (required)
#   VAL           eval JSONL (required)
#   BUNDLE        tokenizer bundle dir (required)
#   CKPT_KEEP     dir for the best-checkpoint copy (required)
#   BEST_NAME     name of the best-copy dir (default: joint_best)
#   LIMIT         rows per eval (default: 200)
#   POLL_SEC      poll interval (default: 300)
#   ONCE          1 = evaluate current latest once and exit (launch gate)
#   STOP_BELOW    contribution threshold in CER points; unset = observe only
#   STOP_GRACE_SEC seconds to wait after the graceful stop before pkill
#                 (default: 900 -- one full step + checkpoint write)
#   STOP_PATTERN  pkill fallback pattern (default: scripts.train_rdt)
#   MAX_NEW       max new tokens per row (default: 480)
#
# NEVER set EVICT-style page-cache eviction here: the trainer is alive on
# the same GB10 unified memory and an 80GB evict allocation would OOM it.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-python3}"
LIMIT="${LIMIT:-200}"
POLL_SEC="${POLL_SEC:-300}"
ONCE="${ONCE:-0}"
STOP_GRACE_SEC="${STOP_GRACE_SEC:-900}"
STOP_PATTERN="${STOP_PATTERN:-scripts.train_rdt}"
MAX_NEW="${MAX_NEW:-480}"
BEST_NAME="${BEST_NAME:-joint_best}"

readonly IMAGE_SIZE=224
readonly D_VISION=512
readonly N_IMAGE_TOKENS=256
readonly PATCH_PRESET=prod

for var in RUN_DIR VAL BUNDLE CKPT_KEEP; do
    if [ -z "${!var:-}" ]; then
        echo "ocr_sentinel: $var env var must be set" >&2
        exit 2
    fi
done
mkdir -p "$CKPT_KEEP"

LOG="$RUN_DIR/sentinel.log"
BEST_FILE="$RUN_DIR/sentinel_best.txt"
mkdir -p "$RUN_DIR"
touch "$LOG"

_log() {
    echo "[sentinel $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

_real_cer() {
    # grapheme_cer from the real-image aggregate line: "[eval] n=... grapheme_cer=..."
    grep '^\[eval\] n=' "$1" | grep -o "grapheme_cer=[0-9.]*" | head -1 | cut -d= -f2
}

_blank_cer() {
    # grapheme_cer from the blank-baseline aggregate line: "[eval/blank] n=..."
    grep '^\[eval/blank\] n=' "$1" | grep -o "grapheme_cer=[0-9.]*" | head -1 | cut -d= -f2
}

_stop_training() {
    _log "requesting graceful stop via control plane ($RUN_DIR)"
    "$PY" -c "
from Model.training.status import StatusReporter
StatusReporter.request('$RUN_DIR', 'save', 'stop')
" || true
    touch "$RUN_DIR/SENTINEL_STOP"
    local waited=0
    while pgrep -f "$STOP_PATTERN" >/dev/null 2>&1; do
        if [ "$waited" -ge "$STOP_GRACE_SEC" ]; then
            _log "trainer still alive after ${STOP_GRACE_SEC}s; pkill -f $STOP_PATTERN"
            pkill -f "$STOP_PATTERN" || true
            break
        fi
        sleep 30
        waited=$((waited + 30))
    done
    _log "training stopped (SENTINEL_STOP left in $RUN_DIR)"
}

last_step=""
below_count=0
# Numeric sentinel floor, NOT an empty string: this feeds a Python comparison
# below, and quoting inside ${...:-} would leak literal quotes into the
# expression (first-round TypeError).
best_contrib="-1e9"
if [ -f "$BEST_FILE" ]; then
    best_contrib="$(cut -d' ' -f1 "$BEST_FILE" 2>/dev/null || echo "-1e9")"
fi
_log "watching $RUN_DIR (limit=$LIMIT poll=${POLL_SEC}s once=$ONCE stop_below=${STOP_BELOW:-off} best=$best_contrib)"

while true; do
    step_dir=""
    if [ -e "$RUN_DIR/latest" ]; then
        step_dir="$(readlink -f "$RUN_DIR/latest")"
    fi
    if [ -n "$step_dir" ] && [ "$step_dir" != "$last_step" ] && [ -d "$step_dir" ]; then
        last_step="$step_dir"
        tag="$(basename "$step_dir")"
        out="$RUN_DIR/sentinel_${tag}.txt"
        set +e
        "$PY" -m scripts.eval_vlm_ocr \
            --checkpoint "$step_dir" --data "$VAL" --tokenizer-bundle "$BUNDLE" \
            --config two_stage_pretrain --mamba official --device cuda \
            --image-size "$IMAGE_SIZE" --d-vision "$D_VISION" \
            --n-image-tokens "$N_IMAGE_TOKENS" --patch-preset "$PATCH_PRESET" \
            --limit "$LIMIT" --batch-size 8 --max-new-tokens "$MAX_NEW" \
            --blank-baseline \
            >"$out" 2>&1
        rc=$?
        set -e
        if [ "$rc" -ne 0 ]; then
            _log "$tag eval FAILED (rc=$rc), see $out"
        else
            real_cer="$(_real_cer "$out")"
            blank_cer="$(_blank_cer "$out")"
            if [ -z "$real_cer" ] || [ -z "$blank_cer" ]; then
                _log "$tag eval output missing aggregate lines (real='$real_cer' blank='$blank_cer'), see $out"
            else
                contrib="$("$PY" -c "print(f'{(${blank_cer} - ${real_cer}) * 100:.1f}')")"
                _log "$tag real_cer=$real_cer blank_cer=$blank_cer contribution=${contrib}pts"
                is_best="$("$PY" -c "print(1 if $best_contrib < $contrib else 0)")"
                if [ "$is_best" = "1" ]; then
                    best_contrib="$contrib"
                    echo "$contrib $tag $(date '+%m-%d %H:%M:%S')" >"$BEST_FILE"
                    rm -rf "$CKPT_KEEP/$BEST_NAME.tmp"
                    cp -aL "$step_dir" "$CKPT_KEEP/$BEST_NAME.tmp"
                    rm -rf "$CKPT_KEEP/$BEST_NAME"
                    mv "$CKPT_KEEP/$BEST_NAME.tmp" "$CKPT_KEEP/$BEST_NAME"
                    _log "$tag is new best (contribution=${contrib}pts) -> $CKPT_KEEP/$BEST_NAME"
                fi
                if [ -n "${STOP_BELOW:-}" ]; then
                    is_below="$("$PY" -c "print(1 if $contrib < $STOP_BELOW else 0)")"
                    if [ "$is_below" = "1" ]; then
                        below_count=$((below_count + 1))
                        _log "$tag below threshold ($contrib < $STOP_BELOW), count=$below_count/2"
                        if [ "$below_count" -ge 2 ]; then
                            _stop_training
                            exit 3
                        fi
                    else
                        below_count=0
                    fi
                fi
            fi
        fi
        if [ "$ONCE" = "1" ]; then
            if [ "$rc" -ne 0 ]; then
                _log "ONCE mode: eval failed (rc=$rc)"
                exit "$rc"
            fi
            _log "ONCE mode: exiting after one evaluation"
            exit 0
        fi
    elif [ "$ONCE" = "1" ]; then
        _log "ONCE mode: no checkpoint found under $RUN_DIR"
        exit 4
    fi
    sleep "$POLL_SEC"
done
