#!/usr/bin/env bash
# Visual-contribution sentinel for Phase 3 alignment runs.
#
# Watches a train_vlm_align run dir; whenever a new step checkpoint appears,
# runs a small generative eval twice (real images + --blank-baseline) and
# logs contribution = blank_cer - real_cer. A model that reads the image has
# contribution >> 0; a model that collapsed onto its text prior has ~0 (the
# v1 failure signature: 121.9 vs blank 120.0 at 20k).
#
# With STOP_BELOW set (Phase 3b), two consecutive checkpoints under the
# threshold terminate the training process and leave a SENTINEL_STOP marker,
# so a collapsing full fine-tune cannot burn the remaining GPU budget.
#
# Usage:
#   RUN_DIR=~/dolocr/runs/align_frozen_v2 VAL=~/dolocr/data_v1/val.jsonl \
#   BUNDLE=~/dolocr/bundle_v3b scripts/ocr_sentinel.sh
#
# Env:
#   RUN_DIR      train_vlm_align output dir to watch (required)
#   VAL          eval JSONL (required)
#   BUNDLE       tokenizer bundle dir (required)
#   EVICT        page-cache evict command before each eval (default: true)
#   LIMIT        rows per eval (default: 200)
#   POLL_SEC     poll interval (default: 300)
#   STOP_BELOW   contribution threshold in CER points; unset disables
#                auto-stop (frozen stage), set e.g. 10 for Phase 3b
#   STOP_PATTERN pkill -f pattern for auto-stop (default: scripts.train_vlm_align)
#   MAX_NEW      max new tokens per row (default: 480)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-python3}"
EVICT="${EVICT:-true}"
LIMIT="${LIMIT:-200}"
POLL_SEC="${POLL_SEC:-300}"
STOP_PATTERN="${STOP_PATTERN:-scripts.train_vlm_align}"
MAX_NEW="${MAX_NEW:-480}"

readonly IMAGE_SIZE=224
readonly D_VISION=512
readonly N_IMAGE_TOKENS=256
readonly PATCH_PRESET=prod

for var in RUN_DIR VAL BUNDLE; do
    if [ -z "${!var:-}" ]; then
        echo "ocr_sentinel: $var env var must be set" >&2
        exit 2
    fi
done

LOG="$RUN_DIR/sentinel.log"
mkdir -p "$RUN_DIR"
touch "$LOG"

_log() {
    echo "[sentinel $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

_cer_from() {
    # extract grapheme_cer from an eval_vlm_ocr report line
    grep -o "grapheme_cer=[0-9.]*" "$1" | head -1 | cut -d= -f2
}

last_step=""
below_count=0
_log "watching $RUN_DIR (limit=$LIMIT poll=${POLL_SEC}s stop_below=${STOP_BELOW:-off})"

while true; do
    step_dir=""
    if [ -e "$RUN_DIR/latest" ]; then
        step_dir="$(readlink -f "$RUN_DIR/latest")"
    fi
    if [ -n "$step_dir" ] && [ "$step_dir" != "$last_step" ] && [ -d "$step_dir" ]; then
        last_step="$step_dir"
        tag="$(basename "$step_dir")"
        real_out="$RUN_DIR/sentinel_${tag}_real.txt"
        blank_out="$RUN_DIR/sentinel_${tag}_blank.txt"
        bash -c "$EVICT" || true
        set +e
        "$PY" -m scripts.eval_vlm_ocr \
            --checkpoint "$step_dir" --data "$VAL" --tokenizer-bundle "$BUNDLE" \
            --config two_stage_pretrain --mamba official --device cuda \
            --image-size "$IMAGE_SIZE" --d-vision "$D_VISION" \
            --n-image-tokens "$N_IMAGE_TOKENS" --patch-preset "$PATCH_PRESET" \
            --limit "$LIMIT" --batch-size 8 --max-new-tokens "$MAX_NEW" \
            >"$real_out" 2>&1
        real_rc=$?
        "$PY" -m scripts.eval_vlm_ocr \
            --checkpoint "$step_dir" --data "$VAL" --tokenizer-bundle "$BUNDLE" \
            --config two_stage_pretrain --mamba official --device cuda \
            --image-size "$IMAGE_SIZE" --d-vision "$D_VISION" \
            --n-image-tokens "$N_IMAGE_TOKENS" --patch-preset "$PATCH_PRESET" \
            --limit "$LIMIT" --batch-size 8 --max-new-tokens "$MAX_NEW" \
            --blank-baseline \
            >"$blank_out" 2>&1
        blank_rc=$?
        set -e
        if [ "$real_rc" -ne 0 ] || [ "$blank_rc" -ne 0 ]; then
            _log "$tag eval FAILED (real_rc=$real_rc blank_rc=$blank_rc), see $real_out / $blank_out"
        else
            real_cer="$(_cer_from "$real_out")"
            blank_cer="$(_cer_from "$blank_out")"
            contrib="$("$PY" -c "print(f'{(${blank_cer} - ${real_cer}) * 100:.1f}')")"
            _log "$tag real_cer=$real_cer blank_cer=$blank_cer contribution=${contrib}pts"
            if [ -n "${STOP_BELOW:-}" ]; then
                is_below="$("$PY" -c "print(1 if $contrib < $STOP_BELOW else 0)")"
                if [ "$is_below" = "1" ]; then
                    below_count=$((below_count + 1))
                    _log "$tag below threshold ($contrib < $STOP_BELOW), count=$below_count/2"
                    if [ "$below_count" -ge 2 ]; then
                        _log "two consecutive checkpoints below threshold; stopping training ($STOP_PATTERN)"
                        pkill -f "$STOP_PATTERN" || true
                        touch "$RUN_DIR/SENTINEL_STOP"
                        exit 3
                    fi
                else
                    below_count=0
                fi
            fi
        fi
    fi
    sleep "$POLL_SEC"
done
