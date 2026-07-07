#!/usr/bin/env bash
# Borrow the GPU from the running mn pretraining for one frozen-alignment
# round, then hand it back. Single command, idempotent stages:
#
#   1. graceful-stop the pretrain run (control plane; saves a checkpoint)
#   2. evict page cache (safe now -- no live GPU process)
#   3. val/train disjointness check (image-level hard gate)
#   4. generation sanity check on the pretrain checkpoint (GPU, prints
#      held-out prefix continuations for eyeballing)
#   5. phase3 frozen alignment (RDT_CHECKPOINT = pretrain latest,
#      SSL tower init, FROZEN_STEPS default 6000, ~7h)
#   6. sentinel ONCE eval on the alignment result (real vs blank CER)
#   7. resume pretraining (train_rdt --resume, exact stream replay)
#
# Usage (on the box):
#   bash scripts/swap_to_align.sh          # full sequence
#   RESUME_ONLY=1 bash scripts/swap_to_align.sh   # just step 7 (recovery)
#
# Env:
#   PRETRAIN_RUN   pretrain output dir (default ~/dolocr/runs/mn_pretrain_v1)
#   FROZEN_STEPS   alignment steps (default 6000)
#   SKIP_GEN       1 = skip step 4
set -uo pipefail

V=${V:-$HOME/jupyterlab/.venv/bin/python3}
PRETRAIN_RUN="${PRETRAIN_RUN:-$HOME/dolocr/runs/mn_pretrain_v1}"
PRETRAIN_DATA="${PRETRAIN_DATA:-$HOME/dolocr/pretrain_data}"
DATA="$HOME/dolocr/data_v1"
RUNS="$HOME/dolocr/runs"
BUNDLE="$HOME/dolocr/bundle_v3b"
SSL="$HOME/dolocr/runs/omvt_ssl_v1/latest"
CKPT_KEEP="$HOME/dolocr/ckpt_keep"
FROZEN_STEPS="${FROZEN_STEPS:-6000}"
LOG() { echo "[swap $(date '+%m-%d %H:%M:%S')] $*"; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

_resume_pretrain() {
    LOG "resuming pretraining from $PRETRAIN_RUN/latest"
    nohup "$V" -m scripts.train_rdt \
        --config two_stage_pretrain --mamba official \
        --data "$PRETRAIN_DATA/mn_part_*.jsonl" \
        --eval-data "$PRETRAIN_DATA/eval/mn_part_15.jsonl" \
        --eval-every 4000 --eval-max-batches 25 \
        --seq-len 2048 --micro-batch-size 8 --grad-accum-steps 4 \
        --grad-ckpt-recurrent off --grad-ckpt-prelude-coda off \
        --precision bf16 --learning-rate 3e-4 --warmup-steps 2000 \
        --lr-schedule wsd --wsd-stable-ratio 0.9 --max-steps 300000 \
        --save-every 2000 --keep-last-n 4 --log-every 20 \
        --output "$PRETRAIN_RUN" --resume "$PRETRAIN_RUN/latest" \
        >> "$HOME/dolocr/mn_pretrain_v1.log" 2>&1 &
    LOG "PRETRAIN_PID=$!"
}

if [ "${RESUME_ONLY:-0}" = "1" ]; then
    _resume_pretrain
    exit 0
fi

# --- 1. graceful stop ---
if pgrep -f "scripts\.train_rdt" >/dev/null 2>&1; then
    LOG "requesting graceful stop (save+stop)"
    "$V" -m scripts.rdt_monitor control save --run "$PRETRAIN_RUN" || true
    "$V" -m scripts.rdt_monitor control stop --run "$PRETRAIN_RUN" || true
    waited=0
    while pgrep -f "scripts\.train_rdt" >/dev/null 2>&1; do
        sleep 30
        waited=$((waited + 30))
        if [ "$waited" -ge 1800 ]; then
            LOG "trainer still alive after 30min; aborting swap (NOT killing it)"
            exit 1
        fi
    done
    LOG "pretraining stopped gracefully at $(readlink -f "$PRETRAIN_RUN/latest")"
else
    LOG "no trainer running"
fi

# --- 2. evict page cache (no GPU process alive now) ---
for ev in "$HOME/dolocr/evict_cache.py" "$HOME/evict_cache.py"; do
    [ -f "$ev" ] && { LOG "evict via $ev"; "$V" "$ev" || true; break; }
done

# --- 3. val/train disjointness (image-level hard gate) ---
LOG "val disjointness check"
"$V" - "$DATA" <<'EOF'
import glob
import json
import sys

data = sys.argv[1]
val_imgs = set()
with open(f"{data}/val.jsonl") as f:
    for line in f:
        row = json.loads(line)
        val_imgs.add((row.get("images") or [None])[0])
overlap = 0
paths = sorted(glob.glob(f"{data}/jsonl/*.jsonl")) or sorted(
    glob.glob(f"{data}/align_*.jsonl")
)
assert paths, "no align shards found"
for p in paths:
    with open(p) as f:
        for line in f:
            row = json.loads(line)
            if (row.get("images") or [None])[0] in val_imgs:
                overlap += 1
print(f"val_rows={len(val_imgs)} train_files={len(paths)} image_overlap={overlap}")
assert overlap == 0, "VAL LEAK: refuse to align"
print("disjointness OK")
EOF

# --- 4. generation sanity check (GPU free now) ---
if [ "${SKIP_GEN:-0}" != "1" ] && [ -f "$HOME/gen_check_box.py" ]; then
    LOG "generation check on pretrain checkpoint"
    "$V" "$HOME/gen_check_box.py" || LOG "gen check failed (non-fatal, continuing)"
fi

# --- 5. frozen alignment ---
LOG "starting phase3 frozen (RDT_CHECKPOINT=$PRETRAIN_RUN/latest, steps=$FROZEN_STEPS)"
RDT_CHECKPOINT="$(readlink -f "$PRETRAIN_RUN/latest")" \
SSL_CHECKPOINT="$SSL" DATA="$DATA" RUNS="$RUNS" CKPT_KEEP="$CKPT_KEEP" \
BUNDLE="$BUNDLE" FROZEN_STEPS="$FROZEN_STEPS" EVICT="true" \
    bash scripts/run_dol_ocr_phase3.sh frozen
rc=$?
if [ "$rc" -ne 0 ]; then
    LOG "frozen alignment FAILED (rc=$rc); resuming pretraining anyway"
    _resume_pretrain
    exit "$rc"
fi

# --- 6. sentinel ONCE eval ---
LOG "sentinel ONCE eval on alignment result"
RUN_DIR="$RUNS/align_frozen_v2" VAL="$DATA/val.jsonl" BUNDLE="$BUNDLE" \
CKPT_KEEP="$CKPT_KEEP" ONCE=1 LIMIT=100 \
    bash scripts/ocr_sentinel.sh || LOG "sentinel eval failed (rc=$?)"

# --- 7. hand the GPU back ---
_resume_pretrain
LOG "SWAP_DONE (alignment results in $RUNS/align_frozen_v2/sentinel.log)"
