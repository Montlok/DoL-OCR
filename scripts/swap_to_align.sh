#!/usr/bin/env bash
# Borrow the GPU from the running mn pretraining for one frozen-alignment
# round, then hand it back. Stages:
#
#   1. graceful-stop the pretrain run (control plane; saves a checkpoint),
#      then arm an EXIT trap so ANY failure path resumes pretraining
#   2. val/train disjointness check over the real align shards (hard gate)
#   3. evict page cache (after the big sequential read, before GPU work)
#   4. generation sanity check on the pretrain checkpoint (GPU, prints
#      held-out prefix continuations for eyeballing; non-fatal)
#   5. phase3 frozen alignment (RDT_CHECKPOINT = pretrain latest,
#      SSL tower init, FROZEN_STEPS default 6000, ~7h at measured 4.2s/step)
#   6. sentinel ONCE eval on the alignment result (real vs blank CER)
#   7. resume pretraining and VERIFY it is alive and stepping
#
# Total GPU borrow ~8-8.5h. Launch detached so an ssh drop cannot orphan it:
#
#   nohup bash scripts/swap_to_align.sh > ~/dolocr/swap.log 2>&1 &
#
# Recovery-only mode (just hand the GPU back):
#   RESUME_ONLY=1 bash scripts/swap_to_align.sh
#
# Env:
#   PRETRAIN_RUN   pretrain output dir (default ~/dolocr/runs/mn_pretrain_v1)
#   FROZEN_STEPS   alignment steps (default 6000)
#   SKIP_GEN       1 = skip step 4
set -uo pipefail

V=${V:-$HOME/jupyterlab/.venv/bin/python3}
PRETRAIN_RUN="${PRETRAIN_RUN:-$HOME/dolocr/runs/mn_pretrain_v1}"
PRETRAIN_DATA="${PRETRAIN_DATA:-$HOME/dolocr/pretrain_data}"
DATA_ROOT="$HOME/dolocr/data_v1"                 # val.jsonl lives here
ALIGN_DATA="$DATA_ROOT/jsonl/align"              # the 354 align shards live here
RUNS="$HOME/dolocr/runs"
BUNDLE="$HOME/dolocr/bundle_v3b"
SSL="$HOME/dolocr/runs/omvt_ssl_v1/latest"
CKPT_KEEP="$HOME/dolocr/ckpt_keep"
FROZEN_STEPS="${FROZEN_STEPS:-6000}"
RUN_TAG="v2"
ALIGN_RUN="$RUNS/align_frozen_$RUN_TAG"
EVICT_PY="$HOME/dolocr/evict_cache.py"
LOG() { echo "[swap $(date '+%m-%d %H:%M:%S')] $*"; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

_resume_pretrain() {
    # Idempotent: never double-start.
    if pgrep -f "scripts\.train_rdt" >/dev/null 2>&1; then
        LOG "trainer already running; not starting another"
        return 0
    fi
    LOG "resuming pretraining from $PRETRAIN_RUN/latest"
    local before
    before=$(grep -c "step=" "$HOME/dolocr/mn_pretrain_v1.log" 2>/dev/null || true)
    before=${before:-0}
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
    LOG "PRETRAIN_PID=$! -- verifying liveness (fast-forward takes minutes)"
    local waited=0 ff_seen=0
    while [ "$waited" -lt 1800 ]; do
        sleep 60
        waited=$((waited + 60))
        if ! pgrep -f "scripts\.train_rdt" >/dev/null 2>&1; then
            LOG "RESUME FAILED: trainer died within ${waited}s; check log tail:"
            tail -5 "$HOME/dolocr/mn_pretrain_v1.log"
            return 1
        fi
        local now
        now=$(grep -c "step=" "$HOME/dolocr/mn_pretrain_v1.log" 2>/dev/null || true)
        now=${now:-0}
        if [ "$now" -gt "$before" ]; then
            LOG "resume verified: new step lines appearing"
            return 0
        fi
        if [ "$ff_seen" = "0" ] && tail -3 "$HOME/dolocr/mn_pretrain_v1.log" 2>/dev/null \
                | grep -q "resume fast-forward"; then
            ff_seen=1
            LOG "fast-forward in progress (data replay); still waiting for first step line"
        fi
    done
    LOG "resume liveness unconfirmed after 30min (process alive, no step line yet); check panel.sh"
    return 0
}

_on_exit() {
    trap - EXIT
    LOG "EXIT trap: ensuring the GPU goes back to pretraining"
    _resume_pretrain
}

if [ "${RESUME_ONLY:-0}" = "1" ]; then
    _resume_pretrain
    exit $?
fi

# --- 0. preflight: evict helper must exist (GB10 page cache vs CUDA) ---
if [ ! -f "$EVICT_PY" ]; then
    alt="$HOME/glmocr-ft-work/real_scan_eval/evict_cache.py"
    if [ -f "$alt" ]; then
        cp "$alt" "$EVICT_PY"
        LOG "copied evict helper from $alt"
    else
        LOG "PREFLIGHT FAILED: no evict helper at $EVICT_PY (page-cache evict is"
        LOG "mandatory before GPU stages on this box); aborting before stopping trainer"
        exit 2
    fi
fi
[ -d "$ALIGN_DATA" ] || { LOG "PREFLIGHT FAILED: $ALIGN_DATA missing"; exit 2; }

# --- 1. graceful stop, then arm the give-back trap ---
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
trap _on_exit EXIT

# --- 2. val/train disjointness over the REAL align shards (hard gate) ---
LOG "val disjointness check over $ALIGN_DATA"
"$V" - "$DATA_ROOT" "$ALIGN_DATA" <<'EOF'
import glob
import json
import sys

data_root, align_dir = sys.argv[1], sys.argv[2]
val_imgs = set()
with open(f"{data_root}/val.jsonl") as f:
    for line in f:
        row = json.loads(line)
        val_imgs.add((row.get("images") or [None])[0])
paths = sorted(glob.glob(f"{align_dir}/*.jsonl"))
assert paths, f"no align shards under {align_dir}"
overlap = 0
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
rc=$?
if [ "$rc" -ne 0 ]; then
    LOG "disjointness check FAILED (rc=$rc); giving the GPU back"
    exit "$rc"
fi

# --- 3. evict the page cache the 64G read just filled (no GPU process alive) ---
LOG "evicting page cache"
"$V" "$EVICT_PY" || LOG "evict returned non-zero (continuing)"

# --- 4. generation sanity check (GPU free now; non-fatal) ---
if [ "${SKIP_GEN:-0}" != "1" ]; then
    LOG "generation check on $(readlink -f "$PRETRAIN_RUN/latest")"
    CKPT="$(readlink -f "$PRETRAIN_RUN/latest")" \
        "$V" scripts/gen_check_box.py || LOG "gen check failed rc=$? (non-fatal)"
fi

# --- 5. frozen alignment (phase3 runs its own EVICT before the GPU stage) ---
LOG "starting phase3 frozen (RDT_CHECKPOINT=$PRETRAIN_RUN/latest, steps=$FROZEN_STEPS)"
PY="$V" RDT_CHECKPOINT="$(readlink -f "$PRETRAIN_RUN/latest")" \
SSL_CHECKPOINT="$SSL" DATA="$ALIGN_DATA" RUNS="$RUNS" CKPT_KEEP="$CKPT_KEEP" \
FROZEN_STEPS="$FROZEN_STEPS" RUN_TAG="$RUN_TAG" EVICT="$V $EVICT_PY" \
    bash scripts/run_dol_ocr_phase3.sh frozen
rc=$?
if [ "$rc" -ne 0 ]; then
    LOG "frozen alignment FAILED (rc=$rc); giving the GPU back"
    exit "$rc"
fi

# --- 6. sentinel ONCE eval (no evict here: keep it light, GPU idle after) ---
LOG "sentinel ONCE eval on $ALIGN_RUN"
PY="$V" RUN_DIR="$ALIGN_RUN" VAL="$DATA_ROOT/val.jsonl" BUNDLE="$BUNDLE" \
CKPT_KEEP="$CKPT_KEEP" BEST_NAME="align_frozen_${RUN_TAG}_best" ONCE=1 LIMIT=100 \
    bash scripts/ocr_sentinel.sh || LOG "sentinel eval failed rc=$? (results may be missing)"

# --- 7. hand the GPU back and verify ---
trap - EXIT
_resume_pretrain
rc=$?
LOG "SWAP_DONE (alignment results: $ALIGN_RUN/sentinel.log)"
exit "$rc"
