#!/usr/bin/env bash
# DoL-OCR Phase 3: RDT <-> OMVT alignment (VLM training).
#
# Sub-stages, gated by $1:
#   probe            100-step real-data throughput probe at micro-batch
#                    16/32/48/64, printing s/step for each (no checkpoint).
#   frozen           Phase 3a: freeze RDT, train projector+tower from the
#                    Phase 2 SSL checkpoint. Writes ckpt_keep/align_frozen_final
#                    on completion -- the rollback point before unfreezing.
#   unfreeze         Phase 3b: full fine-tune from align_frozen_final. Runs
#                    the mandatory verify-tower-restore pre-check first and
#                    refuses to proceed if it fails.
#   resume-unfreeze  Continue an interrupted 3b run via --resume latest.
#                    Mutually exclusive with any --init flag by construction.
#   verify-tower-restore
#                    Standalone: run the pre-check that "unfreeze" also runs
#                    automatically. Useful to re-verify without starting 3b.
#
# Idempotent: each sub-stage has its own "done" marker; a clean rerun of the
# same sub-stage is a no-op. Failures are NOT retried -- a FAILED marker is
# left and the script exits non-zero (NaN / OOM investigation is manual).
#
# Usage:
#   DATA=/nvme/dolocr/data RUNS=/nvme/dolocr/runs CKPT_KEEP=/nvme/dolocr/ckpt_keep \
#       scripts/run_dol_ocr_phase3.sh frozen
#
# Env:
#   PY             python executable (default: python3)
#   DATA           dir holding align.jsonl / probe data (required for probe/frozen/unfreeze)
#   RUNS           output root for run dirs (required)
#   CKPT_KEEP      long-lived checkpoint root (required for frozen/unfreeze)
#   EVICT          page-cache evict command, run before every GPU stage
#                  (default: "true"; the box sets this to its evict_cache.py)
#   FROZEN_STEPS   Phase 3a step count (default: 2000)
#   KEEP_LAST_N    checkpoint-pruning helper: how many most-recent step dirs
#                  to keep, in addition to every-25000 milestones (default: 3)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-python3}"
EVICT="${EVICT:-true}"
FROZEN_STEPS="${FROZEN_STEPS:-2000}"
KEEP_LAST_N="${KEEP_LAST_N:-3}"

# ---------------------------------------------------------------------------
# Geometry: single source of truth. Every invocation below derives its flags
# from this block; no literal image/vision/patch/seq-len number is repeated
# anywhere else in this file. Must match scripts/run_dol_ocr_phase2.sh's
# block exactly -- Phase 2's SSL checkpoint embeds its own omvt_config and
# is the authoritative geometry source at load time, but this block is what
# derives the CLI-side flags (--image-size, --seq-len, ...) for a
# from-scratch RDT and must not drift from what Phase 2 actually built.
# ---------------------------------------------------------------------------
readonly IMAGE_SIZE=224
readonly D_VISION=512
readonly N_IMAGE_TOKENS=256
readonly PATCH_PRESET=prod
readonly SEQ_LEN=512

STAGE="${1:-}"
if [ -z "$STAGE" ]; then
    echo "usage: $0 {probe|frozen|unfreeze|resume-unfreeze|verify-tower-restore}" >&2
    exit 2
fi

if [ -z "${RUNS:-}" ]; then
    echo "run_dol_ocr_phase3: RUNS env var must be set (output root)" >&2
    exit 2
fi

FROZEN_RUN_DIR="$RUNS/align_frozen_v1"
UNFREEZE_RUN_DIR="$RUNS/align_unfreeze_v1"
VERIFY_RUN_DIR="$RUNS/verify_tower_restore"
mkdir -p "$RUNS"

_require_data() {
    if [ -z "${DATA:-}" ]; then
        echo "run_dol_ocr_phase3: DATA env var must be set" >&2
        exit 2
    fi
}

_require_ckpt_keep() {
    if [ -z "${CKPT_KEEP:-}" ]; then
        echo "run_dol_ocr_phase3: CKPT_KEEP env var must be set" >&2
        exit 2
    fi
    mkdir -p "$CKPT_KEEP"
}

_fail_marker_for() {
    echo "$1/.FAILED"
}

_check_no_prior_failure() {
    local marker
    marker="$(_fail_marker_for "$1")"
    if [ -f "$marker" ]; then
        echo "run_dol_ocr_phase3: $marker exists from a previous failed run; " \
             "remove it manually after investigating before rerunning." >&2
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# probe: 100-step real-data throughput probe at micro 16/32/48/64
# ---------------------------------------------------------------------------
_stage_probe() {
    _require_data
    local probe_dir="$RUNS/throughput_probe"
    _check_no_prior_failure "$probe_dir"
    mkdir -p "$probe_dir"
    trap 'touch "$(_fail_marker_for "$probe_dir")"; echo "run_dol_ocr_phase3: probe FAILED" >&2' ERR

    bash -c "$EVICT"
    for micro in 16 32 48 64; do
        echo "==> probe micro-batch-size=$micro"
        local t0 t1 dt
        t0=$(date +%s.%N)
        "$PY" -m scripts.train_vlm_align \
            --config two_stage_pretrain \
            --data "$DATA" \
            --image-size "$IMAGE_SIZE" --n-image-tokens "$N_IMAGE_TOKENS" \
            --seq-len "$SEQ_LEN" \
            --batch-size "$micro" \
            --steps 100 --smoke \
            --device cuda --precision bf16 \
            --output "$probe_dir/micro_$micro"
        t1=$(date +%s.%N)
        dt=$("$PY" -c "print(($t1 - $t0) / 100.0)")
        echo "micro=$micro s/step=$dt"
    done
    trap - ERR
    echo "run_dol_ocr_phase3: probe OK (re-budget from the s/step numbers above)"
}

# ---------------------------------------------------------------------------
# frozen: Phase 3a
# ---------------------------------------------------------------------------
_stage_frozen() {
    _require_data
    _require_ckpt_keep
    if [ -z "${SSL_CHECKPOINT:-}" ]; then
        echo "run_dol_ocr_phase3: SSL_CHECKPOINT env var must be set to the " \
             "Phase 2 SSL 'latest' checkpoint path" >&2
        exit 2
    fi
    _check_no_prior_failure "$FROZEN_RUN_DIR"
    mkdir -p "$FROZEN_RUN_DIR"
    local done_marker="$FROZEN_RUN_DIR/.done"
    local final_ckpt="$CKPT_KEEP/align_frozen_final"

    if [ -f "$done_marker" ] && [ -d "$final_ckpt" ]; then
        echo "frozen stage already done ($final_ckpt exists), skipping"
        return 0
    fi
    trap 'touch "$(_fail_marker_for "$FROZEN_RUN_DIR")"; echo "run_dol_ocr_phase3: frozen FAILED" >&2' ERR

    bash -c "$EVICT"
    RESUME_ARGS=()
    INIT_ARGS=(--init-omvt-checkpoint "$SSL_CHECKPOINT" --use-ema-tower)
    if [ -e "$FROZEN_RUN_DIR/latest" ]; then
        echo "found existing $FROZEN_RUN_DIR/latest, resuming (dropping --init flags)"
        RESUME_ARGS=(--resume "$FROZEN_RUN_DIR/latest")
        INIT_ARGS=()
    fi
    "$PY" -m scripts.train_vlm_align \
        --config two_stage_pretrain \
        --data "$DATA" \
        --image-size "$IMAGE_SIZE" --n-image-tokens "$N_IMAGE_TOKENS" \
        --seq-len "$SEQ_LEN" \
        --device cuda --precision bf16 \
        "${INIT_ARGS[@]}" \
        --freeze-rdt --steps "$FROZEN_STEPS" --batch-size 32 --grad-ckpt \
        --lr 3e-4 --warmup-steps 500 --save-every 1000 \
        --output "$FROZEN_RUN_DIR" \
        "${RESUME_ARGS[@]}"

    rm -rf "$final_ckpt"
    cp -aL "$FROZEN_RUN_DIR/latest" "$final_ckpt"
    touch "$done_marker"
    trap - ERR
    echo "run_dol_ocr_phase3: frozen OK -> $final_ckpt"
}

# ---------------------------------------------------------------------------
# verify-tower-restore: the mandatory pre-check before unfreezing.
#
# Runs a tiny (2-step) frozen run that actually saves a checkpoint (--smoke
# never checkpoints, see train_vlm_align.py), then loads that checkpoint the
# same way Phase 3b's --init-rdt-checkpoint path does and asserts every
# vision.omvt.tower.* tensor is bit-identical (torch.equal) to what was
# saved. This guards against the OMVT tower being silently constructed with
# strict=False and dropping its trained weights into "unexpected" instead of
# actually loading them.
# ---------------------------------------------------------------------------
_stage_verify_tower_restore() {
    _require_data
    if [ -z "${SSL_CHECKPOINT:-}" ]; then
        echo "run_dol_ocr_phase3: SSL_CHECKPOINT env var must be set to the " \
             "Phase 2 SSL 'latest' checkpoint path" >&2
        exit 2
    fi
    _check_no_prior_failure "$VERIFY_RUN_DIR"
    rm -rf "$VERIFY_RUN_DIR"
    mkdir -p "$VERIFY_RUN_DIR"
    trap 'touch "$(_fail_marker_for "$VERIFY_RUN_DIR")"; echo "run_dol_ocr_phase3: verify-tower-restore FAILED" >&2' ERR

    bash -c "$EVICT"
    echo "==> saving a tiny 2-step frozen checkpoint"
    "$PY" -m scripts.train_vlm_align \
        --config two_stage_pretrain \
        --data "$DATA" \
        --image-size "$IMAGE_SIZE" --n-image-tokens "$N_IMAGE_TOKENS" \
        --seq-len "$SEQ_LEN" \
        --device cuda --precision bf16 \
        --init-omvt-checkpoint "$SSL_CHECKPOINT" --use-ema-tower \
        --freeze-rdt --steps 2 --batch-size 8 --grad-ckpt \
        --lr 3e-4 --warmup-steps 1 --save-every 2 \
        --output "$VERIFY_RUN_DIR"

    echo "==> loading it the Phase 3b way and diffing vision.omvt.tower.*"
    SAVED_CKPT="$VERIFY_RUN_DIR/latest" \
    IMAGE_SIZE="$IMAGE_SIZE" N_IMAGE_TOKENS="$N_IMAGE_TOKENS" \
    D_VISION="$D_VISION" PATCH_PRESET="$PATCH_PRESET" SEQ_LEN="$SEQ_LEN" \
    "$PY" - <<'PYEOF'
import os
import sys
from dataclasses import replace

import torch

sys.path.insert(0, os.getcwd())

from Model.config import two_stage_pretrain_config
from Model.model import RDTForCausalLM
from Model.omvt import OMVTInjector
from Model.training.checkpoint import load_checkpoint
from Model.training.multimodal_cli import make_omvt_cfg

saved_ckpt = os.environ["SAVED_CKPT"]
image_size = int(os.environ["IMAGE_SIZE"])
n_image_tokens = int(os.environ["N_IMAGE_TOKENS"])
d_vision = int(os.environ["D_VISION"])
patch_preset = os.environ["PATCH_PRESET"]
seq_len = int(os.environ["SEQ_LEN"])

# The tensors we compare against: the tower state as it was actually saved
# inside the RDT checkpoint (vision.omvt.tower.* keys of model.pt).
saved_payload = load_checkpoint(saved_ckpt)
saved_state = saved_payload.model_state
saved_tower_keys = {
    k: v for k, v in saved_state.items() if k.startswith("vision.omvt.tower.")
}
if not saved_tower_keys:
    print(
        "verify-tower-restore: FAIL -- saved checkpoint has zero "
        "vision.omvt.tower.* keys (tower was never installed?)",
        file=sys.stderr,
    )
    sys.exit(1)

# Rebuild a model exactly the way Phase 3b's --init-rdt-checkpoint path does:
# construct RDT + a matching-size OMVTInjector, then load_state_dict the
# saved checkpoint (this is what train_vlm_align.py's _load_rdt_init does).
rdt_cfg = replace(two_stage_pretrain_config(), max_seq_len=seq_len)
omvt_cfg = make_omvt_cfg(image_size, d_vision, n_image_tokens, preset=patch_preset)
model = RDTForCausalLM(rdt_cfg)
model.vision._omvt_cfg = omvt_cfg
model.vision.omvt = OMVTInjector(rdt_cfg, omvt_cfg)

missing, unexpected = model.load_state_dict(saved_state, strict=False)
tower_in_unexpected = [k for k in unexpected if k.startswith("vision.omvt.tower.")]
tower_in_missing = [k for k in missing if k.startswith("vision.omvt.tower.")]
if tower_in_unexpected or tower_in_missing:
    print(
        "verify-tower-restore: FAIL -- tower keys landed in "
        f"missing={tower_in_missing[:5]} unexpected={tower_in_unexpected[:5]} "
        "instead of being loaded",
        file=sys.stderr,
    )
    sys.exit(1)

loaded_state = model.state_dict()
mismatches = []
for key, saved_tensor in saved_tower_keys.items():
    loaded_tensor = loaded_state.get(key)
    if loaded_tensor is None:
        mismatches.append((key, "missing from loaded model"))
        continue
    if not torch.equal(loaded_tensor, saved_tensor):
        mismatches.append((key, "value differs"))

if mismatches:
    print(
        f"verify-tower-restore: FAIL -- {len(mismatches)}/"
        f"{len(saved_tower_keys)} vision.omvt.tower.* tensors do not match "
        "the saved checkpoint:",
        file=sys.stderr,
    )
    for key, reason in mismatches[:20]:
        print(f"  {key}: {reason}", file=sys.stderr)
    sys.exit(1)

print(
    f"verify-tower-restore: OK -- all {len(saved_tower_keys)} "
    "vision.omvt.tower.* tensors match exactly"
)
PYEOF
    touch "$VERIFY_RUN_DIR/.done"
    trap - ERR
    echo "run_dol_ocr_phase3: verify-tower-restore OK"
}

# ---------------------------------------------------------------------------
# unfreeze: Phase 3b (runs verify-tower-restore first, refuses on failure)
# ---------------------------------------------------------------------------
_stage_unfreeze() {
    _require_data
    _require_ckpt_keep
    local final_ckpt="$CKPT_KEEP/align_frozen_final"
    if [ ! -d "$final_ckpt" ]; then
        echo "run_dol_ocr_phase3: $final_ckpt does not exist; run the " \
             "'frozen' stage first" >&2
        exit 2
    fi
    _check_no_prior_failure "$UNFREEZE_RUN_DIR"
    mkdir -p "$UNFREEZE_RUN_DIR"

    if [ -e "$UNFREEZE_RUN_DIR/latest" ]; then
        echo "run_dol_ocr_phase3: $UNFREEZE_RUN_DIR/latest already exists; " \
             "use 'resume-unfreeze' instead of 'unfreeze' to continue it " \
             "(--resume and --init-rdt-checkpoint are mutually exclusive)" >&2
        exit 2
    fi

    echo "==> mandatory pre-check: verify-tower-restore"
    _stage_verify_tower_restore

    _check_no_prior_failure "$UNFREEZE_RUN_DIR"
    trap 'touch "$(_fail_marker_for "$UNFREEZE_RUN_DIR")"; echo "run_dol_ocr_phase3: unfreeze FAILED" >&2' ERR

    bash -c "$EVICT"
    "$PY" -m scripts.train_vlm_align \
        --config two_stage_pretrain \
        --data "$DATA" \
        --image-size "$IMAGE_SIZE" --n-image-tokens "$N_IMAGE_TOKENS" \
        --seq-len "$SEQ_LEN" \
        --device cuda --precision bf16 \
        --init-rdt-checkpoint "$final_ckpt" \
        --grad-ckpt --lr 2e-4 --warmup-steps 4000 --steps 120000 \
        --batch-size 32 --save-every 5000 \
        --output "$UNFREEZE_RUN_DIR"
    trap - ERR
    echo "run_dol_ocr_phase3: unfreeze OK -> $UNFREEZE_RUN_DIR"
}

# ---------------------------------------------------------------------------
# resume-unfreeze: continue an interrupted 3b run
# ---------------------------------------------------------------------------
_stage_resume_unfreeze() {
    _require_data
    if [ ! -e "$UNFREEZE_RUN_DIR/latest" ]; then
        echo "run_dol_ocr_phase3: $UNFREEZE_RUN_DIR/latest does not exist; " \
             "nothing to resume -- run 'unfreeze' first" >&2
        exit 2
    fi
    _check_no_prior_failure "$UNFREEZE_RUN_DIR"
    trap 'touch "$(_fail_marker_for "$UNFREEZE_RUN_DIR")"; echo "run_dol_ocr_phase3: resume-unfreeze FAILED" >&2' ERR

    bash -c "$EVICT"
    "$PY" -m scripts.train_vlm_align \
        --config two_stage_pretrain \
        --data "$DATA" \
        --image-size "$IMAGE_SIZE" --n-image-tokens "$N_IMAGE_TOKENS" \
        --seq-len "$SEQ_LEN" \
        --device cuda --precision bf16 \
        --grad-ckpt --lr 2e-4 --warmup-steps 4000 --steps 120000 \
        --batch-size 32 --save-every 5000 \
        --output "$UNFREEZE_RUN_DIR" \
        --resume "$UNFREEZE_RUN_DIR/latest"
    trap - ERR
    echo "run_dol_ocr_phase3: resume-unfreeze OK -> $UNFREEZE_RUN_DIR"
}

# ---------------------------------------------------------------------------
# checkpoint pruning helper: keep last KEEP_LAST_N step dirs + every 25000
# ---------------------------------------------------------------------------
_prune_checkpoints() {
    local run_dir="${1:?usage: _prune_checkpoints <run_dir>}"
    if [ ! -d "$run_dir" ]; then
        echo "run_dol_ocr_phase3: prune target does not exist: $run_dir" >&2
        exit 2
    fi
    local step_dirs
    step_dirs=$(find "$run_dir" -maxdepth 1 -type d -name 'step_*' | sort)
    if [ -z "$step_dirs" ]; then
        echo "no step_* checkpoint dirs under $run_dir"
        return 0
    fi
    local total keep_recent
    total=$(echo "$step_dirs" | wc -l | tr -d ' ')
    keep_recent=$(echo "$step_dirs" | tail -n "$KEEP_LAST_N")
    while IFS= read -r d; do
        [ -z "$d" ] && continue
        local step_num
        step_num=$(basename "$d" | sed 's/^step_0*//')
        step_num=${step_num:-0}
        local is_recent=0
        while IFS= read -r r; do
            [ "$r" = "$d" ] && is_recent=1
        done <<< "$keep_recent"
        local is_milestone=0
        if [ "$step_num" -gt 0 ] && [ "$((step_num % 25000))" -eq 0 ]; then
            is_milestone=1
        fi
        if [ "$is_recent" -eq 0 ] && [ "$is_milestone" -eq 0 ]; then
            echo "pruning $d"
            rm -rf "$d"
        fi
    done <<< "$step_dirs"
    echo "run_dol_ocr_phase3: pruned $run_dir (kept last $KEEP_LAST_N + every 25000, out of $total)"
}

case "$STAGE" in
    probe)
        _stage_probe
        ;;
    frozen)
        _stage_frozen
        _prune_checkpoints "$FROZEN_RUN_DIR"
        ;;
    verify-tower-restore)
        _stage_verify_tower_restore
        ;;
    unfreeze)
        _stage_unfreeze
        _prune_checkpoints "$UNFREEZE_RUN_DIR"
        ;;
    resume-unfreeze)
        _stage_resume_unfreeze
        _prune_checkpoints "$UNFREEZE_RUN_DIR"
        ;;
    prune)
        _prune_checkpoints "${2:?usage: $0 prune <run_dir>}"
        ;;
    *)
        echo "unknown stage: $STAGE (expected probe|frozen|unfreeze|resume-unfreeze|verify-tower-restore|prune)" >&2
        exit 2
        ;;
esac
