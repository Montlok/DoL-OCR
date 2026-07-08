#!/bin/bash
# A800 server-side driver. Runs INSIDE or OUTSIDE the container:
#   outside: bash docker/a800_run.sh probe|full   (wraps docker run)
#   inside:  IN_CONTAINER=1 bash docker/a800_run.sh probe|full
#
# Expected host layout (adjust HOST_* below or via env):
#   $HOST_ROOT/repo   DoL-OCR checkout (this repo)
#   $HOST_ROOT/data   mn_part_*.jsonl + eval/mn_part_15.jsonl
#   $HOST_ROOT/runs   output (persisted)
#   $HOST_ROOT/ckpt   optional: step_XXXX checkpoint from the box to resume
#
# probe: 40-step throughput ladder -- tries micro16/no-ckpt, micro8/no-ckpt,
#        micro8/ckpt-on; prints tok/s for the survivors. A800 has 80G
#        dedicated VRAM (not unified) -- the GB10 memory ceiling does not
#        transfer, so measure rather than assume.
# full:  long run with the chosen config (MICRO/ACCUM/CKPT env), WSD,
#        resumes from $HOST_ROOT/ckpt if present, else from scratch.
set -uo pipefail

HOST_ROOT="${HOST_ROOT:-$HOME/dol}"
IMAGE="${IMAGE:-dol-pretrain:cu124}"
STAGE="${1:-}"
[ -z "$STAGE" ] && { echo "usage: $0 {probe|full}"; exit 2; }

if [ "${IN_CONTAINER:-0}" != "1" ]; then
    exec docker run --rm --gpus all --shm-size=16g \
        -v "$HOST_ROOT/repo:/work/repo" \
        -v "$HOST_ROOT/data:/work/data" \
        -v "$HOST_ROOT/runs:/work/runs" \
        $( [ -d "$HOST_ROOT/ckpt" ] && echo "-v $HOST_ROOT/ckpt:/work/ckpt" ) \
        -e IN_CONTAINER=1 -e MICRO -e ACCUM -e CKPT -e MAX_STEPS \
        "$IMAGE" bash /work/repo/docker/a800_run.sh "$STAGE"
fi

cd /work/repo
export PYTHONPATH=/work/repo
LOG() { echo "[a800 $(date '+%m-%d %H:%M:%S')] $*"; }

train() {
    local micro="$1" accum="$2" ckpt="$3" steps="$4" out="$5" extra=("${@:6}")
    python3 -m scripts.train_rdt \
        --config two_stage_pretrain --mamba official \
        --data "/work/data/mn_part_*.jsonl" \
        --eval-data /work/data/eval/mn_part_15.jsonl \
        --eval-every 4000 --eval-max-batches 25 \
        --seq-len 2048 --micro-batch-size "$micro" --grad-accum-steps "$accum" \
        --grad-ckpt-recurrent "$ckpt" --grad-ckpt-prelude-coda "$ckpt" \
        --precision bf16 --learning-rate 3e-4 --warmup-steps 2000 \
        --lr-schedule wsd --wsd-stable-ratio 0.9 --max-steps "$steps" \
        --save-every 2000 --keep-last-n 4 --log-every 10 \
        --output "$out" "${extra[@]}"
}

case "$STAGE" in
probe)
    for cfg in "16 2 off" "8 4 off" "8 4 on"; do
        set -- $cfg
        LOG "probe micro=$1 accum=$2 ckpt=$3"
        rm -rf /work/runs/probe
        if train "$1" "$2" "$3" 40 /work/runs/probe \
                --warmup-steps 5 --save-every 1000000 --eval-every 1000000 \
                > "/work/runs/probe_${1}_${3}.log" 2>&1; then
            grep "step=" "/work/runs/probe_${1}_${3}.log" | tail -2
        else
            LOG "config micro=$1 ckpt=$3 FAILED (likely OOM), see probe_${1}_${3}.log"
        fi
    done
    rm -rf /work/runs/probe
    LOG "probe done -- pick the fastest surviving config, then: MICRO=.. ACCUM=.. CKPT=.. $0 full"
    ;;
full)
    MICRO="${MICRO:-8}"; ACCUM="${ACCUM:-4}"; CKPT="${CKPT:-off}"
    RESUME=()
    if [ -e /work/ckpt ]; then
        # host-provided box checkpoint: seed the run dir once
        if [ ! -e /work/runs/mn_pretrain_a800/latest ]; then
            mkdir -p /work/runs/mn_pretrain_a800
            step_dir=$(ls -d /work/ckpt/step_* 2>/dev/null | tail -1)
            if [ -n "$step_dir" ]; then
                cp -a "$step_dir" /work/runs/mn_pretrain_a800/
                ln -sfn "$(basename "$step_dir")" /work/runs/mn_pretrain_a800/latest
                LOG "seeded from box checkpoint $(basename "$step_dir")"
            fi
        fi
    fi
    [ -e /work/runs/mn_pretrain_a800/latest ] && RESUME=(--resume /work/runs/mn_pretrain_a800/latest)
    LOG "full run micro=$MICRO accum=$ACCUM ckpt=$CKPT resume=${RESUME[*]:-no}"
    train "$MICRO" "$ACCUM" "$CKPT" "${MAX_STEPS:-300000}" \
        /work/runs/mn_pretrain_a800 "${RESUME[@]}" \
        2>&1 | tee -a /work/runs/mn_pretrain_a800.log
    ;;
*)
    echo "usage: $0 {probe|full}"; exit 2 ;;
esac
