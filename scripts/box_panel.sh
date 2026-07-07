#!/bin/bash
# Live training panel for the mn pretrain run. Refreshes every 5s.
# Usage: bash ~/panel.sh    (Ctrl-C to exit; training is unaffected)
LOG=$HOME/dolocr/mn_pretrain_v1.log
EPOCH_TOKENS=2140000000   # 2.14B tokens per epoch (mn v3, measured 5.61 B/tok)

while true; do
    clear
    echo "=== DoL mn_pretrain_v1 ($(date '+%m-%d %H:%M:%S')) ==="
    echo
    # --- hardware, real-time ---
    gpu=$(nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw --format=csv,noheader 2>/dev/null)
    echo "GPU  util/temp/power : $gpu"
    mem=$(free -g | awk 'NR==2 {printf "%dG used / %dG total, %dG avail", $3, $2, $7}')
    echo "MEM  $mem"
    swap=$(free -g | awk 'NR==3 {printf "%dG / %dG", $3, $2}')
    echo "SWAP $swap"
    disk=$(df -h / | awk 'NR==2 {printf "%s used / %s (%s)", $3, $2, $5}')
    echo "DISK $disk"
    alive=$(pgrep -c -f "scripts\.train_rdt" 2>/dev/null || echo 0)
    echo "PROC train_rdt processes: $alive (0 = DEAD, investigate)"
    echo
    # --- training, from log (one line per ~10 min) ---
    echo "--- last 4 step lines (new line every ~10 min at current speed) ---"
    grep "step=" "$LOG" 2>/dev/null | tail -4
    echo
    ev=$(grep "eval_loss" "$LOG" 2>/dev/null | tail -1)
    echo "last eval : ${ev:-<first eval at step 4000>}"
    ck=$(ls -dt "$HOME"/dolocr/runs/mn_pretrain_v1/step_* 2>/dev/null | head -1)
    echo "last ckpt : ${ck:-<first checkpoint at step 2000>}"
    echo
    # --- progress arithmetic from the latest step line ---
    last=$(grep "step=" "$LOG" 2>/dev/null | tail -1)
    if [ -n "$last" ]; then
        step=$(echo "$last" | grep -o "step=[0-9]*" | cut -d= -f2)
        tps=$(echo "$last" | grep -o "throughput=[0-9.]*" | cut -d= -f2)
        # tokens so far: step * 64k nominal (micro8 x accum4 x seq2048)
        python3 - "$step" "${tps:-2.0}" "$EPOCH_TOKENS" <<'EOF'
import sys
step, tps_k, epoch = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
toks = step * 65536
tps = tps_k * 1000
pct = toks / epoch * 100
remain_s = (epoch - toks) / tps if tps > 0 else 0
print(f"progress  : step={step}  ~{toks/1e6:.0f}M tokens  = {pct:.2f}% of epoch 1")
print(f"speed     : {tps_k}K tok/s -> {tps*86400/1e9:.2f}B tokens/day")
print(f"epoch 1 ETA: {remain_s/86400:.1f} days from now")
EOF
    fi
    echo
    echo "(Ctrl-C exits panel only; training keeps running)"
    sleep 5
done
