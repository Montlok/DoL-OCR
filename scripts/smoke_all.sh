#!/usr/bin/env bash
# Run the pretraining-acceptance smoke workflows end-to-end.
# All paths use synthetic data and complete on CPU.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="${OUT:-/tmp/llm4mglian_smoke}"
mkdir -p "$OUT"

echo "==> [1/5] text RDT smoke (single process)"
python3 -m scripts.train_rdt --config tiny --smoke --precision fp32 --output "$OUT/rdt_single"

echo "==> [2/5] text RDT smoke (DDP 2 workers, gloo)"
python3 -m torch.distributed.run --nproc_per_node=2 --master_port=29512 \
    scripts/train_rdt.py --config tiny --smoke --dist ddp --precision fp32 \
    --output "$OUT/rdt_ddp"

echo "==> [3/5] VLM align smoke (MLP fallback dispatcher + OMVT path)"
python3 -m scripts.train_vlm_align --smoke --steps 2 --seq-len 16 --n-image-tokens 4 \
    --output "$OUT/vlm_align"

echo "==> [4/5] OMVT vision-tower SSL smoke"
python3 -m scripts.train_omvt_ssl --smoke --steps 3 --output "$OUT/omvt_ssl"

if python3 -c "import PIL" >/dev/null 2>&1; then
    echo "==> [5/5] multimodal end-to-end (PIL images → JSONL → trainers)"
    OUT="$OUT/mm" bash "$ROOT/scripts/smoke_multimodal.sh"
else
    echo "==> [5/5] multimodal smoke skipped (Pillow not installed; pip install -e .[image])"
fi

echo "All smoke runs OK."
