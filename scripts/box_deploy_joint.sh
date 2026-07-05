#!/bin/bash
# Deploy sequence for the joint run on the training box. Run from anywhere
# on the box after pulling the repo. Each numbered step is idempotent;
# rerunning after a crash continues where it left off.
#
# Box path conventions: data/runs/bundles under ~/dolocr, repo at ~/DoL-OCR
# (override with REPO=). EVICT must point at the box's page-cache evict
# invocation and must never run while a GPU process is alive.
set -euo pipefail

# Two phases, gated by $1:
#   prep    steps 0-3 (stop CTC, val check, build, probe) -- ends after the
#           probe so the step budget is recomputed from measured s/step
#           BEFORE any long run starts
#   train   steps 4-5 (launch full run, print sentinel gate instructions)
PHASE="${1:-}"
if [ "$PHASE" != "prep" ] && [ "$PHASE" != "train" ]; then
    echo "usage: $0 {prep|train}" >&2
    exit 2
fi

REPO="${REPO:-$HOME/DoL-OCR}"
DATA="$HOME/dolocr/data_v1"
RUNS="$HOME/dolocr/runs"
CKPT_KEEP="$HOME/dolocr/ckpt_keep"
BUNDLE="$HOME/dolocr/bundle_v3b"
SSL="$HOME/dolocr/runs/omvt_ssl_v1/latest"
EVICT="${EVICT:-python3 $HOME/dolocr/evict_cache.py}"
ALIGN_GLOB="${ALIGN_GLOB:-align_*.jsonl}"     # must not match val/test files

cd "$REPO"

if [ "$PHASE" = "prep" ]; then

echo "=== step 0: stop CTC training (user order) ==="
pkill -f "scripts.train_ctc_head" || true
sleep 3
pgrep -af "scripts.train_ctc_head" && { echo "CTC still alive"; exit 1; } || echo "CTC stopped"

echo "=== step 1: val/train disjointness check ==="
python3 - "$DATA" "$ALIGN_GLOB" <<'EOF'
import glob
import hashlib
import json
import sys

data, pattern = sys.argv[1], sys.argv[2]


def row_keys(path):
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            img = (row.get("images") or [None])[0]
            tgt = tuple(
                t for t, l in zip(row["input_ids"], row["labels"]) if l != -100
            )
            yield img, hashlib.sha1(repr(tgt).encode()).hexdigest()


val_imgs, val_tgts = set(), set()
for img, h in row_keys(f"{data}/val.jsonl"):
    val_imgs.add(img)
    val_tgts.add(h)

train_paths = sorted(glob.glob(f"{data}/{pattern}"))
assert train_paths, f"no train shards match {pattern}"
img_overlap = 0
tgt_overlap = 0
for p in train_paths:
    for img, h in row_keys(p):
        if img in val_imgs:
            img_overlap += 1
        if h in val_tgts:
            tgt_overlap += 1

print(
    f"val_rows={len(val_imgs)} train_shards={len(train_paths)} "
    f"image_overlap={img_overlap} target_text_overlap={tgt_overlap}"
)
# Image overlap is a hard leak: refuse to train.
assert img_overlap == 0, "VAL LEAK: val images present in training glob"
# Target-text overlap is inherent to multi-font rendering of the same
# sentences -- report it (it depresses blank_cer via text memorization, so
# contribution reads conservative), do not gate on it.
print("disjointness OK (image-level); note target_text_overlap when reading contribution")
EOF

echo "=== step 2: build packed text shards (CPU; BUNDLE required per QA; no evict in parallel) ==="
DATA="$DATA" ALIGN_GLOB="$ALIGN_GLOB" BUNDLE="$BUNDLE" SEQ_LEN=1024 \
    bash scripts/run_dol_ocr_joint.sh build
# QA gate: compare the [qa] index0 decode line above with the source ref
# text before proceeding.

echo "=== step 3: throughput probe (100 joint steps) ==="
DATA="$DATA" RUNS="$RUNS" SSL_CHECKPOINT="$SSL" EVICT="$EVICT" \
    ALIGN_GLOB="$ALIGN_GLOB" MICRO="${MICRO:-8}" ACCUM="${ACCUM:-2}" \
    bash scripts/run_dol_ocr_joint.sh probe
echo "PREP_DONE. Report s/step above, recompute the step budget, then run: $0 train"
exit 0
fi

echo "=== step 4: launch full joint run (background) ==="
nohup env DATA="$DATA" RUNS="$RUNS" SSL_CHECKPOINT="$SSL" EVICT="$EVICT" \
    ALIGN_GLOB="$ALIGN_GLOB" MICRO="${MICRO:-8}" ACCUM="${ACCUM:-2}" \
    MIX_EVERY="${MIX_EVERY:-3}" MAX_STEPS="${MAX_STEPS:-60000}" KEEP_LAST_N=8 \
    bash scripts/run_dol_ocr_joint.sh full \
    > "$HOME/dolocr/joint_v1.log" 2>&1 &
echo "JOINT_PID=$!"
sleep 90
tail -5 "$HOME/dolocr/joint_v1.log"

echo "=== step 5: sentinel launch gate ==="
cat <<'NOTE'
When runs/joint_v1 has its FIRST checkpoint (step_002000), run ONE manual
sentinel round and verify: load succeeds, real_cer and blank_cer are both
non-zero and differ (a failed eval now exits non-zero in ONCE mode):

  RUN_DIR=$HOME/dolocr/runs/joint_v1 VAL=$HOME/dolocr/data_v1/val.jsonl \
  BUNDLE=$HOME/dolocr/bundle_v3b CKPT_KEEP=$HOME/dolocr/ckpt_keep \
  ONCE=1 LIMIT=100 bash scripts/ocr_sentinel.sh

Only after that gate passes, arm the continuous observe-only sentinel:

  nohup env RUN_DIR=$HOME/dolocr/runs/joint_v1 VAL=$HOME/dolocr/data_v1/val.jsonl \
    BUNDLE=$HOME/dolocr/bundle_v3b CKPT_KEEP=$HOME/dolocr/ckpt_keep \
    LIMIT=200 POLL_SEC=600 bash scripts/ocr_sentinel.sh \
    > $HOME/dolocr/sentinel_joint_v1.log 2>&1 &

Add STOP_BELOW=5 only after the first checkpoint shows positive contribution.
Planned stops (manifest swap / deadline): bash scripts/run_dol_ocr_joint.sh stop
NOTE
echo "DEPLOY_READY"
