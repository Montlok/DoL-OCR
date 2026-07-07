# -*- coding: utf-8 -*-
"""Generation sanity check for a pretrain checkpoint (box, GPU, bf16, batch 1).

Continues two held-out prefixes greedily and prints prefix vs continuation
for eyeballing. Run only when the GPU is otherwise idle (see RUNBOOK).

Env:
  CKPT    checkpoint dir (default: resolved ~/dolocr/runs/mn_pretrain_v1/latest)
  EVAL    held-out packed jsonl (default: ~/dolocr/pretrain_data/eval/mn_part_15.jsonl)
  BUNDLE  tokenizer bundle dir (default: ~/dolocr/bundle_v3b)
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, REPO)

from Model.config import RDTConfig  # noqa: E402
from Model.model import RDTForCausalLM  # noqa: E402
from Tokenizer.unified import TokenizerBundle  # noqa: E402

CKPT = os.path.realpath(
    os.environ.get("CKPT", os.path.expanduser("~/dolocr/runs/mn_pretrain_v1/latest"))
)
EVAL = os.environ.get(
    "EVAL", os.path.expanduser("~/dolocr/pretrain_data/eval/mn_part_15.jsonl")
)
BUNDLE = os.environ.get("BUNDLE", os.path.expanduser("~/dolocr/bundle_v3b"))
print(f"[gen_check] checkpoint: {CKPT}")

meta = torch.load(f"{CKPT}/meta.pt", map_location="cpu", weights_only=False)
cfg_dict = meta["metadata"]["rdt_config"] if "metadata" in meta else meta["rdt_config"]
cfg = RDTConfig(**cfg_dict)

model = RDTForCausalLM(cfg)
state = torch.load(f"{CKPT}/model.pt", map_location="cpu", weights_only=False)
if isinstance(state, dict) and "model" in state:
    state = state["model"]
model.load_state_dict(state)
del state
model = model.to(torch.bfloat16).cuda().eval()
print("[gen_check] loaded on cuda bf16")

bundle = TokenizerBundle.from_dir(BUNDLE)
row = json.loads(open(EVAL).readline())["input_ids"]

for i, pre in enumerate([row[0:24], row[900:924]]):
    ids = torch.tensor([pre], dtype=torch.long, device="cuda")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=40, greedy=True, use_cache=True,
            repetition_penalty=1.1, eos_id=cfg.eos_id,
        )
    dt = time.time() - t0
    cont = out[0, len(pre):].tolist()
    print(f"\n=== held-out prefix {i} (greedy, {dt:.1f}s) ===", flush=True)
    print("PREFIX:", bundle.tokenizer.decode([t for t in pre if t > 3]))
    print("MODEL :", bundle.tokenizer.decode([t for t in cont if t > 3]))

print("\nGEN_CHECK_DONE")
