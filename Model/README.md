# RDT Model — Pretraining Guide

Recurrent Depth Transformer with Mamba3 + Multi-head Latent Attention (MLA), an
ACT (PonderNet) controller, optional bidirectional auxiliary head, and a
two-tier vision pathway (MLP fallback + OMVT — Orientation-aware
Multiscript Vision Tower).

## 1. Module layout

```
Model/
  config.py            # RDTConfig, TrainingConfig, OMVTConfig, tiny/small/base/pretrain
  model.py             # RDTForCausalLM (LM head, vision injection, ACT)
  recurrent.py         # RecurrentCore (fixed + ACT loops, activation checkpointing)
  blocks.py            # StandardBlock, RecurrentBlock, AttnSubLayer, MambaSubLayer
  layers/
    mamba3_layer.py    # Official mamba_ssm.Mamba3 with NaiveSSM fallback
    mla.py             # MLA + MorphologicalRoPE
    ...
  vision.py            # MLPVisionEncoder + dispatcher VisionInjector
  omvt/
    patcher.py         # MultiScalePatcher: vertical/horizontal/square/layout
    router.py          # GeometricRouter (sobel-based, LM-free)
    mixers.py          # VerticalSSM, HorizontalSSM, LocalAttention, LayoutMixer
    compressor.py      # PerceiverCompressor → fixed-N visual tokens
    tower.py           # OMVTVisionTower (full pipeline)
    injector.py        # OMVTInjector: tower + projector + <image_patch> replacement
    native_patcher.py  # packed per-sample arbitrary-resolution patch geometry
    native_planner.py  # budgeted overlap windows + exact ownership proof
    native_tower.py    # ragged native-detail tower (v1 path remains unchanged)
    heads.py / losses.py  # OCR / masked-patch / orientation / layout-order SSL
  posttrain/
    ocr_anyres_builder.py # review-pack and immutable READY dataset producer
    ocr_joint_trainer.py  # visual/joint SFT optimizer cycles
    ocr_anyres_grpo.py    # admitted cache-free anyres GRPO objective
  training/
    data.py            # JSONL + StreamingJsonlDataset + pixel-aware collator + dataloader
    optim.py           # AdamW / Adam-atan2 / Muon (+CombinedOptimizer) + warmup/cosine/WSD
    dist.py            # init_distributed + wrap_ddp + wrap_fsdp
    checkpoint.py      # FSDP-aware save / resume
    loop.py            # train_one_step + evaluate (autocast + grad accum + clip)
    logging.py         # RankZeroLogger (+ optional tensorboard)
    multimodal_cli.py  # shared --multimodal / --image-size / --n-image-tokens helpers
```

## 2. Configs

```python
from Model.config import (
    tiny_config,
    small_config,
    base_config,
    pretrain_config,
    segmented_pretrain_config,
)
cfg = pretrain_config()  # ~1.1B params: d_model=2048, 16 heads, 8 recurrent steps
```

| Config     | d_model | heads | head_dim | layers (prelude / coda) | recurrent steps | seq_len |
|------------|---------|-------|----------|-------------------------|-----------------|---------|
| tiny       | 512     | 8     | 64       | 2 / 2                   | 4               | 2048    |
| small      | 1024    | 16    | 64       | 3 / 3                   | 8               | 4096    |
| base       | 2048    | 32    | 64       | 4 / 4                   | 16              | 8192    |
| pretrain   | 2048    | 16    | 128      | 3 / 3                   | 8               | 4096    |
| segmented_pretrain | 2048 | 16 | 128     | 3 / 3                   | random 2-8      | 4096    |

Activation-memory controls (in `RDTConfig`):

- `grad_ckpt_recurrent`, `grad_ckpt_blocks`, `grad_ckpt_prelude_coda`.
- `bptt_window > 0` truncates BPTT (older recurrent steps are detached).
- `use_act` switches to PonderNet-style adaptive halting with
  `act_max_steps` upper bound; the loop runs the full bound without
  host-syncs so CUDA streams stay pipelined.

### Two-stage core (`core_type="two_stage"`)

`TwoStageCore` (`Model/two_stage.py`) is a drop-in replacement for
`RecurrentCore` with the same forward signature/return contract. It splits the
recurrent core into a **pure-Mamba encoding stage** (order-preserving,
equal-length compression of the *raw* context) followed by a **pure-attention
refinement stage** (recurrent MLA on top of the Mamba backbone — the
Transformer never touches raw context).

```python
from Model.config import two_stage_tiny_config, two_stage_pretrain_config
cfg = two_stage_pretrain_config()  # mHC drift control, official Mamba on CUDA
```

Drift control for the refinement loop (`recurrent_drift_mode`):

- `none` / `norm` / `decay` / `both` — plain recurrent attention with optional
  boundary RMSNorm and/or decayed Stage-1 injection (`recurrent_inject_decay`).
- `mhc` — Manifold-Constrained Hyper-Connections (arXiv:2512.24880) sunk into
  **every attention/ffn residual** via `MHCAttnSubLayer`. Streams are expanded
  once, kept across all steps/layers, and collapsed once; stability comes from
  the per-layer doubly-stochastic (Sinkhorn) constraint, not from loop-level
  injection or boundary norm. Tunables: `mhc_n_streams` (default 4),
  `mhc_sinkhorn_iters` (default 20).

Constraints: `core_type="two_stage"` requires `use_act=False`, and
`two_stage_downsample` must stay `False` (the downsample path is not implemented
for the causal pretraining core — pooling a word's characters would leak
intra-word future, so `True` raises). Tests run under the standard runner:
`python3 -m unittest discover Model` (covers `Model/tests/test_mhc.py` and
`Model/tests/test_two_stage_integration.py`). Optional manual smoke:
`python3 smoke_two_stage.py`.

### Segmented causal core (`core_type="segmented"`)

`SegmentedCore` (`Model/segmented.py`) is the **Block-Transformer-style**
(arXiv:2406.02657) core that runs the expensive attention + RDT recurrence over
**`n_seg` block summaries instead of all `n_tok` tokens**, cutting the quadratic
cost (`n_seg = ceil(L / segment_len) << n_tok`). It is a drop-in replacement for
`RecurrentCore` (same forward signature / `(hidden, info)` contract) and is fully
causal end-to-end — zero future leakage is the red line, guarded by
`Model/tests/test_segmented.py`.

```python
from Model.config import segmented_tiny_config, segmented_pretrain_config
cfg = segmented_pretrain_config()  # segment_len=8, official Mamba on CUDA
```

Pipeline (all forward-only / block-causal):

1. **S1 segment encoding** — causal Mamba over every token; each block's summary
   is the causal hidden state at its boundary (sees only `<=` that token).
2. **S2 block refinement** — block-causal MLA + shared-weight RDT recurrent depth
   over the `n_seg` summaries (Huginn arXiv:2502.05171; random-r + truncated
   BPTT via `recurrent_random_r`, `bptt_window`).
3. **S3 local head** — token `t` (block `s = t // segment_len`) is decoded from
   the refined summary of block `s-1` (block 0 → learned `start_ctx`) plus a
   causal local Mamba decoder; a summary therefore only ever conditions the
   *next* block.

New knobs:

- `segment_len` — block length `L_B` (tiny default 4, pretrain 8; smaller blocks
  model more easily, BD3LM/Block-Transformer).
- `segmented_local_layers` — depth of the S3 local causal decoder.
- `kv_share_budget` — reserved for future cached-decode recurrent KV sharing.
  Keep this at `0` today: cache-equivalent decode needs one MLA cache per
  `(step, layer)`, and the cached path rejects `>0` rather than silently mixing
  recurrent-step histories.
- `kl_exit_threshold` — zero-shot KL early-exit for `generate()` (Huginn §6.1):
  `>0` adaptively stops the recurrent-depth loop once successive step
  distributions converge (cache-free path only); `0` (default) keeps fixed depth
  and is bit-exact. Naturally spends more depth on rare/hard Mongolian segments.

Cached incremental decoding (`generate(..., use_cache=True)`) is supported for
`segmented` (in addition to `two_stage`) and matches the cache-free path within
fp tolerance; it requires the NaiveSSM backend (`--mamba=naive`). Tests:
`Model/tests/test_segmented.py`, `test_early_exit.py`, `test_multilingual.py`.


## 3. Training entry points

All three scripts are CLI-driven and accept `--smoke` for a synthetic
in-memory smoke run. Every non-smoke `train_rdt` run must identify both the
tokenizer bundle and the immutable producer receipt for the exact JSONL shards.
That receipt also names one registered row producer and fingerprints its
current source; unknown or drifted producers are rejected before model
allocation.
If `--eval-data` is set, its separately built `--eval-data-receipt` is required
as well.
Every production/non-smoke receipt-backed train, eval, or mix JSONL row must
persist both `word_pos` and `morph_depth`; model-side reconstruction/fallback is
legacy/smoke-only and is not a production data path.

```bash
# Text RDT pretraining (single process)
python -m scripts.train_rdt --config pretrain \
    --tokenizer-bundle path/to/tokenizer/bundle \
    --data "path/to/shards/*.jsonl" \
    --data-receipt path/to/train.receipt.json \
    --output runs/rdt

# DDP / FSDP
torchrun --nproc_per_node=8 scripts/train_rdt.py --config pretrain \
    --dist fsdp --precision bf16 \
    --tokenizer-bundle path/to/tokenizer/bundle \
    --data "path/to/shards/*.jsonl" \
    --data-receipt path/to/train.receipt.json \
    --output runs/rdt

# OMVT vision-tower SSL (Phase 1 — OCR / masked-patch / orientation / layout-order)
python -m scripts.train_omvt_ssl --output runs/omvt_ssl
#   → real data: --data path/to/mm_shards/*.jsonl

# OMVT → RDT alignment (Phase 3 — vision tokens injected at <image_patch>)
python -m scripts.train_vlm_align --output runs/vlm_align [--freeze-rdt]
#   → real data: --data path/to/mm_shards/*.jsonl [--frozen-vision]

# Joint multimodal RDT pretraining
python -m scripts.train_rdt --config pretrain --multimodal \
    --image-size 64 --n-image-tokens 9 \
    --tokenizer-bundle path/to/tokenizer/bundle \
    --data "path/to/mm_shards/*.jsonl" \
    --data-receipt path/to/mm_train.receipt.json \
    --output runs/rdt_mm
```

Resume: `--resume runs/rdt/latest` (auto-detects FSDP / DDP / single).

### Learning framework (optimizer + LR schedule)

`build_optimizer` / `build_scheduler` (`Model/training/optim.py`) are config-gated
so defaults stay **AdamW + warmup/cosine** (no regression to alignment phases):

- `adam_use_atan2` — replace the Adam update with **Adam-atan2** (arXiv:2407.05872):
  `update = -lr·a·atan2(m̂, b·√v̂)`, eps-free and bf16-underflow-proof. Drop-in.
- `lr_schedule` — `cosine` (default) or `wsd` (**Warmup-Stable-Decay**,
  MiniCPM arXiv:2404.06395): constant plateau then a short decay tail
  controlled by `wsd_stable_ratio` and `wsd_decay_shape`, ideal for a late
  Mongolian-domain decay phase.
- `muon_*` — route 2-D non-embedding weights through **Muon** (Moonlight
  arXiv:2502.16982; quintic Newton-Schulz orthogonalization) while
  embeddings/lm_head/norms/biases stay on AdamW via `CombinedOptimizer`.
  **Experimental only** — shared-weight RDT amplifies gradients `r×`, so the
  Muon×muP interaction is an open question; keep it off by default and document
  any run that enables it. Tests: `Model/tests/test_optim_framework.py`.

## 4. From cold-start to formal pretraining

The end-to-end workflow is `bundle → JSONL → encoded JSONL → train`.

### 4.1 Environment

```bash
pip install -e .[model,train,dist,log]      # core
pip install -e .[image]                     # + Pillow for multimodal
```

`torch>=2.1` is required; `mamba-ssm>=2.2` is required when any production
config (`small`/`base`/`pretrain`) sets `use_official_mamba=True`. CPU
smoke runs use the `NaiveSSM` fallback automatically.

### 4.2 Build a tokenizer bundle

```python
# build_bundle.py
from Tokenizer.unified.bundle import TokenizerBundle

bundle = TokenizerBundle.from_files(
    morphbpe_path="artifacts/morphbpe.json",
    zh_source="Qwen/Qwen2.5-0.5B",   # HF id or local dir
    en_source="meta-llama/Llama-3.2-1B",
    patch_size=14,
    merge_size=2,
)
bundle.save("artifacts/bundle/")
```

Reload later with `TokenizerBundle.from_dir("artifacts/bundle/")`. For
smoke runs, reuse `Tokenizer.tests.test_pretraining_builder.build_smoke_bundle`.

### 4.3 Prepare data

**Text:** point `Tokenizer/tools/build_pretraining_data.py` at a directory
of raw `.txt` / `.jsonl` files; it emits sharded `<name>.jsonl` rows with
`input_ids / attention_mask / labels / word_pos / morph_depth`.

> 文本预训练 JSON 的完整字段级规范（原始输入与编码后分片的各种文本格式、
> 必填/可选字段、`labels` 的 `-100` 语义、长度与同批约束）见
> [`Tokenizer/docs/pretraining_text_format.md`](../Tokenizer/docs/pretraining_text_format.md)。

**Multimodal:** two-stage flow.

```bash
# 1. Pair {stem.png, stem.txt|json} → raw multimodal JSONL.
python -m Tokenizer.tools.build_ocr_data \
    --input  data/raw_ocr/ \
    --output data/raw_mm.jsonl

# 2. Encode the raw rows through the tokenizer bundle, preserving images /
#    ocr_labels / reading_order (they flow through EncodedSample as-is).
python -m Tokenizer.tools.build_pretraining_data \
    --tokenizer-bundle artifacts/bundle/ \
    --input  data/raw_mm.jsonl \
    --output data/mm_shards/shard_00.jsonl \
    --receipt data/mm_shards/train.receipt.json
```

Row schema is documented in
[`Tokenizer/docs/multimodal_data_format.md`](../Tokenizer/docs/multimodal_data_format.md).

**Generative OCR (distinct from the raw-pairing tool above):**
`scripts/build_ocr_data.py` renders synthetic transcription lines to images
*and* tokenizes them in one step (`Tokenizer.tools.build_ocr_data` above only
pairs pre-existing `{image, label}` files, it does not render or tokenize).
For a frozen language model its production target uses the exact native
pretraining route and persists the same `word_pos` / `morph_depth` features.
Every admitted label must be `<unk>`-free and round-trip exactly, including
FVS/MVS/contextual-NNBSP, digits, and punctuation; Mongolian text that would
silently fall back to the general track is rejected. Byte fallback remains an
explicit conversion-only mode for experiments that also retrain the language
side. `--max-seq-len` rejects oversized rows before the expensive render step.
Evaluation (`scripts/eval_ocr.py`, `scripts/eval_vlm_ocr.py`) reports grapheme
CER as the headline metric alongside normalized/raw CER (see
`Model/ocr/metrics.py`).

### 4.4 Pick `--n-image-tokens` carefully (multimodal only)

`MultimodalProcessor` expands every `<image>` placeholder into
`image_patch_count(W, H, patch_size=14, merge_size=2)` `<image_patch>`
slots in the text stream. `OMVTConfig.compress_to` **must** equal this
count, otherwise `inject_visual_features` will refuse the batch.

The current pixel-aware collator also **requires exactly one image per
row**; mixed-cardinality or multi-image batches are rejected explicitly
with a `ValueError`. Use bucketed dataloaders to split multi-image rows
into singletons.

| Image size | Patches per image |
|------------|-------------------|
| 56 × 56    | 4                 |
| 64 × 64    | 9                 |
| 112 × 112  | 16                |
| 224 × 224  | 64                |

CLI: set `--image-size <S>` and `--n-image-tokens <count>` so that
`count == ceil(ceil(S/14)/2)²`. The smoke uses 64 / 9.

### 4.5 Launch pretraining

```bash
# Text-only formal run
torchrun --nproc_per_node=8 scripts/train_rdt.py \
    --config pretrain --dist fsdp --precision bf16 \
    --grad-ckpt-recurrent on --bptt-window 4 \
    --tokenizer-bundle artifacts/bundle/ \
    --data "data/text_shards/*.jsonl" \
    --data-receipt data/text_shards/train.receipt.json \
    --output runs/rdt_pretrain

# Multimodal formal run (pre-aligned OMVT injector + pixel collator)
torchrun --nproc_per_node=8 scripts/train_rdt.py \
    --config pretrain --dist fsdp --precision bf16 \
    --multimodal --image-size 224 --n-image-tokens 64 \
    --grad-ckpt-recurrent on --bptt-window 4 \
    --tokenizer-bundle artifacts/bundle/ \
    --data "data/mm_shards/*.jsonl" \
    --data-receipt data/mm_shards/train.receipt.json \
    --output runs/rdt_pretrain_mm
```

Recommended P0/P1 order (multimodal): warm OMVT alone via
`train_omvt_ssl --data`, then run a short `train_vlm_align --data
--frozen-vision` to settle the projector, then unfreeze for the joint
`train_rdt --multimodal` run.

## 5. Activation-memory strategy (P0)

The recurrent core dominates activation memory:
`recurrent_steps × block_layers × L × d_model`. We control it with three
levers, applied in order from cheapest to most aggressive:

1. **BPTT window** (`cfg.bptt_window=k`): steps before the last `k` run
   under `torch.no_grad`-style detach; activations released immediately.
2. **Recurrent checkpoint** (`grad_ckpt_recurrent=True`): wrap each step's
   block call in `torch.utils.checkpoint(use_reentrant=False)`; trades
   one extra forward for a `~recurrent_steps×` activation-memory cut.
3. **Block / prelude+coda checkpoint** (`grad_ckpt_blocks`,
   `grad_ckpt_prelude_coda`): per-block checkpointing for the static
   prelude/coda stack as well.

Equivalence with non-checkpointed training is covered by
`Model/tests/test_grad_ckpt.py` (loss and embed-grad atol=1e-5).

FSDP uses `transformer_auto_wrap_policy` over
`{StandardBlock, RecurrentBlock, AttnSubLayer, MambaSubLayer}` so the
recurrent steps do **not** share a single shard — that would erase the
sharding benefit.

## 6. Vision pathway

`VisionInjector` is a dispatcher:

- `pixel_values: Tensor`  → `MLPVisionEncoder` (smoke / fallback only).
- `pixel_values: Mapping` → `OMVTInjector` (production path).

### OMVT — Orientation-aware Multiscript Vision Tower

Three-phase roadmap:

1. **Phase 1 (SSL pretraining)** — `scripts/train_omvt_ssl.py`.
   Four heads: OCR reconstruction, masked-patch reconstruction,
   orientation (4-way), layout-order (permutation prediction).
2. **Phase 2 (text-only RDT pretraining)** — `scripts/train_rdt.py`.
   OMVT frozen / unused.
3. **Phase 3 (joint VLM alignment)** — `scripts/train_vlm_align.py`.
   `OMVTVisionTower → PerceiverCompressor → projector → <image_patch>`.
   RDT can be frozen with `--freeze-rdt`.

`GeometricRouter` is intentionally **LM-free**: it derives stream weights
from Sobel-edge statistics (vertical/horizontal/square/layout), so the
vision tower stays self-contained during Phase 1 SSL.

## 7. Testing & smoke

```bash
./scripts/test_all.sh           # Tokenizer + Model unittests + Rust normalizer
./scripts/smoke_all.sh          # 6 acceptance smoke runs (text + DDP + VLM + SSL + multimodal)
./scripts/smoke_multimodal.sh   # multimodal alone (PIL imgs → JSONL → trainers)
```

Acceptance items covered by `smoke_all.sh`:

1. Text RDT pretraining (single process).
2. Text RDT pretraining (DDP × 2, gloo backend, runs on CPU).
3. VLM alignment with OMVT vision tower injected into RDT.
4. OMVT vision-tower SSL with all four heads + losses.
5. OMVT → RDT end-to-end (covered by #3 above).
6. Multimodal end-to-end: PIL images → `build_ocr_data` →
   `build_pretraining_data` → `train_omvt_ssl --data` →
   `train_vlm_align --data` (covered by `smoke_multimodal.sh`).

## 8. Mamba official fail-fast

`Mamba3Layer._build_official` no longer silently swallows constructor
mismatches: a missing parameter raises with an "upgrade `mamba-ssm`"
hint. Production configs (`small_config`, `base_config`,
`pretrain_config`) all set `use_official_mamba=True`. The CPU-only smoke
configs and tests use the `NaiveSSM` fallback explicitly.

## 9. GPU 集群预训练验证清单

These checks must run on the CUDA cluster before the ~1.1B `pretrain_config`
run; a macOS/CPU development host cannot validate them.

Set `DATA_GLOB`, `TOKENIZER_BUNDLE`, and `DATA_RECEIPT` to the real shard glob,
the producing tokenizer bundle, and the builder-emitted receipt before running
the commands below, e.g. `export DATA_GLOB="data/pretrain/*.jsonl"`,
`export TOKENIZER_BUNDLE="artifacts/bundle"`, and
`export DATA_RECEIPT="data/pretrain/train.receipt.json"`.

- [ ] **官方 Mamba** — install CUDA Mamba and prove production configs build
  the upstream backend, not `NaiveSSM`:
  ```bash
  pip install 'mamba-ssm>=2.2'
  python - <<'PY'
  from Model.config import small_config, base_config, pretrain_config
  from Model.layers.mamba3_layer import Mamba3Layer, official_available
  assert official_available(), 'mamba_ssm.modules.mamba3.Mamba3 not importable'
  for make in (small_config, base_config, pretrain_config):
      cfg = make()
      assert cfg.use_official_mamba is True
      layer = Mamba3Layer(cfg, layer_idx=0).cuda()
      assert layer.backend == 'official', layer.backend
      assert layer.mamba.__class__.__name__ == 'Mamba3'
      print(make.__name__, layer.backend)
  PY
  ```
  If constructor mismatch fails in `Mamba3Layer._build_official`, stop and
  upgrade `mamba-ssm`; do not fall back for `small` / `base` / `pretrain`.

- [ ] **FSDP + bf16 + nccl** — run a multi-GPU smoke with the exact distributed
  mode, then inspect that FSDP keeps `use_orig_params=True` and wraps
  `{StandardBlock, RecurrentBlock, AttnSubLayer, MambaSubLayer}`:
  ```bash
  torchrun --standalone --nproc_per_node=2 scripts/train_rdt.py \
      --config tiny --smoke --dist fsdp --precision bf16 --dist-backend nccl \
      --output runs/phase_f_fsdp_smoke
  python - <<'PY'
  import inspect
  from Model.training.dist import wrap_fsdp, _transformer_block_classes
  src = inspect.getsource(wrap_fsdp)
  assert 'use_orig_params=True' in src
  print(sorted(cls.__name__ for cls in _transformer_block_classes()))
  PY
  ```

- [ ] **optimizer 构建顺序 (M6)** — verify optimizer state remains valid across
  pre-wrap build → `apply_parallelism` because FSDP uses `use_orig_params=True`:
  ```bash
  torchrun --standalone --nproc_per_node=2 scripts/train_rdt.py \
      --config tiny --smoke --dist fsdp --precision bf16 --dist-backend nccl \
      --output runs/phase_f_optimizer_order
  ```
  If `use_orig_params` ever changes to `False`, rebuild the optimizer and
  scheduler after `apply_parallelism` in `scripts/train_rdt.py`.

- [ ] **recurrent steps** — confirm distributed runs execute the full scheduled
  recurrent bound (`pretrain_config().recurrent_steps == 8`), not one collapsed
  shard:
  ```bash
  mkdir -p runs
  cat > runs/phase_f_recurrent_check.py <<'PY'
  import torch
  from Model.config import TrainingConfig, pretrain_config
  from Model.model import RDTForCausalLM
  from Model.training.dist import apply_parallelism, init_distributed

  rank, world_size, local_rank = init_distributed('nccl')
  assert world_size > 1
  device = torch.device(f'cuda:{local_rank}')
  cfg = pretrain_config()
  model = RDTForCausalLM(cfg).to(device).train()
  model = apply_parallelism(model, TrainingConfig(parallel='fsdp'), local_rank)
  x = torch.randint(300, 320, (1, 16), device=device)
  out = model(x, labels=x, steps=cfg.recurrent_steps, bptt_window=4)
  assert out['rec_info']['steps_used'] == cfg.recurrent_steps
  if rank == 0:
      print('steps_used', out['rec_info']['steps_used'])
  PY
  torchrun --standalone --nproc_per_node=2 runs/phase_f_recurrent_check.py
  ```

- [ ] **VRAM / throughput 基线** — record max VRAM and tokens/sec for
  `bptt_window` values such as `2`, `4`, `8` across three tiers:
  - full checkpoint: `grad_ckpt_recurrent=True`, `grad_ckpt_prelude_coda=True`
  - partial checkpoint: `grad_ckpt_recurrent=True`, `grad_ckpt_prelude_coda=False`
  - none: `grad_ckpt_recurrent=False`, `grad_ckpt_prelude_coda=False`
  ```bash
  for tier in full partial none; do
    case "$tier" in
      full)    rec=on;  pc=on  ;;
      partial) rec=on;  pc=off ;;
      none)    rec=off; pc=off ;;
    esac
    for bptt in 2 4 8; do
      torchrun --standalone --nproc_per_node=8 scripts/train_rdt.py \
        --config pretrain --dist fsdp --precision bf16 --dist-backend nccl \
        --grad-ckpt-recurrent $rec --grad-ckpt-prelude-coda $pc \
        --bptt-window $bptt --max-steps 20 --save-every 0 \
        --tokenizer-bundle "$TOKENIZER_BUNDLE" \
        --data "$DATA_GLOB" --data-receipt "$DATA_RECEIPT" \
        --output "runs/bench_${tier}_${bptt}"
    done
  done
  ```
  Track `RDTConfig.grad_ckpt_recurrent`, `RDTConfig.grad_ckpt_blocks`,
  `RDTConfig.grad_ckpt_prelude_coda`, and `TrainingConfig.bptt_window`
  (`--bptt-window`). If a branch exposes `use_activation_checkpointing`, include
  it in the matrix; current `RDTConfig` does not define that field.

- [ ] **多卡 checkpoint save/resume e2e** — verify FSDP save → resume is stable
  after the M1 RNG fix (`rng.pt` saves Python, NumPy, torch CPU, and CUDA RNG;
  shuffle seeding uses seed + rank + worker, not `os.getpid()`):
  ```bash
  RDT_RESUME_ARGS=(
      --config tiny --dist fsdp --precision bf16 --dist-backend nccl
      --tokenizer-bundle "$TOKENIZER_BUNDLE"
      --data "$DATA_GLOB" --data-receipt "$DATA_RECEIPT"
      --max-steps 8 --save-every 3
  )
  torchrun --standalone --nproc_per_node=2 scripts/train_rdt.py \
      "${RDT_RESUME_ARGS[@]}" \
      --output runs/phase_f_resume
  # From another terminal after step 3 is durable:
  python -m scripts.rdt_monitor control stop --run runs/phase_f_resume
  # After the first process exits, resume with the identical configuration:
  torchrun --standalone --nproc_per_node=2 scripts/train_rdt.py \
      "${RDT_RESUME_ARGS[@]}" \
      --resume runs/phase_f_resume/latest --output runs/phase_f_resume
  python - <<'PY'
  import torch
  rng = torch.load('runs/phase_f_resume/latest/rng.pt', map_location='cpu', weights_only=False)
  assert {'python', 'numpy', 'cpu', 'cuda'} <= set(rng)
  print(sorted(rng))
  PY
  ```
  `StreamingJsonlDataset` does not serialize a file cursor. Resume verifies the
  exact ordered shard-byte receipt, restores RNG, rebuilds the deterministic
  iterator, and fast-forwards `step × grad_accum_steps` batches before the next
  update. Never use `--no-resume-skip-data` for a production continuation.
