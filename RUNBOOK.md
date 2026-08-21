# DoL-OCR operations runbook

This runbook covers the supported production lineage only: receipt-bound pretraining and the AnyRes Visual SFT → Joint SFT → GRPO → locked-evaluation sequence. Keep machine addresses, mount points, credentials, active run IDs, incident notes, and data locations outside Git.

## 1. Required run record

Before allocating a GPU, record in the external operations system:

- repository commit SHA and clean/dirty state;
- exact Python, PyTorch, CUDA, driver and `mamba-ssm` versions;
- model, tokenizer, OMVT and AnyRes preprocess configuration;
- every input path, receipt path and SHA-256 identity;
- source checkpoint path, repository revision and file hashes;
- train/validation/selection/monitor/locked split policy and exclusions;
- optimizer, precision, batch, schedule, save, eval and stop settings;
- output directory, log destination, operator and start time.

Do not launch when a path is ambiguous, a required hash is missing, a previous output directory would be overwritten, or any split overlaps by source ID, content hash, document group or resolved path.

## 2. Environment and repository preflight

Use explicit external paths. The repository itself must not contain datasets or weights.

```bash
export PYTHON_BIN="${PYTHON_BIN:?set production Python}"
export TOKENIZER_DIR="${TOKENIZER_DIR:?set external tokenizer bundle}"
export SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:?set downloaded DoL-1.2-OCR checkpoint}"
export OUTPUT_DIR="${OUTPUT_DIR:?set a new or resumable output directory}"

git status --short
git rev-parse HEAD
"$PYTHON_BIN" scripts/check_repository_hygiene.py
"$PYTHON_BIN" -m pytest -q Model/tests Tokenizer/tests
```

For a production GPU run, also verify accelerator visibility, free storage, input readability and that no unrelated process owns the target GPU. Production RDT configuration must load official Mamba; NaiveSSM is a CPU test fallback, not a production substitute.

AnyRes initialization must authenticate exactly:

```text
Montlok/DoL-1.2-OCR@bee908ab2a9376f6224dff514564ebb0ae99a643
```

The source bytes must match [`Model/posttrain/release_locks/dol_1_2_ocr.json`](Model/posttrain/release_locks/dol_1_2_ocr.json). Do not select a checkpoint by filename, `latest`, step alias or directory order.

## 3. Receipt-bound pretraining

### 3.1 Tokenizer and text rows

Build a tokenizer bundle and encoded shards outside the repository. Retain the receipt emitted by the builder:

```bash
"$PYTHON_BIN" -m Tokenizer.tools.build_pretraining_data \
  --tokenizer-bundle "$TOKENIZER_DIR" \
  --input "$RAW_TEXT_INPUT" \
  --output "$ENCODED_TRAIN_SHARD" \
  --receipt "$TRAIN_DATA_RECEIPT" \
  --max-length "$MAX_SEQUENCE_LENGTH" \
  --pack
```

Build validation shards independently and use a different receipt. A formal run must not synthesize or guess receipt paths; it consumes the exact builder output. The receipt binds the tokenizer implementation, bundle files, producer fingerprint, ordered shard names, sizes and hashes.

Run the data gate before training:

```bash
"$PYTHON_BIN" -m Tokenizer.evals.pretraining_gate \
  --tokenizer-bundle "$TOKENIZER_DIR" \
  --input "$ENCODED_TRAIN_SHARD" \
  --max-length "$MAX_SEQUENCE_LENGTH" \
  --json
```

### 3.2 RDT language pretraining

```bash
"$PYTHON_BIN" -m scripts.train_rdt \
  --config two_stage_pretrain \
  --mamba official \
  --tokenizer-bundle "$TOKENIZER_DIR" \
  --data "$TRAIN_DATA_GLOB" \
  --data-receipt "$TRAIN_DATA_RECEIPT" \
  --eval-data "$VALIDATION_DATA_GLOB" \
  --eval-data-receipt "$VALIDATION_DATA_RECEIPT" \
  --output "$OUTPUT_DIR" \
  "${EXTRA_ARGS[@]}"
```

DDP/FSDP launches use the same data and receipt contract. Record the exact `torchrun` command and world size. Resume only with an unchanged model shape, tokenizer identity, ordered data receipts, optimizer schedule, recurrence settings and seed.

### 3.3 OMVT and VLM alignment

The supported visual preparation sequence is:

1. OMVT SSL or CTC-supervised tower training;
2. OMVT-to-RDT alignment using native OCR targets;
3. a final checkpoint with a terminal stop reason and authenticated frozen-language lineage.

Representative launches:

```bash
"$PYTHON_BIN" -m scripts.train_omvt_ssl \
  --data "$VISION_TRAIN_DATA" \
  --output "$OMVT_OUTPUT" \
  "${EXTRA_ARGS[@]}"

"$PYTHON_BIN" -m scripts.train_vlm_align \
  --freeze-rdt \
  --ocr-native-targets \
  --ocr-tokenizer-bundle "$TOKENIZER_DIR" \
  --ocr-data-contract "$ALIGNMENT_DATA_CONTRACT" \
  --data "$ALIGNMENT_DATA" \
  --init-rdt-checkpoint "$LANGUAGE_CHECKPOINT" \
  --init-omvt-checkpoint "$OMVT_CHECKPOINT" \
  --output "$ALIGNMENT_OUTPUT" \
  "${EXTRA_ARGS[@]}"
```

The alignment path refuses tokenizer drift, non-native targets and incomplete visual runs. A periodic checkpoint is not automatically eligible merely because its files exist.

## 4. Checkpoint, resume and monitoring

A resumable checkpoint must contain the model, optimizer, scheduler, RNG, data cursor, precision state and configuration required by that stage. Never repoint `latest` to a partially written directory.

```bash
"$PYTHON_BIN" -m scripts.rdt_monitor tui --run "$OUTPUT_DIR"
"$PYTHON_BIN" -m scripts.rdt_monitor control save --run "$OUTPUT_DIR"
"$PYTHON_BIN" -m scripts.rdt_monitor control stop --run "$OUTPUT_DIR"
```

Use the control plane for planned stops. Wait for an atomically published complete checkpoint and process exit before changing mounts or releasing the accelerator. After resume, verify new optimizer-step progress rather than process liveness alone.

## 5. AnyRes dataset admission

AnyRes accepts arbitrary source image dimensions. It preserves a lossless post-EXIF canonical image, plans macro windows with overlapping context and disjoint ownership, and requires manual cut-QA. Internal preprocessing must never silently stretch the full source into a square.

Prepare an immutable review pack:

```bash
"$PYTHON_BIN" -m scripts.build_ocr_anyres_dataset prepare-review \
  --sources "$SOURCES_JSONL" \
  --source-root "$SOURCE_ROOT" \
  --preprocess-contract "$ANYRES_PREPROCESS" \
  --out "$REVIEW_PACK"
```

After human review of Unicode, provenance, difficulty, reading order and every cut decision, finalize into a new directory:

```bash
"$PYTHON_BIN" -m scripts.build_ocr_anyres_dataset finalize \
  --sources "$SOURCES_JSONL" \
  --approved-reviews "$APPROVED_REVIEWS_JSONL" \
  --source-root "$SOURCE_ROOT" \
  --preprocess-contract "$ANYRES_PREPROCESS" \
  --tokenizer "$TOKENIZER_DIR" \
  --split-policy "$SPLIT_POLICY" \
  --split-lock "$SPLIT_LOCK" \
  --out "$ANYRES_DATASET"
```

Validate the complete READY chain before model allocation:

```bash
"$PYTHON_BIN" -m scripts.validate_ocr_anyres_dataset \
  --root "$ANYRES_DATASET" \
  --assets "$ANYRES_DATASET/assets.jsonl" \
  --views "$ANYRES_DATASET/views.jsonl" \
  --train-samples "$ANYRES_DATASET/train.jsonl" \
  --sft-validation-samples "$ANYRES_DATASET/sft_validation.jsonl" \
  --kl-selection-samples "$ANYRES_DATASET/kl_selection.jsonl" \
  --formal-monitor-samples "$ANYRES_DATASET/formal_monitor.jsonl" \
  --preprocess-contract "$ANYRES_PREPROCESS" \
  --tokenizer "$TOKENIZER_DIR" \
  --text-replay "$TEXT_REPLAY_JSONL" \
  --reviewed-exclusions "$REVIEWED_EXCLUSIONS" \
  --out "$DATASET_VALIDATION_REPORT"
```

The four public splits are non-interchangeable:

- `train`: the only split used for parameter updates;
- `sft_validation`: Visual/Joint SFT selection only;
- `kl_selection`: three KL pilots only;
- `formal_monitor`: formal GRPO monitoring only.

The locked benchmark is a physically separate fifth dataset. Training and selection processes must not have read access to its labels or pixels.

## 6. Visual SFT

Visual SFT freezes the language model and frozen global anchor while training the AnyRes native-detail tower, bridge and projector. Start only from the authenticated official release:

```bash
"$PYTHON_BIN" -m scripts.train_ocr_anyres_sft \
  --stage visual \
  --source-checkpoint "$SOURCE_CHECKPOINT" \
  --tokenizer "$TOKENIZER_DIR" \
  --root "$ANYRES_DATASET" \
  --assets "$ANYRES_DATASET/assets.jsonl" \
  --views "$ANYRES_DATASET/views.jsonl" \
  --train-samples "$ANYRES_DATASET/train.jsonl" \
  --sft-validation-samples "$ANYRES_DATASET/sft_validation.jsonl" \
  --kl-selection-samples "$ANYRES_DATASET/kl_selection.jsonl" \
  --formal-monitor-samples "$ANYRES_DATASET/formal_monitor.jsonl" \
  --preprocess-contract "$ANYRES_PREPROCESS" \
  --text-replay "$TEXT_REPLAY_JSONL" \
  --reviewed-exclusions "$REVIEWED_EXCLUSIONS" \
  --output "$VISUAL_SFT_ROOT" \
  --cycles "$VISUAL_CYCLES" \
  --global-batch-size "$GLOBAL_BATCH_SIZE" \
  --tower-lr "$TOWER_LR" --bridge-lr "$BRIDGE_LR" \
  --projector-lr "$PROJECTOR_LR" --lm-lr "$LM_LR"
```

Do not promote a checkpoint on training loss alone. Preserve the stage result and the exact selected checkpoint hash for Joint SFT admission.

## 7. Joint SFT

Joint SFT uses four OCR microbatches plus one independent `0.2 ×` text-CE replay term. The text replay source must be independent, receipt-bound and excluded from all OCR evaluation splits.

```bash
"$PYTHON_BIN" -m scripts.train_ocr_anyres_joint_sft \
  --visual-stage-result "$VISUAL_STAGE_RESULT" \
  --tokenizer "$TOKENIZER_DIR" \
  --root "$ANYRES_DATASET" \
  --assets "$ANYRES_DATASET/assets.jsonl" \
  --views "$ANYRES_DATASET/views.jsonl" \
  --train-samples "$ANYRES_DATASET/train.jsonl" \
  --sft-validation-samples "$ANYRES_DATASET/sft_validation.jsonl" \
  --kl-selection-samples "$ANYRES_DATASET/kl_selection.jsonl" \
  --formal-monitor-samples "$ANYRES_DATASET/formal_monitor.jsonl" \
  --preprocess-contract "$ANYRES_PREPROCESS" \
  --text-replay "$TEXT_REPLAY_JSONL" \
  --reviewed-exclusions "$REVIEWED_EXCLUSIONS" \
  --output "$JOINT_SFT_ROOT" \
  --cycles "$JOINT_CYCLES" \
  --global-batch-size "$GLOBAL_BATCH_SIZE" \
  --text-batch-size "$TEXT_BATCH_SIZE" \
  --lm-lr "$LM_LR" --tower-lr "$TOWER_LR" \
  --projector-lr "$PROJECTOR_LR" --bridge-lr "$BRIDGE_LR"
```

GRPO may start only from the hash-bound eligible joint best referenced by `JOINT_STAGE_RESULT`. It must not start from a periodic checkpoint or a Visual-SFT-only checkpoint.

## 8. KL pilots and formal GRPO

Run exactly three isolated 200-attempt pilots with `kl_coef ∈ {0.04, 0.01, 0}`. Every pilot restarts from the same joint best and uses the same registered OCR batches, text batches and per-attempt RNG seeds.

```bash
for KL in 0.04 0.01 0; do
  "$PYTHON_BIN" -m scripts.train_ocr_anyres_grpo pilot \
    --joint-stage-result "$JOINT_STAGE_RESULT" \
    --tokenizer "$TOKENIZER_DIR" \
    --root "$ANYRES_DATASET" \
    --assets "$ANYRES_DATASET/assets.jsonl" \
    --views "$ANYRES_DATASET/views.jsonl" \
    --train-samples "$ANYRES_DATASET/train.jsonl" \
    --sft-validation-samples "$ANYRES_DATASET/sft_validation.jsonl" \
    --kl-selection-samples "$ANYRES_DATASET/kl_selection.jsonl" \
    --formal-monitor-samples "$ANYRES_DATASET/formal_monitor.jsonl" \
    --preprocess-contract "$ANYRES_PREPROCESS" \
    --text-replay "$TEXT_REPLAY_JSONL" \
    --reviewed-exclusions "$REVIEWED_EXCLUSIONS" \
    --kl-coef "$KL" --output-dir "$PILOT_ROOT/kl-$KL" \
    --lm-lr "$LM_LR" --tower-lr "$TOWER_LR" \
    --projector-lr "$PROJECTOR_LR" --bridge-lr "$BRIDGE_LR" \
    --max-behavior-log-ratio "$MAX_BEHAVIOR_LOG_RATIO"
done
```

Authenticate checkpoint bytes and select KL:

```bash
"$PYTHON_BIN" -m scripts.train_ocr_anyres_grpo select-kl \
  --pilot-result "$PILOT_ROOT/kl-0.04/PILOT_RESULT.json" \
  --pilot-result "$PILOT_ROOT/kl-0.01/PILOT_RESULT.json" \
  --pilot-result "$PILOT_ROOT/kl-0/PILOT_RESULT.json" \
  --output-dir "$KL_SELECTION_ROOT"
```

A no-candidate result is a NO-GO. Formal training re-authenticates all pilot results but restarts from the original joint best; it never resumes a pilot checkpoint:

```bash
"$PYTHON_BIN" -m scripts.train_ocr_anyres_grpo formal \
  --pilot-result "$PILOT_ROOT/kl-0.04/PILOT_RESULT.json" \
  --pilot-result "$PILOT_ROOT/kl-0.01/PILOT_RESULT.json" \
  --pilot-result "$PILOT_ROOT/kl-0/PILOT_RESULT.json" \
  --kl-selection-receipt "$KL_SELECTION_ROOT/KL_SELECTION.json" \
  --joint-stage-result "$JOINT_STAGE_RESULT" \
  --locked-golden-anchor-sha256 "$LOCKED_GOLDEN_ANCHOR_SHA256" \
  --tokenizer "$TOKENIZER_DIR" \
  --root "$ANYRES_DATASET" \
  --assets "$ANYRES_DATASET/assets.jsonl" \
  --views "$ANYRES_DATASET/views.jsonl" \
  --train-samples "$ANYRES_DATASET/train.jsonl" \
  --sft-validation-samples "$ANYRES_DATASET/sft_validation.jsonl" \
  --kl-selection-samples "$ANYRES_DATASET/kl_selection.jsonl" \
  --formal-monitor-samples "$ANYRES_DATASET/formal_monitor.jsonl" \
  --preprocess-contract "$ANYRES_PREPROCESS" \
  --text-replay "$TEXT_REPLAY_JSONL" \
  --reviewed-exclusions "$REVIEWED_EXCLUSIONS" \
  --output-dir "$FORMAL_ROOT" --seed "$FORMAL_SEED" \
  --max-rollout-attempts "$FORMAL_MAX_ATTEMPTS" \
  --max-optimizer-steps "$FORMAL_MAX_UPDATES" \
  --eval-every "$FORMAL_EVAL_EVERY" \
  --save-every "$FORMAL_SAVE_EVERY" \
  --early-stop-patience "$FORMAL_PATIENCE" \
  --lm-lr "$LM_LR" --tower-lr "$TOWER_LR" \
  --projector-lr "$PROJECTOR_LR" --bridge-lr "$BRIDGE_LR" \
  --max-behavior-log-ratio "$MAX_BEHAVIOR_LOG_RATIO"
```

Pilot and formal owners persist model, optimizer, scheduler, scaler, RNG, sampler/text cursors, no-update journals and source/data/protocol hashes. Resume must be deterministic and must fail on contract drift.

## 9. One-shot locked evaluation

Only the finalized formal selection may claim the sealed benchmark:

```bash
"$PYTHON_BIN" -m scripts.eval_ocr_anyres_locked \
  --checkpoint "$FORMAL_SELECTED_CHECKPOINT" \
  --selection-receipt "$FORMAL_ROOT/best/SELECTION_FINALIZED.json" \
  --pilot-result "$PILOT_ROOT/kl-0.04/PILOT_RESULT.json" \
  --pilot-result "$PILOT_ROOT/kl-0.01/PILOT_RESULT.json" \
  --pilot-result "$PILOT_ROOT/kl-0/PILOT_RESULT.json" \
  --kl-selection-receipt "$KL_SELECTION_ROOT/KL_SELECTION.json" \
  --joint-stage-result "$JOINT_STAGE_RESULT" \
  --dataset-admission "$FORMAL_ROOT/DATA_ADMISSION.json" \
  --tokenizer "$TOKENIZER_DIR" \
  --golden-anchor "$LOCKED_GOLDEN_ANCHOR" \
  --build-receipt "$SEALED_ROOT/LOCKED_BUILD_RECEIPT.json" \
  --sealed-root "$SEALED_ROOT" \
  --ledger-dir "$LOCKED_LEDGER" \
  --out "$LOCKED_REPORT"
```

Any failure after the atomic claim leaves a permanent incomplete report and consumes the one-shot. Never delete or move the ledger marker to retry. Do not tune the model after reading the locked report.

Promotion requires, at minimum:

- grapheme CER reported with exact checkpoint and dataset hashes;
- selected policy beats the immutable pre-RL reference;
- selected policy beats its own blank-image ablation;
- validation EOS rate and invalid-output gates satisfy the formal protocol;
- no train/validation/locked overlap evidence remains unresolved.

## 10. Failure handling

1. Stop new work on the accelerator and preserve logs and the latest complete checkpoint.
2. Classify the failure as data, storage, process, numerical, distributed, checkpoint or contract related.
3. Protect the last complete checkpoint before inspecting partial output.
4. Reproduce with the narrowest relevant check on the same code SHA.
5. Resume only after revalidating checkpoint completeness and unchanged data identities.
6. Record the cause, recovery checkpoint, exact command and post-resume progress evidence.

Machine-specific deletion, process termination and storage recovery procedures are intentionally not encoded as copy-paste commands in this repository.
