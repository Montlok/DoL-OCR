# DoL-OCR operations runbook

This runbook defines platform-independent procedures. Keep machine addresses,
credentials, mount points, active run identifiers, and incident timelines in an
untracked `RUNBOOK.local.md` or the external operations system.

## 1. Required run record

Before launch, record the following outside Git:

- repository commit SHA and whether the worktree is clean;
- model, tokenizer, and vision-tower configuration;
- input manifest paths and cryptographic hashes;
- train/validation split policy and exclusion list;
- source checkpoint path and hash;
- optimizer, precision, batch, schedule, save, evaluation, and stop settings;
- output directory, log destination, operator, and start time.

Do not launch when any required path is ambiguous or when validation data overlaps the
training set by source ID, file hash, document group, or resolved path.

## 2. Preflight

Run from the repository root with an explicit Python interpreter:

```bash
export PYTHON_BIN="${PYTHON_BIN:?set the production Python interpreter}"
export OUTPUT_DIR="${OUTPUT_DIR:?set a new or resumable output directory}"

git status --short
git rev-parse HEAD
"$PYTHON_BIN" -m pytest -q Model/tests Tokenizer/tests
```

For a production GPU run, also verify accelerator visibility, free storage, input
readability, tokenizer/checkpoint hashes, and that no other GPU process is active.
Never run a memory probe or cache eviction concurrently with training.

## 3. Launch

Use the Python entry point that matches the phase. Pass all data, checkpoint, and output
paths explicitly; do not rely on workstation-specific defaults.

```bash
# Text pretraining
"$PYTHON_BIN" -m scripts.train_rdt \
  --config two_stage_pretrain \
  --tokenizer-bundle "$TOKENIZER_DIR" \
  --data "$TRAIN_DATA" \
  --data-receipt "$TRAIN_DATA_RECEIPT" \
  --eval-data "$VALIDATION_DATA" \
  --eval-data-receipt "$VALIDATION_DATA_RECEIPT" \
  --output "$OUTPUT_DIR" \
  "${EXTRA_ARGS[@]}"

# Vision-tower SSL
"$PYTHON_BIN" -m scripts.train_omvt_ssl \
  --data "$VISION_DATA" \
  --output "$OUTPUT_DIR" \
  "${EXTRA_ARGS[@]}"

# OMVT-to-RDT alignment
"$PYTHON_BIN" -m scripts.train_vlm_align \
  --data "$ALIGNMENT_DATA" \
  --output "$OUTPUT_DIR" \
  "${EXTRA_ARGS[@]}"
```

Build every text shard set with `Tokenizer.tools.build_pretraining_data` and retain
the emitted `receipt` path. Non-smoke RDT training refuses raw or unreceipted
pre-tokenized JSONL. Each receipt binds the tokenizer implementation/runtime,
one registered row producer and its current source fingerprint, every tokenizer
bundle file actually resolved by `config.json`, and the ordered shard names,
sizes, per-file SHA-256 values, total bytes, and aggregate digest.
`--mix-data-receipt` accepts either this receipt or the
`pretokenized_ocr_alignment` receipt produced by the native OCR data builder.

For frozen-language OCR alignment, generic pre-tokenized data is not enough.
Build it with `scripts.build_ocr_data` and pass its immutable receipt:

```bash
"$PYTHON_BIN" -m scripts.train_vlm_align \
  --freeze-rdt \
  --ocr-native-targets \
  --ocr-tokenizer-bundle "$TOKENIZER_DIR" \
  --ocr-data-contract "$ALIGNMENT_DATA_DIR/ocr_data_contract.json" \
  --data "$ALIGNMENT_DATA_DIR/data.jsonl" \
  --init-rdt-checkpoint "$LANGUAGE_CHECKPOINT" \
  --output "$OUTPUT_DIR" \
  "${EXTRA_ARGS[@]}"
```

This mode refuses a tokenizer whose file fingerprints differ from those recorded
by the frozen language checkpoint. The receipt binds every JSONL shard by name,
size, SHA-256, tokenizer implementation/runtime identity, and the id-to-morphology
table used to reproduce pretraining `word_pos`/`morph_depth`. The strict collator
requires those two fields in every OCR row. OCR-GRPO accepts only a final visual
checkpoint with a terminal `stop_reason`, native tokenization contract v3, the
complete shard receipt, and that frozen-LM lineage. A periodic `COMPLETE`
checkpoint is not a completed visual run.

In shell automation, store optional arguments in an array as shown above. Do not place a
quoted list of arguments in one scalar variable.

For long VLM alignment runs, an optional smoothed training-loss plateau may be used as
an operational stop condition while ``--steps`` remains the hard ceiling:

```bash
"$PYTHON_BIN" -m scripts.train_vlm_align \
  --data "$ALIGNMENT_DATA" \
  --output "$OUTPUT_DIR" \
  --steps "$MAX_STEPS" \
  --early-stop-min-steps "$EARLY_STOP_MIN_STEPS" \
  --early-stop-patience "$EARLY_STOP_PATIENCE" \
  --early-stop-min-delta "$EARLY_STOP_MIN_DELTA" \
  --early-stop-smoothing ema \
  --early-stop-ema-alpha "$EARLY_STOP_EMA_ALPHA" \
  "${EXTRA_ARGS[@]}"
```

Set ``--early-stop-patience 0`` to disable the condition. Patience is counted in
completed optimizer steps, after ``--early-stop-min-steps``. The smoother, best loss,
patience counter, and last observed step are checkpointed; a resume must use the same
early-stop configuration. A checkpoint whose ``stop_reason`` is ``loss_plateau`` is a
completed run, not a continuation target. Use it as an explicit initialization source
only when starting a deliberately new schedule.

## 4. Monitoring and graceful control

The run directory is the control and status boundary:

```bash
"$PYTHON_BIN" -m scripts.rdt_monitor tui --run "$OUTPUT_DIR"
"$PYTHON_BIN" -m scripts.rdt_monitor control save --run "$OUTPUT_DIR"
"$PYTHON_BIN" -m scripts.rdt_monitor control stop --run "$OUTPUT_DIR"
```

Track optimizer step, loss, validation metric, gradient norm, throughput, learning rate,
checkpoint age, free storage, and accelerator utilization. A running process without
step or checkpoint progress is not healthy merely because it still owns the GPU.

Use the control plane for planned stops. Wait until `latest` points to the new step and
all required checkpoint files are present, then confirm process exit before releasing
the accelerator or changing data mounts.

## 5. Checkpoint and resume

A resumable checkpoint must contain model, optimizer, scheduler, RNG, data cursor,
configuration, and metadata required by the selected precision mode. Never repoint
`latest` to a partially written directory.

Resume with the same model shape, tokenizer, data order contract, world size when
required, and training semantics:

```bash
"$PYTHON_BIN" -m scripts.train_rdt \
  --config two_stage_pretrain \
  --tokenizer-bundle "$TOKENIZER_DIR" \
  --data "$TRAIN_DATA" \
  --data-receipt "$TRAIN_DATA_RECEIPT" \
  --eval-data "$VALIDATION_DATA" \
  --eval-data-receipt "$VALIDATION_DATA_RECEIPT" \
  --output "$OUTPUT_DIR" \
  --resume "$OUTPUT_DIR/latest" \
  "${EXTRA_ARGS[@]}"
```

Resume ignores only mount/output/checkpoint path relocation. It compares the exact
model and training configurations, tokenizer identities, mix schedule, ordered
train/eval/mix shard byte receipts, optimizer schedule, recurrence settings, and
seed before loading model or optimizer state.

After resume, verify both process liveness and new optimizer-step progress. Preserve the
last known complete checkpoint until the resumed run has produced a newer complete one.

## 6. Image-conditioned OCR GRPO

> This section documents the legacy line-image v1 path. It must not be used for
> `dol_ocr_anyres_v2` data or checkpoints. The anyres path below has separate
> READY, planner, cut-QA, tokenizer, promotion, and checkpoint contracts.

Keep `rl_train`, `rl_val`, and locked `golden` manifests separate. The training
account receives only train/validation labels and a transcript-free golden identity
registry. Rows from the same page, capture sequence, or document share a `group_id`
and remain in one split. Reject overlap by ID, resolved path, file hash, or group.
Each labeled JSONL row has this form:

```json
{"id":"camera-0001-line-01","group_id":"camera-0001","split":"rl_train","image":"photos/camera-0001-line-01.jpg","sha256":"<64 hex>","reference":"ᠮᠣᠩᠭᠤᠯ","domain":"photo"}
```

Build manifests from the reviewed TSV and annotation-pack provenance. The builder
ignores the TSV image path, selects the pack's `raw_path` rather than its 224-pixel
preview, verifies strict-native tokenizer round-trips, and preserves document-level
assignments in `split_lock.json`. It publishes a new immutable dataset directory:
train/validation pixels are copied under `public/images`, while golden pixels are
copied under `locked/images`. The original annotation-pack path is not serialized.
Never feed a nominal-normalized annotation export into this builder: FVS, MVS, and
NNBSP must remain in the reviewed raw transcription.

When producing a pack from camera images, declare capture provenance explicitly:

```bash
"$PYTHON_BIN" -m scripts.build_annotation_pack \
  --image-dir "$CAPTURE_IMAGE_DIR" \
  --capture-session "$CAPTURE_SESSION_ID" \
  --out "$ANNOTATION_PACK"
```

All pages in one directory remain one split group. Repeat the paired
`--image-dir`/`--capture-session` flags to build one pack from several sessions.
Never manufacture a different session id per page to force desired split ratios;
collect at least three genuinely independent capture/document groups before
building train, validation, and golden.

```bash
"$PYTHON_BIN" -m scripts.build_ocr_rl_manifests \
  --annotations "$REVIEWED_ANNOTATIONS_TSV" \
  --pack-manifest "$ANNOTATION_PACK/manifest.json" \
  --tokenizer "$TOKENIZER_DIR" \
  --out "$RL_DATASET_DIR" \
  --split-lock "$RL_SPLIT_LOCK"
```

Grant the trainer read access to `$RL_DATASET_DIR/public` only. Keep
`$RL_DATASET_DIR/locked` readable by the evaluation account. Audit every public
image and manifest before allocating an accelerator. The validator checks the
visual checkpoint's native-token contract, image decoding, SHA-256 values, split
isolation, and required generation length without opening golden labels.

```bash
"$PYTHON_BIN" -m scripts.validate_ocr_rl_manifests \
  --checkpoint "$INIT_CHECKPOINT" \
  --tokenizer "$TOKENIZER_DIR" \
  --train "$RL_DATASET_DIR/public/rl_train.jsonl" \
  --validation "$RL_DATASET_DIR/public/rl_val.jsonl" \
  --golden-identity "$RL_DATASET_DIR/public/golden_identity.jsonl" \
  --dataset-contract "$RL_DATASET_DIR/public/dataset_contract.json" \
  --out "$MANIFEST_REPORT"
```

Launch only after the validator succeeds. OCR mode freezes the language model by
default and updates the OMVT tower and projector. The dense reward is raw grapheme
`-CER`, with penalties for empty output, reserved tokens, and excessive length.
The validator is the mandatory full-manifest decode/SHA preflight. Training ranks
only check image existence at dataset construction; each consumed batch then reads
every image once, verifies its manifest SHA-256 on those same bytes, and passes the
verified bytes to the image processor. This avoids a duplicate full NAS scan on
every rank without weakening the hot-path integrity check.

```bash
"$PYTHON_BIN" -m scripts.train_grpo \
  --task ocr \
  --tokenizer "$TOKENIZER_DIR" \
  --data "$RL_DATASET_DIR/public/rl_train.jsonl" \
  --validation-manifest "$RL_DATASET_DIR/public/rl_val.jsonl" \
  --golden-identity-manifest "$RL_DATASET_DIR/public/golden_identity.jsonl" \
  --dataset-contract "$RL_DATASET_DIR/public/dataset_contract.json" \
  --init-checkpoint "$INIT_CHECKPOINT" \
  --output "$OUTPUT_DIR" \
  --train-scope vision \
  --group-size 4 \
  --prompts-per-step 2 \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --learning-rate 1e-6 \
  --kl-coef 0.04 \
  --eval-every 50 \
  --early-stop-patience 5 \
  --max-steps 1000 \
  --save-every 200 \
  --keep-last-n 3
```

Rollouts use `temperature=1.0` without top-p truncation. Their exact sampled-token
log-probabilities are retained and checked against the differentiable full-forward
score before every update. OCR advantages use `reward - group_mean` without
standard-deviation amplification. The policy takes one synchronous completion-only
on-policy step with a k3 KL penalty against an immutable step-zero reference; PPO
ratio clipping is intentionally disabled because there is no replay or multi-epoch
update. Real-image validation must outperform the blank-image baseline before
training starts. A batch with no group-relative reward signal takes no optimizer
step; a sustained no-signal streak triggers a full validation pass and a clean,
resumable `stop_reason=no_reward_spread` final checkpoint.

`$OUTPUT_DIR/latest` is the resumable training state; `$OUTPUT_DIR/best/latest` is the
eligible checkpoint with the lowest validation grapheme CER. Resume with the identical
configuration and data contract:

```bash
"$PYTHON_BIN" -m scripts.train_grpo \
  --task ocr \
  --resume "$OUTPUT_DIR/latest" \
  "${EXTRA_ARGS[@]}"
```

Checkpoints are staged and published atomically. Resume restores optimizer, scheduler,
scaler, data cursor, per-rank RNG, health counters, early-stop state, and the immutable
reference identity. A changed manifest, tokenizer, or reference hash aborts resume.
An all-zero-advantage rollout does not advance the optimizer or scheduler. At the start
of such a streak the trainer ensures that the current policy has a full checkpoint;
subsequent zero-update rollouts durably update only
`$OUTPUT_DIR/NO_UPDATE_PROGRESS.pt`, anchored to that checkpoint's metadata. Resume
applies this small cursor/RNG journal only to the exact anchor, so outage recovery
preserves no-signal patience without rewriting multi-gigabyte weights after every
skipped rollout.
After a terminal checkpoint is durable, the trainer writes
`$OUTPUT_DIR/best/SELECTION_FINALIZED.json`. The locked evaluator refuses to run
without this receipt, so golden data cannot be opened while training is still active.

Evaluate the locked golden set once, after checkpoint selection is complete:

```bash
"$PYTHON_BIN" -m scripts.eval_ocr_grpo \
  --checkpoint "$OUTPUT_DIR/best/latest" \
  --tokenizer "$TOKENIZER_DIR" \
  --golden-manifest "$RL_DATASET_DIR/locked/golden.jsonl" \
  --golden-identity-manifest "$RL_DATASET_DIR/public/golden_identity.jsonl" \
  --dataset-contract "$RL_DATASET_DIR/public/dataset_contract.json" \
  --golden-receipt "$RL_DATASET_DIR/locked/build_receipt.json" \
  --selection-receipt "$OUTPUT_DIR/best/SELECTION_FINALIZED.json" \
  --out "$OUTPUT_DIR/golden_report.json"
```

The evaluator first validates the public contract's SHA-256 anchor for the canonical
locked `build_receipt.json` without opening or hashing `golden.jsonl`, prepares both
models and the output path, then atomically claims
`$RL_DATASET_DIR/locked/golden_evaluation_ledger/<locked-receipt-sha256>.json`.
The ledger location is derived from the resolved locked-golden root and cannot be
overridden per run, so a different output directory cannot consume the same sealed
golden set again. The evaluator rejects a symlink or non-directory in place of the
ledger and creates the receipt with an exclusive open before directory-fsyncing it.
It immediately publishes and directory-fsyncs a
`claimed_evaluation_incomplete` report stub. Only after both durable records exist
does it hash and validate `golden.jsonl`, then parse labels and images. A receipt,
label-hash, or inference failure still consumes the one-shot audit; do not delete
the ledger or rerun. Do not tune hyperparameters after reading the locked report.

## 7. Anyres v2 data and supervised admission

The only admitted source checkpoint is the immutable release
`Montlok/DoL-1.2-OCR@bee908ab2a9376f6224dff514564ebb0ae99a643`.
Do not initialize this path from a filename, a step alias, DoL-1.5, or a final
checkpoint selected by directory order.

Generate an immutable review pack before transcription review:

```bash
"$PYTHON_BIN" -m scripts.build_ocr_anyres_dataset prepare-review \
  --sources "$SOURCES_JSONL" \
  --source-root "$SOURCE_ROOT" \
  --preprocess-contract "$ANYRES_PREPROCESS" \
  --out "$REVIEW_PACK"
```

The pack contains lossless post-EXIF canonical PNGs, exact planner windows,
cut-QA evidence, three reading-order proposals, and a pending review template.
It never approves labels. After human Unicode, provenance, difficulty, reading
order, and cut review, finalize into a new output directory:

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

Training accepts only the dataset-root `READY` chain, including the build
receipt, all manifest hashes, and every structured manual cut adjudication.
The native detail path retains original aspect ratio and pixels; internal
macro-window crops have overlapping context and disjoint ownership, and cut-QA
must prove every connected component is fully visible in at least one view.
The 224-square image is retained only as the frozen v1 global anchor.

The supervised sequence is:

1. visual SFT: freeze LM and the v1 global anchor; train native detail tower and bridge;
2. joint SFT: four OCR microbatches plus one independent `0.2 ×` text-CE replay term;
3. select only the hash-bound eligible joint best;
4. run three isolated 200-attempt KL pilots (`0.04`, `0.01`, `0`), each restarted
   from the same joint best;
5. restart the selected formal GRPO run from that same joint best.

Image and text data each have four public, non-interchangeable splits:
`train`, `sft_validation`, `kl_selection`, and `formal_monitor`. Visual/joint
SFT may read only `sft_validation`; KL pilots may evaluate only `kl_selection`;
formal GRPO may monitor only `formal_monitor`. All updates use `train`. The
locked benchmark is a physically separate fifth dataset and is never a public
split.

Run each KL pilot in a separate output root. The owner registers the same 200
OCR batches, text batches, and per-attempt RNG seeds for all three candidates:

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

Authenticate real checkpoint bytes and select KL. A no-candidate result writes
only `KL_SELECTION_NO_GO.json` and exits nonzero:

```bash
"$PYTHON_BIN" -m scripts.train_ocr_anyres_grpo select-kl \
  --pilot-result "$PILOT_ROOT/kl-0.04/PILOT_RESULT.json" \
  --pilot-result "$PILOT_ROOT/kl-0.01/PILOT_RESULT.json" \
  --pilot-result "$PILOT_ROOT/kl-0/PILOT_RESULT.json" \
  --output-dir "$KL_SELECTION_ROOT"
```

Formal training must re-authenticate those same three results, then restart from
the original joint best. It never resumes a pilot checkpoint:

```bash
"$PYTHON_BIN" -m scripts.train_ocr_anyres_grpo formal \
  --pilot-result "$PILOT_ROOT/kl-0.04/PILOT_RESULT.json" \
  --pilot-result "$PILOT_ROOT/kl-0.01/PILOT_RESULT.json" \
  --pilot-result "$PILOT_ROOT/kl-0/PILOT_RESULT.json" \
  --kl-selection-receipt "$KL_SELECTION_ROOT/KL_SELECTION.json" \
  --joint-stage-result "$JOINT_STAGE_RESULT" \
  --locked-golden-anchor-sha256 "$LOCKED_GOLDEN_ANCHOR_SHA256" \
  --tokenizer "$TOKENIZER_DIR" --root "$ANYRES_DATASET" \
  --assets "$ANYRES_DATASET/assets.jsonl" --views "$ANYRES_DATASET/views.jsonl" \
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
  --eval-every "$FORMAL_EVAL_EVERY" --save-every "$FORMAL_SAVE_EVERY" \
  --early-stop-patience "$FORMAL_PATIENCE" \
  --lm-lr "$LM_LR" --tower-lr "$TOWER_LR" \
  --projector-lr "$PROJECTOR_LR" --bridge-lr "$BRIDGE_LR" \
  --max-behavior-log-ratio "$MAX_BEHAVIOR_LOG_RATIO"
```

After formal selection, the sealed evaluator performs all public lineage and
checkpoint checks before claiming the benchmark. The claim marker is keyed by
the locked build-receipt SHA, so a second model cannot rerun the same benchmark:

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
  --sealed-root "$SEALED_ROOT" --ledger-dir "$LOCKED_LEDGER" \
  --out "$LOCKED_REPORT"
```

Any failure after the atomic claim leaves a permanent
`claimed_evaluation_incomplete` report and consumes the one-shot. Never delete
the ledger marker to retry.

Pilot and formal owners persist full model/optimizer/scheduler/RNG/scaler state,
strict sampler/text cursors, no-update journals, and source/data/protocol hashes.
They support deterministic `--resume`. Code-level admission does not replace a
real CUDA BF16 smoke on the exact AnyRes checkpoint and hardware.

## 8. Evaluation gates

- Use grapheme CER as the headline OCR metric and retain normalized/code-point metrics
  only as diagnostics.
- Report real-image and blank-image results together so visual contribution is visible.
- Keep validation and locked golden data outside all training and tuning paths.
- Record the evaluated checkpoint hash, dataset hash, generation settings, and raw
  report. Do not overwrite an earlier evaluation report.
- Report raw and normalized exact-match, the actual normalization backend, and
  digit, punctuation, FVS1-4, MVS, and NNBSP error/support counts.
- Do not promote a checkpoint on training loss alone. Plateau stopping only controls
  training duration; it does not replace validation or the locked evaluation gate.
- Require validation EOS rate of at least 99% and an invalid-output rate of zero before
  an OCR GRPO checkpoint can replace `best`.
- For the final locked evaluation, require the selected policy to beat both the
  immutable pre-RL reference and its own blank-image ablation.

## 9. Failure handling

1. Stop new work on the accelerator and preserve logs and the latest complete checkpoint.
2. Determine whether the failure is data, storage, process, numerical, distributed, or
   checkpoint related.
3. Do not delete partial output until its exact path is verified and the previous complete
   checkpoint is protected.
4. Reproduce with the narrowest relevant check before restarting the full run.
5. Resume only after validating checkpoint completeness and the unchanged data contract.
6. Record the cause, recovery checkpoint, command, and post-resume progress evidence.

Destructive cleanup and forced process termination require a machine-specific incident
procedure; they are intentionally not encoded as copy-paste commands in this repository.
