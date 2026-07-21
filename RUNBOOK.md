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
  --data "$TRAIN_DATA" \
  --eval-data "$VALIDATION_DATA" \
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
  --data "$TRAIN_DATA" \
  --eval-data "$VALIDATION_DATA" \
  --output "$OUTPUT_DIR" \
  --resume "$OUTPUT_DIR/latest" \
  "${EXTRA_ARGS[@]}"
```

After resume, verify both process liveness and new optimizer-step progress. Preserve the
last known complete checkpoint until the resumed run has produced a newer complete one.

## 6. Image-conditioned OCR GRPO

Keep `rl_train`, `rl_val`, and locked `golden` manifests separate. Rows from the same
page, capture sequence, or document must share a `group_id` and remain in one split.
Reject overlap by ID, resolved path, file hash, or group. Each JSONL row has this form:

```json
{"id":"camera-0001-line-01","group_id":"camera-0001","split":"rl_train","image":"photos/camera-0001-line-01.jpg","sha256":"<64 hex>","reference":"ᠮᠣᠩᠭᠤᠯ","domain":"photo"}
```

Audit every image and manifest before allocating an accelerator. The validator checks
image decoding, SHA-256 values, split isolation, and required generation length. Locked
golden labels are not tokenized or used for model selection during this audit.

```bash
"$PYTHON_BIN" -m scripts.validate_ocr_rl_manifests \
  --checkpoint "$INIT_CHECKPOINT" \
  --tokenizer "$TOKENIZER_DIR" \
  --train "$RL_TRAIN_MANIFEST" \
  --validation "$RL_VALIDATION_MANIFEST" \
  --golden "$GOLDEN_MANIFEST" \
  --image-root "$IMAGE_ROOT" \
  --out "$MANIFEST_REPORT"
```

Launch only after the validator succeeds. OCR mode freezes the language model by
default and updates the OMVT tower and projector. The dense reward is raw grapheme
`-CER`, with penalties for empty output, reserved tokens, and excessive length.

```bash
"$PYTHON_BIN" -m scripts.train_grpo \
  --task ocr \
  --tokenizer "$TOKENIZER_DIR" \
  --data "$RL_TRAIN_MANIFEST" \
  --validation-manifest "$RL_VALIDATION_MANIFEST" \
  --golden-manifest "$GOLDEN_MANIFEST" \
  --image-root "$IMAGE_ROOT" \
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

Rollouts use `temperature=1.0` without top-p truncation so rollout and scoring
distributions match. Advantages are normalized within each prompt group. The policy
uses a completion-only clipped surrogate and a k3 KL penalty against an immutable
step-zero reference. Real-image validation must outperform the blank-image baseline
before training starts.

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

Evaluate the locked golden set once, after checkpoint selection is complete:

```bash
"$PYTHON_BIN" -m scripts.eval_ocr_grpo \
  --checkpoint "$OUTPUT_DIR/best/latest" \
  --tokenizer "$TOKENIZER_DIR" \
  --golden-manifest "$GOLDEN_MANIFEST" \
  --out "$OUTPUT_DIR/golden_report.json"
```

Do not tune hyperparameters after reading the locked report.

## 7. Evaluation gates

- Use grapheme CER as the headline OCR metric and retain normalized/code-point metrics
  only as diagnostics.
- Report real-image and blank-image results together so visual contribution is visible.
- Keep validation and locked golden data outside all training and tuning paths.
- Record the evaluated checkpoint hash, dataset hash, generation settings, and raw
  report. Do not overwrite an earlier evaluation report.
- Do not promote a checkpoint on training loss alone. Plateau stopping only controls
  training duration; it does not replace validation or the locked evaluation gate.
- Require validation EOS rate of at least 99% and an invalid-output rate of zero before
  an OCR GRPO checkpoint can replace `best`.
- For the final locked evaluation, require the selected policy to beat both the
  immutable pre-RL reference and its own blank-image ablation.

## 8. Failure handling

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
