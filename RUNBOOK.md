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

## 6. Evaluation gates

- Use grapheme CER as the headline OCR metric and retain normalized/code-point metrics
  only as diagnostics.
- Report real-image and blank-image results together so visual contribution is visible.
- Keep validation and locked golden data outside all training and tuning paths.
- Record the evaluated checkpoint hash, dataset hash, generation settings, and raw
  report. Do not overwrite an earlier evaluation report.
- Do not promote a checkpoint on training loss alone.

## 7. Failure handling

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
