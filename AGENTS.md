# DoL-OCR engineering guidance

This file contains durable repository rules only. Machine addresses, credentials,
personal paths, active run identifiers, current loss values, and incident notes belong
in an untracked `AGENTS.local.md` or the external operations system.

## Repository boundaries

- The production target is Traditional Mongolian OCR. Multiscript tokenizer coverage
  supports mixed-page handling and is not evidence of equivalent language training.
- Commit source code, configuration, tests, and small public examples only. Never
  commit corpora, scans, full transcriptions, manifests that disclose private data,
  model weights, checkpoints, logs, credentials, or host-specific configuration.
- Real labeled photographs are reserved for reinforcement learning and a permanently
  locked evaluation set. They are not SFT data. Split by source group before training
  so related pages cannot cross train, validation, or golden boundaries.
- Keep annotation applications in their dedicated repository. This repository may
  contain only the model-side import, validation, and evaluation contracts.

## Training safety

- Do not start side GPU workloads, page-cache eviction, or hardware probes while a
  production training process owns the accelerator.
- Stop through the trainer control plane so a complete checkpoint is written. Use
  process signals only after the control path has failed and the recovery target has
  been identified.
- Resume only from a complete checkpoint and preserve optimizer, scheduler, scaler,
  RNG, data cursor, model configuration, and corpus provenance.
- Official-Mamba and fallback-Mamba checkpoint shapes are not interchangeable. Validate
  production checkpoints with the same backend family used to train them.

## Change discipline

- Keep each pull request to one primary responsibility. Do not mix repository cleanup,
  model behavior, data contracts, and deployment state in one commit series.
- Add regression tests for changed behavior and run the narrow tests first, including
  index zero and boundary cases, before the full suite.
- Report measured results and the exact command used. Do not select only favorable
  samples or infer production quality from smoke data.
- Do not hard-code IP addresses, user home directories, private repository paths, or
  live run names in tracked code. Use explicit CLI arguments or environment variables.
- Update `RUNBOOK.md` when a change affects launch, monitoring, checkpoint, resume,
  evaluation, or rollback behavior.
