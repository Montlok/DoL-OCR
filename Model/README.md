# DoL-OCR model stack

`Model/` contains the supported RDT language core, OMVT vision stack, receipt-bound training runtime and AnyRes post-training contracts. It intentionally does not contain unrelated chat-alignment pipelines or alternative experimental model families.

## 1. Supported architecture

### RDT language core

`Model.model.RDTForCausalLM` is the causal language model owner. The production configuration uses the two-stage recurrent-depth core:

1. causal prelude blocks encode the token stream;
2. shared recurrent blocks refine hidden state for a configured or sampled number of steps;
3. causal coda blocks and the tied LM head produce next-token logits.

`Model.layers.mhc` provides manifold-constrained hyper-connections used by the reviewed release. `Model.layers.mamba3_layer` must select official Mamba on the production CUDA host. The NaiveSSM implementation exists for bounded CPU tests only.

The official OCR release uses `boundary_v1` position semantics. Position, morphology and target-encoding contracts are part of checkpoint compatibility, not tunable preprocessing conveniences.

### OMVT vision stack

`Model.omvt` owns the visual pathway:

- multi-scale patch extraction and geometry planning;
- orientation/layout-aware mixing;
- visual compression;
- projection into the RDT hidden dimension;
- injection at authenticated image-token positions.

The released v1 visual path remains the frozen global anchor. AnyRes adds a native-detail path that preserves the source aspect ratio and plans internal windows. The user-facing input is the original image; arbitrary source dimensions are accepted only after dataset admission and cut-QA.

### CTC and alignment

CTC training remains part of the admitted OMVT provenance. The supported preparation path can train/evaluate the tower and CTC head, then align OMVT to a frozen RDT through native OCR targets. These stages are not interchangeable with AnyRes SFT, and their checkpoints must carry the configuration and terminal-state metadata required by downstream admission.

## 2. Official source release

AnyRes training accepts only:

```text
Montlok/DoL-1.2-OCR@bee908ab2a9376f6224dff514564ebb0ae99a643
```

[`posttrain/release_locks/dol_1_2_ocr.json`](posttrain/release_locks/dol_1_2_ocr.json) fixes:

- model and metadata SHA-256 values;
- tokenizer manifest, vocabulary and token-ID identities;
- model parameter count and vocabulary capacity;
- `boundary_v1` position contract;
- native OCR tokenization contract;
- source training phase, stop reason and checkpoint step.

Loading code must fail closed when any bound file or contract differs. A path with the right filename is not proof of identity.

## 3. Pretraining modules

| Area | Owner |
| --- | --- |
| model/config | `model.py`, `config.py`, `two_stage.py`, `recurrent.py`, `blocks.py` |
| transformer layers | `layers/` |
| visual stack | `omvt/`, `vision.py` |
| training runtime | `training/` |
| OCR data/position/alignment contracts | `ocr/` |
| checkpoint and release verification | `posttrain/checkpointing.py`, `posttrain/release_contract.py` |

Formal RDT data is pre-tokenized and receipt-bound. Every production row persists `input_ids`, `attention_mask`, `labels`, `word_pos` and `morph_depth`; data receipts bind the exact tokenizer and producer source. Multimodal rows also bind image payloads and image-token spans.

Supported production entry points are documented in the root [`RUNBOOK.md`](../RUNBOOK.md). The core sequence is:

```text
Tokenizer bundle + receipts
  -> train_rdt
  -> train_omvt_ssl / train_ctc_head
  -> train_vlm_align with native OCR targets
  -> authenticated DoL-1.2-OCR release
```

## 4. AnyRes dataset contract

`posttrain/ocr_anyres_builder.py`, `ocr_anyres_manifest.py`, `ocr_anyres_data.py` and the `ocr/anyres_*` contracts enforce the dataset boundary.

Each admitted source provides:

- canonical post-EXIF image bytes and SHA-256;
- original width and height;
- macro-window plan with overlapping context and disjoint ownership;
- reading-order proposal and human selection;
- Unicode-exact transcription review;
- provenance and difficulty review;
- structured cut-QA for every connected text component;
- immutable split assignment and exclusion evidence.

The dataset root becomes usable only after the complete `READY` chain validates. The four public splits have separate consumers:

| Split | Consumer |
| --- | --- |
| `train` | all parameter updates |
| `sft_validation` | Visual/Joint SFT evaluation |
| `kl_selection` | three KL pilots |
| `formal_monitor` | formal GRPO monitoring |

The sealed benchmark is not a fifth public manifest. It is physically isolated and opened only by the locked evaluator after formal checkpoint selection.

## 5. Visual and Joint SFT

`posttrain/ocr_joint_*` owns supervised stage execution and admission.

### Visual SFT

- source must authenticate against the official release lock;
- RDT language parameters and the global anchor remain frozen;
- native-detail tower, bridge and projector are trainable;
- evaluation reads only `sft_validation`;
- stage result binds the selected checkpoint bytes and all input contracts.

### Joint SFT

- starts from the eligible Visual-SFT result;
- combines four OCR microbatches with one independent text replay microbatch;
- applies the text CE replay term at weight `0.2` under the stage contract;
- records OCR/text cursors, RNG and optimizer state for deterministic resume;
- publishes a hash-bound eligible joint best for GRPO admission.

Periodic or merely complete checkpoints are not automatically eligible. Downstream code follows the stage result receipt, never directory order.

## 6. AnyRes GRPO

The GRPO implementation is split by responsibility:

- `posttrain/ocr_anyres_grpo.py`: differentiable objective and model-facing operations;
- `posttrain/ocr_anyres_reward.py`: OCR reward semantics;
- `posttrain/ocr_anyres_grpo_protocol.py`: pilot/formal protocol;
- `posttrain/ocr_anyres_grpo_owner.py`: selection and ownership checks;
- `posttrain/ocr_anyres_grpo_run.py`: durable run state;
- `posttrain/ocr_anyres_grpo_trainer.py`: training orchestration;
- `posttrain/ocr_anyres_grpo_formal_protocol.py`: formal gates;
- `posttrain/grpo.py`, `logprobs.py`, `masking.py`, `rewards.py`: shared mathematical primitives.

The supported algorithm contract includes:

- group-relative OCR advantages;
- dense raw grapheme-CER reward with invalid/empty/length penalties;
- one synchronous completion-only on-policy update per rollout;
- immutable step-zero reference identity;
- completion-only logprob calculation while preserving token alignment;
- unique-image visual encoding followed by mathematically equivalent token repetition;
- no optimizer/scheduler advance when a group has no usable reward spread;
- durable no-update journal anchored to the exact full checkpoint;
- deterministic resume of RNG, sampler, text replay and health counters.

Three isolated 200-attempt pilots compare `kl_coef` values `0.04`, `0.01` and `0`. They must start from the same joint best and share registered batches/seeds. The formal run restarts from that same joint best after authenticating all pilot results; it never continues a pilot checkpoint.

## 7. Locked evaluation

`posttrain/ocr_locked_*` and `scripts/eval_ocr_anyres_locked.py` enforce one-shot evaluation.

Before reading sealed pixels or labels, the evaluator authenticates:

- selected formal checkpoint bytes;
- formal selection receipt;
- all three pilot results and KL selection receipt;
- joint-stage result;
- public dataset admission;
- locked build receipt and golden anchor;
- tokenizer and release identities.

It then creates an exclusive ledger claim. A failure after claim leaves an incomplete audit and still consumes the benchmark. Removing the ledger to rerun is a protocol violation.

Headline reporting uses raw grapheme CER. Exact match, code-point diagnostics, Unicode control support, EOS rate, invalid-output rate, immutable reference comparison and blank-image ablation remain part of promotion evidence.

## 8. Checkpoint and runtime invariants

- Checkpoints publish atomically; partial directories are never eligible.
- Resume compares model, tokenizer, optimizer, data, protocol and seed identities before state loading.
- Path relocation is allowed only where the contract explicitly treats paths as relocatable; content hashes remain unchanged.
- Frozen parameters are checked, not assumed from optimizer construction.
- The reference policy and source release are authenticated by bytes.
- Validation and locked reports are append-only evidence, not mutable dashboards.

## 9. Testing

Run from the repository root:

```bash
python scripts/check_repository_hygiene.py
ruff check Model Tokenizer scripts
python -m pytest -q Model/tests Tokenizer/tests
python -m build
```

CPU tests cover tensor shapes, mathematical equivalence, contracts, corruption cases and resume state. They do not prove official Mamba, CUDA BF16, GPU memory headroom or production throughput. Changes to those areas require a matching Linux/CUDA validation record.

## 10. Repository boundary

`Model/` contains source and tests only. Do not add weights, checkpoints, media fixtures, JSONL datasets, profiler dumps or run logs. Tests should construct minimal fixtures in temporary directories or memory. A new model module or CLI must prove a direct consumer in the supported lineage; self-references from tests or documentation are insufficient.
