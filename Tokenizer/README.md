# DoL-OCR Tokenizer

The tokenizer uses one stable token-ID space with two routed content tracks:

- morphology-aware MorphBPE for traditional Mongolian;
- a multilingual byte-level BPE for Chinese, English, Japanese, Cyrillic, digits, punctuation and other text.

Multimodal special tokens and image spans are part of the official DoL-1.2-OCR checkpoint identity. Token IDs, bundle files and normalization behavior must not be changed independently of the model release contract.

## Modules

| Path | Responsibility |
| --- | --- |
| `traditional_mongolian/` | Unicode normalization, suffix inventory and morphology analysis |
| `morphbpe/` | boundary-aware traditional-Mongolian BPE |
| `generic_bpe/` | multilingual byte-level BPE and lossless byte fallback |
| `unified/` | routing, stable ID space, serialization and bundle identity |
| `multimodal/` | image placeholder expansion and token-span metadata |
| `pretraining/` | encoded-row construction, packing and immutable receipts |
| `evals/` | tokenizer and pretraining-data gates |
| `tools/` | explicitly admitted tokenizer/data-build CLIs only |

## Install and test

```bash
python -m pip install -e ".[dev,image,model]"
python -m pytest -q Tokenizer/tests
```

## Build a tokenizer bundle

All inputs and outputs below are external to the Git working tree.

```bash
python -m Tokenizer.tools.prepare_corpus \
  --source mongolian \
  --input "$RAW_MONGOLIAN_CORPUS" \
  --output "$CLEAN_MONGOLIAN_ROWS"

python -m Tokenizer.tools.build_morphbpe \
  --input "$CLEAN_MONGOLIAN_ROWS" \
  --output "$MORPHBPE_MODEL" \
  --vocab-size 24000

python -m Tokenizer.tools.build_general_bpe \
  --input "$GENERAL_TEXT_ROWS" \
  --output "$GENERAL_BPE_MODEL" \
  --vocab-size 40000

python -m Tokenizer.tools.build_unified_tokenizer \
  --morphbpe "$MORPHBPE_MODEL" \
  --general "$GENERAL_BPE_MODEL" \
  --output "$TOKENIZER_BUNDLE"
```

`tokenizers` is a base dependency because bundle loading and encoding require it. Corpus preparation and training may additionally need the `tokenizer-build` extra.

## Encode pretraining rows

The builder accepts external `.txt` or `.jsonl` input, emits encoded shards and writes the exact receipt consumed by formal RDT training:

```bash
python -m Tokenizer.tools.build_pretraining_data \
  --tokenizer-bundle "$TOKENIZER_BUNDLE" \
  --input "$RAW_PRETRAIN_INPUT" \
  --output "$ENCODED_SHARD" \
  --receipt "$ENCODED_RECEIPT" \
  --max-length 2048 \
  --pack
```

Formal rows persist `input_ids`, `attention_mask`, `labels`, `word_pos` and `morph_depth`. The receipt binds exact shard bytes, the tokenizer bundle, tokenizer algorithm and registered row-producer source fingerprint. Formal `train_rdt` runs fail closed on a missing receipt, unknown producer or producer drift.

Validate before model allocation:

```bash
python -m Tokenizer.evals.pretraining_gate \
  --tokenizer-bundle "$TOKENIZER_BUNDLE" \
  --input "$ENCODED_SHARD" \
  --max-length 2048 \
  --json
```

## Multimodal processor

```python
from Tokenizer.multimodal import MultimodalProcessor

processor = MultimodalProcessor(tokenizer)
encoding = processor(
    "<image>",
    images=[image],
    image_sizes=[(height, width)],
)
print(encoding.input_ids)
print(encoding.image_token_spans)
```

`MultimodalProcessor` expands placeholders and records token-index spans. It does not own the vision tower. Pixel loading, AnyRes planning and visual feature extraction remain model/data-pipeline responsibilities.

AnyRes source dimensions are not forced into a user-visible fixed crop. The dataset builder preserves canonical original pixels and passes a reviewed multi-window plan to the native-detail visual path.

## Evaluation

```bash
python -m Tokenizer.evals.roundtrip_check --json
python -m Tokenizer.evals.offset_check --json
python -m Tokenizer.evals.chars_per_token --json
python -m Tokenizer.evals.mongolian_boundary_recall --json
python -m Tokenizer.evals.compare_baselines --json
```

Evaluation CLIs use built-in smoke strings when no external input is supplied. Do not commit an evaluation corpus or sample media merely to exercise these commands.

## Repository boundary

`Tokenizer/tools/` is protected by an exact allowlist. New tools require a current pretraining or AnyRes consumer and tests. This repository does not accept crawlers, corpus downloaders, tracked JSONL/CSV/TSV, images, generated tokenizer bundles or vocabulary artifacts.

Further contracts:

- [`docs/tokenizer_architecture.md`](docs/tokenizer_architecture.md)
- [`docs/offset_contract.md`](docs/offset_contract.md)
- [`docs/multimodal_contract.md`](docs/multimodal_contract.md)
- [`docs/pretraining_text_format.md`](docs/pretraining_text_format.md)
- [`docs/training_pipeline.md`](docs/training_pipeline.md)
