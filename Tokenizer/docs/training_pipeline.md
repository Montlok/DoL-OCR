# Tokenizer training pipeline

This document describes the supported tokenizer and receipt-bound pretraining path. Corpus files, generated tokenizer artifacts and encoded shards live outside the Git worktree.

## 1. Prepare external text

Use the admitted Python preparation tools to normalize source-specific records into UTF-8 text rows:

```bash
python -m Tokenizer.tools.prepare_corpus \
  --source mongolian \
  --input "$RAW_MONGOLIAN_CORPUS" \
  --output "$CLEAN_MONGOLIAN_ROWS"
```

Traditional-Mongolian Unicode behavior is owned by `Tokenizer/traditional_mongolian/unicode_norm.py`. Do not add an independent encoding converter or silently fold contextual FVS/MVS/NNBSP distinctions required by OCR targets.

Every external corpus must have documented provenance, authorization, deletion policy and train/evaluation exclusion evidence. Data acquisition and crawling do not belong in this repository.

## 2. Train MorphBPE

```bash
python -m Tokenizer.tools.build_morphbpe \
  --input "$CLEAN_MONGOLIAN_ROWS" \
  --output "$MORPHBPE_MODEL" \
  --vocab-size 24000 \
  --min-pair-freq 2 \
  --min-boundary-confidence 0.60
```

`MorphBPETrainer` uses `MongolStemmer.analyze(word).skeleton_boundaries` as forbidden merge boundaries when analysis confidence is high enough. Low-confidence analyses remain lexical words rather than forcing a false morphology split.

## 3. Train the general track and assemble a bundle

```bash
python -m Tokenizer.tools.build_general_bpe \
  --input "$GENERAL_TEXT_ROWS" \
  --output "$GENERAL_BPE_MODEL" \
  --vocab-size 40000

python -m Tokenizer.tools.build_unified_tokenizer \
  --morphbpe "$MORPHBPE_MODEL" \
  --general "$GENERAL_BPE_MODEL" \
  --output "$TOKENIZER_BUNDLE"
```

The general track is byte-level so non-Mongolian scripts encode without `<unk>`. The unified bundle fixes the global ID layout and special-token identities used by RDT and OMVT.

## 4. Check routed encoding

```python
from Tokenizer.unified.bundle import TokenizerBundle

bundle = TokenizerBundle.from_dir(tokenizer_bundle_path)
result = bundle.tokenizer.encode_with_spans("ᠮᠣᠩᠭᠣᠯ 文字 hello <image>")
assert len(result.input_ids) == len(result.tokens)
```

For official DoL-1.2-OCR continuation, the bundle bytes and token-ID map must match the release lock exactly. Rebuilding a “similar” vocabulary is not compatible.

## 5. Build receipt-bound pretraining rows

`Tokenizer.pretraining.PretrainingDataBuilder` writes causal-LM rows containing `input_ids`, `attention_mask`, `labels`, `word_pos` and `morph_depth`. Structural and placeholder labels use `IGNORE_INDEX` (`-100`); EOS remains supervised.

```bash
python -m Tokenizer.tools.build_pretraining_data \
  --tokenizer-bundle "$TOKENIZER_BUNDLE" \
  --input "$RAW_PRETRAIN_INPUT" \
  --output "$ENCODED_SHARD" \
  --receipt "$ENCODED_RECEIPT" \
  --max-length 2048 \
  --pack \
  --pack-max-length 2048
```

Packing combines text-only rows. Multimodal rows remain standalone, and truncation must never leave a partial image-token span. The emitted receipt binds:

- exact tokenizer bundle files and algorithm identity;
- registered row-producer name and source fingerprint;
- ordered output shard paths, byte sizes and SHA-256 values;
- aggregate shard digest.

Build validation data separately. Training and validation may not share a receipt.

## 6. Gate encoded rows

```bash
python -m Tokenizer.evals.pretraining_gate \
  --tokenizer-bundle "$TOKENIZER_BUNDLE" \
  --input "$ENCODED_SHARD" \
  --max-length 2048 \
  --max-unk-rate 0.01 \
  --min-supervised-rate 0.01 \
  --json
```

The gate validates IDs, sequence lengths, labels, morphology fields, multimodal spans, loss masks, unknown rate and supervised-token rate. Formal training must still validate the receipt; passing the row gate alone is not admission.

## 7. Launch RDT pretraining

```bash
python -m scripts.train_rdt \
  --config two_stage_pretrain \
  --mamba official \
  --tokenizer-bundle "$TOKENIZER_BUNDLE" \
  --data "$ENCODED_SHARD" \
  --data-receipt "$ENCODED_RECEIPT" \
  --eval-data "$ENCODED_VALIDATION_SHARD" \
  --eval-data-receipt "$ENCODED_VALIDATION_RECEIPT" \
  --output "$RDT_OUTPUT"
```

Formal training fails on missing receipts, producer drift, tokenizer drift or rows missing required morphology fields. CPU smoke inputs may exercise bounded fallbacks, but those fallbacks are not a production data contract.

## 8. Multimodal alignment

Pre-rendered OCR pairs are built through `scripts.build_ocr_data_from_pairs`; synthetic vertical lines, when explicitly required for pretraining, are built through `scripts.build_ocr_data`. Both outputs remain outside Git and are bound by their own data contracts.

OMVT/VLM preparation then uses:

```bash
python -m scripts.train_omvt_ssl --data "$VISION_DATA" --output "$OMVT_OUTPUT"

python -m scripts.train_vlm_align \
  --freeze-rdt \
  --ocr-native-targets \
  --ocr-tokenizer-bundle "$TOKENIZER_BUNDLE" \
  --ocr-data-contract "$OCR_DATA_CONTRACT" \
  --data "$ALIGNMENT_DATA" \
  --init-rdt-checkpoint "$RDT_CHECKPOINT" \
  --init-omvt-checkpoint "$OMVT_CHECKPOINT" \
  --output "$ALIGNMENT_OUTPUT"
```

Frozen-language OCR alignment requires native target encoding and exact tokenizer round-trip. Unknown tokens or contextual-control drift are errors, not fallback opportunities.

## 9. Tests

```bash
python -m pytest -q Tokenizer/tests
python scripts/check_repository_hygiene.py
```

Tests construct fixtures in temporary directories or memory. Do not add tracked sample corpora or media.
