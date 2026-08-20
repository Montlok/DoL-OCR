# Training Pipeline

End-to-end recipe for producing the v0.8 unified tokenizer artefact.

## 1. Normalize raw Mongolian text

Use the Rust normalizer to fold Menksoft PUA and MW presentation
variants down to nominal Unicode before any tokenizer training touches
the text:

```bash
python -m Tokenizer.tools.normalize_mongolian --nominal \
    --input data/raw_mongolian.txt --output data/clean_mongolian.txt
```

The Python tool shells out to the Rust `normalize` example
(`cd "Encoding Mapping" && cargo run --example normalize -- --nominal`).
A PyO3 binding is a planned follow-up; the CLI bridge is the v0.8
contract.

## 2. Train MorphBPE

```bash
python -m Tokenizer.tools.build_morphbpe \
    --input Tokenizer/data/sample_text.jsonl \
    --output artefacts/morphbpe.json \
    --vocab-size 4096 \
    --min-pair-freq 2 \
    --min-boundary-confidence 0.60
```

`MorphBPETrainer` uses `MongolStemmer.analyze(word).skeleton_boundaries`
as forbidden merge boundaries only when the stemmer confidence meets the
configured threshold. Low-confidence analyses are treated as lexical words,
which prevents false roots like splitting `ᠮᠣᠩᠭᠣᠯ` at final `ᠯ`.
The trainer also ignores non-Mongolian whitespace-delimited words in mixed
corpora; all non-Mongolian text is handled by the general byte-level BPE track
outside MorphBPE. The JSON output follows `serialization.SCHEMA_VERSION == 1`.

## 3. Build the unified tokenizer

Train the general multilingual byte-level BPE track, then combine it with
MorphBPE through `Tokenizer.tools.build_unified_tokenizer` (or programmatically
via `build_unified_vocab` + `DualTrackTokenizer`):

```bash
python3 -m Tokenizer.tools.build_general_bpe \
    --input general.jsonl --output general.json --vocab-size 40000
python3 -m Tokenizer.tools.build_unified_tokenizer \
    --morphbpe morphbpe.json --general general.json --output bundle/
```

The general track is byte-level, so Chinese, English, Japanese, Cyrillic, digits
and punctuation all encode without `<unk>`. Omitting `--general` falls back to a
training-free minimal byte-level model.

## 4. Encode mixed text

```python
from Tokenizer.unified.dual_tokenizer import DualTrackTokenizer
result = tokenizer.encode_with_spans("ᠮᠣᠩᠭᠣᠯ 文字 hello <image>")
result.input_ids
result.tokens         # list[EncodedToken]
result.spans          # language-routed spans
result.special_tokens_mask
```

## 5. Multimodal

```python
from Tokenizer.multimodal import MultimodalProcessor
proc = MultimodalProcessor(tokenizer)
enc = proc("describe <image>", images=[img], image_sizes=[(448, 448)])
```

## 6. Build pretraining JSONL

`Tokenizer.pretraining.PretrainingDataBuilder` writes causal-LM style
`input_ids` / `attention_mask` / `labels` rows, plus `word_pos` and
`morph_depth` for `Model.layers.rope.MorphologicalRoPE`. Labels equal input
ids for learnable text tokens, but structural tokens are masked with
`IGNORE_INDEX` (`-100`): `<pad>`, `<bos>`, multimodal placeholders,
image/video/audio patch tokens, OCR/layout tags, and `<unk>`. `<eos>` remains
supervised so sequence endings can be learned.

```bash
python -m Tokenizer.tools.build_pretraining_data \
    --tokenizer-bundle artefacts/tokenizer_bundle \
    --input Tokenizer/data/sample_multimodal.jsonl \
    --output artefacts/pretrain.jsonl \
    --receipt artefacts/pretrain.receipt.json \
    --max-length 2048 \
    --pack \
    --pack-max-length 2048 \
    --pad-to-max-length
```

Packing only combines text-only rows. Multimodal rows stay as standalone
samples, and truncation never leaves a partial image/video token span. If a
span cannot fit, the whole span is removed; empty samples are skipped. Packed
rows shift `word_pos` between samples so recurrent attention does not see two
independent examples at the same morphological position.

`Model.training.PretrainingCollator` can read these rows in a PyTorch
`DataLoader` and right-pad `input_ids`, `attention_mask`, `labels`, `word_pos`,
and `morph_depth`. For lower training memory, call the model with
`return_logits=False` and set `loss_chunk_size` either in `RDTConfig` or the
forward call.

## 7. Gate pretraining data

Run the gate before launching a real training job. It validates ids, labels,
offsets, `word_pos` / `morph_depth`, multimodal span markers, loss masks inside
image/video spans, unknown rate, and supervised-token rate.

```bash
python -m Tokenizer.evals.pretraining_gate \
    --tokenizer-bundle artefacts/tokenizer_bundle \
    --input artefacts/pretrain.jsonl \
    --max-length 2048 \
    --max-unk-rate 0.01 \
    --min-supervised-rate 0.01 \
    --json
```

The builder receipt binds the emitted shard bytes to the exact tokenizer
bundle, shared tokenizer algorithm, and the current source fingerprint of the
named row producer. `train_rdt` accepts only registered producer kinds and
fails closed on producer drift. Pass the receipt to every formal RDT run;
never construct a receipt path independently of the builder output:

```bash
python -m scripts.train_rdt --config pretrain \
    --tokenizer-bundle artefacts/tokenizer_bundle \
    --data artefacts/pretrain.jsonl \
    --data-receipt artefacts/pretrain.receipt.json \
    --output runs/rdt
```

When validation data is configured, build it separately and pass its emitted
receipt with `--eval-data-receipt` alongside `--eval-data`.
Every production/non-smoke receipt-backed train, eval, or mix JSONL row must
persist `word_pos` and `morph_depth`; model-side derivation/fallback is only for
legacy/smoke inputs, never production.

## 8. Evaluate

```bash
python -m Tokenizer.evals.roundtrip_check --json
python -m Tokenizer.evals.offset_check --json
python -m Tokenizer.evals.chars_per_token --json
python -m Tokenizer.evals.mongolian_boundary_recall --json
python -m Tokenizer.evals.compare_baselines --json
```

All evals support `--input` (jsonl or txt) and fall back to built-in
smoke samples when no input is provided.

## 9. Run the test suite

```bash
scripts/test_all.sh
```
