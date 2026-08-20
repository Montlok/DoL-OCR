# Tokenizer

Routed two-track tokenizer for traditional Mongolian (MorphBPE) and a general
multilingual byte-level BPE covering Chinese, English, Japanese and Cyrillic,
plus multimodal placeholders. Byte-level coverage means non-Mongolian text never
falls back to `<unk>`.

## Run tests

```bash
python3 -m unittest discover Tokenizer
```

## Segment text

```bash
python3 -m Tokenizer.unified.dual_tokenizer segment "ᠮᠣᠩᠭᠣᠯ 中文 test <image>"
```

## Build MorphBPE, the general BPE, and the unified tokenizer

```bash
# 1. Normalize heterogeneous corpora into UTF-8 JSONL ({"text": ...})
python3 -m Tokenizer.tools.prepare_corpus --source mongolian \
    --input "1000 traditional_mongolian_corpus(DO NOT GIT IT)" --output mn.jsonl
python3 -m Tokenizer.tools.prepare_corpus --source chinese \
    --input "CHINESE(DO NOT GIT IT)" --output general.jsonl
python3 -m Tokenizer.tools.prepare_corpus --source wiki \
    --lang ja --limit 20000 --output general.jsonl --append

# 2. Train the two tracks
python3 -m Tokenizer.tools.build_morphbpe --input mn.jsonl \
    --output morphbpe.json --vocab-size 24000
python3 -m Tokenizer.tools.build_general_bpe --input general.jsonl \
    --output general.json --vocab-size 40000

# 3. Assemble + validate the unified bundle
python3 -m Tokenizer.tools.build_unified_tokenizer \
    --morphbpe morphbpe.json --general general.json --output bundle/
```

The general BPE trainer needs `tokenizers`; corpus preparation from Wikipedia
needs `datasets`. Install both with `pip install -e '.[tokenizer-build]'`. If
`--general` is omitted, a training-free minimal byte-level fallback is used.

## One-click first pretraining

```bash
# Local self-test (seconds, CPU, synthetic corpus, NaiveSSM fallback)
SMOKE=1 scripts/pretrain_e2e.sh

# Real run (set the corpus paths to your local data)
MN_CORPUS="1000 traditional_mongolian_corpus(DO NOT GIT IT)" \
CN_CORPUS="CHINESE(DO NOT GIT IT)" \
scripts/pretrain_e2e.sh
```

The script is stage-gated and idempotent: corpus prep → tokenizer training →
packed shards → data gate → two-stage mHC pretraining (`two_stage_pretrain`).

## Use MultimodalProcessor

```python
from Tokenizer.multimodal import MultimodalProcessor

processor = MultimodalProcessor(tokenizer)
encoding = processor(
    "Describe <image>",
    images=[image],
    image_sizes=[(224, 224)],
)
print(encoding.input_ids)
print(encoding.image_token_spans)
```

`MultimodalProcessor` expands placeholders and records token spans. It leaves
actual vision feature extraction to an optional external image processor.

## Further documentation

- [docs/tokenizer_architecture.md](docs/tokenizer_architecture.md) — overall design.
- [docs/offset_contract.md](docs/offset_contract.md) — EncodedToken offset semantics.
- [docs/multimodal_contract.md](docs/multimodal_contract.md) — image/video processor contract.
- [docs/training_pipeline.md](docs/training_pipeline.md) — full training pipeline.
- [docs/external_implementation_notes.md](docs/external_implementation_notes.md) — design influences.

## Build pretraining rows

```bash
python3 -m Tokenizer.tools.build_pretraining_data \
  --tokenizer-bundle artefacts/tokenizer_bundle \
  --input Tokenizer/data/sample_multimodal.jsonl \
  --output artefacts/pretrain.jsonl \
  --receipt artefacts/pretrain.receipt.json \
  --max-length 2048 \
  --pack \
  --pad-to-max-length

python3 -m Tokenizer.evals.pretraining_gate \
  --tokenizer-bundle artefacts/tokenizer_bundle \
  --input artefacts/pretrain.jsonl \
  --max-length 2048 \
  --json

python3 -m scripts.train_rdt --config pretrain \
  --tokenizer-bundle artefacts/tokenizer_bundle \
  --data artefacts/pretrain.jsonl \
  --data-receipt artefacts/pretrain.receipt.json \
  --output runs/rdt
```

The pretraining builder masks structural labels with `-100`, writes
`word_pos` / `morph_depth` for the model's morphological RoPE, keeps
image/video spans aligned, and can pack text-only rows while leaving
multimodal rows standalone. It also writes a receipt binding the exact shard
bytes to the tokenizer bundle, shared tokenizer algorithm, and the named row
producer's current source fingerprint (by default `<output>.receipt.json`).
Formal `train_rdt` runs fail closed for an unknown producer, producer drift, or
a missing receipt; `--eval-data` likewise requires its own
`--eval-data-receipt`.
All production/non-smoke receipt-backed train, eval, and mix JSONL rows must
persist `word_pos` and `morph_depth`; model-side fallback exists only for
legacy/smoke inputs and must not be used in production.

## Evaluation

```bash
python3 -m Tokenizer.evals.roundtrip_check --json
python3 -m Tokenizer.evals.offset_check --json
python3 -m Tokenizer.evals.chars_per_token --json
python3 -m Tokenizer.evals.mongolian_boundary_recall --json
python3 -m Tokenizer.evals.compare_baselines --json
python3 -m Tokenizer.evals.tokenizer_hit_rate \
  --train-input Tokenizer/data/sample_text.jsonl \
  --input Tokenizer/data/menksoft_mt_phrases.jsonl \
  --json
```

All evals support `--input <path.jsonl|.txt>` and fall back to built-in smoke
samples when no input is provided.

`tokenizer_hit_rate` reports `<unk>` rate, token hit rate, byte fallback rate,
per-track character density, Mongolian word hit rate, and morpheme-boundary
respect. It can load a persisted `TokenizerBundle` or train a small experimental
MorphBPE tokenizer from `--train-input`.

## Collect Menksoft MT Phrase Samples

```bash
python3 -m Tokenizer.tools.menksoft_collect_phrases \
  --output Tokenizer/data/menksoft_mt_phrases.jsonl \
  --limit 80 \
  --batch-size 20
```

The collector calls Menksoft's public machine-translation endpoint and writes
silver-quality short phrases for tokenizer coverage experiments. Use
`--insecure` only in local environments whose Python TLS certificate store
cannot validate the site certificate.

## Normalize Mongolian via Rust bridge

```bash
python3 -m Tokenizer.tools.normalize_mongolian --nominal < raw.txt > clean.txt
```
