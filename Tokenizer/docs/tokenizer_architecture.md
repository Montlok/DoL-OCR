# Tokenizer architecture

## Current contract

DoL-OCR uses a routed two-track tokenizer with one stable global token-ID space. Traditional Mongolian is handled by morphology-aware MorphBPE; all other text is handled by a multilingual byte-level BPE. Every emitted token records source offsets for downstream position and OCR alignment.

The official DoL-1.2-OCR bundle identity is immutable. AnyRes continuation must use the exact tokenizer files and ID map bound by `Model/posttrain/release_locks/dol_1_2_ocr.json`.

## Module responsibilities

- `Tokenizer/traditional_mongolian/` owns Unicode normalization, suffix inventories, reverse stemming and morphology boundaries.
- `Tokenizer/morphbpe/` implements boundary-constrained BPE for traditional Mongolian and seeds the valid letter inventory so unseen valid letters remain representable.
- `Tokenizer/generic_bpe/` implements the general byte-level track and lossless byte fallback.
- `Tokenizer/unified/` routes spans, owns bundle serialization and provides one ID space across both tracks.
- `Tokenizer/multimodal/` expands image/video placeholders and records token-index spans. It does not encode pixels.
- `Tokenizer/pretraining/` builds encoded rows, packs text-only samples and emits immutable producer/data receipts.
- `Tokenizer/evals/` validates round-trip, offsets, coverage and formal pretraining rows.

Traditional-Mongolian normalization has one Python owner in `traditional_mongolian/unicode_norm.py`. OCR native-target validation preserves contextual FVS, MVS and NNBSP distinctions and fails closed when a transcription cannot round-trip through the frozen tokenizer.

## Token-ID layout

```python
SEGMENT = {
    "special": (0, 256),
    "mongolian": (256, 24576),
    "general": (24576, 65536),
}
```

Core special IDs include `<pad>`, `<unk>`, `<bos>`, `<eos>`, image boundary/patch tokens, OCR boundary tokens and the shared word-boundary token. Reserved IDs are checkpoint ABI: changing their order or meaning requires a new tokenizer/model release, not a local migration.

## Routing

`DualTrackTokenizer.encode_with_spans()` splits the input into:

- traditional-Mongolian spans routed to MorphBPE;
- non-Mongolian spans routed to the general byte-level model;
- exact special tokens routed before either content track;
- explicit whitespace/newline behavior defined by the unified contract.

The result contains `input_ids`, `EncodedToken` records, routing spans, attention mask and special-token mask. Every content token uses Python-character offsets into the original input.

## Multimodal placeholders

`MultimodalProcessor` expands `<image>` into image-start, N image-patch and image-end tokens, then returns `image_token_spans` as input-ID index ranges. An optional caller-owned image processor may provide pixels, but feature extraction and AnyRes planning belong to the model/data pipeline.

The tokenizer reserves other multimodal IDs for checkpoint compatibility. Reserved IDs do not by themselves claim an implemented video or serving capability.

## Pretraining rows

`PretrainingDataBuilder` generates the model-facing fields and masks structural labels with `-100`. Formal rows persist morphology features and are distributed with an immutable receipt binding the exact tokenizer, producer source and output shard bytes.

## Tests

```bash
python -m pytest -q Tokenizer/tests
python scripts/check_repository_hygiene.py
```

Related documents:

- [offset_contract.md](offset_contract.md)
- [multimodal_contract.md](multimodal_contract.md)
- [pretraining_text_format.md](pretraining_text_format.md)
- [training_pipeline.md](training_pipeline.md)
