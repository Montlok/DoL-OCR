# Offset Contract

Every `EncodedToken` produced by `Tokenizer.unified.DualTrackTokenizer` and
its sub-tokenizers carries `start` and `end` integer fields. This document
specifies what those fields mean.

## Coordinate system

- `start` and `end` are **Python character offsets** into the original
  input string passed to `encode_with_spans(text)`. They are not byte
  offsets and not UTF-16 code-unit offsets.
- The slice `text[token.start:token.end]` should round-trip — when the
  tokenizer can do so — to a string that re-encodes to the same token.
- `[start, end)` is half-open. `start == end` is allowed for zero-width
  tokens such as inserted boundary markers.

## Special cases

- BOS, EOS, and other synthetic specials use `start = end = -1`.
- Byte-fallback tokens point at the **original character span** that
  produced the bytes. A 3-byte emoji becomes 3 byte tokens that share
  the same `(start, end)` covering the source character.
- MorphBPE tokens cover the original-text character range, including any
  `MVS` / `NNBSP` / `FVS` control characters that fall inside the
  morpheme. When precise mapping is not possible the tokenizer uses
  conservative wider spans rather than raising.
- Image / video patch tokens use:
  - `start, end` = the span of the `<image>` (or `<video>`)
    placeholder in the original text, or `-1, -1` if the placeholder
    was injected programmatically.
  - `metadata = {"image_index": N}` (or `video_index`) so consumers can
    correlate patches to a specific image in the `images=` argument.

## Invariants

For every encoded sequence:

1. All non-special token spans satisfy `0 <= start <= end <= len(text)`.
2. The non-special spans are weakly monotonic: each `start` is `>=` the
   previous token's `start`.
3. `len(input_ids) == len(tokens) == len(attention_mask)`.
4. `special_tokens_mask[i] == 1` iff `tokens[i].track == "special"`.

## Why this matters

The contract enables:

- Re-rendering token-level model outputs back onto the source string
  (highlighting, NER, alignment).
- Sanity-checking that BPE merges never silently dropped or duplicated
  characters.
- Aligning multimodal placeholder ranges with raw image/video inputs.
