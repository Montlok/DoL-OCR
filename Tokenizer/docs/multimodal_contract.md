# Multimodal Contract

`Tokenizer.multimodal.MultimodalProcessor` is the entry point for
text + images + videos. It does **not** load pixels — it only:

1. Expands `<image>` and `<video>` placeholders into
   `<image_start> <image_patch>*N <image_end>` (and the video
   equivalent).
2. Routes the resulting text through the unified tokenizer.
3. Reports `image_token_spans` / `video_token_spans` as inclusive-start /
   exclusive-end **index ranges into `input_ids`**, not character spans.
4. Annotates every patch token with `metadata["image_index"]` (or
   `video_index`) so downstream model code can match patches to vision
   features.

## Patch counts

- `image_patch_count(w, h, patch_size=14, merge_size=2)` rounds up to a
  whole number of patches and divides by `merge_size**2` to mimic
  Qwen2-VL / LLaVA-style spatial merging.
- `video_patch_count(num_frames, w, h, patch_size=14, temporal_patch_size=2, merge_size=2)`
  applies the same logic with an additional temporal merge factor.

## Count validation

`expand_image_placeholders_by_sizes(text, image_sizes)` raises
`ValueError` when the number of `<image>` tokens in `text` does not
match `len(image_sizes)`. Same contract for the video helper. This
mirrors the placeholder/feature-count check used by vLLM.

## Bounding boxes

`Tokenizer.multimodal.bbox` encodes `[x0, y0, x1, y1]` (pixel
coordinates) as `<bbox_xxx_yyy_xxx_yyy>` with each coordinate
quantised into `bins` (default 1000) buckets. Round-tripping is lossy
by at most one bin width per coordinate.

## What this layer does not do

- It does not produce `pixel_values`. A future `image_processor`
  implementation will plug into the `image_processor=` argument and
  populate `MultimodalEncoding.pixel_values`.
- It does not own the vision encoder. The tokenizer-side contract only
  guarantees a stable mapping from placeholder text to `input_ids`
  ranges and back.
