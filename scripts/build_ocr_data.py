# -*- coding: utf-8 -*-

"""Build a generative-OCR training set for traditional Mongolian.

Renders each transcription string to a vertical-script image and emits a
pre-tokenized JSONL row (see :mod:`Model.ocr.data` for the token contract) that
``scripts/train_vlm_align.py --data ...`` consumes directly. The rendered text
*is* the label, so the ground truth is exact and free of annotation cost.

Input: a UTF-8 text file (one transcription per line) or a JSONL with a
``text`` field per line. Output: ``<out>/images/*.png`` + ``<out>/data.jsonl``.

Vertical rendering needs Pillow built with **libraqm** (for ``direction="ttb"``)
and a traditional-Mongolian font (e.g. Menksoft Qagan / Noto Sans Mongolian).
This environment may lack both; the renderer fails with an actionable message
rather than producing wrong images. The token-row contract (:mod:`Model.ocr.data`)
is independently unit-tested without rendering.

Target encoding contract (pretraining-compatible and lossless)
---------------------------------------------------------------
The OCR target is the supervision signal, not free text. It must preserve the
transcription *and* the token representation learned by the language
checkpoint. :func:`make_ocr_target_encoder` therefore uses the tokenizer's
native MorphBPE/general path first and accepts it only when it contains no
``<unk>`` and round-trips exactly as both Unicode text and UTF-8 bytes, after
the tokenizer's documented NBSP-to-word-boundary normalization. Contextual
Mongolian NNBSP remains byte-exact.

``mode="native"`` is the fail-closed production contract for a frozen language
model: any text the pretrained tokenizer cannot represent aborts rather than
silently changing the output language. ``mode="native_fallback"`` retains a
lossless byte fallback for data-building workflows that can train the language
side on fallback tokens. ``mode="byte_fallback"`` is the legacy character/byte
representation and must not be used to align a frozen pretrained LM.

A lone surrogate or otherwise undecodable input fails the round-trip check and
aborts loudly (by design — a silently-corrupted target is worse than a stopped
build). Instruction text is masked from the loss, but it still uses the same
span-aware tokenizer path so its ``word_pos``/``morph_depth`` features match
language pretraining.

``--max-seq-len`` guards the other side effect of exact byte-fallback
encoding: worst case it can inflate a target to ~3x its MorphBPE-routed
length (one token per UTF-8 byte instead of one token per morpheme). A row
whose total length (BOS + image slots + instruction + target + EOS) exceeds
this budget is skipped *before* the (expensive) render step. Without the
flag the builder still runs, but the summary line warns if any row would
already exceed a 4096-token budget (:class:`Model.training.TrainingConfig`'s
default ``seq_len``) — multimodal rows over ``seq_len`` are a hard crash in
the collator (:mod:`Model.training.data`), not a soft truncation, because
truncating text would desync ``<image_patch>`` slots from the image payload.

Usage::

    python -m scripts.build_ocr_data \
        --input lines.txt --out data/ocr_synth \
        --font /path/to/MongolianFont.ttf \
        --tokenizer-bundle outputs/tok_build/tokenizer/bundle \
        --image-size 224 --max-seq-len 4096
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Model.config import (  # noqa: E402
    BOS_ID,
    EOS_ID,
    IMAGE_END_ID,
    IMAGE_PATCH_ID,
    IMAGE_START_ID,
)
from Model.ocr.alignment_contract import (  # noqa: E402
    build_ocr_alignment_data_contract,
    write_ocr_alignment_data_contract,
)
from Model.ocr.data import build_ocr_row  # noqa: E402
from Model.ocr.tokenization import (  # noqa: E402
    encode_lm_text_features,
    make_ocr_target_encoder,
    native_tokenization_contract,
)

# BOS + <image_start> + <image_end> + EOS; matches build_ocr_row(add_eos=True)
# with a single image (n_image_tokens slots are additional and counted
# separately by the caller).
_PROMPT_FIXED_OVERHEAD = 4


def _check_vertical_support() -> None:
    """Raise an actionable error if Pillow cannot render vertical text."""
    try:
        from PIL import features
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(f"Pillow is required for rendering: {exc}") from exc
    if not features.check("raqm"):
        raise RuntimeError(
            "Pillow lacks libraqm, so direction='ttb' vertical rendering is "
            "unavailable. Install a Pillow build with raqm (e.g. system libraqm "
            "+ 'pip install --force-reinstall pillow') and rerun on that host."
        )


def render_vertical_line(
    text: str,
    font_path: str,
    *,
    image_size: int = 224,
    font_size: int = 28,
    padding: int = 12,
    bg: int = 255,
    fg: int = 0,
):
    """Render ``text`` as a top-to-bottom Mongolian line on a square canvas.

    Returns a grayscale ``PIL.Image`` of side ``image_size``. Requires libraqm.
    """
    _check_vertical_support()
    from PIL import Image, ImageDraw, ImageFont

    if not Path(font_path).exists():
        raise FileNotFoundError(f"font not found: {font_path}")
    font = ImageFont.truetype(font_path, font_size)

    img = Image.new("L", (image_size, image_size), bg)
    draw = ImageDraw.Draw(img)
    # direction="ttb" needs raqm; layout columns left-to-right is the script's
    # natural flow and must not be pre-rotated to horizontal.
    draw.text(
        (padding, padding),
        text,
        font=font,
        fill=fg,
        direction="ttb",
    )
    return img


def _iter_input_text(path: str) -> Iterator[tuple[int, str]]:
    """Yield ``(line_number, text)`` from plain text or JSONL without full reads."""

    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            if line[0] in "{[":
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{input_path}:{lineno}: invalid JSONL: {exc.msg}"
                    ) from exc
                if not isinstance(obj, dict):
                    raise ValueError(
                        f"{input_path}:{lineno}: JSONL row must be an object"
                    )
                text = obj.get("text")
                if not isinstance(text, str) or not text:
                    raise ValueError(
                        f"{input_path}:{lineno}: JSONL row missing non-empty 'text'"
                    )
                yield lineno, text
            else:
                yield lineno, line


def main() -> int:
    ap = argparse.ArgumentParser(description="Build Mongolian generative-OCR data")
    ap.add_argument("--input", required=True, help="text (one line/sample) or JSONL")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--font", required=True, help="traditional Mongolian .ttf/.otf")
    ap.add_argument("--tokenizer-bundle", required=True)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--font-size", type=int, default=28)
    ap.add_argument(
        "--n-image-tokens",
        type=int,
        default=None,
        help="<image_patch> slots per image; must equal OMVT compress_to. "
        "Defaults to image_patch_count(image_size, image_size).",
    )
    ap.add_argument(
        "--instruction",
        default="",
        help="optional prompt text inserted before the transcription target",
    )
    ap.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="skip rows whose total length would exceed this budget before "
        "rendering. Total length = 4 (BOS + <image_start> + <image_end> + "
        "EOS) + n_image_tokens + len(instruction_ids) + len(target_ids), "
        "matching Model/ocr/data.py:build_ocr_row(add_eos=True). "
        "Multimodal rows longer than TrainingConfig.seq_len are refused by "
        "Model/training/data.py at train time, not truncated, so this guard "
        "must run at build time. Unset = no skipping (see the summary "
        "warning instead).",
    )
    args = ap.parse_args()

    from Tokenizer.multimodal.image_placeholders import image_patch_count
    from Tokenizer.unified.bundle import TokenizerBundle

    n_image_tokens = args.n_image_tokens or image_patch_count(
        args.image_size, args.image_size
    )

    bundle = TokenizerBundle.from_dir(args.tokenizer_bundle)
    bundle_issues = bundle.validate()
    if bundle_issues:
        raise ValueError(
            "invalid tokenizer bundle:\n  - " + "\n  - ".join(bundle_issues)
        )
    instruction_features = (
        encode_lm_text_features(
            bundle.tokenizer,
            args.instruction,
            interpret_special_tokens=True,
        )
        if args.instruction
        else None
    )
    instruction_ids = (
        instruction_features.input_ids if instruction_features is not None else []
    )
    instruction_track_ids = (
        instruction_features.morphology_track_ids
        if instruction_features is not None
        else []
    )
    encode_target = make_ocr_target_encoder(bundle.tokenizer)
    prompt_overhead = _PROMPT_FIXED_OVERHEAD + n_image_tokens + len(instruction_ids)

    out_dir = Path(args.out)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    n_seen = 0
    n_written = 0
    n_skipped_long = 0
    max_row_len = 0
    with (out_dir / "data.jsonl").open("w", encoding="utf-8") as fh:
        for lineno, text in _iter_input_text(args.input):
            n_seen += 1
            try:
                target_features = encode_target.encode_with_features(text)
                target_ids = target_features.input_ids
            except ValueError as exc:
                raise ValueError(f"{args.input}:{lineno}: {exc}") from exc
            if not target_ids:
                continue

            row_len = prompt_overhead + len(target_ids)
            if args.max_seq_len is not None and row_len > args.max_seq_len:
                n_skipped_long += 1
                continue

            img = render_vertical_line(
                text,
                args.font,
                image_size=args.image_size,
                font_size=args.font_size,
            )
            rel = f"images/{n_written:08d}.png"
            image_path = out_dir / rel
            encoded_image = io.BytesIO()
            img.save(encoded_image, format="PNG")
            image_bytes = encoded_image.getvalue()
            image_path.write_bytes(image_bytes)
            image_size_bytes = len(image_bytes)
            image_sha256 = hashlib.sha256(image_bytes).hexdigest()

            try:
                row = build_ocr_row(
                    target_ids,
                    n_image_tokens,
                    str(image_path.resolve()),
                    bos_id=BOS_ID,
                    image_start_id=IMAGE_START_ID,
                    image_patch_id=IMAGE_PATCH_ID,
                    image_end_id=IMAGE_END_ID,
                    eos_id=EOS_ID,
                    instruction_ids=instruction_ids,
                    instruction_track_ids=instruction_track_ids,
                    target_track_ids=target_features.morphology_track_ids,
                    image_sha256=image_sha256,
                    image_size_bytes=image_size_bytes,
                )
            except ValueError as exc:
                raise ValueError(f"{args.input}:{lineno}: {exc}") from exc
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1
            max_row_len = max(max_row_len, len(row["input_ids"]))

    if n_seen == 0:
        print("[build-ocr] no input lines")
        return 1
    if n_written == 0:
        print("[build-ocr] no rows written after tokenization")
        return 1
    data_path = out_dir / "data.jsonl"
    data_contract = build_ocr_alignment_data_contract(
        data_path,
        native_tokenization_contract(bundle.tokenizer, args.tokenizer_bundle),
    )
    contract_path = out_dir / "ocr_data_contract.json"
    write_ocr_alignment_data_contract(contract_path, data_contract)

    print(
        f"[build-ocr] wrote {n_written} rows -> {data_path} "
        f"(n_image_tokens={n_image_tokens}, image_size={args.image_size}) "
        f"max_row_len={max_row_len} skipped_too_long={n_skipped_long} "
        f"contract={contract_path}"
    )
    if n_skipped_long > 0:
        print(
            f"[build-ocr] WARNING: {n_skipped_long} row(s) skipped for "
            f"exceeding --max-seq-len={args.max_seq_len}"
        )
    elif args.max_seq_len is None and max_row_len > 4096:
        print(
            "[build-ocr] WARNING: max_row_len "
            f"({max_row_len}) exceeds the TrainingConfig default seq_len "
            "(4096) and --max-seq-len was not set; pass --max-seq-len to "
            "skip over-budget rows at build time, or raise "
            "train_vlm_align.py --seq-len to at least max_row_len"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
