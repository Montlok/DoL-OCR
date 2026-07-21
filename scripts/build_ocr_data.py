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

Target encoding contract (lossless byte fallback)
--------------------------------------------------
The OCR target is the supervision signal, not free text: encoding it through
the plain MorphBPE track (``bundle.encode``) would route rare morphemes to
``<unk>`` or lossy merges, quietly corrupting the label. ``main()`` instead
routes the target through :func:`make_ocr_target_encoder`, which forces every
character through the tokenizer's existing byte-fallback machinery
(:func:`Tokenizer.generic_bpe.encode_byte_fallback`) against a filtered "safe"
vocab view, then verifies the row round-trips **exactly** — code points and
UTF-8 bytes — before it is written. This holds for every character including
FVS1-4, MVS, and NNBSP. Building the id space or adding vocabulary is out of
scope: the encoder only rearranges which existing ids a character maps to. A
lone surrogate or otherwise undecodable input fails the round-trip check and
aborts the build loudly (by design — a silently-corrupted target is worse than
a stopped build). Instruction text is unaffected: it still goes through
``bundle.encode`` unchanged, since it is a masked prompt, not supervision.

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
from Model.ocr.data import build_ocr_row  # noqa: E402
from Tokenizer.generic_bpe import encode_byte_fallback  # noqa: E402
from Tokenizer.unified.vocab import SPECIAL_TOKENS, make_byte_tokens  # noqa: E402

# BOS + <image_start> + <image_end> + EOS; matches build_ocr_row(add_eos=True)
# with a single image (n_image_tokens slots are additional and counted
# separately by the caller).
_PROMPT_FIXED_OVERHEAD = 4


def make_ocr_target_encoder(tokenizer):
    """Build a lossless byte-fallback encoder for OCR *targets* only.

    Returns a closure ``encode_target(text) -> list[int]`` that guarantees,
    for any ``text``:

    1. no ``<unk>`` id in the output,
    2. ``tokenizer.decode(encode_target(text)) == text`` — both as Python
       str equality and as UTF-8 byte equality (``surrogatepass`` so lone
       surrogates are covered instead of silently raising).

    Two failure modes are handled explicitly rather than silently:

    - Missing byte tokens: production bundles carry all 256 ``<0xNN>``
      tokens (see ``Tokenizer/tests/test_stream_decode.py``), but nothing
      in the build tooling *writes* them, so a hand-rolled or stale bundle
      can be missing some. Preflighting here turns that into one clear
      error instead of scattered per-row ``<unk>`` corruption.
    - The ByteLevel-alphabet trap: ``GeneralBPEModel.minimal()`` (and any
      general-BPE vocab trained the same way) assigns direct single-char
      ids to the raw ByteLevel alphabet, e.g. ``"ä"`` or ``"Ġ"``. Handing
      the *full* unified vocab to ``encode_byte_fallback`` would direct-hit
      those ids, but ``DualTrackTokenizer.decode`` reinterprets anything in
      ``general_global_to_local`` by calling ``general.decode`` on it — for
      a ByteLevel model that reinterprets the character as raw *bytes*,
      mangling it. The safe vocab below excludes every id that decode would
      route through the general segment, so those characters are forced
      through the (verified-safe) ``<0xNN>`` byte path instead. The same
      exclusion also catches ``"▁"`` (id 17) and ``"◈"`` (id 18):
      ``decode`` special-cases them to ``" "`` / drop, so they must not be
      used as direct single-char hits either.
    """
    byte_tokens = make_byte_tokens()
    missing = [tok for tok in byte_tokens if tok not in tokenizer.vocab]
    if missing:
        raise ValueError(
            f"tokenizer vocab is missing {len(missing)}/256 byte-fallback "
            f"tokens (e.g. {missing[0]!r}); lossless OCR target encoding "
            "requires all <0xNN> tokens to be present. This bundle was not "
            "built with byte-token coverage — regenerate it, or use a "
            "production bundle (see Tokenizer/tests/test_stream_decode.py "
            "for the expected layout). Adding the missing tokens to an "
            "existing bundle is out of scope for this encoder (it must not "
            "grow the vocabulary)."
        )

    safe_vocab = {
        token: idx
        for token, idx in tokenizer.vocab.items()
        if len(token) == 1
        and idx not in tokenizer.general_global_to_local
        and token not in SPECIAL_TOKENS
    }
    for tok in byte_tokens:
        safe_vocab[tok] = tokenizer.vocab[tok]

    unk_id = tokenizer.unk_id

    def encode_target(text: str) -> list[int]:
        encoded = encode_byte_fallback(text, safe_vocab, unk_id)
        ids = [tok.id for tok in encoded]
        if unk_id in ids:
            raise ValueError(
                f"OCR target encoding produced <unk> (id={unk_id}) for "
                f"text={text!r}; this should be impossible with a complete "
                "byte-fallback vocab — check byte-token coverage"
            )
        decoded = tokenizer.decode(ids)
        if decoded != text:
            raise ValueError(
                "OCR target round-trip mismatch (str): "
                f"decode(encode(text)) != text\n  text={text!r}\n"
                f"  decoded={decoded!r}"
            )
        want_bytes = text.encode("utf-8", "surrogatepass")
        got_bytes = decoded.encode("utf-8", "surrogatepass")
        if got_bytes != want_bytes:
            raise ValueError(
                "OCR target round-trip mismatch (utf-8 bytes): "
                f"text={text!r} bytes={want_bytes!r}\n"
                f"  decoded={decoded!r} bytes={got_bytes!r}"
            )
        return ids

    return encode_target


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
    instruction_ids = (
        bundle.encode(args.instruction, add_bos=False, add_eos=False)
        if args.instruction
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
                target_ids = encode_target(text)
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
            img.save(out_dir / rel)

            try:
                row = build_ocr_row(
                    target_ids,
                    n_image_tokens,
                    rel,
                    bos_id=BOS_ID,
                    image_start_id=IMAGE_START_ID,
                    image_patch_id=IMAGE_PATCH_ID,
                    image_end_id=IMAGE_END_ID,
                    eos_id=EOS_ID,
                    instruction_ids=instruction_ids,
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

    print(
        f"[build-ocr] wrote {n_written} rows -> {out_dir/'data.jsonl'} "
        f"(n_image_tokens={n_image_tokens}, image_size={args.image_size}) "
        f"max_row_len={max_row_len} skipped_too_long={n_skipped_long}"
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
