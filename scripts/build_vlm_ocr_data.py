# -*- coding: utf-8 -*-

"""Build VLM OCR-alignment data from existing OMVT SSL rows.

The synthetic render set already pairs each page image with its exact
transcription tokens (``ssl.jsonl`` rows carry ``images`` + clean
``ocr_labels``). Phase-3 VLM alignment — wiring the trained OMVT tower into
the RDT embedding space through a generative ``image -> text`` objective —
needs those same pairs re-expressed in the pre-tokenized row contract that
``scripts.train_vlm_align`` consumes (:func:`Model.ocr.data.build_ocr_row`):
BOS, ``<image_start>`` + N ``<image_patch>`` + ``<image_end>``, an optional
instruction (loss-masked), then the supervised target + EOS.

So this is a pure re-wrapping: no re-render, no re-tokenize. It reads SSL
rows that carry ``ocr_labels`` (synthetic pages — scan pages have none and
are skipped), and emits alignment rows. ``--n-image-tokens`` MUST equal the
tower ``compress_to`` used at train time (256 for the 448px prod config).

Usage::

    PYTHONPATH=. python3 -m scripts.build_vlm_ocr_data \
        --ssl ../corpus/outputs/omvt_synth_v1/ssl.jsonl \
        --out ../corpus/outputs/vlm_ocr_v1/align.jsonl \
        --n-image-tokens 256 \
        --tokenizer-bundle ../corpus/outputs/tok_build_v2/tokenizer/bundle \
        --instruction ""
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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


def _first_labels(row: dict) -> list[int] | None:
    """Return the page's target token ids from a schema-correct SSL row.

    ``ocr_labels`` is stored as one sequence per image (``[[id, ...], ...]``);
    OCR pages have exactly one image, so the first sequence is the target.
    """
    labels = row.get("ocr_labels")
    if not labels:
        return None
    head = labels[0] if isinstance(labels, (list, tuple)) else None
    if isinstance(head, (list, tuple)):
        return [int(x) for x in head] or None
    if isinstance(head, int):
        return [int(x) for x in labels]
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--ssl", required=True, action="append",
                    help="input SSL jsonl with ocr_labels (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-image-tokens", type=int, required=True,
                    help="must equal the OMVT tower compress_to (e.g. 256)")
    ap.add_argument("--tokenizer-bundle", default="",
                    help="needed only when --instruction is non-empty")
    ap.add_argument("--instruction", default="",
                    help="optional prompt tokens after <image_end>, "
                    "masked from the loss")
    ap.add_argument("--max-target-tokens", type=int, default=512)
    args = ap.parse_args(argv)

    instruction_ids: list[int] = []
    if args.instruction:
        if not args.tokenizer_bundle:
            ap.error("--instruction requires --tokenizer-bundle")
        from Tokenizer.unified.bundle import TokenizerBundle
        bundle = TokenizerBundle.from_dir(args.tokenizer_bundle)
        instruction_ids = bundle.encode(
            args.instruction, add_bos=False, add_eos=False
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_in = n_out = n_skip = 0
    with out_path.open("w", encoding="utf-8") as out_fh:
        for ssl_path in args.ssl:
            with open(ssl_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    n_in += 1
                    row = json.loads(line)
                    images = row.get("images")
                    target = _first_labels(row)
                    if not images or not target:
                        n_skip += 1  # scan page or empty: no OCR supervision
                        continue
                    target = target[: args.max_target_tokens]
                    align = build_ocr_row(
                        target,
                        args.n_image_tokens,
                        images[0],
                        bos_id=BOS_ID,
                        image_start_id=IMAGE_START_ID,
                        image_patch_id=IMAGE_PATCH_ID,
                        image_end_id=IMAGE_END_ID,
                        eos_id=EOS_ID,
                        instruction_ids=instruction_ids,
                    )
                    out_fh.write(json.dumps(align, ensure_ascii=False) + "\n")
                    n_out += 1

    print(f"[vlm-data] {n_in} rows in -> {n_out} alignment rows "
          f"({n_skip} skipped, no labels) -> {out_path}")
    return 0 if n_out else 1


if __name__ == "__main__":
    raise SystemExit(main())
