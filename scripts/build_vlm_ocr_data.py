# -*- coding: utf-8 -*-

"""Retired legacy VLM-OCR row re-wrapper.

This entry point used to copy token ids out of OMVT SSL rows.  Those ids do
not carry the native tokenizer morphology route, an image-byte binding, or an
immutable data receipt, so the resulting rows are unsafe for a frozen language
model and are rejected by the strict training/evaluation gates.

Use :mod:`scripts.build_ocr_data_from_pairs` for existing image/text pairs or
:mod:`scripts.build_ocr_data` for rendered text.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    del argv
    print(
        "scripts.build_vlm_ocr_data is retired because it cannot prove the "
        "native frozen-LM token route or image bytes. Rebuild with "
        "scripts.build_ocr_data_from_pairs (existing pairs) or "
        "scripts.build_ocr_data (rendered text); both emit the required "
        "native OCR alignment receipt.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
