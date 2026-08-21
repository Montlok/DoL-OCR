# -*- coding: utf-8 -*-
"""Traditional Mongolian code points that should never become unknown.

MorphBPE is trained from corpus evidence, but a small corpus may not contain
every legal Mongolian letter. Seeding the base alphabet keeps valid Mongolian
text encodable at character level even before enough data exists for good BPE
merges.
"""

from __future__ import annotations

import unicodedata

MONGOLIAN_RANGES = [
    (0x1800, 0x18AF),
    (0x11660, 0x1167F),
]

MONGOLIAN_LETTERS = tuple(
    chr(cp)
    for start, end in MONGOLIAN_RANGES
    for cp in range(start, end + 1)
    if "MONGOLIAN" in unicodedata.name(chr(cp), "")
    and unicodedata.category(chr(cp)).startswith("L")
)


def is_mongolian_codepoint(ch: str) -> bool:
    cp = ord(ch)
    return any(start <= cp <= end for start, end in MONGOLIAN_RANGES)
