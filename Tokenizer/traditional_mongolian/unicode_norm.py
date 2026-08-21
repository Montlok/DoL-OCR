# -*- coding: utf-8 -*-
"""Unicode controls and offset helpers for traditional Mongolian text."""

NNBSP = "\u202F"
NIRUGU = "\u180A"
FVS1 = "\u180B"
FVS2 = "\u180C"
FVS3 = "\u180D"
MVS = "\u180E"
FVS4 = "\u180F"

CONTROL_CHARS = {
    "NNBSP": NNBSP,
    "NIRUGU": NIRUGU,
    "FVS1": FVS1,
    "FVS2": FVS2,
    "FVS3": FVS3,
    "MVS": MVS,
    "FVS4": FVS4,
}

CTRL_NO_NNBSP = {FVS1, FVS2, FVS3, FVS4, MVS, NIRUGU}
CTRL_ALL = {FVS1, FVS2, FVS3, FVS4, MVS, NIRUGU, NNBSP}


def strip_controls(text: str) -> str:
    """Strip glyph controls while preserving NNBSP word-internal spacing."""
    return "".join(ch for ch in text if ch not in CTRL_NO_NNBSP)


def strip_all(text: str) -> str:
    """Strip every control ignored by suffix matching."""
    return "".join(ch for ch in text if ch not in CTRL_ALL)


def strip_all_with_map(text: str) -> tuple[str, list[int]]:
    """Return control-stripped text and skeleton-to-original boundary offsets."""
    chars: list[str] = []
    boundary_map: list[int] = [0]

    for i, ch in enumerate(text):
        if ch in CTRL_ALL:
            continue
        chars.append(ch)
        boundary_map.append(i + 1)

    if boundary_map:
        boundary_map[-1] = len(text)

    return "".join(chars), boundary_map


def control_boundaries(text: str) -> set[int]:
    """Return stripped-text offsets preceded by ignored controls in source."""
    boundaries: set[int] = set()
    skeleton_pos = 0
    pending_control = False

    for ch in text:
        if ch in CTRL_ALL:
            if skeleton_pos > 0:
                pending_control = True
            continue

        if pending_control:
            boundaries.add(skeleton_pos)
            pending_control = False
        skeleton_pos += 1

    return boundaries
