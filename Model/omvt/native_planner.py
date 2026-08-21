# -*- coding: utf-8 -*-

"""Deterministic macro-window geometry planner for native OMVT-v2.

The planner owns only image geometry.  It does not inspect ink, connected
components, characters, or labels; dataset/manifest QA remains responsible
for deciding whether a planned asset is safe to train on.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import operator
from typing import Any, Mapping, Sequence

from Model.omvt.patcher import PATCH_KINDS


BBoxYXxy = tuple[int, int, int, int]
PatchShapeItems = tuple[tuple[str, tuple[int, int]], ...]
PLAN_CONTRACT = "native_macro_window_plan_v1"
COVERAGE_PROOF_METHOD = "exact_integer_sweep_line_v1"


def _as_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _normalize_patch_shapes(
    patch_shapes: Mapping[str, Sequence[int]],
) -> PatchShapeItems:
    if not isinstance(patch_shapes, Mapping):
        raise TypeError("patch_shapes must be a mapping keyed by patch kind")
    if set(patch_shapes) != set(PATCH_KINDS):
        raise ValueError(
            "patch_shapes must contain exactly vertical/horizontal/square/layout"
        )
    normalized: list[tuple[str, tuple[int, int]]] = []
    for kind in PATCH_KINDS:
        shape = patch_shapes[kind]
        if len(shape) != 2:
            raise ValueError(f"{kind} patch shape must have two entries")
        patch_h = _as_int(shape[0], name=f"{kind}.patch_h")
        patch_w = _as_int(shape[1], name=f"{kind}.patch_w")
        if patch_h <= 0 or patch_w <= 0:
            raise ValueError("patch dimensions must be positive")
        normalized.append((kind, (patch_h, patch_w)))
    return tuple(normalized)


def raw_patch_token_count(
    height: int,
    width: int,
    patch_shapes: Mapping[str, Sequence[int]] | PatchShapeItems,
) -> int:
    """Count all four padded patch streams for a rectangular pixel region."""

    height = _as_int(height, name="height")
    width = _as_int(width, name="width")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if isinstance(patch_shapes, Mapping):
        shapes = _normalize_patch_shapes(patch_shapes)
    else:
        shapes = patch_shapes
    return sum(
        ((height + patch_h - 1) // patch_h)
        * ((width + patch_w - 1) // patch_w)
        for _, (patch_h, patch_w) in shapes
    )


@dataclass(frozen=True)
class NativeMacroWindow:
    index: int
    asset_bbox_yxxy: BBoxYXxy
    ownership_bbox_yxxy: BBoxYXxy
    halo_tlbr: tuple[int, int, int, int]
    raw_patch_tokens: int

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "asset_bbox_yxxy": list(self.asset_bbox_yxxy),
            "ownership_bbox_yxxy": list(self.ownership_bbox_yxxy),
            "halo_tlbr": list(self.halo_tlbr),
            "raw_patch_tokens": self.raw_patch_tokens,
        }


@dataclass(frozen=True)
class NativeCoverageProof:
    method: str
    asset_pixels: int
    ownership_union_pixels: int
    ownership_gap_pixels: int
    ownership_overlap_pixels: int
    asset_window_union_pixels: int
    asset_window_gap_pixels: int
    ownership_regions_checked: int
    asset_window_regions_checked: int
    max_window_raw_patch_tokens: int
    all_window_budgets_valid: bool
    verified: bool

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "asset_pixels": self.asset_pixels,
            "ownership_union_pixels": self.ownership_union_pixels,
            "ownership_gap_pixels": self.ownership_gap_pixels,
            "ownership_overlap_pixels": self.ownership_overlap_pixels,
            "asset_window_union_pixels": self.asset_window_union_pixels,
            "asset_window_gap_pixels": self.asset_window_gap_pixels,
            "ownership_regions_checked": self.ownership_regions_checked,
            "asset_window_regions_checked": self.asset_window_regions_checked,
            "max_window_raw_patch_tokens": self.max_window_raw_patch_tokens,
            "all_window_budgets_valid": self.all_window_budgets_valid,
            "verified": self.verified,
        }


@dataclass(frozen=True)
class NativeMacroWindowPlan:
    asset_hw: tuple[int, int]
    patch_shapes: PatchShapeItems
    max_raw_patch_tokens: int
    max_windows: int
    halo_px: int
    identity: bool
    windows: tuple[NativeMacroWindow, ...]
    coverage_proof: NativeCoverageProof
    contract: str = PLAN_CONTRACT

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "asset_hw": list(self.asset_hw),
            "patch_shapes": {
                kind: list(shape) for kind, shape in self.patch_shapes
            },
            "max_raw_patch_tokens": self.max_raw_patch_tokens,
            "max_windows": self.max_windows,
            "halo_px": self.halo_px,
            "identity": self.identity,
            "windows": [window.canonical_payload() for window in self.windows],
            "coverage_proof": self.coverage_proof.canonical_payload(),
        }

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def canonical_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()


def _bbox_hw(bbox: BBoxYXxy) -> tuple[int, int]:
    y0, x0, y1, x1 = bbox
    return y1 - y0, x1 - x0


def _expand_with_halo(
    ownership_bbox: BBoxYXxy,
    *,
    asset_hw: tuple[int, int],
    halo_px: int,
) -> tuple[BBoxYXxy, tuple[int, int, int, int]]:
    asset_h, asset_w = asset_hw
    y0, x0, y1, x1 = ownership_bbox
    asset_bbox = (
        max(0, y0 - halo_px),
        max(0, x0 - halo_px),
        min(asset_h, y1 + halo_px),
        min(asset_w, x1 + halo_px),
    )
    ay0, ax0, ay1, ax1 = asset_bbox
    return asset_bbox, (y0 - ay0, x0 - ax0, ay1 - y1, ax1 - x1)


def _window_token_count(
    ownership_bbox: BBoxYXxy,
    *,
    asset_hw: tuple[int, int],
    patch_shapes: PatchShapeItems,
    halo_px: int,
) -> int:
    asset_bbox, _ = _expand_with_halo(
        ownership_bbox,
        asset_hw=asset_hw,
        halo_px=halo_px,
    )
    window_h, window_w = _bbox_hw(asset_bbox)
    return raw_patch_token_count(window_h, window_w, patch_shapes)


def _split_candidates(
    ownership_bbox: BBoxYXxy,
    *,
    asset_hw: tuple[int, int],
    patch_shapes: PatchShapeItems,
    halo_px: int,
) -> list[tuple[tuple[int, int, int, int], BBoxYXxy, BBoxYXxy]]:
    y0, x0, y1, x1 = ownership_bbox
    height, width = _bbox_hw(ownership_bbox)
    candidates: list[tuple[tuple[int, int, int, int], BBoxYXxy, BBoxYXxy]] = []
    if height > 1:
        split = y0 + height // 2
        first = (y0, x0, split, x1)
        second = (split, x0, y1, x1)
        counts = (
            _window_token_count(
                first,
                asset_hw=asset_hw,
                patch_shapes=patch_shapes,
                halo_px=halo_px,
            ),
            _window_token_count(
                second,
                asset_hw=asset_hw,
                patch_shapes=patch_shapes,
                halo_px=halo_px,
            ),
        )
        candidates.append(((max(counts), sum(counts), -height, 0), first, second))
    if width > 1:
        split = x0 + width // 2
        first = (y0, x0, y1, split)
        second = (y0, split, y1, x1)
        counts = (
            _window_token_count(
                first,
                asset_hw=asset_hw,
                patch_shapes=patch_shapes,
                halo_px=halo_px,
            ),
            _window_token_count(
                second,
                asset_hw=asset_hw,
                patch_shapes=patch_shapes,
                halo_px=halo_px,
            ),
        )
        candidates.append(((max(counts), sum(counts), -width, 1), first, second))
    return candidates


class _CoverageLengthTree:
    """Range-add tree tracking x-length covered at least once/twice."""

    def __init__(self, coordinates: Sequence[int]) -> None:
        self.coordinates = tuple(coordinates)
        self.interval_count = len(self.coordinates) - 1
        size = max(1, self.interval_count * 4)
        self.cover = [0] * size
        self.covered_once = [0] * size
        self.covered_twice = [0] * size

    def _pull(self, node: int, left: int, right: int) -> None:
        total = self.coordinates[right] - self.coordinates[left]
        if self.cover[node] >= 2:
            self.covered_once[node] = total
            self.covered_twice[node] = total
        elif self.cover[node] == 1:
            self.covered_once[node] = total
            if right - left == 1:
                self.covered_twice[node] = 0
            else:
                self.covered_twice[node] = (
                    self.covered_once[node * 2]
                    + self.covered_once[node * 2 + 1]
                )
        elif right - left == 1:
            self.covered_once[node] = 0
            self.covered_twice[node] = 0
        else:
            self.covered_once[node] = (
                self.covered_once[node * 2]
                + self.covered_once[node * 2 + 1]
            )
            self.covered_twice[node] = (
                self.covered_twice[node * 2]
                + self.covered_twice[node * 2 + 1]
            )

    def _add(
        self,
        query_left: int,
        query_right: int,
        delta: int,
        *,
        node: int,
        left: int,
        right: int,
    ) -> None:
        if query_left <= left and right <= query_right:
            self.cover[node] += delta
            if self.cover[node] < 0:
                raise RuntimeError("coverage sweep produced a negative count")
            self._pull(node, left, right)
            return
        midpoint = (left + right) // 2
        if query_left < midpoint:
            self._add(
                query_left,
                query_right,
                delta,
                node=node * 2,
                left=left,
                right=midpoint,
            )
        if query_right > midpoint:
            self._add(
                query_left,
                query_right,
                delta,
                node=node * 2 + 1,
                left=midpoint,
                right=right,
            )
        self._pull(node, left, right)

    def add(self, left: int, right: int, delta: int) -> None:
        if left >= right:
            raise ValueError("coverage interval must be non-empty")
        self._add(
            left,
            right,
            delta,
            node=1,
            left=0,
            right=self.interval_count,
        )

    @property
    def union_length(self) -> int:
        return self.covered_once[1]

    @property
    def overlap_length(self) -> int:
        return self.covered_twice[1]


def _rectangle_union_and_overlap_area(
    rectangles: Sequence[BBoxYXxy],
) -> tuple[int, int]:
    x_coordinates = sorted(
        {coordinate for _, x0, _, x1 in rectangles for coordinate in (x0, x1)}
    )
    x_indices = {coordinate: index for index, coordinate in enumerate(x_coordinates)}
    events: dict[int, list[tuple[int, int, int]]] = {}
    for y0, x0, y1, x1 in rectangles:
        events.setdefault(y0, []).append((x_indices[x0], x_indices[x1], 1))
        events.setdefault(y1, []).append((x_indices[x0], x_indices[x1], -1))
    tree = _CoverageLengthTree(x_coordinates)
    union_area = 0
    overlap_area = 0
    previous_y = min(events)
    for y in sorted(events):
        delta_y = y - previous_y
        union_area += tree.union_length * delta_y
        overlap_area += tree.overlap_length * delta_y
        for left, right, delta in events[y]:
            tree.add(left, right, delta)
        previous_y = y
    if tree.union_length != 0 or tree.overlap_length != 0:
        raise RuntimeError("coverage sweep did not close all rectangles")
    return union_area, overlap_area


def _validate_window_geometry(
    window: NativeMacroWindow,
    *,
    asset_hw: tuple[int, int],
    patch_shapes: PatchShapeItems,
    max_raw_patch_tokens: int,
) -> None:
    asset_h, asset_w = asset_hw
    ay0, ax0, ay1, ax1 = window.asset_bbox_yxxy
    oy0, ox0, oy1, ox1 = window.ownership_bbox_yxxy
    if not (0 <= ay0 < ay1 <= asset_h and 0 <= ax0 < ax1 <= asset_w):
        raise RuntimeError("macro-window asset bbox is outside the source asset")
    if not (ay0 <= oy0 < oy1 <= ay1 and ax0 <= ox0 < ox1 <= ax1):
        raise RuntimeError("macro-window asset bbox must contain its ownership region")
    expected_halo = (oy0 - ay0, ox0 - ax0, ay1 - oy1, ax1 - ox1)
    if window.halo_tlbr != expected_halo:
        raise RuntimeError("macro-window halo does not match its global bboxes")
    window_h, window_w = _bbox_hw(window.asset_bbox_yxxy)
    expected_tokens = raw_patch_token_count(window_h, window_w, patch_shapes)
    if window.raw_patch_tokens != expected_tokens:
        raise RuntimeError("macro-window patch budget was recorded incorrectly")
    if window.raw_patch_tokens > max_raw_patch_tokens:
        raise RuntimeError("macro-window exceeds max_raw_patch_tokens")


def prove_native_macro_window_coverage(
    *,
    asset_hw: tuple[int, int],
    windows: Sequence[NativeMacroWindow],
    patch_shapes: Mapping[str, Sequence[int]] | PatchShapeItems,
    max_raw_patch_tokens: int,
) -> NativeCoverageProof:
    """Prove exact pixel ownership and processing coverage analytically."""

    asset_h, asset_w = asset_hw
    if asset_h <= 0 or asset_w <= 0:
        raise ValueError("asset dimensions must be positive")
    if not windows:
        raise ValueError("macro-window plan must contain at least one window")
    shapes = (
        _normalize_patch_shapes(patch_shapes)
        if isinstance(patch_shapes, Mapping)
        else patch_shapes
    )
    for expected_index, window in enumerate(windows):
        if window.index != expected_index:
            raise RuntimeError("macro-window indices must be contiguous and ordered")
        _validate_window_geometry(
            window,
            asset_hw=asset_hw,
            patch_shapes=shapes,
            max_raw_patch_tokens=max_raw_patch_tokens,
        )

    ownership_rectangles = [window.ownership_bbox_yxxy for window in windows]
    asset_rectangles = [window.asset_bbox_yxxy for window in windows]
    ownership_union, ownership_overlap = _rectangle_union_and_overlap_area(
        ownership_rectangles
    )
    asset_window_union, _ = _rectangle_union_and_overlap_area(asset_rectangles)
    asset_pixels = asset_h * asset_w
    ownership_gap = asset_pixels - ownership_union
    asset_window_gap = asset_pixels - asset_window_union
    all_budgets_valid = all(
        window.raw_patch_tokens <= max_raw_patch_tokens for window in windows
    )
    verified = (
        ownership_gap == 0
        and ownership_overlap == 0
        and asset_window_gap == 0
        and all_budgets_valid
    )
    return NativeCoverageProof(
        method=COVERAGE_PROOF_METHOD,
        asset_pixels=asset_pixels,
        ownership_union_pixels=ownership_union,
        ownership_gap_pixels=ownership_gap,
        ownership_overlap_pixels=ownership_overlap,
        asset_window_union_pixels=asset_window_union,
        asset_window_gap_pixels=asset_window_gap,
        ownership_regions_checked=len(ownership_rectangles),
        asset_window_regions_checked=len(asset_rectangles),
        max_window_raw_patch_tokens=max(
            window.raw_patch_tokens for window in windows
        ),
        all_window_budgets_valid=all_budgets_valid,
        verified=verified,
    )


def plan_native_macro_windows(
    *,
    height: int,
    width: int,
    patch_shapes: Mapping[str, Sequence[int]],
    max_raw_patch_tokens: int,
    max_windows: int,
    halo_px: int,
) -> NativeMacroWindowPlan:
    """Plan deterministic overlapping crops with disjoint ownership regions."""

    height = _as_int(height, name="height")
    width = _as_int(width, name="width")
    max_raw_patch_tokens = _as_int(
        max_raw_patch_tokens,
        name="max_raw_patch_tokens",
    )
    max_windows = _as_int(max_windows, name="max_windows")
    halo_px = _as_int(halo_px, name="halo_px")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if max_raw_patch_tokens <= 0:
        raise ValueError("max_raw_patch_tokens must be positive")
    if max_windows <= 0:
        raise ValueError("max_windows must be positive")
    if halo_px < 0:
        raise ValueError("halo_px must be non-negative")
    shapes = _normalize_patch_shapes(patch_shapes)
    asset_hw = (height, width)
    full_bbox = (0, 0, height, width)
    full_tokens = raw_patch_token_count(height, width, shapes)

    if full_tokens <= max_raw_patch_tokens:
        leaves = [full_bbox]
        identity = True
    else:
        minimum_halo_height = min(height, 2 * halo_px + 1)
        minimum_halo_width = min(width, 2 * halo_px + 1)
        minimum_tokens = raw_patch_token_count(
            minimum_halo_height,
            minimum_halo_width,
            shapes,
        )
        if minimum_tokens > max_raw_patch_tokens:
            raise ValueError(
                "minimum one-pixel ownership window with requested halo exceeds "
                "max_raw_patch_tokens"
            )

        stack = [full_bbox]
        leaves = []
        while stack:
            ownership_bbox = stack.pop()
            tokens = _window_token_count(
                ownership_bbox,
                asset_hw=asset_hw,
                patch_shapes=shapes,
                halo_px=halo_px,
            )
            if tokens <= max_raw_patch_tokens:
                leaves.append(ownership_bbox)
                if len(leaves) + len(stack) > max_windows:
                    raise ValueError("macro-window plan exceeds max_windows")
                continue
            candidates = _split_candidates(
                ownership_bbox,
                asset_hw=asset_hw,
                patch_shapes=shapes,
                halo_px=halo_px,
            )
            if not candidates:
                raise ValueError(
                    "minimum one-pixel ownership window exceeds "
                    "max_raw_patch_tokens"
                )
            _, first, second = min(candidates, key=lambda candidate: candidate[0])
            stack.append(second)
            stack.append(first)
            if len(leaves) + len(stack) > max_windows:
                raise ValueError("macro-window plan exceeds max_windows")
        leaves.sort()
        identity = False

    windows: list[NativeMacroWindow] = []
    for index, ownership_bbox in enumerate(leaves):
        asset_bbox, actual_halo = _expand_with_halo(
            ownership_bbox,
            asset_hw=asset_hw,
            halo_px=halo_px,
        )
        window_h, window_w = _bbox_hw(asset_bbox)
        windows.append(
            NativeMacroWindow(
                index=index,
                asset_bbox_yxxy=asset_bbox,
                ownership_bbox_yxxy=ownership_bbox,
                halo_tlbr=actual_halo,
                raw_patch_tokens=raw_patch_token_count(
                    window_h,
                    window_w,
                    shapes,
                ),
            )
        )
    proof = prove_native_macro_window_coverage(
        asset_hw=asset_hw,
        windows=windows,
        patch_shapes=shapes,
        max_raw_patch_tokens=max_raw_patch_tokens,
    )
    if not proof.verified:
        raise RuntimeError("native macro-window coverage proof failed")
    return NativeMacroWindowPlan(
        asset_hw=asset_hw,
        patch_shapes=shapes,
        max_raw_patch_tokens=max_raw_patch_tokens,
        max_windows=max_windows,
        halo_px=halo_px,
        identity=identity,
        windows=tuple(windows),
        coverage_proof=proof,
    )


__all__ = [
    "COVERAGE_PROOF_METHOD",
    "NativeCoverageProof",
    "NativeMacroWindow",
    "NativeMacroWindowPlan",
    "PLAN_CONTRACT",
    "plan_native_macro_windows",
    "prove_native_macro_window_coverage",
    "raw_patch_token_count",
]
