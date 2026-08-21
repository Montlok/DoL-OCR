# -*- coding: utf-8 -*-

"""Connected-component evidence for OCR crop and macro-window boundaries."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

CUT_QA_CONTRACT = "connected_component_cut_qa_v2"


@dataclass(frozen=True)
class ComponentBox:
    component_id: int
    box_xyxy: tuple[int, int, int, int]
    pixels: int


@dataclass(frozen=True)
class ViewCutEvidence:
    view_id: str
    box_xyxy: tuple[int, int, int, int]
    edge_ink_fraction: Mapping[str, float]
    cc_crossing_count: int
    cc_uncovered_count: int
    coverage: float
    coverage_proof_sha256: str
    cut_suspect: bool
    review_status: str

    def manifest_qa(self) -> dict[str, object]:
        return {
            "cut_suspect": self.cut_suspect,
            "edge_ink_fraction": dict(self.edge_ink_fraction),
            "cc_crossing_count": self.cc_crossing_count,
            "cc_uncovered_count": self.cc_uncovered_count,
            "coverage": self.coverage,
            "coverage_proof_sha256": self.coverage_proof_sha256,
            "review_status": self.review_status,
            "adjudication_sha256": None,
        }


@dataclass(frozen=True)
class CutQAEvidence:
    contract: str
    canonical_pixel_sha256: str
    plan_sha256: str
    ink_threshold: int
    min_component_pixels: int
    edge_band_px: int
    max_edge_ink_fraction: float
    component_count: int
    uncovered_component_ids: tuple[int, ...]
    coverage_proof_sha256: str
    proof_payload: Mapping[str, object]
    views: tuple[ViewCutEvidence, ...]


def connected_component_boxes(
    grayscale: np.ndarray,
    *,
    ink_threshold: int = 200,
    min_component_pixels: int = 2,
) -> list[ComponentBox]:
    if grayscale.ndim != 2:
        raise ValueError("grayscale image must have shape [H,W]")
    if grayscale.size == 0:
        raise ValueError("grayscale image must not be empty")
    if not 0 <= ink_threshold <= 255:
        raise ValueError("ink_threshold must be in [0,255]")
    if min_component_pixels <= 0:
        raise ValueError("min_component_pixels must be positive")
    try:
        from scipy import ndimage
    except ImportError as exc:  # pragma: no cover - production dependency gate
        raise RuntimeError("OCR cut QA requires scipy") from exc

    ink = np.asarray(grayscale) < ink_threshold
    labels, count = ndimage.label(ink, structure=np.ones((3, 3), dtype=np.uint8))
    objects = ndimage.find_objects(labels)
    components: list[ComponentBox] = []
    for label_id in range(1, count + 1):
        slices = objects[label_id - 1]
        if slices is None:
            continue
        ys, xs = slices
        pixels = int(np.count_nonzero(labels[ys, xs] == label_id))
        if pixels < min_component_pixels:
            continue
        components.append(
            ComponentBox(
                component_id=label_id,
                box_xyxy=(int(xs.start), int(ys.start), int(xs.stop), int(ys.stop)),
                pixels=pixels,
            )
        )
    return components


def analyze_cut_qa(
    grayscale: np.ndarray,
    view_boxes: Mapping[str, Sequence[int]],
    *,
    canonical_pixel_sha256: str,
    plan_sha256: str,
    ink_threshold: int = 200,
    min_component_pixels: int = 2,
    edge_band_px: int = 2,
    max_edge_ink_fraction: float = 0.01,
) -> CutQAEvidence:
    for name, value in (
        ("canonical_pixel_sha256", canonical_pixel_sha256),
        ("plan_sha256", plan_sha256),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    if edge_band_px <= 0:
        raise ValueError("edge_band_px must be positive")
    if not 0.0 <= max_edge_ink_fraction <= 1.0:
        raise ValueError("max_edge_ink_fraction must be in [0,1]")
    height, width = grayscale.shape
    normalized_boxes = {
        str(view_id): _validate_box(box, width=width, height=height)
        for view_id, box in view_boxes.items()
    }
    if not normalized_boxes:
        raise ValueError("cut QA requires at least one view")

    components = connected_component_boxes(
        grayscale,
        ink_threshold=ink_threshold,
        min_component_pixels=min_component_pixels,
    )
    complete_owners = {
        component.component_id: [
            view_id
            for view_id, box in normalized_boxes.items()
            if _contains(box, component.box_xyxy)
        ]
        for component in components
    }
    uncovered = tuple(
        component_id
        for component_id, owners in complete_owners.items()
        if not owners
    )
    proof_payload = {
        "contract": CUT_QA_CONTRACT,
        "canonical_pixel_sha256": canonical_pixel_sha256,
        "plan_sha256": plan_sha256,
        "parameters": {
            "ink_threshold": ink_threshold,
            "min_component_pixels": min_component_pixels,
            "edge_band_px": edge_band_px,
            "max_edge_ink_fraction": max_edge_ink_fraction,
        },
        "image_hw": [height, width],
        "components": [
            {
                "component_id": component.component_id,
                "box_xyxy": list(component.box_xyxy),
                "pixels": component.pixels,
            }
            for component in components
        ],
        "view_boxes": {key: list(value) for key, value in sorted(normalized_boxes.items())},
        "complete_owners": {
            str(key): value for key, value in sorted(complete_owners.items())
        },
        "uncovered": list(uncovered),
    }
    proof_sha = _canonical_sha256(proof_payload)
    component_coverage = (
        1.0
        if not components
        else (len(components) - len(uncovered)) / len(components)
    )

    ink = np.asarray(grayscale) < ink_threshold
    view_evidence: list[ViewCutEvidence] = []
    for view_id, box in sorted(normalized_boxes.items()):
        crossing = [
            component.component_id
            for component in components
            if _intersects(box, component.box_xyxy)
            and not _contains(box, component.box_xyxy)
        ]
        unresolved = [component_id for component_id in crossing if component_id in uncovered]
        edges = _edge_ink_fraction(ink, box, edge_band_px)
        contained_touching_edge = any(
            _contains(box, component.box_xyxy)
            and (
                component.box_xyxy[0] == box[0]
                or component.box_xyxy[1] == box[1]
                or component.box_xyxy[2] == box[2]
                or component.box_xyxy[3] == box[3]
            )
            for component in components
        )
        edge_suspect = any(
            value > max_edge_ink_fraction for value in edges.values()
        ) and contained_touching_edge
        suspect = bool(unresolved or edge_suspect)
        view_evidence.append(
            ViewCutEvidence(
                view_id=view_id,
                box_xyxy=box,
                edge_ink_fraction=edges,
                cc_crossing_count=len(crossing),
                cc_uncovered_count=len(unresolved),
                coverage=component_coverage,
                coverage_proof_sha256=proof_sha,
                cut_suspect=suspect,
                review_status="quarantine" if suspect else "accepted",
            )
        )
    return CutQAEvidence(
        contract=CUT_QA_CONTRACT,
        canonical_pixel_sha256=canonical_pixel_sha256,
        plan_sha256=plan_sha256,
        ink_threshold=ink_threshold,
        min_component_pixels=min_component_pixels,
        edge_band_px=edge_band_px,
        max_edge_ink_fraction=max_edge_ink_fraction,
        component_count=len(components),
        uncovered_component_ids=uncovered,
        coverage_proof_sha256=proof_sha,
        proof_payload=proof_payload,
        views=tuple(view_evidence),
    )


def _validate_box(
    box: Sequence[int],
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    if len(box) != 4 or any(type(value) is not int for value in box):
        raise ValueError("view boxes must contain four integers")
    x0, y0, x1, y1 = (int(value) for value in box)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("view box is outside the image")
    return x0, y0, x1, y1


def _contains(
    outer: tuple[int, int, int, int],
    inner: tuple[int, int, int, int],
) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def _intersects(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> bool:
    return not (
        left[2] <= right[0]
        or right[2] <= left[0]
        or left[3] <= right[1]
        or right[3] <= left[1]
    )


def _edge_ink_fraction(
    ink: np.ndarray,
    box: tuple[int, int, int, int],
    edge_band_px: int,
) -> dict[str, float]:
    x0, y0, x1, y1 = box
    band_y = min(edge_band_px, y1 - y0)
    band_x = min(edge_band_px, x1 - x0)
    crop = ink[y0:y1, x0:x1]
    return {
        "top": float(crop[:band_y, :].mean()),
        "bottom": float(crop[-band_y:, :].mean()),
        "left": float(crop[:, :band_x].mean()),
        "right": float(crop[:, -band_x:].mean()),
    }


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "CUT_QA_CONTRACT",
    "ComponentBox",
    "CutQAEvidence",
    "ViewCutEvidence",
    "analyze_cut_qa",
    "connected_component_boxes",
]
