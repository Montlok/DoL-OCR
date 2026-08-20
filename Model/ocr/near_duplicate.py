# -*- coding: utf-8 -*-

"""Versioned perceptual-hash evidence for OCR split leakage review."""

from __future__ import annotations

import io
import hashlib
import json
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from PIL import Image, ImageOps


PHASH_CONTRACT = "pil_lanczos32_scipy_dct8_median_v1"


def perceptual_hash(image: bytes | Image.Image) -> str:
    try:
        from scipy.fft import dctn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OCR perceptual hash requires scipy") from exc
    if isinstance(image, Image.Image):
        source = ImageOps.exif_transpose(image).convert("L")
    else:
        with Image.open(io.BytesIO(bytes(image))) as raw:
            raw.load()
            source = ImageOps.exif_transpose(raw).convert("L")
    resample = (
        Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    )
    values = np.asarray(source.resize((32, 32), resample), dtype=np.float32)
    low = dctn(values, type=2, norm="ortho")[:8, :8]
    flat = low.reshape(-1)
    median = float(np.median(flat[1:]))
    bits = flat >= median
    bits[0] = False
    result = 0
    for bit in bits:
        result = (result << 1) | int(bool(bit))
    return f"{result:016x}"


def phash_hamming_distance(left: str, right: str) -> int:
    for name, value in (("left", left), ("right", right)):
        if (
            not isinstance(value, str)
            or len(value) != 16
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise ValueError(f"{name} pHash must be 16 lowercase hex characters")
    return (int(left, 16) ^ int(right, 16)).bit_count()


@dataclass(frozen=True)
class NearDuplicateCluster:
    cluster_id: str
    member_ids: tuple[str, ...]


def near_duplicate_cluster_id(member_ids: tuple[str, ...]) -> str:
    if not member_ids or tuple(sorted(set(member_ids))) != member_ids:
        raise ValueError("near-duplicate member_ids must be sorted and unique")
    encoded = json.dumps(
        list(member_ids),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"phash:{PHASH_CONTRACT}:{hashlib.sha256(encoded).hexdigest()}"


def cluster_perceptual_hashes(
    hashes: Mapping[str, str],
    *,
    max_hamming_distance: int,
    max_records: int = 20_000,
) -> tuple[NearDuplicateCluster, ...]:
    if not 0 <= max_hamming_distance <= 64:
        raise ValueError("max_hamming_distance must be in [0,64]")
    if max_records <= 0 or len(hashes) > max_records:
        raise ValueError("pHash review set exceeds max_records")
    ids = sorted(hashes)
    for sample_id in ids:
        if not sample_id or sample_id != sample_id.strip():
            raise ValueError("pHash sample ids must be non-empty stripped strings")
        phash_hamming_distance(hashes[sample_id], hashes[sample_id])

    parent = list(range(len(ids)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for left in range(len(ids)):
        for right in range(left + 1, len(ids)):
            if (
                phash_hamming_distance(hashes[ids[left]], hashes[ids[right]])
                <= max_hamming_distance
            ):
                union(left, right)

    members: dict[int, list[str]] = {}
    for index, sample_id in enumerate(ids):
        members.setdefault(find(index), []).append(sample_id)
    groups = sorted(
        (tuple(values) for values in members.values()),
        key=lambda row: row,
    )
    return tuple(
        NearDuplicateCluster(
            cluster_id=near_duplicate_cluster_id(group),
            member_ids=group,
        )
        for group in groups
    )


__all__ = [
    "NearDuplicateCluster",
    "PHASH_CONTRACT",
    "cluster_perceptual_hashes",
    "perceptual_hash",
    "near_duplicate_cluster_id",
    "phash_hamming_distance",
]
