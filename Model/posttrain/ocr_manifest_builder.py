# -*- coding: utf-8 -*-

"""Deterministic construction of physically separated OCR-RL manifests.

The annotation TSV is authoritative only for text.  Image provenance comes
from an annotation-pack ``manifest.json`` and always selects ``raw_path``;
the 224-pixel preview path shown to annotators is never a training source.

Splits are assigned at document/capture-group granularity and persisted in a
split lock.  Existing assignments are immutable when new groups are added.
The public golden registry contains identities and image hashes only.  Golden
paths, pixels, and transcripts remain in the locked manifest.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from Model.ocr.metrics import symbol_metrics
from Model.ocr.tokenization import canonicalize_native_ocr_text

RL_SPLITS = ("rl_train", "rl_val", "golden")
SPLIT_LOCK_SCHEMA_VERSION = 1
DATASET_MANIFEST_SCHEMA_VERSION = 1
DEFAULT_SPLIT_FRACTIONS = {
    "rl_train": 0.63,
    "rl_val": 0.07,
    "golden": 0.30,
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_order_key(seed: int, domain: str, group_id: str) -> str:
    payload = f"{seed}\0{domain}\0{group_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_split_fractions(
    fractions: Mapping[str, float],
) -> dict[str, float]:
    if set(fractions) != set(RL_SPLITS):
        raise ValueError(
            "split fractions must define exactly " + ", ".join(RL_SPLITS)
        )
    normalized = {name: float(fractions[name]) for name in RL_SPLITS}
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in normalized.values()
    ):
        raise ValueError("all split fractions must be finite and positive")
    if not math.isclose(sum(normalized.values()), 1.0, abs_tol=1e-9):
        raise ValueError("split fractions must sum to 1")
    return normalized


def read_annotation_tsv(
    path: str | Path,
    *,
    reference_column: str = "auto",
) -> dict[str, str]:
    """Read unique annotation ids without trusting the TSV image-path column."""

    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        if "id" not in fieldnames:
            raise ValueError(f"annotation TSV has no id column: {source}")
        if reference_column == "auto":
            candidates = [
                name
                for name in ("reference", "transcription")
                if name in fieldnames
            ]
            if len(candidates) != 1:
                raise ValueError(
                    "annotation TSV must contain exactly one of reference or "
                    "transcription, or pass --reference-column"
                )
            selected_reference = candidates[0]
        else:
            selected_reference = reference_column
            if selected_reference not in fieldnames:
                raise ValueError(
                    f"annotation TSV has no {selected_reference!r} column"
                )

        annotations: dict[str, str] = {}
        for line_no, row in enumerate(reader, start=2):
            sample_id = str(row.get("id") or "").strip()
            if not sample_id:
                raise ValueError(f"{source}:{line_no}: id must not be empty")
            if sample_id in annotations:
                raise ValueError(
                    f"{source}:{line_no}: duplicate id {sample_id!r}"
                )
            reference = row.get(selected_reference)
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError(
                    f"{source}:{line_no}: reference must not be empty"
                )
            if "\x00" in reference:
                raise ValueError(
                    f"{source}:{line_no}: reference contains NUL"
                )
            if "\ufffd" in reference:
                raise ValueError(
                    f"{source}:{line_no}: reference contains U+FFFD; the "
                    "annotation is already lossy"
                )
            annotations[sample_id] = reference
    if not annotations:
        raise ValueError(f"annotation TSV is empty: {source}")
    return annotations


def load_annotation_pack(
    path: str | Path,
    *,
    domain: str,
) -> tuple[Path, dict[str, dict[str, Any]]]:
    """Load pack provenance and resolve only portable, in-root raw crops."""

    if not domain.strip():
        raise ValueError("domain must not be empty")
    source = Path(path)
    root = source.parent.resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid annotation-pack JSON: {source}: {exc}") from exc
    lines = payload.get("lines") if isinstance(payload, dict) else None
    if not isinstance(lines, list) or not lines:
        raise ValueError("annotation-pack manifest must contain non-empty lines")

    provenance: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(lines):
        if not isinstance(row, dict):
            raise ValueError(f"annotation-pack lines[{index}] is not an object")
        sample_id = row.get("id")
        source_id = (
            row.get("source_id")
            or row.get("pdf")
            or row.get("capture_session")
            or row.get("group_id")
        )
        raw_path = row.get("raw_path")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(f"annotation-pack lines[{index}] has invalid id")
        if sample_id in provenance:
            raise ValueError(f"annotation-pack duplicate id {sample_id!r}")
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError(
                f"annotation-pack line {sample_id!r} has no source document"
            )
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(
                f"annotation-pack line {sample_id!r} has no raw_path"
            )
        relative = Path(raw_path)
        if relative.is_absolute():
            raise ValueError(
                f"annotation-pack raw_path must be relative: {raw_path!r}"
            )
        resolved = (root / relative).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(
                f"annotation-pack raw_path escapes pack root: {raw_path!r}"
            )
        explicit_group = row.get("group_id") or row.get("capture_session")
        group_id = (
            str(explicit_group).strip()
            if explicit_group is not None
            else f"document:{source_id}"
        )
        row_domain = str(row.get("domain") or domain).strip()
        if not group_id or not row_domain:
            raise ValueError(
                f"annotation-pack line {sample_id!r} has invalid group/domain"
            )
        provenance[sample_id] = {
            "id": sample_id,
            "group_id": group_id,
            "domain": row_domain,
            "source_id": source_id,
            "source_page": row.get("page"),
            "raw_path": resolved.relative_to(root).as_posix(),
            "resolved_raw_path": resolved,
        }
    return root, provenance


def load_split_lock(
    path: str | Path | None,
    *,
    seed: int,
    fractions: Mapping[str, float],
) -> dict[str, Any]:
    """Load and validate an existing split lock, or return an empty lock."""

    normalized = validate_split_fractions(fractions)
    if path is None or not Path(path).exists():
        return {
            "schema_version": SPLIT_LOCK_SCHEMA_VERSION,
            "seed": int(seed),
            "fractions": normalized,
            "assignments": {},
            "domains": {},
        }
    source = Path(path)
    try:
        lock = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid split-lock JSON: {source}: {exc}") from exc
    if not isinstance(lock, dict):
        raise ValueError("split lock must be a JSON object")
    if lock.get("schema_version") != SPLIT_LOCK_SCHEMA_VERSION:
        raise ValueError("split lock schema_version differs from runtime")
    if lock.get("seed") != int(seed):
        raise ValueError("split lock seed differs from requested seed")
    if lock.get("fractions") != normalized:
        raise ValueError("split lock fractions differ from requested fractions")
    assignments = lock.get("assignments")
    domains = lock.get("domains")
    if not isinstance(assignments, dict) or not isinstance(domains, dict):
        raise ValueError("split lock assignments/domains must be objects")
    if any(split not in RL_SPLITS for split in assignments.values()):
        raise ValueError("split lock contains an unknown split")
    if any(
        not isinstance(group, str)
        or not isinstance(split, str)
        or not isinstance(domains.get(group), str)
        for group, split in assignments.items()
    ):
        raise ValueError("split lock group assignments/domains are malformed")
    return {
        "schema_version": SPLIT_LOCK_SCHEMA_VERSION,
        "seed": int(seed),
        "fractions": normalized,
        "assignments": dict(assignments),
        "domains": dict(domains),
    }


def assign_group_splits(
    group_domains: Mapping[str, str],
    *,
    seed: int,
    fractions: Mapping[str, float] = DEFAULT_SPLIT_FRACTIONS,
    existing_lock: Mapping[str, Any] | None = None,
    group_sizes: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Assign groups deterministically while balancing sample counts."""

    normalized = validate_split_fractions(fractions)
    if not group_domains:
        raise ValueError("no groups available for split assignment")
    sizes = {
        group_id: int((group_sizes or {}).get(group_id, 1))
        for group_id in group_domains
    }
    if any(size <= 0 for size in sizes.values()):
        raise ValueError("group sizes must be positive integers")
    lock = (
        {
            "schema_version": SPLIT_LOCK_SCHEMA_VERSION,
            "seed": int(seed),
            "fractions": normalized,
            "assignments": {},
            "domains": {},
        }
        if existing_lock is None
        else {
            "schema_version": existing_lock.get("schema_version"),
            "seed": existing_lock.get("seed"),
            "fractions": dict(existing_lock.get("fractions", {})),
            "assignments": dict(existing_lock.get("assignments", {})),
            "domains": dict(existing_lock.get("domains", {})),
        }
    )
    if lock["schema_version"] != SPLIT_LOCK_SCHEMA_VERSION:
        raise ValueError("split lock schema_version differs from runtime")
    if lock["seed"] != int(seed) or lock["fractions"] != normalized:
        raise ValueError("split lock seed/fractions differ from requested contract")

    assignments: dict[str, str] = lock["assignments"]
    domains: dict[str, str] = lock["domains"]
    for group_id, domain in group_domains.items():
        if not group_id or not domain:
            raise ValueError("group ids and domains must be non-empty")
        if group_id in assignments:
            if assignments[group_id] not in RL_SPLITS:
                raise ValueError(f"locked group {group_id!r} has unknown split")
            if domains.get(group_id) != domain:
                raise ValueError(
                    f"locked group {group_id!r} changed domain from "
                    f"{domains.get(group_id)!r} to {domain!r}"
                )

    groups_by_domain: dict[str, list[str]] = defaultdict(list)
    for group_id, domain in group_domains.items():
        groups_by_domain[domain].append(group_id)

    for domain, groups in sorted(groups_by_domain.items()):
        stable_order = sorted(
            groups,
            key=lambda group: _stable_order_key(seed, domain, group),
        )
        new_groups = [
            group
            for group in sorted(
                stable_order,
                key=lambda group: (
                    -sizes[group],
                    _stable_order_key(seed, domain, group),
                ),
            )
            if group not in assignments
        ]
        if not new_groups:
            continue
        existing_current = [
            group for group in stable_order if group in assignments
        ]
        counts = Counter()
        for group in existing_current:
            counts[assignments[group]] += sizes[group]
        total = sum(sizes[group] for group in stable_order)
        targets = {name: normalized[name] * total for name in RL_SPLITS}
        for index, group in enumerate(new_groups):
            empty = [
                name
                for name in RL_SPLITS
                if not any(
                    assignments.get(candidate) == name
                    for candidate in stable_order
                )
            ]
            groups_left = len(new_groups) - index
            candidates = (
                empty
                if empty and groups_left <= len(empty)
                else list(RL_SPLITS)
            )
            split = max(
                candidates,
                key=lambda name: (
                    targets[name] - counts[name],
                    normalized[name],
                    name,
                ),
            )
            assignments[group] = split
            domains[group] = domain
            counts[split] += sizes[group]

    current_splits = {assignments[group] for group in group_domains}
    if len(group_domains) >= len(RL_SPLITS) and current_splits != set(RL_SPLITS):
        raise ValueError(
            "group-level split assignment produced an empty split; provide at "
            "least three independently lockable groups per domain or a reviewed "
            "split lock"
        )
    return {
        "schema_version": SPLIT_LOCK_SCHEMA_VERSION,
        "seed": int(seed),
        "fractions": normalized,
        "assignments": dict(sorted(assignments.items())),
        "domains": dict(sorted(domains.items())),
    }


def build_manifest_rows(
    annotations: Mapping[str, str],
    provenance: Mapping[str, Mapping[str, Any]],
    assignments: Mapping[str, str],
    *,
    image_root: str | Path,
    encode_reference: Callable[[str], Sequence[int]],
    decode_reference: Callable[[Sequence[int]], str],
) -> list[dict[str, Any]]:
    """Join labels to raw-image provenance and verify the native text contract."""

    from PIL import Image

    missing = sorted(set(annotations) - set(provenance))
    if missing:
        raise ValueError(
            "annotation ids missing from pack manifest: " + ", ".join(missing[:8])
        )
    root = Path(image_root).resolve()
    seen_paths: dict[Path, str] = {}
    seen_hashes: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for sample_id in sorted(annotations):
        item = provenance[sample_id]
        group_id = str(item["group_id"])
        split = assignments.get(group_id)
        if split not in RL_SPLITS:
            raise ValueError(f"group {group_id!r} has no valid split assignment")
        resolved = Path(item["resolved_raw_path"]).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"raw image escapes image root: {resolved}")
        if not resolved.is_file():
            raise FileNotFoundError(f"raw image is missing: {resolved}")
        previous_path = seen_paths.get(resolved)
        if previous_path is not None:
            raise ValueError(
                f"duplicate raw image path for ids {previous_path!r} and "
                f"{sample_id!r}: {resolved}"
            )
        try:
            with Image.open(resolved) as image:
                image.verify()
        except Exception as exc:
            raise ValueError(f"raw image is not decodable: {resolved}: {exc}") from exc
        digest = file_sha256(resolved)
        previous_hash = seen_hashes.get(digest)
        if previous_hash is not None:
            raise ValueError(
                f"duplicate raw image SHA-256 for ids {previous_hash!r} and "
                f"{sample_id!r}: {digest}"
            )
        seen_paths[resolved] = sample_id
        seen_hashes[digest] = sample_id

        raw_reference = annotations[sample_id]
        canonical_reference = canonicalize_native_ocr_text(raw_reference)
        ids = [int(token) for token in encode_reference(raw_reference)]
        if not ids:
            raise ValueError(f"native reference encoding is empty for {sample_id!r}")
        decoded = decode_reference(ids)
        if decoded != canonical_reference:
            raise ValueError(
                f"native reference round-trip differs for {sample_id!r}: "
                f"{decoded!r} != {canonical_reference!r}"
            )
        rows.append(
            {
                "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
                "id": sample_id,
                "group_id": group_id,
                "split": split,
                "image": resolved.relative_to(root).as_posix(),
                "sha256": digest,
                "reference": canonical_reference,
                "domain": str(item["domain"]),
                "source_id": str(item["source_id"]),
                "source_page": item.get("source_page"),
            }
        )
    return rows


def partition_manifest_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Return train, validation, locked golden, and public golden identity."""

    by_split = {name: [] for name in RL_SPLITS}
    group_to_split: dict[str, str] = {}
    for source in rows:
        row = dict(source)
        split = row.get("split")
        if split not in by_split:
            raise ValueError(f"unknown manifest split {split!r}")
        group_id = str(row.get("group_id", ""))
        previous = group_to_split.setdefault(group_id, str(split))
        if previous != split:
            raise ValueError(
                f"group {group_id!r} crosses splits {previous!r}/{split!r}"
            )
        by_split[str(split)].append(row)
    if any(not by_split[name] for name in RL_SPLITS):
        raise ValueError("train, validation, and golden must all be non-empty")
    for split_rows in by_split.values():
        split_rows.sort(key=lambda row: str(row["id"]))
    identity = [
        {
            "schema_version": 1,
            "id": row["id"],
            "group_id": row["group_id"],
            "split": "golden",
            "image_sha256": row["sha256"],
        }
        for row in by_split["golden"]
    ]
    return (
        by_split["rl_train"],
        by_split["rl_val"],
        by_split["golden"],
        identity,
    )


def materialize_manifest_images(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_root: str | Path,
    public_root: str | Path,
    locked_root: str | Path,
) -> list[dict[str, Any]]:
    """Copy train/val and golden pixels into disjoint content-addressed roots."""

    source_base = Path(source_root).resolve()
    public_base = Path(public_root)
    locked_base = Path(locked_root)
    materialized: list[dict[str, Any]] = []
    for source_row in rows:
        row = dict(source_row)
        split = str(row.get("split", ""))
        destination_base = locked_base if split == "golden" else public_base
        directory_mode = 0o700 if split == "golden" else 0o755
        file_mode = 0o600 if split == "golden" else 0o644
        digest = str(row["sha256"]).lower()
        source_path = (source_base / str(row["image"])).resolve()
        if not source_path.is_relative_to(source_base):
            raise ValueError(f"source image escapes pack root: {source_path}")
        suffix = source_path.suffix.lower() or ".img"
        relative = Path("images") / digest[:2] / f"{digest}{suffix}"
        destination = destination_base / relative
        images_root = destination_base / "images"
        images_root.mkdir(parents=True, exist_ok=True)
        os.chmod(images_root, directory_mode)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(destination.parent, directory_mode)
        if destination.exists():
            raise FileExistsError(
                f"duplicate materialized image target: {destination}"
            )
        shutil.copyfile(source_path, destination)
        os.chmod(destination, file_mode)
        if file_sha256(destination) != digest:
            raise RuntimeError(
                f"materialized image hash changed during copy: {destination}"
            )
        row["image"] = relative.as_posix()
        materialized.append(row)
    return materialized


def reference_symbol_support(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    refs = [str(row["reference"]) for row in rows]
    metrics = symbol_metrics(refs, refs)
    support: dict[str, dict[str, int]] = {}
    for name in ("digit", "punctuation", "fvs", "mvs", "nnbsp"):
        support[name] = {
            "n_ref": int(metrics[name]["n_ref"]),
            "line_support": int(metrics[name]["line_support"]),
        }
    return support


__all__ = [
    "DATASET_MANIFEST_SCHEMA_VERSION",
    "DEFAULT_SPLIT_FRACTIONS",
    "RL_SPLITS",
    "SPLIT_LOCK_SCHEMA_VERSION",
    "assign_group_splits",
    "build_manifest_rows",
    "file_sha256",
    "load_annotation_pack",
    "load_split_lock",
    "materialize_manifest_images",
    "partition_manifest_rows",
    "read_annotation_tsv",
    "reference_symbol_support",
    "validate_split_fractions",
]
