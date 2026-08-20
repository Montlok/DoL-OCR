# -*- coding: utf-8 -*-

"""Strict, non-transforming dataset loader for DoL OCR anyres v2 data."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from Model.ocr.cut_qa import analyze_cut_qa
from Model.ocr.near_duplicate import perceptual_hash
from Model.posttrain.ocr_anyres_manifest import (
    ANYRES_PUBLIC_SPLITS,
    canonical_json_sha256,
    quota_bucket,
    quota_counts,
    validate_anyres_assets_views_samples,
)


ImageDelivery = Literal["bytes", "path"]
_READY_KIND = "dol_ocr_anyres_ready_v1"
_BUILD_RECEIPT_KIND = "dol_ocr_anyres_finalize_receipt_v1"
_CUT_ADJUDICATION_KIND = "dol_ocr_anyres_cut_adjudication_v1"


def _strict_json_object(raw: str, *, source: Path, line_no: int) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(
                    f"{source}:{line_no}: duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=object_pairs)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source}:{line_no}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{source}:{line_no}: row must be a JSON object")
    return value


def load_anyres_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load one non-symlink JSONL manifest with duplicate-key rejection."""

    source = Path(path)
    if source.is_symlink():
        raise ValueError(f"anyres manifest must not be a symlink: {source}")
    rows: list[dict[str, Any]] = []
    try:
        with source.open("r", encoding="utf-8", errors="strict") as handle:
            for line_no, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                rows.append(
                    _strict_json_object(raw, source=source, line_no=line_no)
                )
    except FileNotFoundError as exc:
        raise ValueError(f"anyres manifest does not exist: {source}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"anyres manifest is not valid UTF-8: {source}") from exc
    return rows


def _rgba_image_sha256(image: Image.Image) -> tuple[str, tuple[int, int]]:
    rgba = image.convert("RGBA")
    width, height = rgba.size
    header = f"dol-ocr-rgba-v1\0{width}\0{height}\0".encode("ascii")
    return hashlib.sha256(header + rgba.tobytes()).hexdigest(), (width, height)


def rgba_pixel_sha256(
    image_bytes: bytes,
    *,
    max_pixels: int | None = None,
) -> tuple[str, tuple[int, int]]:
    """Hash decoded RGBA pixels with dimensions bound into the digest."""

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
            if max_pixels is not None and width * height > max_pixels:
                raise ValueError(
                    f"decoded image exceeds max pixels: {width * height} > "
                    f"{max_pixels}"
                )
            image.load()
            return _rgba_image_sha256(image)
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("image bytes are not a complete PIL-decodable image") from exc


def _canonicalized_raw_pixel_sha256(
    image_bytes: bytes,
    *,
    max_pixels: int,
) -> tuple[str, tuple[int, int]]:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
            if width * height > max_pixels:
                raise ValueError("raw image exceeds max_decode_pixels")
            canonical = ImageOps.exif_transpose(image)
            width, height = canonical.size
            if width * height > max_pixels:
                raise ValueError("canonical image exceeds max_decode_pixels")
            canonical.load()
            return _rgba_image_sha256(canonical)
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("raw image cannot be canonicalized losslessly") from exc


def _resolve_data_file(root: Path, relpath: str, *, where: str) -> Path:
    candidate = root.joinpath(*relpath.split("/"))
    relative = candidate.relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{where} must not traverse a symlink: {relpath}")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"{where} does not exist: {relpath}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{where} escapes the dataset root: {relpath}") from exc
    mode = resolved.stat().st_mode
    if not stat.S_ISREG(mode):
        raise ValueError(f"{where} must be a regular file: {relpath}")
    return resolved


def _read_verified_file(path: Path, expected_sha256: str, *, where: str) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{where} must be a regular file")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError(f"{where} changed while it was being verified")
        actual = digest.hexdigest()
        if actual != expected_sha256:
            raise ValueError(
                f"{where} SHA-256 mismatch: expected {expected_sha256}, got {actual}"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_stable_file(path: Path, *, where: str) -> tuple[bytes, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{where} must be a regular non-symlink file")
    initial = path.read_bytes()
    digest = hashlib.sha256(initial).hexdigest()
    verified = _read_verified_file(path, digest, where=where)
    return verified, digest


def _strict_json_bytes(raw: bytes, *, where: str) -> dict[str, Any]:
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{where} has duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=hook,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{where} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{where} must contain one JSON object")
    return value


def validate_anyres_ready_dataset(
    root: str | Path,
    *,
    expected_preprocess_contract_sha256: str,
    assets_manifest: str | Path,
    views_manifest: str | Path,
    train_manifest: str | Path,
    sft_validation_manifest: str | Path,
    kl_selection_manifest: str | Path,
    formal_monitor_manifest: str | Path,
    normalized: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify READY, build receipt, manifests, and manual adjudications."""

    resolved_root = Path(root).resolve(strict=True)
    expected_paths = {
        "assets.jsonl": Path(assets_manifest).resolve(strict=True),
        "views.jsonl": Path(views_manifest).resolve(strict=True),
        "train.jsonl": Path(train_manifest).resolve(strict=True),
        "sft_validation.jsonl": Path(sft_validation_manifest).resolve(strict=True),
        "kl_selection.jsonl": Path(kl_selection_manifest).resolve(strict=True),
        "formal_monitor.jsonl": Path(formal_monitor_manifest).resolve(strict=True),
        "quarantine.jsonl": resolved_root / "quarantine.jsonl",
    }
    for name, path in expected_paths.items():
        if path != (resolved_root / name).resolve(strict=True):
            raise ValueError(f"{name} must be consumed from the READY dataset root")

    ready_raw, ready_file_sha = _read_stable_file(
        resolved_root / "READY",
        where="anyres READY",
    )
    ready = _strict_json_bytes(ready_raw, where="anyres READY")
    if set(ready) != {
        "schema_version",
        "kind",
        "build_receipt_sha256",
        "preprocess_contract_canonical_sha256",
        "manifests_sha256",
        "accepted_samples",
    } or ready.get("schema_version") != 1 or ready.get("kind") != _READY_KIND:
        raise ValueError("anyres READY fields differ from contract")
    if ready["preprocess_contract_canonical_sha256"] != (
        expected_preprocess_contract_sha256
    ):
        raise ValueError("READY preprocess contract differs from runtime")

    build_raw, build_sha = _read_stable_file(
        resolved_root / "receipts" / "build.json",
        where="anyres build receipt",
    )
    if build_sha != ready["build_receipt_sha256"]:
        raise ValueError("READY build receipt SHA256 mismatch")
    build = _strict_json_bytes(build_raw, where="anyres build receipt")
    required_build_fields = {
        "schema_version",
        "kind",
        "inputs",
        "contracts",
        "counts",
        "quota_counts",
        "manifests_sha256",
        "cut_adjudications_sha256",
        "split_lock_sha256",
        "components",
    }
    if set(build) != required_build_fields or build.get("schema_version") != 1 or (
        build.get("kind") != _BUILD_RECEIPT_KIND
    ):
        raise ValueError("anyres build receipt fields differ from contract")
    if build["contracts"].get("preprocess_contract_canonical_sha256") != (
        expected_preprocess_contract_sha256
    ):
        raise ValueError("build receipt preprocess contract differs")
    if build["manifests_sha256"] != ready["manifests_sha256"]:
        raise ValueError("READY/build manifest hashes differ")

    manifest_hashes: dict[str, str] = {}
    for name, path in expected_paths.items():
        _raw, digest = _read_stable_file(path, where=f"READY manifest {name}")
        manifest_hashes[name] = digest
    if manifest_hashes != ready["manifests_sha256"]:
        raise ValueError("READY manifest SHA256 mismatch")
    samples = normalized.get("samples")
    if not isinstance(samples, list) or ready["accepted_samples"] != len(samples):
        raise ValueError("READY accepted sample count differs from manifests")

    adjudication_path = resolved_root / "receipts" / "cut_adjudications.jsonl"
    adjudication_raw, adjudication_sha = _read_stable_file(
        adjudication_path,
        where="cut adjudications",
    )
    if adjudication_sha != build["cut_adjudications_sha256"]:
        raise ValueError("cut adjudications SHA256 differs from build receipt")
    records: dict[str, dict[str, Any]] = {}
    for line_no, raw_line in enumerate(adjudication_raw.splitlines(), start=1):
        if not raw_line:
            raise ValueError("cut adjudications must not contain blank lines")
        record = _strict_json_bytes(
            raw_line,
            where=f"cut adjudications:{line_no}",
        )
        digest = record.get("canonical_sha256")
        unhashed = {key: value for key, value in record.items() if key != "canonical_sha256"}
        if (
            record.get("schema_version") != 1
            or record.get("kind") != _CUT_ADJUDICATION_KIND
            or not isinstance(digest, str)
            or canonical_json_sha256(unhashed) != digest
            or digest in records
        ):
            raise ValueError("cut adjudication record is not canonical")
        records[digest] = record

    assets = {
        str(asset["asset_id"]): asset
        for asset in normalized.get("assets", [])
    }
    manual_digests: set[str] = set()
    for view in normalized.get("views", []):
        qa = view["qa"]
        if qa["review_status"] != "manual_accepted":
            continue
        digest = str(qa["adjudication_sha256"])
        record = records.get(digest)
        asset = assets[str(view["asset_id"])]
        if record is None:
            raise ValueError(f"manual view {view['view_id']!r} has no adjudication")
        expected = {
            "view_id": view["view_id"],
            "window_index": view["transform"]["window_index"],
            "native_plan_sha256": asset["native_plan"]["plan_sha256"],
            "asset_pixel_sha256": asset["pixel_sha256"],
            "asset_raw_sha256": asset["raw_sha256"],
            "cut_qa_coverage_proof_sha256": asset["cut_qa"][
                "coverage_proof_sha256"
            ],
            "decision": "manual_accepted",
        }
        for field, value in expected.items():
            if record.get(field) != value:
                raise ValueError(
                    f"manual view {view['view_id']!r} adjudication {field} differs"
                )
        manual_digests.add(digest)
    if set(records) != manual_digests:
        raise ValueError("cut adjudication receipt contains unreferenced records")
    return {
        "schema_version": 1,
        "kind": "dol_ocr_anyres_ready_admission_v1",
        "ready_file_sha256": ready_file_sha,
        "build_receipt_sha256": build_sha,
        "cut_adjudications_sha256": adjudication_sha,
        "manifests_sha256": manifest_hashes,
        "accepted_samples": len(samples),
        "manual_adjudications": len(records),
        "canonical_sha256": canonical_json_sha256(
            {
                "ready_file_sha256": ready_file_sha,
                "build_receipt_sha256": build_sha,
                "cut_adjudications_sha256": adjudication_sha,
                "manifests_sha256": manifest_hashes,
                "accepted_samples": len(samples),
                "manual_adjudications": len(records),
            }
        ),
    }


def _verify_image(
    image_bytes: bytes,
    *,
    expected_size: tuple[int, int],
    expected_pixel_sha256: str | None,
    max_decode_pixels: int,
    where: str,
) -> tuple[str, tuple[int, int]]:
    try:
        pixel_sha256, size = rgba_pixel_sha256(
            image_bytes,
            max_pixels=max_decode_pixels,
        )
    except ValueError as exc:
        raise ValueError(f"{where}: {exc}") from exc
    if size != expected_size:
        raise ValueError(
            f"{where} PIL size mismatch: expected {expected_size}, got {size}"
        )
    if expected_pixel_sha256 is not None and pixel_sha256 != expected_pixel_sha256:
        raise ValueError(
            f"{where} decoded pixel SHA-256 mismatch: expected "
            f"{expected_pixel_sha256}, got {pixel_sha256}"
        )
    return pixel_sha256, size


def _verify_lossless_png_without_orientation(
    image_bytes: bytes,
    *,
    where: str,
) -> None:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.format != "PNG":
                raise ValueError(f"{where} must be encoded as PNG")
            orientation = image.getexif().get(274, 1)
            if orientation not in (None, 1):
                raise ValueError(
                    f"{where} must not carry EXIF orientation metadata"
                )
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"{where} cannot be inspected as canonical PNG") from exc


class AnyresOCRDataset:
    """Load admitted anyres rows and return untouched bytes or verified paths.

    The loader performs no crop, resize, normalization, tensor conversion, or
    tokenization.  Every returned sample carries its deterministic quota bucket
    so a collator/sampler does not need to infer that classification.
    """

    def __init__(
        self,
        *,
        root: str | Path,
        assets_manifest: str | Path,
        views_manifest: str | Path,
        samples_manifest: str | Path,
        split: str,
        expected_preprocess_contract_sha256: str,
        max_decode_pixels: int,
        max_views_per_sample: int,
        image_delivery: ImageDelivery = "bytes",
    ) -> None:
        if image_delivery not in {"bytes", "path"}:
            raise ValueError("image_delivery must be 'bytes' or 'path'")
        if split not in ANYRES_PUBLIC_SPLITS:
            raise ValueError(
                "split must be train, sft_validation, kl_selection, or "
                "formal_monitor"
            )
        if (
            not isinstance(expected_preprocess_contract_sha256, str)
            or len(expected_preprocess_contract_sha256) != 64
            or any(
                char not in "0123456789abcdef"
                for char in expected_preprocess_contract_sha256
            )
        ):
            raise ValueError(
                "expected_preprocess_contract_sha256 must be a lowercase digest"
            )
        for name, value in (
            ("max_decode_pixels", max_decode_pixels),
            ("max_views_per_sample", max_views_per_sample),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        try:
            resolved_root = Path(root).resolve(strict=True)
        except FileNotFoundError as exc:
            raise ValueError(f"anyres dataset root does not exist: {root}") from exc
        if not resolved_root.is_dir():
            raise ValueError("anyres dataset root must be a directory")

        assets = load_anyres_jsonl(assets_manifest)
        views = load_anyres_jsonl(views_manifest)
        samples = load_anyres_jsonl(samples_manifest)
        normalized, quarantine = validate_anyres_assets_views_samples(
            assets, views, samples
        )
        selected = [row for row in normalized["samples"] if row["split"] == split]
        if not selected:
            raise ValueError(f"no accepted anyres samples for split {split!r}")
        mismatched_preprocess = [
            str(row["sample_id"])
            for row in selected
            if row["preprocess_contract_sha256"]
            != expected_preprocess_contract_sha256
        ]
        if mismatched_preprocess:
            raise ValueError(
                "anyres sample preprocess contract differs from the requested "
                f"runtime: {mismatched_preprocess[:8]}"
            )
        excessive_views = [
            str(row["sample_id"])
            for row in selected
            if len(row["view_ids"]) > max_views_per_sample
        ]
        if excessive_views:
            raise ValueError(
                "anyres samples exceed max_views_per_sample: "
                f"{excessive_views[:8]}"
            )

        assets_by_id = {row["asset_id"]: row for row in normalized["assets"]}
        views_by_id = {row["view_id"]: row for row in normalized["views"]}
        selected_asset_ids = {str(row["asset_id"]) for row in selected}
        selected_view_ids = {
            str(view_id) for row in selected for view_id in row["view_ids"]
        }
        selected_views_by_asset: dict[str, list[str]] = {
            asset_id: [] for asset_id in selected_asset_ids
        }
        for view_id in sorted(selected_view_ids):
            view = views_by_id.get(view_id)
            if view is None:
                raise ValueError(
                    f"accepted sample references quarantined view {view_id!r}"
                )
            selected_views_by_asset[str(view["asset_id"])].append(view_id)

        asset_paths: dict[str, dict[str, Path]] = {}
        view_paths: dict[str, Path] = {}
        for asset_id in sorted(selected_asset_ids):
            asset = assets_by_id[asset_id]
            for role in ("raw", "canonical"):
                width = int(asset[f"{role}_width"])
                height = int(asset[f"{role}_height"])
                if width * height > max_decode_pixels:
                    raise ValueError(
                        f"asset {asset_id!r} {role} image exceeds "
                        f"max_decode_pixels: {width * height} > "
                        f"{max_decode_pixels}"
                    )
            raw_path = _resolve_data_file(
                resolved_root,
                str(asset["raw_relpath"]),
                where=f"asset {asset_id!r} raw file",
            )
            canonical_path = _resolve_data_file(
                resolved_root,
                str(asset["canonical_relpath"]),
                where=f"asset {asset_id!r} canonical file",
            )
            raw_bytes = _read_verified_file(
                raw_path,
                str(asset["raw_sha256"]),
                where=f"asset {asset_id!r} raw file",
            )
            canonical_bytes = _read_verified_file(
                canonical_path,
                str(asset["canonical_sha256"]),
                where=f"asset {asset_id!r} canonical file",
            )
            _verify_lossless_png_without_orientation(
                canonical_bytes,
                where=f"asset {asset_id!r} canonical file",
            )
            _verify_image(
                raw_bytes,
                expected_size=(int(asset["raw_width"]), int(asset["raw_height"])),
                expected_pixel_sha256=None,
                max_decode_pixels=max_decode_pixels,
                where=f"asset {asset_id!r} raw file",
            )
            _verify_image(
                canonical_bytes,
                expected_size=(
                    int(asset["canonical_width"]),
                    int(asset["canonical_height"]),
                ),
                expected_pixel_sha256=str(asset["pixel_sha256"]),
                max_decode_pixels=max_decode_pixels,
                where=f"asset {asset_id!r} canonical file",
            )
            actual_phash = perceptual_hash(canonical_bytes)
            if actual_phash != asset["near_duplicate"]["value"]:
                raise ValueError(
                    f"asset {asset_id!r} perceptual hash differs from manifest"
                )
            canonicalized_raw_sha, canonicalized_raw_size = (
                _canonicalized_raw_pixel_sha256(
                    raw_bytes,
                    max_pixels=max_decode_pixels,
                )
            )
            expected_canonical_size = (
                int(asset["canonical_width"]),
                int(asset["canonical_height"]),
            )
            if canonicalized_raw_size != expected_canonical_size:
                raise ValueError(
                    f"asset {asset_id!r} raw EXIF canonical size differs from "
                    "canonical asset"
                )
            if canonicalized_raw_sha != str(asset["pixel_sha256"]):
                raise ValueError(
                    f"asset {asset_id!r} raw EXIF canonical pixels differ from "
                    "canonical asset"
                )
            asset_paths[asset_id] = {
                "raw": raw_path,
                "canonical": canonical_path,
            }
            try:
                with Image.open(io.BytesIO(canonical_bytes)) as canonical_image:
                    canonical_image.load()
                    canonical_rgba = canonical_image.convert("RGBA")
                    for view_id in selected_views_by_asset[asset_id]:
                        view = views_by_id[view_id]
                        view_pixels = int(view["derived_width"]) * int(
                            view["derived_height"]
                        )
                        if view_pixels > max_decode_pixels:
                            raise ValueError(
                                f"view {view_id!r} exceeds max_decode_pixels: "
                                f"{view_pixels} > {max_decode_pixels}"
                            )
                        derived_path = _resolve_data_file(
                            resolved_root,
                            str(view["derived_relpath"]),
                            where=f"view {view_id!r} derived file",
                        )
                        derived_bytes = _read_verified_file(
                            derived_path,
                            str(view["derived_sha256"]),
                            where=f"view {view_id!r} derived file",
                        )
                        _verify_lossless_png_without_orientation(
                            derived_bytes,
                            where=f"view {view_id!r} derived file",
                        )
                        _verify_image(
                            derived_bytes,
                            expected_size=(
                                int(view["derived_width"]),
                                int(view["derived_height"]),
                            ),
                            expected_pixel_sha256=str(view["pixel_sha256"]),
                            max_decode_pixels=max_decode_pixels,
                            where=f"view {view_id!r} derived file",
                        )
                        x0, y0, x1, y1 = (
                            int(value) for value in view["box_xyxy"]
                        )
                        crop_sha, crop_size = _rgba_image_sha256(
                            canonical_rgba.crop((x0, y0, x1, y1))
                        )
                        expected_size = (
                            int(view["derived_width"]),
                            int(view["derived_height"]),
                        )
                        if crop_size != expected_size or crop_sha != str(
                            view["pixel_sha256"]
                        ):
                            raise ValueError(
                                f"view {view_id!r} pixels are not the declared "
                                "no-resize canonical crop"
                            )
                        view_paths[view_id] = derived_path
                    cut_contract = asset["cut_qa"]
                    cut_payload = cut_contract["payload"]
                    parameters = cut_payload["parameters"]
                    cut_evidence = analyze_cut_qa(
                        np.asarray(canonical_rgba.convert("L"), dtype=np.uint8),
                        {
                            view_id: views_by_id[view_id]["box_xyxy"]
                            for view_id in selected_views_by_asset[asset_id]
                        },
                        canonical_pixel_sha256=str(asset["pixel_sha256"]),
                        plan_sha256=str(asset["native_plan"]["plan_sha256"]),
                        ink_threshold=int(parameters["ink_threshold"]),
                        min_component_pixels=int(
                            parameters["min_component_pixels"]
                        ),
                        edge_band_px=int(parameters["edge_band_px"]),
                        max_edge_ink_fraction=float(
                            parameters["max_edge_ink_fraction"]
                        ),
                    )
                    if (
                        cut_evidence.coverage_proof_sha256
                        != cut_contract["coverage_proof_sha256"]
                        or cut_evidence.proof_payload != cut_payload
                    ):
                        raise ValueError(
                            f"asset {asset_id!r} cut-QA proof is not reproducible"
                        )
                    evidence_by_view = {
                        evidence.view_id: evidence
                        for evidence in cut_evidence.views
                    }
                    for view_id in selected_views_by_asset[asset_id]:
                        actual = views_by_id[view_id]["qa"]
                        recomputed = evidence_by_view[view_id].manifest_qa()
                        for field in (
                            "cut_suspect",
                            "edge_ink_fraction",
                            "cc_crossing_count",
                            "cc_uncovered_count",
                            "coverage",
                            "coverage_proof_sha256",
                        ):
                            if actual[field] != recomputed[field]:
                                raise ValueError(
                                    f"view {view_id!r} cut-QA field {field} "
                                    "differs from recomputation"
                                )
                        if (
                            actual["review_status"] == "accepted"
                            and recomputed["review_status"] != "accepted"
                        ):
                            raise ValueError(
                                f"view {view_id!r} requires manual cut-QA review"
                            )
            except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
                raise ValueError(
                    f"asset {asset_id!r} canonical image cannot be reopened"
                ) from exc

        self.root = resolved_root
        self.image_delivery = image_delivery
        self.preprocess_contract_sha256 = expected_preprocess_contract_sha256
        self.quarantine = copy.deepcopy(quarantine)
        self._assets = assets_by_id
        self._views = views_by_id
        self._samples = selected
        self.sample_ids = tuple(str(row["sample_id"]) for row in selected)
        self._index_by_sample_id = {
            sample_id: index for index, sample_id in enumerate(self.sample_ids)
        }
        self._asset_paths = asset_paths
        self._view_paths = view_paths
        self.quota_counts = quota_counts(selected)
        self.quota_buckets = tuple(quota_bucket(row) for row in selected)
        contract_base = {
            "schema_version": 1,
            "kind": "dol_ocr_anyres_dataset_selection_v1",
            "split": split,
            "preprocess_contract_sha256": expected_preprocess_contract_sha256,
            "max_decode_pixels": max_decode_pixels,
            "max_views_per_sample": max_views_per_sample,
            "normalized_assets_sha256": canonical_json_sha256(
                normalized["assets"]
            ),
            "normalized_views_sha256": canonical_json_sha256(
                normalized["views"]
            ),
            "selected_samples_sha256": canonical_json_sha256(selected),
            "sample_ids": list(self.sample_ids),
            "quota_counts": dict(self.quota_counts),
            "quarantine_sha256": canonical_json_sha256(quarantine),
        }
        self._dataset_contract = {
            **contract_base,
            "contract_sha256": canonical_json_sha256(contract_base),
        }

    def __len__(self) -> int:
        return len(self._samples)

    def index_for_sample_id(self, sample_id: str) -> int:
        try:
            return self._index_by_sample_id[sample_id]
        except KeyError as exc:
            raise KeyError(f"unknown anyres sample_id {sample_id!r}") from exc

    def get_by_sample_id(self, sample_id: str) -> dict[str, Any]:
        return self[self.index_for_sample_id(sample_id)]

    @property
    def dataset_contract(self) -> dict[str, Any]:
        return copy.deepcopy(self._dataset_contract)

    @property
    def contract_sha256(self) -> str:
        return str(self._dataset_contract["contract_sha256"])

    def _payload(self, path: Path, sha256: str, *, where: str) -> dict[str, Any]:
        current_bytes = _read_verified_file(path, sha256, where=where)
        if self.image_delivery == "bytes":
            return {"delivery": "bytes", "bytes": current_bytes}
        return {"delivery": "path", "path": str(path)}

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._samples[index]
        asset_id = str(sample["asset_id"])
        asset = self._assets[asset_id]
        canonical = self._payload(
            self._asset_paths[asset_id]["canonical"],
            str(asset["canonical_sha256"]),
            where=f"asset {asset_id!r} canonical file",
        )
        canonical["metadata"] = {
            "asset_id": asset_id,
            "relpath": asset["canonical_relpath"],
            "sha256": asset["canonical_sha256"],
            "pixel_sha256": asset["pixel_sha256"],
            "width": asset["canonical_width"],
            "height": asset["canonical_height"],
        }

        derived: list[dict[str, Any]] = []
        for view_id in sample["reading_order"]:
            view = self._views[str(view_id)]
            payload = self._payload(
                self._view_paths[str(view_id)],
                str(view["derived_sha256"]),
                where=f"view {view_id!r} derived file",
            )
            payload["metadata"] = copy.deepcopy(view)
            derived.append(payload)

        return {
            "dataset_contract_sha256": self.contract_sha256,
            "sample": copy.deepcopy(sample),
            "asset": copy.deepcopy(asset),
            "canonical_image": canonical,
            "derived_images": derived,
            "quota_bucket": self.quota_buckets[index],
        }


__all__ = [
    "AnyresOCRDataset",
    "ImageDelivery",
    "load_anyres_jsonl",
    "rgba_pixel_sha256",
    "validate_anyres_ready_dataset",
]
