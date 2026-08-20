# -*- coding: utf-8 -*-

"""Identity-only manifests for locked OCR golden data.

The training process needs enough information to reject leakage, but it must
not be able to read golden images, paths, or transcripts.  A public identity
manifest therefore contains only sample/group ids and image content hashes.
The labeled manifest is opened solely by the final evaluator.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from Model.ocr.tokenization import canonical_json_sha256
from Model.posttrain.ocr_manifest_builder import file_sha256 as _file_sha256

GOLDEN_IDENTITY_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")
_FORBIDDEN_GOLDEN_FIELDS = frozenset(
    {"reference", "transcription", "image", "image_path", "images", "instruction"}
)


def load_ocr_dataset_contract(
    path: str | Path,
    *,
    train_manifest: str | Path,
    validation_manifest: str | Path,
    golden_identity_manifest: str | Path,
    tokenization_contract: dict,
) -> dict:
    """Bind all public manifests, pixels, and tokenizer semantics before RL."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"OCR dataset contract does not exist: {source}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid OCR dataset contract JSON: {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("OCR dataset contract must be a JSON object")
    errors: list[str] = []
    if payload.get("schema_version") != 2:
        errors.append("schema_version must be 2")
    if payload.get("image_root") != "." or payload.get(
        "images_materialized"
    ) is not True:
        errors.append(
            "public images must be materialized beneath the contract directory"
        )
    if payload.get("ocr_tokenization_contract") != tokenization_contract:
        errors.append("OCR tokenization contract differs from runtime")
    locked_receipt_sha256 = payload.get("locked_golden_receipt_sha256")
    if (
        not isinstance(locked_receipt_sha256, str)
        or not _SHA256_RE.fullmatch(locked_receipt_sha256)
    ):
        errors.append("locked golden receipt anchor is missing or invalid")
    manifests = payload.get("manifests")
    if not isinstance(manifests, dict):
        errors.append("manifests must be an object")
        manifests = {}
    expected = {
        "rl_train_sha256": _file_sha256(train_manifest),
        "rl_val_sha256": _file_sha256(validation_manifest),
        "golden_identity_sha256": _file_sha256(golden_identity_manifest),
    }
    for key, digest in expected.items():
        if manifests.get(key) != digest:
            errors.append(f"{key} differs from the public manifest")
    identity_rows = load_golden_identity_manifest(golden_identity_manifest)
    if manifests.get(
        "golden_identity_semantic_sha256"
    ) != golden_identity_semantic_sha256(identity_rows):
        errors.append("golden identity semantic SHA-256 differs")
    symbol_support = payload.get("reference_symbol_support")
    if not isinstance(symbol_support, dict) or set(symbol_support) != {
        "rl_train",
        "rl_val",
    }:
        errors.append(
            "public symbol support must contain train/validation only"
        )
    split_lock = source.parent / "split_lock.json"
    if not split_lock.is_file() or payload.get(
        "split_lock_sha256"
    ) != _file_sha256(split_lock):
        errors.append("split lock is missing or differs from the contract")
    if errors:
        raise ValueError(
            "unsafe OCR dataset contract:\n  - " + "\n  - ".join(errors)
        )
    return {
        **payload,
        "contract_file_sha256": _file_sha256(source),
        "resolved_image_root": str(source.parent.resolve()),
    }


def load_locked_golden_receipt_anchor(
    path: str | Path,
    *,
    golden_manifest: str | Path,
    golden_identity_manifest: str | Path,
    identity_rows: list[dict[str, str | int]],
    public_dataset_contract: str | Path,
) -> dict:
    """Verify the sealed receipt anchor without opening the golden manifest.

    This is the only locked-receipt operation allowed before the one-shot
    ledger is claimed.  In particular, it must not open or hash
    ``golden_manifest`` bytes: doing so would expose labels before the durable
    consumption record exists.
    """

    source = Path(path)
    golden = Path(golden_manifest)
    canonical_receipt = golden.parent / "build_receipt.json"
    if source.is_symlink() or golden.is_symlink():
        raise ValueError("locked golden manifest/receipt must not be symlinks")
    if source.absolute() != canonical_receipt.absolute():
        raise ValueError(
            "locked golden receipt must be canonical build_receipt.json beside "
            "the golden manifest"
        )
    if golden.name != "golden.jsonl":
        raise ValueError("locked golden manifest must be named golden.jsonl")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"locked golden receipt does not exist: {source}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid locked golden receipt JSON: {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("locked golden receipt must be a JSON object")
    try:
        public_payload = json.loads(
            Path(public_dataset_contract).read_text(encoding="utf-8")
        )
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(
            "cannot read the public dataset contract for locked receipt anchoring"
        ) from exc
    if not isinstance(public_payload, dict):
        raise ValueError("public dataset contract must be an object")
    expected_receipt_sha256 = public_payload.get(
        "locked_golden_receipt_sha256"
    )
    expected = {
        "golden_identity_sha256": _file_sha256(golden_identity_manifest),
        "golden_identity_semantic_sha256": (
            golden_identity_semantic_sha256(identity_rows)
        ),
    }
    errors = []
    if payload.get("schema_version") != 2:
        errors.append("schema_version must be 2")
    if payload.get("kind") != "ocr_locked_golden_build_receipt":
        errors.append("kind is not an OCR locked-golden build receipt")
    locked_golden_sha256 = payload.get("locked_golden_sha256")
    if (
        not isinstance(locked_golden_sha256, str)
        or not _SHA256_RE.fullmatch(locked_golden_sha256)
    ):
        errors.append("locked_golden_sha256 is missing or invalid")
    if expected_receipt_sha256 != _file_sha256(source):
        errors.append(
            "receipt SHA-256 differs from the training-anchored public contract"
        )
    for key, digest in expected.items():
        if payload.get(key) != digest:
            errors.append(f"{key} differs from the sealed build")
    if errors:
        raise ValueError(
            "unsafe locked golden receipt:\n  - " + "\n  - ".join(errors)
        )
    return {**payload, "receipt_file_sha256": _file_sha256(source)}


def verify_locked_golden_manifest(
    receipt: dict,
    *,
    golden_manifest: str | Path,
) -> str:
    """Hash and verify golden labels after the one-shot claim is durable."""

    expected = receipt.get("locked_golden_sha256")
    if not isinstance(expected, str) or not _SHA256_RE.fullmatch(expected):
        raise ValueError(
            "unsafe locked golden receipt:\n  - "
            "locked_golden_sha256 is missing or invalid"
        )
    actual = _file_sha256(golden_manifest)
    if actual != expected.lower():
        raise ValueError(
            "unsafe locked golden receipt:\n  - "
            "locked_golden_sha256 differs from the sealed build"
        )
    return actual


def load_golden_identity_manifest(
    path: str | Path,
    *,
    required_split: str = "golden",
) -> list[dict[str, str | int]]:
    """Load a transcript-free golden identity registry with strict schema."""

    source = Path(path)
    rows: list[dict[str, str | int]] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{source}:{line_no}: expected a JSON object")
            forbidden = sorted(_FORBIDDEN_GOLDEN_FIELDS & set(obj))
            if forbidden:
                raise ValueError(
                    f"{source}:{line_no}: identity manifest must not contain "
                    "golden payload fields: " + ", ".join(forbidden)
                )
            if obj.get("schema_version") != GOLDEN_IDENTITY_SCHEMA_VERSION:
                raise ValueError(
                    f"{source}:{line_no}: schema_version must be "
                    f"{GOLDEN_IDENTITY_SCHEMA_VERSION}"
                )
            sample_id = obj.get("id")
            group_id = obj.get("group_id")
            split = obj.get("split")
            digest = obj.get("image_sha256")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError(f"{source}:{line_no}: non-empty string id is required")
            if not isinstance(group_id, str) or not group_id.strip():
                raise ValueError(
                    f"{source}:{line_no}: non-empty string group_id is required"
                )
            if split != required_split:
                raise ValueError(
                    f"{source}:{line_no}: split must be {required_split!r}, got {split!r}"
                )
            if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
                raise ValueError(
                    f"{source}:{line_no}: image_sha256 must be 64 hex characters"
                )
            sample_id = sample_id.strip()
            group_id = group_id.strip()
            digest = digest.lower()
            if sample_id in seen_ids:
                raise ValueError(f"{source}:{line_no}: duplicate id {sample_id!r}")
            if digest in seen_hashes:
                raise ValueError(
                    f"{source}:{line_no}: duplicate image_sha256 {digest}"
                )
            seen_ids.add(sample_id)
            seen_hashes.add(digest)
            rows.append(
                {
                    "schema_version": GOLDEN_IDENTITY_SCHEMA_VERSION,
                    "id": sample_id,
                    "group_id": group_id,
                    "split": required_split,
                    "image_sha256": digest,
                }
            )
    if not rows:
        raise ValueError(f"golden identity manifest is empty: {source}")
    return rows


def golden_identity_semantic_sha256(
    rows: list[dict[str, str | int]],
) -> str:
    """Canonical hash used to match a locked labeled manifest by identity."""

    semantic = [
        {
            "schema_version": GOLDEN_IDENTITY_SCHEMA_VERSION,
            "id": str(row["id"]),
            "group_id": str(row["group_id"]),
            "split": str(row["split"]),
            "image_sha256": str(row["image_sha256"]).lower(),
        }
        for row in sorted(rows, key=lambda item: str(item["id"]))
    ]
    return canonical_json_sha256(semantic)


def golden_identity_keys(
    rows: list[dict[str, str | int]],
) -> tuple[set[str], set[str], set[str]]:
    """Return ids, image hashes, and groups for train/validation exclusions."""

    return (
        {str(row["id"]) for row in rows},
        {str(row["image_sha256"]).lower() for row in rows},
        {str(row["group_id"]) for row in rows},
    )


def assert_labeled_golden_matches_identity(
    dataset,
    identity_rows: list[dict[str, str | int]],
) -> None:
    """Require an exact id/group/image-hash match before final evaluation."""

    expected = {
        str(row["id"]): (
            str(row["group_id"]),
            str(row["image_sha256"]).lower(),
        )
        for row in identity_rows
    }
    actual: dict[str, tuple[str, str]] = {}
    for index in range(len(dataset)):
        row = dataset[index]
        actual[str(row["id"])] = (
            str(row["group_id"]),
            str(row["sha256"]).lower(),
        )
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    mismatched = sorted(
        sample_id
        for sample_id in expected.keys() & actual.keys()
        if expected[sample_id] != actual[sample_id]
    )
    if missing or extra or mismatched:
        details = []
        if missing:
            details.append(f"missing ids={missing[:8]}")
        if extra:
            details.append(f"unexpected ids={extra[:8]}")
        if mismatched:
            details.append(f"group/hash mismatch ids={mismatched[:8]}")
        raise ValueError(
            "locked golden does not match the registered identity manifest: "
            + "; ".join(details)
        )


__all__ = [
    "GOLDEN_IDENTITY_SCHEMA_VERSION",
    "assert_labeled_golden_matches_identity",
    "golden_identity_keys",
    "golden_identity_semantic_sha256",
    "load_golden_identity_manifest",
    "load_locked_golden_receipt_anchor",
    "load_ocr_dataset_contract",
    "verify_locked_golden_manifest",
]
