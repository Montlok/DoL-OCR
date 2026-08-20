# -*- coding: utf-8 -*-

"""Build deterministic, leakage-safe OCR-RL manifests from reviewed annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from Model.ocr.tokenization import (
    OCR_NATIVE_TARGET_ENCODING,
    make_ocr_target_encoder,
    native_tokenization_contract,
)
from Model.posttrain.ocr_manifest_builder import (
    assign_group_splits,
    build_manifest_rows,
    file_sha256,
    load_annotation_pack,
    load_split_lock,
    materialize_manifest_images,
    partition_manifest_rows,
    read_annotation_tsv,
    reference_symbol_support,
    validate_split_fractions,
)
from Model.posttrain.ocr_manifests import golden_identity_semantic_sha256


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, help="reviewed annotation TSV")
    parser.add_argument(
        "--pack-manifest",
        required=True,
        help="annotation-pack manifest.json; raw_path is authoritative",
    )
    parser.add_argument("--tokenizer", required=True, help="TokenizerBundle directory")
    parser.add_argument("--out", required=True, help="new dataset output directory")
    parser.add_argument("--domain", default="scan_book")
    parser.add_argument(
        "--reference-column",
        default="auto",
        help="auto, reference, transcription, or another explicit TSV column",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-frac", type=float, default=0.63)
    parser.add_argument("--validation-frac", type=float, default=0.07)
    parser.add_argument("--golden-frac", type=float, default=0.30)
    parser.add_argument(
        "--split-lock",
        default="",
        help="existing/persistent split_lock.json; defaults to OUT/split_lock.json",
    )
    return parser.parse_args(argv)


def _jsonl(rows: list[dict]) -> str:
    return "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in rows
    )


def _json(payload: dict) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_write(path: Path, text: str, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _refuse_manifest_overwrite(paths: list[Path]) -> None:
    occupied = [path for path in paths if path.exists()]
    if occupied:
        raise FileExistsError(
            "refusing to overwrite an existing OCR-RL dataset artifact: "
            + ", ".join(str(path) for path in occupied)
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    fractions = validate_split_fractions(
        {
            "rl_train": args.train_frac,
            "rl_val": args.validation_frac,
            "golden": args.golden_frac,
        }
    )
    output = Path(args.out).absolute()
    if output.exists():
        raise FileExistsError(
            "OCR-RL dataset versions are immutable; choose a new --out path: "
            f"{output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    external_split_lock = (
        Path(args.split_lock).absolute() if args.split_lock else None
    )
    if (
        external_split_lock is not None
        and external_split_lock.is_relative_to(output)
    ):
        raise ValueError("--split-lock must be outside the new immutable --out")

    annotations = read_annotation_tsv(
        args.annotations,
        reference_column=args.reference_column,
    )
    image_root, provenance = load_annotation_pack(
        args.pack_manifest,
        domain=args.domain,
    )
    selected_provenance = {
        sample_id: provenance[sample_id]
        for sample_id in annotations
        if sample_id in provenance
    }
    missing = sorted(set(annotations) - set(selected_provenance))
    if missing:
        raise ValueError(
            "annotation ids missing from pack manifest: " + ", ".join(missing[:8])
        )
    group_domains: dict[str, str] = {}
    group_sizes: Counter[str] = Counter()
    for item in selected_provenance.values():
        group_id = str(item["group_id"])
        domain = str(item["domain"])
        previous = group_domains.setdefault(group_id, domain)
        if previous != domain:
            raise ValueError(
                f"group {group_id!r} contains multiple domains"
            )
        group_sizes[group_id] += 1
    existing_lock = load_split_lock(
        external_split_lock,
        seed=args.seed,
        fractions=fractions,
    )
    split_lock = assign_group_splits(
        group_domains,
        seed=args.seed,
        fractions=fractions,
        existing_lock=existing_lock,
        group_sizes=group_sizes,
    )

    from Tokenizer.unified.bundle import TokenizerBundle

    bundle = TokenizerBundle.from_dir(args.tokenizer)
    issues = bundle.validate()
    if issues:
        raise ValueError(
            "invalid tokenizer bundle:\n  - " + "\n  - ".join(issues)
        )
    token_contract = native_tokenization_contract(
        bundle.tokenizer,
        args.tokenizer,
    )
    encode_reference = make_ocr_target_encoder(
        bundle.tokenizer,
        mode=OCR_NATIVE_TARGET_ENCODING,
    )
    rows = build_manifest_rows(
        annotations,
        selected_provenance,
        split_lock["assignments"],
        image_root=image_root,
        encode_reference=encode_reference,
        decode_reference=bundle.tokenizer.decode,
    )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.staging-",
            dir=output.parent,
        )
    )
    try:
        public = staging / "public"
        locked = staging / "locked"
        public.mkdir()
        locked.mkdir()
        os.chmod(staging, 0o755)
        os.chmod(public, 0o755)
        os.chmod(locked, 0o700)

        materialized_rows = materialize_manifest_images(
            rows,
            source_root=image_root,
            public_root=public,
            locked_root=locked,
        )
        train_rows, validation_rows, golden_rows, identity_rows = (
            partition_manifest_rows(materialized_rows)
        )

        train_text = _jsonl(train_rows)
        validation_text = _jsonl(validation_rows)
        identity_text = _jsonl(identity_rows)
        golden_text = _jsonl(golden_rows)
        split_lock_text = _json(split_lock)
        split_lock_sha256 = _text_sha256(split_lock_text)
        identity_sha256 = _text_sha256(identity_text)
        identity_semantic_sha256 = golden_identity_semantic_sha256(
            identity_rows
        )
        locked_receipt_payload = {
            "schema_version": 2,
            "kind": "ocr_locked_golden_build_receipt",
            "annotations_sha256": file_sha256(args.annotations),
            "pack_manifest_sha256": file_sha256(args.pack_manifest),
            "split_lock_sha256": split_lock_sha256,
            "locked_golden_sha256": _text_sha256(golden_text),
            "golden_identity_sha256": identity_sha256,
            "golden_identity_semantic_sha256": identity_semantic_sha256,
            "golden_reference_symbol_support": reference_symbol_support(
                golden_rows
            ),
        }
        locked_receipt_text = _json(locked_receipt_payload)
        public_contract_payload = {
            "schema_version": 2,
            "image_root": ".",
            "images_materialized": True,
            "text_normalization_contract": (
                token_contract["reference_canonicalization"]
            ),
            "ocr_tokenization_contract": token_contract,
            "split_lock_sha256": split_lock_sha256,
            "locked_golden_receipt_sha256": _text_sha256(
                locked_receipt_text
            ),
            "manifests": {
                "rl_train_sha256": _text_sha256(train_text),
                "rl_val_sha256": _text_sha256(validation_text),
                "golden_identity_sha256": identity_sha256,
                "golden_identity_semantic_sha256": (
                    identity_semantic_sha256
                ),
            },
            "counts": {
                "rl_train": len(train_rows),
                "rl_val": len(validation_rows),
                "golden": len(golden_rows),
                "groups": len(group_domains),
            },
            "reference_symbol_support": {
                "rl_train": reference_symbol_support(train_rows),
                "rl_val": reference_symbol_support(validation_rows),
            },
        }
        _atomic_write(
            public / "split_lock.json",
            split_lock_text,
            mode=0o644,
        )
        _atomic_write(
            public / "rl_train.jsonl",
            train_text,
            mode=0o644,
        )
        _atomic_write(
            public / "rl_val.jsonl",
            validation_text,
            mode=0o644,
        )
        _atomic_write(
            public / "golden_identity.jsonl",
            identity_text,
            mode=0o644,
        )
        _atomic_write(
            public / "dataset_contract.json",
            _json(public_contract_payload),
            mode=0o644,
        )
        _atomic_write(
            locked / "golden.jsonl",
            golden_text,
            mode=0o600,
        )
        _atomic_write(
            locked / "build_receipt.json",
            locked_receipt_text,
            mode=0o600,
        )
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    if external_split_lock is not None:
        _atomic_write(
            external_split_lock,
            split_lock_text,
            mode=0o644,
        )

    report = {
        "output": str(output),
        "source_pack_image_root": str(image_root),
        "split_lock": str(output / "public" / "split_lock.json"),
        "external_split_lock": (
            str(external_split_lock) if external_split_lock is not None else ""
        ),
        "public": {
            "image_root": str(output / "public"),
            "rl_train": str(output / "public" / "rl_train.jsonl"),
            "rl_val": str(output / "public" / "rl_val.jsonl"),
            "golden_identity": str(
                output / "public" / "golden_identity.jsonl"
            ),
            "dataset_contract": str(
                output / "public" / "dataset_contract.json"
            ),
        },
        "locked": {
            "image_root": str(output / "locked"),
            "golden": str(output / "locked" / "golden.jsonl"),
            "build_receipt": str(output / "locked" / "build_receipt.json"),
        },
        "counts": public_contract_payload["counts"],
    }
    print(_json(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
