#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Prepare OCR anyres review packs or finalize an immutable approved dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Model.ocr.tokenization import (  # noqa: E402
    OCR_NATIVE_TARGET_ENCODING,
    make_ocr_target_encoder,
    native_tokenization_contract,
)
from Model.posttrain.ocr_anyres_builder import (  # noqa: E402
    finalize_anyres_dataset,
    prepare_anyres_review_pack,
)
from Tokenizer.unified.bundle import TokenizerBundle  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare-review",
        help="materialize immutable pixels, plans, previews, and review templates",
    )
    prepare.add_argument("--sources", required=True, help="strict sources.jsonl")
    prepare.add_argument("--source-root", required=True)
    prepare.add_argument("--preprocess-contract", required=True)
    prepare.add_argument("--out", required=True)

    finalize = commands.add_parser(
        "finalize",
        help="finalize completed human reviews into train/validation data",
    )
    finalize.add_argument("--sources", required=True, help="strict sources.jsonl")
    finalize.add_argument(
        "--approved-reviews",
        required=True,
        help="strict approved_reviews.jsonl bound to deterministic plans",
    )
    finalize.add_argument("--source-root", required=True)
    finalize.add_argument("--preprocess-contract", required=True)
    finalize.add_argument("--tokenizer", required=True)
    finalize.add_argument("--split-policy", required=True)
    finalize.add_argument("--split-lock", required=True)
    finalize.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare-review":
        report = prepare_anyres_review_pack(
            sources_path=args.sources,
            preprocess_contract_path=args.preprocess_contract,
            source_root=args.source_root,
            output_dir=args.out,
        )
    else:
        bundle = TokenizerBundle.from_dir(args.tokenizer)
        issues = bundle.validate()
        if issues:
            raise ValueError(
                "invalid tokenizer bundle:\n  - " + "\n  - ".join(issues)
            )
        encoder = make_ocr_target_encoder(
            bundle.tokenizer,
            mode=OCR_NATIVE_TARGET_ENCODING,
        )
        report = finalize_anyres_dataset(
            sources_path=args.sources,
            approved_reviews_path=args.approved_reviews,
            preprocess_contract_path=args.preprocess_contract,
            split_policy_path=args.split_policy,
            split_lock_path=args.split_lock,
            source_root=args.source_root,
            output_dir=args.out,
            native_encoder=encoder,
            decode_reference=bundle.tokenizer.decode,
            tokenizer_contract=native_tokenization_contract(
                bundle.tokenizer,
                args.tokenizer,
            ),
        )
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
