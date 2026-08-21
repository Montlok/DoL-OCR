# -*- coding: utf-8 -*-
"""Build and save a routed tokenizer bundle (MorphBPE + general byte-level BPE)."""

from __future__ import annotations

import argparse
import sys

from Tokenizer.unified.bundle import TokenizerBundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--morphbpe", required=True, help="MorphBPE JSON model path")
    parser.add_argument(
        "--general",
        help="general byte-level BPE (tokenizers JSON); omit for a minimal "
        "training-free byte-level fallback",
    )
    parser.add_argument("--output", required=True, help="output bundle directory")
    args = parser.parse_args()

    try:
        bundle = TokenizerBundle.from_files(args.morphbpe, general_path=args.general)
    except ImportError as exc:
        raise SystemExit(f"Missing dependency: {exc}") from exc
    bundle.save_dir(args.output)
    loaded = TokenizerBundle.from_dir(args.output)
    issues = loaded.validate()
    print(f"vocab_size={loaded.tokenizer.vocab_size}")
    print(f"saved={args.output}")
    if issues:
        print("validate=failed")
        for issue in issues:
            print(f"- {issue}")
        sys.exit(1)
    print("validate=ok")


if __name__ == "__main__":
    main()
