#!/usr/bin/env python3
"""Fail closed when tracked files cross DoL-OCR repository boundaries."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable

MAX_TRACKED_BYTES = 10 * 1024 * 1024

SCRIPT_ALLOWLIST = frozenset(
    {
        "scripts/build_ocr_anyres_dataset.py",
        "scripts/build_ocr_data.py",
        "scripts/build_ocr_data_from_pairs.py",
        "scripts/build_text_rows_from_align.py",
        "scripts/check_repository_hygiene.py",
        "scripts/eval_ctc_head.py",
        "scripts/eval_ocr_anyres_locked.py",
        "scripts/eval_omvt_ssl.py",
        "scripts/eval_vlm_ocr.py",
        "scripts/generate.py",
        "scripts/ingest_scan_pdfs.py",
        "scripts/pretrain_e2e.sh",
        "scripts/rdt_monitor.py",
        "scripts/render_mn_pages.py",
        "scripts/train_ctc_head.py",
        "scripts/train_ocr_anyres_grpo.py",
        "scripts/train_ocr_anyres_joint_sft.py",
        "scripts/train_ocr_anyres_sft.py",
        "scripts/train_omvt_ssl.py",
        "scripts/train_rdt.py",
        "scripts/train_vlm_align.py",
        "scripts/validate_ocr_anyres_dataset.py",
    }
)

TOKENIZER_TOOL_ALLOWLIST = frozenset(
    {
        "Tokenizer/tools/build_corpus_mix.py",
        "Tokenizer/tools/build_general_bpe.py",
        "Tokenizer/tools/build_morphbpe.py",
        "Tokenizer/tools/build_pretraining_data.py",
        "Tokenizer/tools/build_unified_tokenizer.py",
        "Tokenizer/tools/clean_chinese_web.py",
        "Tokenizer/tools/clean_traditional_mongolian.py",
        "Tokenizer/tools/corpus_clean.py",
        "Tokenizer/tools/corpus_filters.py",
        "Tokenizer/tools/prepare_corpus.py",
    }
)

FORBIDDEN_BASENAMES = {
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
    "nohup.out",
}

FORBIDDEN_ENDINGS = (
    # Datasets and media are external to this source repository.
    ".avi",
    ".bmp",
    ".csv",
    ".flac",
    ".gif",
    ".jpeg",
    ".jpg",
    ".jsonl",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ndjson",
    ".ogg",
    ".pdf",
    ".png",
    ".svg",
    ".tif",
    ".tiff",
    ".tsv",
    ".txt",
    ".wav",
    ".webm",
    ".webp",
    # Model, optimizer, array, database and run artifacts.
    ".arrow",
    ".bin",
    ".ckpt",
    ".db",
    ".engine",
    ".ggml",
    ".gguf",
    ".h5",
    ".hdf5",
    ".joblib",
    ".log",
    ".mar",
    ".mlmodel",
    ".model",
    ".npy",
    ".npz",
    ".onnx",
    ".parquet",
    ".pb",
    ".pid",
    ".pickle",
    ".pkl",
    ".prof",
    ".pt",
    ".pth",
    ".safetensors",
    ".sqlite",
    ".sqlite3",
    ".tflite",
    ".trace",
    ".trt",
    ".weight",
    ".weights",
    # Archives and bulk exports.
    ".7z",
    ".bz2",
    ".gz",
    ".rar",
    ".tar",
    ".tar.bz2",
    ".tar.gz",
    ".tar.xz",
    ".tar.zst",
    ".tgz",
    ".xz",
    ".zip",
    ".zst",
)

FORBIDDEN_PATH_PREFIXES = (
    "Encoding Mapping/",
    "Tokenizer/data/",
    "artifacts/",
    "checkpoints/",
    "corpora/",
    "corpus/",
    "data/",
    "datasets/",
    "local_data/",
    "outputs/",
    "private_data/",
    "runs/",
    "weights/",
)

FORBIDDEN_PATH_FRAGMENTS = (
    "do not git it",
    "donotgitit",
)


def tracked_paths(root: Path) -> list[Path]:
    """Return tracked paths exactly as recorded by Git."""

    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    )
    return [Path(raw.decode("utf-8")) for raw in result.stdout.split(b"\0") if raw]


def check_paths(root: Path, paths: Iterable[Path]) -> list[str]:
    """Return deterministic policy violations for ``paths`` under ``root``."""

    violations: list[str] = []
    for relative in sorted(paths, key=lambda item: item.as_posix()):
        rendered = relative.as_posix()
        lowered = rendered.casefold()
        absolute = root / relative

        if relative.is_absolute() or ".." in relative.parts:
            violations.append(f"unsafe tracked path: {rendered!r}")

        if any(character in rendered for character in ("\n", "\r", "\0")):
            violations.append(f"unsafe tracked path: {rendered!r}")

        if relative.name in FORBIDDEN_BASENAMES:
            violations.append(f"local-system file is tracked: {rendered}")

        if relative.name == ".gitkeep":
            violations.append(f"placeholder file is tracked: {rendered}")

        if lowered.startswith(tuple(prefix.casefold() for prefix in FORBIDDEN_PATH_PREFIXES)):
            violations.append(f"forbidden data or retired path is tracked: {rendered}")

        if any(fragment in lowered for fragment in FORBIDDEN_PATH_FRAGMENTS):
            violations.append(f"local-only path marker is tracked: {rendered}")

        if lowered.endswith(FORBIDDEN_ENDINGS):
            violations.append(f"artifact, dataset, or media file is tracked: {rendered}")

        if rendered.startswith("scripts/") and rendered not in SCRIPT_ALLOWLIST:
            violations.append(f"unapproved scripts entry point is tracked: {rendered}")

        if (
            rendered.startswith("Tokenizer/tools/")
            and rendered not in TOKENIZER_TOOL_ALLOWLIST
        ):
            violations.append(f"unapproved tokenizer tool is tracked: {rendered}")

        if absolute.is_symlink():
            violations.append(f"symbolic link is tracked: {rendered}")

        try:
            size = absolute.stat().st_size
        except FileNotFoundError:
            violations.append(f"tracked path is missing from the checkout: {rendered}")
            continue
        if size > MAX_TRACKED_BYTES:
            violations.append(
                f"tracked file exceeds {MAX_TRACKED_BYTES} bytes: {rendered} ({size})"
            )

    return violations


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    paths = tracked_paths(root)
    violations = check_paths(root, paths)
    if violations:
        print("repository hygiene check failed:", file=sys.stderr)
        for violation in violations:
            print(f"- {violation}", file=sys.stderr)
        return 1
    print(f"repository hygiene check passed ({len(paths)} tracked files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
