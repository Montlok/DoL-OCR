#!/usr/bin/env python3
"""Fail when tracked files cross repository data and artifact boundaries."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable

MAX_TRACKED_BYTES = 10 * 1024 * 1024

FORBIDDEN_BASENAMES = {
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
}

FORBIDDEN_ENDINGS = (
    ".7z",
    ".arrow",
    ".bin",
    ".ckpt",
    ".db",
    ".ggml",
    ".gguf",
    ".h5",
    ".hdf5",
    ".joblib",
    ".log",
    ".npy",
    ".npz",
    ".onnx",
    ".parquet",
    ".pid",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
    ".sqlite",
    ".sqlite3",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tflite",
    ".zip",
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
        absolute = root / relative

        if any(character in rendered for character in ("\n", "\r", "\0")):
            violations.append(f"unsafe tracked path: {rendered!r}")

        if relative.name in FORBIDDEN_BASENAMES:
            violations.append(f"local-system file is tracked: {rendered}")

        if relative.name == ".gitkeep":
            violations.append(f"placeholder file is tracked: {rendered}")

        lowered = rendered.lower()
        if lowered.endswith(FORBIDDEN_ENDINGS):
            violations.append(f"artifact or corpus file is tracked: {rendered}")

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
