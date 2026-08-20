# -*- coding: utf-8 -*-

"""Runtime fingerprint for the tokenizer algorithm, independent of its vocab."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
import unicodedata
from pathlib import Path

from Tokenizer.unified.bundle import TokenizerBundle

TOKENIZER_ALGORITHM_CONTRACT_VERSION = 1
TOKENIZER_BUNDLE_CONTRACT_VERSION = 1
_ALGORITHM_FILES = (
    "Tokenizer/generic_bpe/__init__.py",
    "Tokenizer/generic_bpe/byte_fallback.py",
    "Tokenizer/generic_bpe/general_bpe.py",
    "Tokenizer/morphbpe/__init__.py",
    "Tokenizer/morphbpe/offsets.py",
    "Tokenizer/morphbpe/serialization.py",
    "Tokenizer/morphbpe/tokenizer.py",
    "Tokenizer/pretraining/builder.py",
    "Tokenizer/pretraining/morphology.py",
    "Tokenizer/traditional_mongolian/__init__.py",
    "Tokenizer/traditional_mongolian/alphabet.py",
    "Tokenizer/traditional_mongolian/morph_rules.py",
    "Tokenizer/traditional_mongolian/stemmer.py",
    "Tokenizer/traditional_mongolian/suffixes.py",
    "Tokenizer/traditional_mongolian/unicode_norm.py",
    "Tokenizer/unified/__init__.py",
    "Tokenizer/unified/bundle.py",
    "Tokenizer/unified/contract.py",
    "Tokenizer/unified/dual_tokenizer.py",
    "Tokenizer/unified/encoded.py",
    "Tokenizer/unified/stream_decode.py",
    "Tokenizer/unified/vocab.py",
)


def _resolved_bundle_member(root: Path, value: str, *, role: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"tokenizer bundle {role} path is empty")
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"tokenizer bundle {role} must stay inside the bundle directory"
        ) from exc
    if not candidate.is_file():
        raise ValueError(f"tokenizer bundle {role} file is missing: {candidate}")
    return candidate


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def tokenizer_bundle_contract(bundle_dir: str | Path) -> dict[str, object]:
    """Fingerprint every persisted file the tokenizer actually resolves.

    The contract contains no absolute path, so an unchanged bundle may move to
    a different mount.  Configured model files must remain inside the bundle;
    this prevents a receipt from silently depending on unrecorded external
    bytes.
    """

    root = Path(bundle_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"tokenizer bundle directory does not exist: {root}")
    config_path = root / "config.json"
    vocab_path = root / "vocab.json"
    manifest_path = root / "manifest.json"
    for role, path in (
        ("config", config_path),
        ("vocab", vocab_path),
        ("manifest", manifest_path),
    ):
        if not path.is_file():
            raise ValueError(f"tokenizer bundle {role} file is missing: {path}")

    try:
        bundle = TokenizerBundle.from_dir(str(root))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid tokenizer bundle at {root}: {exc}") from exc
    issues = bundle.validate()
    if issues:
        raise ValueError(
            "invalid tokenizer bundle:\n  - " + "\n  - ".join(issues)
        )

    role_paths: list[tuple[str, Path]] = [
        ("config", config_path.resolve()),
        ("vocab", vocab_path.resolve()),
        (
            "morphbpe",
            _resolved_bundle_member(
                root,
                bundle.config.morphbpe_file,
                role="morphbpe",
            ),
        ),
    ]
    if bundle.config.general_file:
        role_paths.append(
            (
                "general",
                _resolved_bundle_member(
                    root,
                    bundle.config.general_file,
                    role="general",
                ),
            )
        )
    role_paths.append(("manifest", manifest_path.resolve()))
    files = [
        {
            "role": role,
            "name": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for role, path in role_paths
    ]
    return {
        "schema_version": TOKENIZER_BUNDLE_CONTRACT_VERSION,
        "files": files,
        "files_canonical_sha256": _canonical_json_sha256(files),
    }


def tokenizer_algorithm_contract() -> dict[str, object]:
    """Hash every implementation file that can change text-to-id routing."""

    root = Path(__file__).resolve().parents[2]
    files: dict[str, str] = {}
    for relative in _ALGORITHM_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"tokenizer algorithm source file is missing: {path}"
            )
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        files[relative] = digest.hexdigest()
    files_digest = hashlib.sha256(
        json.dumps(
            files,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    try:
        tokenizers_version = importlib.metadata.version("tokenizers")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "tokenizers runtime is missing; tokenizer algorithm identity "
            "cannot be established"
        ) from exc
    return {
        "contract_version": TOKENIZER_ALGORITHM_CONTRACT_VERSION,
        "files": files,
        "files_canonical_sha256": files_digest,
        "runtime": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "python_cache_tag": sys.implementation.cache_tag,
            "unicode_database_version": unicodedata.unidata_version,
            "tokenizers_version": tokenizers_version,
        },
        "special_token_interpretation": "enabled",
        "mongolian_general_fallback": "observable",
    }


__all__ = [
    "TOKENIZER_ALGORITHM_CONTRACT_VERSION",
    "TOKENIZER_BUNDLE_CONTRACT_VERSION",
    "tokenizer_algorithm_contract",
    "tokenizer_bundle_contract",
]
