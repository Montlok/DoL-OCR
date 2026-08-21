# -*- coding: utf-8 -*-
"""Persisted tokenizer bundle for reproducible pretraining data builds."""

from __future__ import annotations

import json
import os
import shutil
import warnings
import hashlib
from dataclasses import asdict, dataclass, fields
from typing import Any

from Tokenizer.generic_bpe import GeneralBPEModel
from Tokenizer.morphbpe import MorphBPETokenizer
from Tokenizer.multimodal import MultimodalProcessor
from Tokenizer.traditional_mongolian.stemmer import MongolStemmer

from .dual_tokenizer import DualTrackTokenizer
from .vocab import SPECIAL_TOKENS, build_unified_vocab


BUNDLE_VERSION = 2
CONFIG_NAME = "config.json"
MORPHBPE_NAME = "morphbpe.json"
GENERAL_NAME = "general.json"
VOCAB_NAME = "vocab.json"
MANIFEST_NAME = "manifest.json"


@dataclass
class TokenizerBundleConfig:
    version: int
    morphbpe_file: str
    general_file: str = ""
    patch_size: int = 14
    merge_size: int = 2
    temporal_patch_size: int = 2


# Config keys written by pre-v2 bundles that no longer map to any field. They
# are dropped (with a warning) when loading so old bundles still open.
_LEGACY_CONFIG_KEYS = {"zh_source", "en_source", "use_smoke_hf"}


def _config_from_raw(raw: dict) -> "TokenizerBundleConfig":
    """Build a config from a possibly-legacy ``config.json`` dict.

    Older bundles stored ``zh_source``/``en_source``/``use_smoke_hf``; those
    keys are ignored so previously generated bundles keep loading after the
    schema narrowed. Genuinely unknown keys still raise.
    """

    known = {f.name for f in fields(TokenizerBundleConfig)}
    unknown = set(raw) - known
    legacy = unknown & _LEGACY_CONFIG_KEYS
    if legacy:
        warnings.warn(
            "ignoring legacy tokenizer-bundle config keys "
            f"{sorted(legacy)} from a pre-v2 bundle; they are no longer used.",
            stacklevel=2,
        )
    other_unknown = unknown - _LEGACY_CONFIG_KEYS
    if other_unknown:
        raise TypeError(
            f"unknown tokenizer-bundle config keys: {sorted(other_unknown)}"
        )
    return TokenizerBundleConfig(**{k: v for k, v in raw.items() if k in known})


class TokenizerBundle:
    tokenizer: DualTrackTokenizer
    processor: MultimodalProcessor
    config: TokenizerBundleConfig

    def __init__(
        self,
        tokenizer: DualTrackTokenizer,
        processor: MultimodalProcessor,
        config: TokenizerBundleConfig,
        bundle_dir: str | None = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.bundle_dir = bundle_dir

    @classmethod
    def from_files(
        cls,
        morphbpe_path: str,
        general_path: str | None = None,
        patch_size: int = 14,
        merge_size: int = 2,
        temporal_patch_size: int = 2,
    ) -> "TokenizerBundle":
        config = TokenizerBundleConfig(
            version=BUNDLE_VERSION,
            morphbpe_file=morphbpe_path,
            general_file=general_path or "",
            patch_size=patch_size,
            merge_size=merge_size,
            temporal_patch_size=temporal_patch_size,
        )
        return cls._build(config, morphbpe_path=morphbpe_path, vocab=None)

    @classmethod
    def from_dir(cls, path: str) -> "TokenizerBundle":
        config_path = os.path.join(path, CONFIG_NAME)
        vocab_path = os.path.join(path, VOCAB_NAME)
        with open(config_path, "r", encoding="utf-8") as f:
            config = _config_from_raw(json.load(f))
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = {str(token): int(idx) for token, idx in json.load(f).items()}
        morphbpe_path = os.path.join(path, config.morphbpe_file)
        general_path = (
            os.path.join(path, config.general_file) if config.general_file else None
        )
        bundle = cls._build(
            config,
            morphbpe_path=morphbpe_path,
            vocab=vocab,
            general_path=general_path,
        )
        if os.path.exists(os.path.join(path, MANIFEST_NAME)):
            bundle.bundle_dir = path
        return bundle

    @classmethod
    def _build(
        cls,
        config: TokenizerBundleConfig,
        morphbpe_path: str,
        vocab: dict[str, int] | None,
        general_path: str | None = None,
    ) -> "TokenizerBundle":
        stemmer = MongolStemmer()
        morphbpe = MorphBPETokenizer.from_file(morphbpe_path, stemmer)

        gen_path = general_path if general_path is not None else config.general_file
        if gen_path:
            general = GeneralBPEModel.load(gen_path)
        else:
            general = GeneralBPEModel.minimal()

        if vocab is None:
            vocab = build_unified_vocab(morphbpe.vocab, general.get_vocab())

        tokenizer = DualTrackTokenizer(vocab, morphbpe, general)
        processor = MultimodalProcessor(
            tokenizer,
            patch_size=config.patch_size,
            merge_size=config.merge_size,
            temporal_patch_size=config.temporal_patch_size,
        )
        return cls(tokenizer=tokenizer, processor=processor, config=config)

    def save_dir(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

        dest_morphbpe = os.path.join(path, MORPHBPE_NAME)
        source_morphbpe = self.config.morphbpe_file
        if os.path.abspath(source_morphbpe) != os.path.abspath(dest_morphbpe):
            if os.path.exists(source_morphbpe):
                shutil.copyfile(source_morphbpe, dest_morphbpe)
            else:
                self.tokenizer.morphbpe.save(dest_morphbpe)

        dest_general = os.path.join(path, GENERAL_NAME)
        source_general = self.config.general_file
        if (
            source_general
            and os.path.exists(source_general)
            and os.path.abspath(source_general) != os.path.abspath(dest_general)
        ):
            shutil.copyfile(source_general, dest_general)
        else:
            self.tokenizer.general.save(dest_general)

        config = TokenizerBundleConfig(
            version=self.config.version,
            morphbpe_file=MORPHBPE_NAME,
            general_file=GENERAL_NAME,
            patch_size=self.config.patch_size,
            merge_size=self.config.merge_size,
            temporal_patch_size=self.config.temporal_patch_size,
        )
        with open(os.path.join(path, CONFIG_NAME), "w", encoding="utf-8") as f:
            json.dump(asdict(config), f, ensure_ascii=False, indent=2)
        with open(os.path.join(path, VOCAB_NAME), "w", encoding="utf-8") as f:
            json.dump(self.tokenizer.vocab, f, ensure_ascii=False, indent=2)
        _write_manifest(path, config)
        self.bundle_dir = path

    def encode(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> list[int]:
        return self.tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)

    def encode_with_spans(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ):
        return self.tokenizer.encode_with_spans(text, add_bos=add_bos, add_eos=add_eos)

    def encode_multimodal(
        self,
        text: str,
        images=None,
        image_sizes=None,
        videos=None,
        video_sizes=None,
        add_bos: bool = False,
        add_eos: bool = False,
    ):
        return self.processor(
            text,
            images=images,
            image_sizes=image_sizes,
            videos=videos,
            video_sizes=video_sizes,
            add_bos=add_bos,
            add_eos=add_eos,
        )

    def validate(self) -> list[str]:
        issues: list[str] = []
        if self.config.version != BUNDLE_VERSION:
            issues.append(f"unsupported config version: {self.config.version}")
        if not self.config.morphbpe_file:
            issues.append("config.morphbpe_file is empty")
        values = list(self.tokenizer.vocab.values())
        if len(values) != len(set(values)):
            issues.append("vocab contains duplicate ids")
        for token, expected_id in SPECIAL_TOKENS.items():
            actual = self.tokenizer.vocab.get(token)
            if actual != expected_id:
                issues.append(
                    f"special token {token!r} has id {actual}, expected {expected_id}"
                )
        if "<unk>" not in self.tokenizer.morphbpe.vocab:
            issues.append("morphbpe vocab is missing <unk>")
        try:
            text = "\u182e\u1822\u1828\u182d\u1822\u182f 文字 test \U0001f642"
            result = self.encode_with_spans(text, add_bos=True, add_eos=True)
            if len(result.input_ids) != len(result.tokens):
                issues.append("encode_with_spans produced mismatched ids/tokens")
        except Exception as exc:  # pragma: no cover - reported as validation issue.
            issues.append(f"encode smoke failed: {exc}")
        try:
            mm = self.encode_multimodal(
                "文字 <image> test",
                images=["smoke-image"],
                image_sizes=[(14, 14)],
            )
            if len(mm.input_ids) != len(mm.attention_mask):
                issues.append("multimodal smoke produced mismatched ids/mask")
            if not mm.image_token_spans:
                issues.append("multimodal smoke produced no image span")
        except Exception as exc:  # pragma: no cover - reported as validation issue.
            issues.append(f"multimodal smoke failed: {exc}")
        if self.bundle_dir:
            issues.extend(validate_manifest(self.bundle_dir))
        return issues


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(path: str, config: TokenizerBundleConfig) -> None:
    files = {}
    for name in (CONFIG_NAME, MORPHBPE_NAME, GENERAL_NAME, VOCAB_NAME):
        file_path = os.path.join(path, name)
        if os.path.exists(file_path):
            files[name] = _sha256_file(file_path)
    manifest = {
        "version": BUNDLE_VERSION,
        "files": files,
        "sources": {
            "morphbpe_file": config.morphbpe_file,
            "general_file": config.general_file,
        },
        "multimodal": {
            "patch_size": config.patch_size,
            "merge_size": config.merge_size,
            "temporal_patch_size": config.temporal_patch_size,
        },
    }
    with open(os.path.join(path, MANIFEST_NAME), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)


def read_manifest(path: str) -> dict[str, Any]:
    manifest_path = os.path.join(path, MANIFEST_NAME)
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_manifest(path: str) -> list[str]:
    issues: list[str] = []
    manifest = read_manifest(path)
    if not manifest:
        issues.append(f"{MANIFEST_NAME} is missing")
        return issues
    files = manifest.get("files") or {}
    for name, expected in files.items():
        file_path = os.path.join(path, name)
        if not os.path.exists(file_path):
            issues.append(f"manifest file is missing: {name}")
            continue
        actual = _sha256_file(file_path)
        if actual != expected:
            issues.append(f"manifest hash mismatch for {name}")
    return issues
