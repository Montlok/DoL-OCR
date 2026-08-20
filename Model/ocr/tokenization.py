# -*- coding: utf-8 -*-

"""Shared OCR tokenization contract.

Visual pretraining, OCR reinforcement learning, validation, and final
evaluation must supervise the same token representation.  In particular, a
frozen language model must never be trained against an opportunistic byte
fallback when its pretrained representation is available.

This module is the single owner of that contract.  CLI scripts may re-export
the helpers for compatibility, but must not carry private copies.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from Tokenizer.generic_bpe import encode_byte_fallback
from Tokenizer.pretraining.morphology import (
    MORPH_TRACK_GENERAL,
    MORPH_TRACK_MONGOLIAN,
    MORPH_TRACK_RESET,
    derive_morph_info_from_track_ids,
    derive_morph_info_from_tokens,
)
from Tokenizer.unified.contract import (
    tokenizer_algorithm_contract,
    tokenizer_bundle_contract,
)
from Tokenizer.unified.dual_tokenizer import contextual_char_lang
from Tokenizer.unified.vocab import SPECIAL_TOKENS, make_byte_tokens

OCR_TARGET_ENCODING_MODES = frozenset(
    {"native", "native_fallback", "byte_fallback"}
)
OCR_NATIVE_TARGET_ENCODING = "native"
OCR_TOKENIZATION_CONTRACT_VERSION = 3
OCR_REFERENCE_CANONICALIZATION = "native-pretraining-v3"
OCR_LM_INPUT_REPRESENTATION_VERSION = 1
OCR_ADAPTER_ALGORITHM_VERSION = 1
@dataclass(frozen=True)
class OCRTextEncoding:
    """One lossless text encoding plus its pretrained morphology features."""

    input_ids: list[int]
    morphology_track_ids: list[int]
    word_pos: list[int]
    morph_depth: list[int]


def _is_word_surface(text: str) -> bool:
    return bool(text) and any(
        unicodedata.category(ch)[0] in {"L", "N"} for ch in text
    )


def canonical_morphology_track_for_id(
    tokenizer,
    token_id: int,
    *,
    mn_global_ids: set[int] | None = None,
) -> int:
    """Map one vocabulary id to the deterministic decode-time word track."""

    token_id = int(token_id)
    token = str(tokenizer.id_to_token.get(token_id, ""))
    if token_id < 256 or token in SPECIAL_TOKENS:
        return MORPH_TRACK_RESET

    general_local = tokenizer.general_global_to_local.get(token_id)
    if general_local is not None:
        surface = tokenizer.general.decode([int(general_local)])
        return (
            MORPH_TRACK_GENERAL
            if _is_word_surface(surface)
            else MORPH_TRACK_RESET
        )

    if mn_global_ids is None:
        mn_global_ids = {
            int(value) for value in tokenizer.mn_local_to_global.values()
        }
    if token_id in mn_global_ids:
        return MORPH_TRACK_MONGOLIAN
    return MORPH_TRACK_RESET


def tokenizer_morphology_track_table(tokenizer) -> tuple[int, ...]:
    """Return the cached fixed id->word-track table.

    The table is vocabulary-sized and is consulted for every encoded OCR
    target.  Keep it immutable so cache hits can return the exact same object
    instead of copying tens of thousands of entries per sample.
    """

    cached = getattr(tokenizer, "_ocr_morphology_track_table_v1", None)
    if isinstance(cached, tuple):
        return cached
    extent = max(int(idx) for idx in tokenizer.id_to_token) + 1
    table = [MORPH_TRACK_RESET] * extent
    for global_id in tokenizer.mn_local_to_global.values():
        global_id = int(global_id)
        if 256 <= global_id < extent:
            table[global_id] = MORPH_TRACK_MONGOLIAN

    general_items = sorted(
        (
            int(global_id),
            int(local_id),
        )
        for global_id, local_id in tokenizer.general_global_to_local.items()
        if 256 <= int(global_id) < extent
    )
    local_batches = [[local_id] for _global_id, local_id in general_items]
    native = getattr(tokenizer.general, "_tk", None)
    if native is not None and hasattr(native, "decode_batch"):
        surfaces = native.decode_batch(
            local_batches,
            skip_special_tokens=False,
        )
    else:
        surfaces = [
            tokenizer.general.decode(local_ids) for local_ids in local_batches
        ]
    for (global_id, _local_id), surface in zip(general_items, surfaces):
        table[global_id] = (
            MORPH_TRACK_GENERAL
            if _is_word_surface(surface)
            else MORPH_TRACK_RESET
        )
    for _token, token_id in SPECIAL_TOKENS.items():
        if 0 <= int(token_id) < extent:
            table[int(token_id)] = MORPH_TRACK_RESET
    frozen_table = tuple(table)
    tokenizer._ocr_morphology_track_table_v1 = frozen_table
    return frozen_table


def tokenizer_morphology_track_sha256(tokenizer) -> str:
    table = tokenizer_morphology_track_table(tokenizer)
    return hashlib.sha256(bytes(table)).hexdigest()


def encode_lm_text_features(
    tokenizer,
    text: str,
    *,
    interpret_special_tokens: bool,
) -> OCRTextEncoding:
    """Encode text and prove id-only morphology matches pretraining spans."""

    encoded = tokenizer.encode_with_spans(
        text,
        add_bos=False,
        add_eos=False,
        interpret_special_tokens=interpret_special_tokens,
    )
    return _lm_features_from_encoded(tokenizer, encoded)


def _lm_features_from_encoded(tokenizer, encoded) -> OCRTextEncoding:
    ids = [int(value) for value in encoded.input_ids]
    table = tokenizer_morphology_track_table(tokenizer)
    track_ids = [table[token_id] for token_id in ids]
    span_word_pos, span_morph_depth = derive_morph_info_from_tokens(
        encoded.tokens
    )
    word_pos, morph_depth = derive_morph_info_from_track_ids(track_ids)
    if (word_pos, morph_depth) != (span_word_pos, span_morph_depth):
        raise ValueError(
            "token route has ambiguous morphology: the id-only generation "
            "representation differs from the pretrained span representation"
        )
    return OCRTextEncoding(ids, track_ids, word_pos, morph_depth)


def canonicalize_native_ocr_text(text: str) -> str:
    """Apply only the word-boundary folding learned during pretraining.

    Ordinary NBSP and a non-Mongolian NNBSP share the pretrained tokenizer's
    word-boundary id with ASCII space.  Contextual NNBSP inside Mongolian text
    is meaningful and remains byte-exact.  No other code point is normalized:
    FVS1-4, MVS, presentation forms, and literal control-token surfaces are
    preserved.
    """

    chars = list(text)
    changed = False
    for index, ch in enumerate(text):
        if ch not in {"\u00a0", "\u202f"}:
            continue
        if contextual_char_lang(text, index) != "space":
            continue
        chars[index] = " "
        changed = True
    return "".join(chars) if changed else text


def canonical_json_sha256(value: Any) -> str:
    """SHA-256 of a JSON value independent of whitespace/key formatting."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tokenizer_manifest_canonical_sha256(bundle_dir: str | Path) -> str:
    """Hash ``manifest.json`` with the visual-training canonical JSON rule."""

    path = Path(bundle_dir) / "manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"tokenizer bundle has no manifest.json: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid tokenizer manifest JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"tokenizer manifest must be a non-empty JSON object: {path}")
    return canonical_json_sha256(payload)


def tokenizer_vocab_sha256(tokenizer) -> str:
    """Fingerprint the complete token-to-id mapping, not only vocab size."""

    payload = json.dumps(
        sorted((str(token), int(idx)) for token, idx in tokenizer.vocab.items()),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ocr_adapter_algorithm_contract() -> dict[str, Any]:
    """Fingerprint the OCR-only adapter layered over pretraining tokenization."""

    source_path = Path(__file__)
    digest = hashlib.sha256()
    with source_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "version": OCR_ADAPTER_ALGORITHM_VERSION,
        "tokenization_contract_version": OCR_TOKENIZATION_CONTRACT_VERSION,
        "source_file": "Model/ocr/tokenization.py",
        "source_sha256": digest.hexdigest(),
    }


def make_ocr_target_encoder(tokenizer, *, mode: str = OCR_NATIVE_TARGET_ENCODING):
    """Build a verified OCR-target encoder.

    ``native`` is the production contract for a frozen language model.  It uses
    the tokenizer's pretrained path and fails on ``<unk>`` or any round-trip
    change beyond :func:`canonicalize_native_ocr_text`.

    ``native_fallback`` and ``byte_fallback`` remain explicit conversion tools
    for experiments that also train the language side.  They are never selected
    implicitly.
    """

    if mode not in OCR_TARGET_ENCODING_MODES:
        raise ValueError(
            f"unknown OCR target encoding mode {mode!r}; expected one of "
            f"{sorted(OCR_TARGET_ENCODING_MODES)}"
        )

    unk_id = tokenizer.unk_id
    stats = {
        "native": 0,
        "byte_fallback": 0,
        "canonicalized": 0,
        "rejected_mn_general_fallback": 0,
    }

    def _verify(ids: list[int], text: str, *, route: str) -> None:
        if unk_id in ids:
            raise ValueError(
                f"{route} OCR target encoding produced <unk> (id={unk_id}) "
                f"for text={text!r}"
            )
        decoded = tokenizer.decode(ids)
        if decoded != text:
            raise ValueError(
                f"{route} OCR target round-trip mismatch (str): "
                f"decode(encode(text)) != text\n  text={text!r}\n"
                f"  decoded={decoded!r}"
            )
        want_bytes = text.encode("utf-8", "surrogatepass")
        got_bytes = decoded.encode("utf-8", "surrogatepass")
        if got_bytes != want_bytes:
            raise ValueError(
                f"{route} OCR target round-trip mismatch (utf-8 bytes): "
                f"text={text!r} bytes={want_bytes!r}\n"
                f"  decoded={decoded!r} bytes={got_bytes!r}"
            )

    byte_tokens = make_byte_tokens()
    safe_vocab: dict[str, int] | None = None
    if mode != OCR_NATIVE_TARGET_ENCODING:
        missing = [token for token in byte_tokens if token not in tokenizer.vocab]
        if missing:
            raise ValueError(
                f"tokenizer vocab is missing {len(missing)}/256 byte-fallback "
                f"tokens (e.g. {missing[0]!r}); fallback was requested explicitly "
                "but this bundle cannot provide it losslessly"
            )
        safe_vocab = {
            token: idx
            for token, idx in tokenizer.vocab.items()
            if len(token) == 1
            and idx not in tokenizer.general_global_to_local
            and token not in SPECIAL_TOKENS
        }
        for token in byte_tokens:
            safe_vocab[token] = tokenizer.vocab[token]

    def _native(text: str) -> tuple[OCRTextEncoding, bool]:
        expected = (
            canonicalize_native_ocr_text(text)
            if mode == OCR_NATIVE_TARGET_ENCODING
            else text
        )
        try:
            route_encoder = getattr(tokenizer, "encode_with_spans", None)
            if not callable(route_encoder):
                raise TypeError(
                    "strict native OCR requires encode_with_spans route metadata"
                )
            encoded = route_encoder(
                expected,
                add_bos=False,
                add_eos=False,
                interpret_special_tokens=True,
            )
            route_tokens = getattr(encoded, "tokens", None)
            ids = list(getattr(encoded, "input_ids", ()))
            if route_tokens is None:
                raise TypeError(
                    "encode_with_spans result has no auditable token routes"
                )
            fallback_tokens = [
                token
                for token in route_tokens
                if getattr(token, "track", "") == "mn_general_fallback"
            ]
            if fallback_tokens:
                stats["rejected_mn_general_fallback"] += 1
                surfaces = "".join(
                    str(
                        getattr(token, "surface", None)
                        or getattr(token, "token", "")
                    )
                    for token in fallback_tokens
                )
                raise ValueError(
                    "MorphBPE missed Mongolian source text and would silently "
                    f"route it through the general track: {surfaces!r}"
                )
            features = _lm_features_from_encoded(tokenizer, encoded)
        except Exception as exc:
            raise ValueError(
                f"native OCR target encoding failed for text={text!r}: {exc}"
            ) from exc
        _verify(ids, expected, route="native")
        return features, expected != text

    def _byte_fallback(text: str) -> OCRTextEncoding:
        assert safe_vocab is not None
        encoded = encode_byte_fallback(text, safe_vocab, unk_id)
        ids = [int(token.id) for token in encoded]
        _verify(ids, text, route="byte-fallback")
        table = tokenizer_morphology_track_table(tokenizer)
        track_ids = [table[token_id] for token_id in ids]
        word_pos, morph_depth = derive_morph_info_from_track_ids(track_ids)
        return OCRTextEncoding(
            ids,
            track_ids,
            word_pos,
            morph_depth,
        )

    def encode_with_features(text: str) -> OCRTextEncoding:
        if mode == "byte_fallback":
            features = _byte_fallback(text)
            stats["byte_fallback"] += 1
            return features
        try:
            features, canonicalized = _native(text)
        except ValueError:
            if mode == OCR_NATIVE_TARGET_ENCODING:
                raise
            features = _byte_fallback(text)
            stats["byte_fallback"] += 1
            return features
        stats["native"] += 1
        stats["canonicalized"] += int(canonicalized)
        return features

    def encode_target(text: str) -> list[int]:
        return encode_with_features(text).input_ids

    encode_target.mode = mode  # type: ignore[attr-defined]
    encode_target.stats = stats  # type: ignore[attr-defined]
    encode_target.encode_with_features = encode_with_features  # type: ignore[attr-defined]
    return encode_target


def native_tokenization_contract(tokenizer, bundle_dir: str | Path) -> dict[str, Any]:
    """Return the immutable contract persisted in OCR-GRPO artifacts."""

    morphology_table = tokenizer_morphology_track_table(tokenizer)
    return {
        "target_encoding": OCR_NATIVE_TARGET_ENCODING,
        "tokenization_contract_version": OCR_TOKENIZATION_CONTRACT_VERSION,
        "tokenizer_manifest_canonical_sha256": (
            tokenizer_manifest_canonical_sha256(bundle_dir)
        ),
        "tokenizer_vocab_sha256": tokenizer_vocab_sha256(tokenizer),
        "tokenizer_bundle": tokenizer_bundle_contract(bundle_dir),
        "pretraining_tokenizer_algorithm": tokenizer_algorithm_contract(),
        "ocr_adapter_algorithm": ocr_adapter_algorithm_contract(),
        "lm_input_representation": {
            "version": OCR_LM_INPUT_REPRESENTATION_VERSION,
            "morphology": "span-verified-id-track-v1",
            "track_ids": {
                "reset": MORPH_TRACK_RESET,
                "mongolian_word": MORPH_TRACK_MONGOLIAN,
                "general_word": MORPH_TRACK_GENERAL,
            },
            "track_table_size": len(morphology_table),
            "track_table_sha256": hashlib.sha256(
                bytes(morphology_table)
            ).hexdigest(),
            "producer_fields": ["word_pos", "morph_depth"],
            "runtime_generation": "token-id-track-table",
        },
        "native_route": "dual-track-pretraining-special-aware",
        "mongolian_general_fallback": "forbidden",
        "reference_canonicalization": OCR_REFERENCE_CANONICALIZATION,
    }


__all__ = [
    "OCR_ADAPTER_ALGORITHM_VERSION",
    "OCR_LM_INPUT_REPRESENTATION_VERSION",
    "OCR_NATIVE_TARGET_ENCODING",
    "OCR_REFERENCE_CANONICALIZATION",
    "OCR_TARGET_ENCODING_MODES",
    "OCR_TOKENIZATION_CONTRACT_VERSION",
    "OCRTextEncoding",
    "canonical_morphology_track_for_id",
    "canonical_json_sha256",
    "canonicalize_native_ocr_text",
    "encode_lm_text_features",
    "make_ocr_target_encoder",
    "native_tokenization_contract",
    "ocr_adapter_algorithm_contract",
    "tokenizer_manifest_canonical_sha256",
    "tokenizer_morphology_track_sha256",
    "tokenizer_morphology_track_table",
    "tokenizer_vocab_sha256",
]
