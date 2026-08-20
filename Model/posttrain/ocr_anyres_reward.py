# -*- coding: utf-8 -*-
"""Pure, raw-Unicode OCR rewards for anyres policy optimization."""

from __future__ import annotations

from collections.abc import Callable, Container, Sequence
from dataclasses import dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from Model.config import SEGMENT, WORD_BOUNDARY_ID
from Model.ocr.metrics import symbol_metrics
from Model.posttrain.rewards import grapheme_cer_value, length_excess_penalty


_REVIEWED_FACTORY_TOKEN = object()


@dataclass(frozen=True)
class AnyresOCRRewardConfig:
    grapheme_cer_weight: float = 1.0
    exact_match_bonus: float = 0.10
    supported_symbol_error_weight: float = 0.20
    valid_eos_bonus: float = 0.05
    missing_eos_penalty: float = 0.10
    invalid_output_penalty: float = 0.20
    length_excess_penalty_weight: float = 0.05
    grounding_margin_weight: float = 0.0
    max_length_ratio: float = 2.0
    grounding_margin_clip_min: float = -1.0
    grounding_margin_clip_max: float = 1.0
    require_tokenizer_roundtrip: bool = True

    def __post_init__(self) -> None:
        nonnegative = (
            "grapheme_cer_weight",
            "exact_match_bonus",
            "supported_symbol_error_weight",
            "valid_eos_bonus",
            "missing_eos_penalty",
            "invalid_output_penalty",
            "length_excess_penalty_weight",
            "grounding_margin_weight",
        )
        for name in nonnegative:
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be a finite non-negative number")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be a finite non-negative number")
        for name in (
            "max_length_ratio",
            "grounding_margin_clip_min",
            "grounding_margin_clip_max",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be finite")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.max_length_ratio < 1.0:
            raise ValueError("max_length_ratio must be at least 1")
        if self.grounding_margin_clip_min > self.grounding_margin_clip_max:
            raise ValueError("grounding margin clip bounds are reversed")
        if self.invalid_output_penalty <= 0:
            raise ValueError("invalid_output_penalty must be strictly positive")
        if type(self.require_tokenizer_roundtrip) is not bool:
            raise ValueError("require_tokenizer_roundtrip must be bool")


_SUPPORTED_SYMBOL_PATHS = (
    ("fvs", "variants", "fvs1"),
    ("fvs", "variants", "fvs2"),
    ("fvs", "variants", "fvs3"),
    ("fvs", "variants", "fvs4"),
    ("mvs",),
    ("nnbsp",),
    ("digit",),
    ("punctuation",),
)


def _strict_unicode_reference(value: object, index: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f"references[{index}] must be a string")
    reason = _invalid_unicode_reason(value)
    if reason is not None:
        raise ValueError(f"references[{index}] contains {reason}")
    return value


def _invalid_unicode_reason(value: str) -> str | None:
    for character in value:
        codepoint = ord(character)
        if character == "\x00":
            return "NUL"
        if character == "\ufffd":
            return "U+FFFD"
        if 0xD800 <= codepoint <= 0xDFFF:
            return "a surrogate code point"
    return None


def _completion_values(value: object, index: int) -> list[int]:
    if isinstance(value, torch.Tensor):
        if value.ndim != 1:
            raise ValueError(f"completion_ids[{index}] must be one-dimensional")
        raw = value.detach().cpu().tolist()
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        raw = list(value)
    else:
        raise TypeError(f"completion_ids[{index}] must be an integer sequence")
    if any(isinstance(token, bool) or not isinstance(token, int) for token in raw):
        raise ValueError(f"completion_ids[{index}] must contain only integers")
    return [int(token) for token in raw]


def _supported_symbol_micro_error(response: str, reference: str) -> tuple[float, int, int]:
    metrics = symbol_metrics([response], [reference])
    errors = 0
    support = 0
    for path in _SUPPORTED_SYMBOL_PATHS:
        row: Any = metrics
        for component in path:
            row = row[component]
        n_ref = int(row["n_ref"])
        if n_ref <= 0:
            continue
        support += n_ref
        errors += int(row["substitutions"])
        errors += int(row["deletions"])
        errors += int(row["insertions"])
    return (0.0 if support == 0 else errors / support), errors, support


def _grounding_values(value: object, count: int) -> list[float]:
    if value is None:
        return [0.0] * count
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            raw = [float(value.item())] * count
        elif value.ndim == 1:
            raw = [float(item) for item in value.detach().cpu().tolist()]
        else:
            raise ValueError("grounding_margin tensor must be scalar or one-dimensional")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        raw = [float(value)] * count
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        raw = [float(item) for item in value]
    else:
        raise TypeError("grounding_margin must be external scalar or sequence")
    if len(raw) != count:
        raise ValueError("grounding_margin length must match the response group")
    if any(not math.isfinite(item) for item in raw):
        raise ValueError("grounding_margin values must be finite")
    return raw


def score_anyres_ocr_group(
    responses: Sequence[str],
    references: Sequence[str],
    completion_ids: Sequence[Sequence[int] | torch.Tensor],
    eos_id: int,
    valid_token_ids: Container[int],
    grounding_margin: float | Sequence[float] | torch.Tensor | None = None,
    *,
    tokenizer_encode: Callable[[str], Sequence[int]] | None = None,
    tokenizer_decode: Callable[[Sequence[int]], str] | None = None,
    config: AnyresOCRRewardConfig | None = None,
) -> dict[str, Any]:
    """Score one response group against raw, lossless transcripts.

    ``grounding_margin`` is never estimated here.  It must be supplied by the
    caller from a separately frozen reference-policy comparison.
    """

    cfg = AnyresOCRRewardConfig() if config is None else config
    if not isinstance(cfg, AnyresOCRRewardConfig):
        raise TypeError("config must be an AnyresOCRRewardConfig")
    responses = list(responses)
    references = list(references)
    completion_ids = list(completion_ids)
    if not responses:
        raise ValueError("response group must not be empty")
    if len(responses) != len(references) or len(responses) != len(completion_ids):
        raise ValueError("responses, references, and completion_ids must align")
    if isinstance(eos_id, bool) or not isinstance(eos_id, int) or eos_id < 0:
        raise ValueError("eos_id must be a non-negative integer")
    try:
        eos_is_valid = eos_id in valid_token_ids
    except TypeError as exc:
        raise TypeError("valid_token_ids must support integer membership") from exc
    if not eos_is_valid:
        raise ValueError("eos_id is absent from valid_token_ids")
    if (tokenizer_encode is None) != (tokenizer_decode is None):
        raise ValueError("tokenizer encode/decode callbacks must be supplied together")
    if tokenizer_encode is not None and not callable(tokenizer_encode):
        raise TypeError("tokenizer_encode must be callable")
    if tokenizer_decode is not None and not callable(tokenizer_decode):
        raise TypeError("tokenizer_decode must be callable")
    if cfg.require_tokenizer_roundtrip and tokenizer_encode is None:
        raise ValueError("production anyres OCR reward requires tokenizer roundtrip callbacks")

    margins = _grounding_values(grounding_margin, len(responses))
    samples: list[dict[str, Any]] = []
    totals: list[float] = []
    special_lo, special_hi = SEGMENT["special"]

    for index, (response, reference, raw_ids, raw_margin) in enumerate(
        zip(responses, references, completion_ids, margins, strict=True)
    ):
        if not isinstance(response, str):
            raise TypeError(f"responses[{index}] must be a string")
        reference = _strict_unicode_reference(reference, index)
        ids = _completion_values(raw_ids, index)
        try:
            eos_position = ids.index(eos_id)
        except ValueError:
            eos_position = None
        content_ids = ids if eos_position is None else ids[:eos_position]
        saw_eos = eos_position is not None
        invalid_reasons: list[str] = []

        unicode_reason = _invalid_unicode_reason(response)
        if unicode_reason is not None:
            invalid_reasons.append(unicode_reason)
        invalid_content_token = False
        for token_id in content_ids:
            if token_id not in valid_token_ids:
                invalid_reasons.append(f"unassigned_token:{token_id}")
                invalid_content_token = True
                continue
            if special_lo <= token_id < special_hi and token_id != WORD_BOUNDARY_ID:
                invalid_reasons.append(f"reserved_token:{token_id}")
                invalid_content_token = True

        if (
            tokenizer_encode is not None
            and tokenizer_decode is not None
            and not invalid_content_token
        ):
            encoded_raw = list(tokenizer_encode(response))
            if any(
                isinstance(token, bool) or not isinstance(token, int)
                for token in encoded_raw
            ):
                raise TypeError("tokenizer encoder returned non-integer IDs")
            encoded = [int(token) for token in encoded_raw]
            decoded = tokenizer_decode(encoded)
            if not isinstance(decoded, str):
                raise TypeError("tokenizer decoder must return str")
            if decoded != response:
                invalid_reasons.append("tokenizer_text_roundtrip")
            if encoded != content_ids:
                invalid_reasons.append("tokenizer_completion_roundtrip")

        raw_cer = grapheme_cer_value(
            response,
            reference,
            normalize=False,
            backend="python",
        )
        symbol_error, symbol_errors, symbol_support = (
            _supported_symbol_micro_error(response, reference)
        )
        is_empty = len(response) == 0
        invalid = bool(invalid_reasons)
        exact = response == reference and not is_empty and not invalid
        length_excess = length_excess_penalty(
            response,
            reference,
            max_ratio=cfg.max_length_ratio,
        )
        clipped_margin = min(
            cfg.grounding_margin_clip_max,
            max(cfg.grounding_margin_clip_min, raw_margin),
        )
        breakdown = {
            "raw_grapheme_cer": -cfg.grapheme_cer_weight * raw_cer,
            "exact_match": cfg.exact_match_bonus if exact else 0.0,
            "supported_symbol_error": (
                -cfg.supported_symbol_error_weight * symbol_error
            ),
            "valid_eos": (
                cfg.valid_eos_bonus
                if saw_eos and not is_empty and not invalid
                else 0.0
            ),
            "missing_eos_or_hit_cap": (
                0.0 if saw_eos else -cfg.missing_eos_penalty
            ),
            "invalid_output": (
                -cfg.invalid_output_penalty if invalid else 0.0
            ),
            "length_excess": (
                -cfg.length_excess_penalty_weight * length_excess
            ),
            "grounding_margin": (
                cfg.grounding_margin_weight * clipped_margin
                if not invalid
                else 0.0
            ),
            "empty_output_guard": 0.0,
        }
        total = float(sum(breakdown.values()))
        if not math.isfinite(total) or any(
            not math.isfinite(float(value)) for value in breakdown.values()
        ):
            raise FloatingPointError("anyres OCR reward produced a non-finite value")
        if is_empty and total > 0.0:
            breakdown["empty_output_guard"] = -total
            total = 0.0
        totals.append(total)
        samples.append(
            {
                "total": total,
                "breakdown": breakdown,
                "diagnostics": {
                    "raw_grapheme_cer": raw_cer,
                    "exact": exact,
                    "saw_eos": saw_eos,
                    "hit_cap": not saw_eos,
                    "invalid": invalid,
                    "invalid_reasons": invalid_reasons,
                    "supported_symbol_errors": symbol_errors,
                    "supported_symbol_reference_count": symbol_support,
                    "supported_symbol_micro_error": symbol_error,
                    "length_excess": length_excess,
                    "grounding_margin_raw": raw_margin,
                    "grounding_margin_clipped": clipped_margin,
                },
            }
        )
    return {
        "group_rewards": torch.tensor(totals, dtype=torch.float32),
        "samples": samples,
        "config": {field.name: getattr(cfg, field.name) for field in fields(cfg)},
    }


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AnyresOCRRewardAdapter:
    """The sole production bridge from sampled token IDs to OCR rewards."""

    def __init__(
        self,
        *,
        eos_id: int,
        valid_token_ids: Sequence[int],
        tokenizer_encode: Callable[[str], Sequence[int]],
        tokenizer_decode: Callable[[Sequence[int]], str],
        tokenizer_contract_sha256: str,
        config: AnyresOCRRewardConfig | None = None,
        _factory_token: object | None = None,
    ) -> None:
        if _factory_token is not _REVIEWED_FACTORY_TOKEN:
            raise ValueError(
                "production reward adapters must be built from a reviewed "
                "TokenizerBundle"
            )
        if isinstance(eos_id, bool) or not isinstance(eos_id, int) or eos_id < 0:
            raise ValueError("eos_id must be a non-negative integer")
        ids = tuple(int(value) for value in valid_token_ids)
        if (
            not ids
            or len(ids) != len(set(ids))
            or any(value < 0 for value in ids)
        ):
            raise ValueError("valid_token_ids must be unique non-negative integers")
        if not callable(tokenizer_encode) or not callable(tokenizer_decode):
            raise TypeError("tokenizer callbacks must be callable")
        if (
            not isinstance(tokenizer_contract_sha256, str)
            or len(tokenizer_contract_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in tokenizer_contract_sha256
            )
        ):
            raise ValueError("tokenizer_contract_sha256 must be a lowercase digest")
        self.eos_id = eos_id
        self.valid_token_ids = frozenset(ids)
        if eos_id not in self.valid_token_ids:
            raise ValueError("eos_id is absent from valid_token_ids")
        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode
        self.config = AnyresOCRRewardConfig() if config is None else config
        if not isinstance(self.config, AnyresOCRRewardConfig):
            raise TypeError("config must be an AnyresOCRRewardConfig")
        if self.config.grounding_margin_weight != 0.0:
            raise ValueError(
                "production reward adapter has no frozen grounding-margin "
                "source; grounding_margin_weight must be 0"
            )
        valid_ids_sha = hashlib.sha256(
            b"".join(value.to_bytes(8, "big") for value in sorted(ids))
        ).hexdigest()
        contract_base = {
            "schema_version": 1,
            "kind": "dol_ocr_anyres_reward_adapter_v1",
            "eos_id": eos_id,
            "valid_token_ids_sha256": valid_ids_sha,
            "valid_token_count": len(ids),
            "tokenizer_contract_sha256": tokenizer_contract_sha256,
            "construction": "reviewed_native_tokenizer_bundle_v1",
            "implementation_sha256": _file_sha256(Path(__file__)),
            "config": {
                field.name: getattr(self.config, field.name)
                for field in fields(self.config)
            },
            "grounding_margin": "disabled_no_frozen_control_source_v1",
        }
        self.contract = {
            **contract_base,
            "canonical_sha256": _canonical_sha256(contract_base),
        }

    def decode_completion(self, value: Sequence[int] | torch.Tensor) -> str:
        ids = _completion_values(value, 0)
        try:
            eos_position = ids.index(self.eos_id)
        except ValueError:
            eos_position = len(ids)
        content = ids[:eos_position]
        special_lo, special_hi = SEGMENT["special"]
        pieces: list[str] = []
        valid_segment: list[int] = []

        def flush() -> None:
            if not valid_segment:
                return
            decoded = self.tokenizer_decode(list(valid_segment))
            if not isinstance(decoded, str):
                raise TypeError("tokenizer decoder must return str")
            pieces.append(decoded)
            valid_segment.clear()

        for token_id in content:
            invalid = token_id not in self.valid_token_ids or (
                special_lo <= token_id < special_hi
                and token_id != WORD_BOUNDARY_ID
            )
            if invalid:
                flush()
                pieces.append("\ufffd")
            else:
                valid_segment.append(token_id)
        flush()
        response = "".join(pieces)
        response.encode("utf-8", errors="strict")
        return response

    def score_group(
        self,
        references: Sequence[str],
        completion_ids: torch.Tensor,
        eos_mask: torch.Tensor,
    ) -> dict[str, Any]:
        if completion_ids.ndim != 2 or eos_mask.shape != completion_ids.shape:
            raise ValueError("completion_ids/eos_mask must align as [N,T]")
        expected_eos = completion_ids == self.eos_id
        if not torch.equal(eos_mask.to(dtype=torch.bool), expected_eos):
            raise ValueError("eos_mask differs from completion token IDs")
        responses = [self.decode_completion(row) for row in completion_ids]
        scored = score_anyres_ocr_group(
            responses,
            references,
            completion_ids,
            self.eos_id,
            self.valid_token_ids,
            tokenizer_encode=self.tokenizer_encode,
            tokenizer_decode=self.tokenizer_decode,
            config=self.config,
        )
        return {**scored, "responses": responses}


def build_anyres_ocr_reward_adapter(
    tokenizer_bundle: str | Path | Any,
    *,
    config: AnyresOCRRewardConfig | None = None,
) -> AnyresOCRRewardAdapter:
    """Build the production adapter from the reviewed strict-native bundle."""

    from Model.ocr.tokenization import (
        canonical_json_sha256,
        make_ocr_target_encoder,
        native_tokenization_contract,
    )
    from Tokenizer.unified.bundle import TokenizerBundle

    if isinstance(tokenizer_bundle, (str, Path)):
        bundle = TokenizerBundle.from_dir(str(tokenizer_bundle))
    elif isinstance(tokenizer_bundle, TokenizerBundle):
        bundle = tokenizer_bundle
    else:
        raise TypeError("tokenizer_bundle must be a path or TokenizerBundle")
    issues = bundle.validate()
    if issues:
        raise ValueError("invalid tokenizer bundle: " + "; ".join(issues))
    if not bundle.bundle_dir:
        raise ValueError("reviewed tokenizer bundle must have a persisted bundle_dir")
    bundle_dir = str(Path(bundle.bundle_dir).resolve(strict=True))
    encoder = make_ocr_target_encoder(bundle.tokenizer, mode="native")
    tokenizer_contract = native_tokenization_contract(
        bundle.tokenizer,
        bundle_dir,
    )
    valid_token_ids = sorted(
        {int(value) for value in bundle.tokenizer.vocab.values()}
    )
    from Model.config import EOS_ID

    return AnyresOCRRewardAdapter(
        eos_id=EOS_ID,
        valid_token_ids=valid_token_ids,
        tokenizer_encode=encoder,
        tokenizer_decode=lambda values: bundle.tokenizer.decode(
            [int(value) for value in values]
        ),
        tokenizer_contract_sha256=canonical_json_sha256(tokenizer_contract),
        config=config,
        _factory_token=_REVIEWED_FACTORY_TOKEN,
    )


__all__ = [
    "AnyresOCRRewardAdapter",
    "AnyresOCRRewardConfig",
    "build_anyres_ocr_reward_adapter",
    "score_anyres_ocr_group",
]
