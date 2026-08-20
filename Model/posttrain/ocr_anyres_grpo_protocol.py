# -*- coding: utf-8 -*-

"""Deterministic, KL-independent scheduling for fixed anyres GRPO pilots.

The protocol is deliberately a pure in-memory value.  Building it preflights
an exact copy of the production :class:`OCRQuotaSampler`; it never prepares or
commits a batch on the live sampler.  All 200 rollout attempts are registered
up front so an inactive-reward attempt still consumes its OCR, text, and seed
slots and resume is a direct lookup by rollout-attempt index.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import torch

from Model.ocr.tokenization import canonical_json_sha256
from Model.posttrain.ocr_quota_sampler import OCR_QUOTA_BUCKETS, OCRQuotaSampler


FIXED_KL_PILOT_ROLLOUT_ATTEMPTS = 200
FIXED_KL_PILOT_PROTOCOL_KIND = "dol_ocr_anyres_grpo_kl_pilot_protocol_v1"
FIXED_KL_PILOT_SEED_DERIVATION = "sha256-domain-attempt-u63-v1"
FIXED_KL_PILOT_TEXT_SCHEDULE = "registered-order-contiguous-cyclic-v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_SEED = (1 << 63) - 1
_WRAPPER_KEYS = {"payload", "canonical_sha256"}
_PAYLOAD_KEYS = {
    "schema_version",
    "kind",
    "rollout_attempts",
    "global_batch_size",
    "text_batch_size",
    "world_size",
    "base_seed",
    "ocr_sampler_config_sha256",
    "ocr_sample_list_sha256",
    "ocr_dataset_contract_sha256",
    "ocr_sampler_start_state_sha256",
    "text_dataset_contract_sha256",
    "text_sample_count",
    "text_sample_list_sha256",
    "text_schedule_contract",
    "seed_derivation_contract",
    "attempts",
    "ocr_prompt_schedule_sha256",
    "text_batch_schedule_sha256",
    "rollout_seed_schedule_sha256",
}
_ATTEMPT_KEYS = {
    "attempt_index",
    "ocr_prompt_ids",
    "text_sample_ids",
    "cpu_seed",
    "cuda_seed",
    "sampler_state_before_sha256",
    "sampler_state_after_sha256",
}


def build_fixed_kl_pilot_protocol(
    *,
    sampler: OCRQuotaSampler,
    ocr_dataset_contract_sha256: str,
    text_sample_ids: Sequence[str],
    text_dataset_contract_sha256: str,
    text_batch_size: int,
    base_seed: int,
) -> dict[str, object]:
    """Pre-register the exact 200-attempt OCR/text/seed pilot schedule.

    ``kl_coef`` is intentionally absent from this API and from every seed
    derivation input.  Separate KL trials built from independent samplers with
    the same bindings therefore receive byte-identical schedules.
    """

    _require_fresh_sampler(sampler)
    _positive_int(text_batch_size, "text_batch_size")
    _nonnegative_seed(base_seed, "base_seed")
    ocr_dataset_hash = _sha256(
        ocr_dataset_contract_sha256,
        "ocr_dataset_contract_sha256",
    )
    text_dataset_hash = _sha256(
        text_dataset_contract_sha256,
        "text_dataset_contract_sha256",
    )
    registered_text_ids = _validated_sample_ids(
        text_sample_ids,
        field="text_sample_ids",
    )

    if sampler.world_size != 1:
        raise ValueError("fixed KL pilot protocol requires world_size=1")
    if sampler.global_batch_size % 20 != 0:
        raise ValueError("global_batch_size must be divisible by 20")
    if sampler.pending_global_batch is not None:
        raise ValueError("sampler must be at a committed attempt boundary")

    live_state_before = deepcopy(sampler.state_dict())
    live_state_before_sha256 = canonical_json_sha256(live_state_before)
    planning_sampler = deepcopy(sampler)
    attempts: list[dict[str, object]] = []
    for attempt_index in range(FIXED_KL_PILOT_ROLLOUT_ATTEMPTS):
        sampler_state_before_sha256 = canonical_json_sha256(
            planning_sampler.state_dict()
        )
        ocr_prompt_ids = list(planning_sampler.prepare_global_batch())
        planning_sampler.commit_global_batch()
        sampler_state_after_sha256 = canonical_json_sha256(
            planning_sampler.state_dict()
        )
        text_batch = _text_batch_for_attempt(
            registered_text_ids,
            text_batch_size=text_batch_size,
            attempt_index=attempt_index,
        )
        attempts.append(
            {
                "attempt_index": attempt_index,
                "ocr_prompt_ids": ocr_prompt_ids,
                "text_sample_ids": list(text_batch),
                "cpu_seed": _attempt_seed(
                    base_seed,
                    attempt_index,
                    domain="cpu",
                ),
                "cuda_seed": _attempt_seed(
                    base_seed,
                    attempt_index,
                    domain="cuda",
                ),
                "sampler_state_before_sha256": sampler_state_before_sha256,
                "sampler_state_after_sha256": sampler_state_after_sha256,
            }
        )

    if sampler.state_dict() != live_state_before:
        raise RuntimeError("fixed KL pilot planning mutated the live sampler")

    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": FIXED_KL_PILOT_PROTOCOL_KIND,
        "rollout_attempts": FIXED_KL_PILOT_ROLLOUT_ATTEMPTS,
        "global_batch_size": sampler.global_batch_size,
        "text_batch_size": text_batch_size,
        "world_size": sampler.world_size,
        "base_seed": base_seed,
        "ocr_sampler_config_sha256": sampler.config_sha256,
        "ocr_sample_list_sha256": sampler.sample_list_sha256,
        "ocr_dataset_contract_sha256": ocr_dataset_hash,
        "ocr_sampler_start_state_sha256": live_state_before_sha256,
        "text_dataset_contract_sha256": text_dataset_hash,
        "text_sample_count": len(registered_text_ids),
        "text_sample_list_sha256": canonical_json_sha256(
            list(registered_text_ids)
        ),
        "text_schedule_contract": FIXED_KL_PILOT_TEXT_SCHEDULE,
        "seed_derivation_contract": FIXED_KL_PILOT_SEED_DERIVATION,
        "attempts": attempts,
        "ocr_prompt_schedule_sha256": _ocr_schedule_sha256(attempts),
        "text_batch_schedule_sha256": _text_schedule_sha256(attempts),
        "rollout_seed_schedule_sha256": _seed_schedule_sha256(attempts),
    }
    wrapper = {
        "payload": payload,
        "canonical_sha256": canonical_json_sha256(payload),
    }
    validate_fixed_kl_pilot_protocol(
        wrapper,
        sampler=sampler,
        ocr_dataset_contract_sha256=ocr_dataset_hash,
        text_sample_ids=registered_text_ids,
        text_dataset_contract_sha256=text_dataset_hash,
        sampler_attempt_index=0,
    )
    return wrapper


def rebuild_and_match_fixed_kl_pilot_protocol(
    saved: Mapping[str, object],
    *,
    fresh_sampler: OCRQuotaSampler,
    ocr_dataset_contract_sha256: str,
    text_sample_ids: Sequence[str],
    text_dataset_contract_sha256: str,
    text_batch_size: int,
    base_seed: int,
) -> dict[str, Any]:
    """Authenticate a saved protocol by deterministic, full reconstruction.

    A canonical digest proves only that a payload is internally self-consistent:
    a writer could otherwise edit prompt IDs and recompute every digest.  This
    entrypoint instead rebuilds all 200 rows from a genuinely fresh production
    sampler and the registered text IDs, then requires exact wrapper equality.
    """

    _require_fresh_sampler(fresh_sampler)
    saved_payload = validate_fixed_kl_pilot_protocol(
        saved,
        sampler=fresh_sampler,
        ocr_dataset_contract_sha256=ocr_dataset_contract_sha256,
        text_sample_ids=text_sample_ids,
        text_dataset_contract_sha256=text_dataset_contract_sha256,
        sampler_attempt_index=0,
    )
    rebuilt = build_fixed_kl_pilot_protocol(
        sampler=fresh_sampler,
        ocr_dataset_contract_sha256=ocr_dataset_contract_sha256,
        text_sample_ids=text_sample_ids,
        text_dataset_contract_sha256=text_dataset_contract_sha256,
        text_batch_size=text_batch_size,
        base_seed=base_seed,
    )
    if saved != rebuilt:
        raise ValueError(
            "saved fixed KL pilot protocol differs from deterministic rebuild"
        )
    return saved_payload


def validate_fixed_kl_pilot_protocol(
    protocol: Mapping[str, object],
    *,
    sampler: OCRQuotaSampler | None = None,
    ocr_dataset_contract_sha256: str | None = None,
    text_sample_ids: Sequence[str] | None = None,
    text_dataset_contract_sha256: str | None = None,
    sampler_attempt_index: int | None = None,
) -> dict[str, Any]:
    """Validate structure and, when supplied, all live dataset bindings.

    ``sampler_attempt_index`` verifies that a checkpoint-restored live sampler
    is at the exact committed boundary for the next rollout attempt.  Index
    ``200`` denotes the terminal boundary after all scheduled attempts.
    """

    wrapper = _exact_mapping(protocol, _WRAPPER_KEYS, "protocol")
    payload = _exact_mapping(wrapper["payload"], _PAYLOAD_KEYS, "protocol.payload")
    canonical_sha256 = _sha256(
        wrapper["canonical_sha256"],
        "protocol.canonical_sha256",
    )
    if canonical_json_sha256(payload) != canonical_sha256:
        raise ValueError("protocol canonical_sha256 is invalid")

    if payload["schema_version"] != 1:
        raise ValueError("protocol schema_version is incompatible")
    if payload["kind"] != FIXED_KL_PILOT_PROTOCOL_KIND:
        raise ValueError("protocol kind is incompatible")
    if payload["rollout_attempts"] != FIXED_KL_PILOT_ROLLOUT_ATTEMPTS:
        raise ValueError("fixed KL pilot must contain exactly 200 attempts")

    global_batch_size = _positive_int(
        payload["global_batch_size"],
        "global_batch_size",
    )
    if global_batch_size % 20 != 0:
        raise ValueError("global_batch_size must be divisible by 20")
    text_batch_size = _positive_int(payload["text_batch_size"], "text_batch_size")
    if payload["world_size"] != 1:
        raise ValueError("fixed KL pilot protocol requires world_size=1")
    base_seed = _nonnegative_seed(payload["base_seed"], "base_seed")

    for field in (
        "ocr_sampler_config_sha256",
        "ocr_sample_list_sha256",
        "ocr_dataset_contract_sha256",
        "ocr_sampler_start_state_sha256",
        "text_dataset_contract_sha256",
        "text_sample_list_sha256",
        "ocr_prompt_schedule_sha256",
        "text_batch_schedule_sha256",
        "rollout_seed_schedule_sha256",
    ):
        _sha256(payload[field], field)
    text_sample_count = _positive_int(
        payload["text_sample_count"],
        "text_sample_count",
    )
    if payload["text_schedule_contract"] != FIXED_KL_PILOT_TEXT_SCHEDULE:
        raise ValueError("text schedule contract is incompatible")
    if payload["seed_derivation_contract"] != FIXED_KL_PILOT_SEED_DERIVATION:
        raise ValueError("seed derivation contract is incompatible")

    attempts_value = payload["attempts"]
    if not isinstance(attempts_value, list) or len(attempts_value) != (
        FIXED_KL_PILOT_ROLLOUT_ATTEMPTS
    ):
        raise ValueError("protocol attempts must be an exact 200-row list")
    attempts: list[dict[str, Any]] = []
    previous_after_sha256 = payload["ocr_sampler_start_state_sha256"]
    for attempt_index, value in enumerate(attempts_value):
        row = _exact_mapping(
            value,
            _ATTEMPT_KEYS,
            f"attempts[{attempt_index}]",
        )
        if row["attempt_index"] != attempt_index:
            raise ValueError("attempt indexes must be contiguous from zero")
        prompt_ids = _validated_fixed_batch(
            row["ocr_prompt_ids"],
            size=global_batch_size,
            field=f"attempts[{attempt_index}].ocr_prompt_ids",
        )
        text_ids = _validated_fixed_batch(
            row["text_sample_ids"],
            size=text_batch_size,
            field=f"attempts[{attempt_index}].text_sample_ids",
        )
        cpu_seed = _nonnegative_seed(
            row["cpu_seed"],
            f"attempts[{attempt_index}].cpu_seed",
        )
        cuda_seed = _nonnegative_seed(
            row["cuda_seed"],
            f"attempts[{attempt_index}].cuda_seed",
        )
        before_sha256 = _sha256(
            row["sampler_state_before_sha256"],
            f"attempts[{attempt_index}].sampler_state_before_sha256",
        )
        after_sha256 = _sha256(
            row["sampler_state_after_sha256"],
            f"attempts[{attempt_index}].sampler_state_after_sha256",
        )
        if before_sha256 != previous_after_sha256:
            raise ValueError("OCR sampler state chain is discontinuous")
        previous_after_sha256 = after_sha256
        if cpu_seed != _attempt_seed(base_seed, attempt_index, domain="cpu"):
            raise ValueError("CPU attempt seed is not reproducible")
        if cuda_seed != _attempt_seed(base_seed, attempt_index, domain="cuda"):
            raise ValueError("CUDA attempt seed is not reproducible")
        attempts.append(
            {
                "attempt_index": attempt_index,
                "ocr_prompt_ids": list(prompt_ids),
                "text_sample_ids": list(text_ids),
                "cpu_seed": cpu_seed,
                "cuda_seed": cuda_seed,
                "sampler_state_before_sha256": before_sha256,
                "sampler_state_after_sha256": after_sha256,
            }
        )

    if _ocr_schedule_sha256(attempts) != payload["ocr_prompt_schedule_sha256"]:
        raise ValueError("OCR prompt schedule SHA is invalid")
    if _text_schedule_sha256(attempts) != payload["text_batch_schedule_sha256"]:
        raise ValueError("text batch schedule SHA is invalid")
    if _seed_schedule_sha256(attempts) != payload["rollout_seed_schedule_sha256"]:
        raise ValueError("rollout seed schedule SHA is invalid")

    if sampler is not None:
        _require_sampler(sampler)
        if sampler.world_size != 1:
            raise ValueError("live sampler world_size drift")
        if sampler.global_batch_size != global_batch_size:
            raise ValueError("live sampler global_batch_size drift")
        if sampler.config_sha256 != payload["ocr_sampler_config_sha256"]:
            raise ValueError("live sampler config drift")
        if sampler.sample_list_sha256 != payload["ocr_sample_list_sha256"]:
            raise ValueError("live sampler sample-list drift")

    if ocr_dataset_contract_sha256 is not None and _sha256(
        ocr_dataset_contract_sha256,
        "ocr_dataset_contract_sha256",
    ) != payload["ocr_dataset_contract_sha256"]:
        raise ValueError("OCR dataset contract drift")
    if text_dataset_contract_sha256 is not None and _sha256(
        text_dataset_contract_sha256,
        "text_dataset_contract_sha256",
    ) != payload["text_dataset_contract_sha256"]:
        raise ValueError("text dataset contract drift")
    if text_sample_ids is not None:
        registered = _validated_sample_ids(
            text_sample_ids,
            field="text_sample_ids",
        )
        if len(registered) != text_sample_count:
            raise ValueError("text sample-list count drift")
        if canonical_json_sha256(list(registered)) != payload[
            "text_sample_list_sha256"
        ]:
            raise ValueError("text sample-list drift")
        for attempt_index, row in enumerate(attempts):
            expected = _text_batch_for_attempt(
                registered,
                text_batch_size=text_batch_size,
                attempt_index=attempt_index,
            )
            if row["text_sample_ids"] != list(expected):
                raise ValueError("text batch schedule is not reproducible")

    if sampler_attempt_index is not None:
        if sampler is None:
            raise ValueError("sampler_attempt_index requires a live sampler")
        validate_sampler_resume_state(
            {"payload": payload, "canonical_sha256": canonical_sha256},
            sampler=sampler,
            attempt_index=sampler_attempt_index,
        )
    return deepcopy(payload)


def fixed_kl_pilot_attempt(
    protocol: Mapping[str, object],
    attempt_index: int,
) -> dict[str, Any]:
    """Return one immutable schedule slot by rollout-attempt index."""

    payload = validate_fixed_kl_pilot_protocol(protocol)
    index = _attempt_index(attempt_index, allow_terminal=False)
    return deepcopy(payload["attempts"][index])


def next_fixed_kl_pilot_attempt(
    protocol: Mapping[str, object],
    *,
    completed_rollout_attempts: int,
) -> dict[str, Any]:
    """Replay the next slot after resume, independent of optimizer progress."""

    return fixed_kl_pilot_attempt(protocol, completed_rollout_attempts)


def advance_fixed_kl_pilot_attempt(
    protocol: Mapping[str, object],
    *,
    attempt_index: int,
    active_group_count: int,
) -> int:
    """Commit an attempt cursor; zero active groups still advances one slot."""

    validate_fixed_kl_pilot_protocol(protocol)
    index = _attempt_index(attempt_index, allow_terminal=False)
    if (
        isinstance(active_group_count, bool)
        or not isinstance(active_group_count, int)
        or active_group_count < 0
    ):
        raise ValueError("active_group_count must be a non-negative integer")
    return index + 1


def validate_sampler_resume_state(
    protocol: Mapping[str, object],
    *,
    sampler: OCRQuotaSampler,
    attempt_index: int,
) -> None:
    """Fail closed unless ``sampler`` is at the requested attempt boundary."""

    payload = validate_fixed_kl_pilot_protocol(protocol)
    _require_sampler(sampler)
    index = _attempt_index(attempt_index, allow_terminal=True)
    if sampler.pending_global_batch is not None:
        raise ValueError("resume sampler must be at a committed attempt boundary")
    if sampler.config_sha256 != payload["ocr_sampler_config_sha256"]:
        raise ValueError("resume sampler config drift")
    if sampler.sample_list_sha256 != payload["ocr_sample_list_sha256"]:
        raise ValueError("resume sampler sample-list drift")
    if index == FIXED_KL_PILOT_ROLLOUT_ATTEMPTS:
        expected = payload["attempts"][-1]["sampler_state_after_sha256"]
    else:
        expected = payload["attempts"][index]["sampler_state_before_sha256"]
    if canonical_json_sha256(sampler.state_dict()) != expected:
        raise ValueError("resume sampler is not at the requested attempt boundary")


def apply_attempt_seed(
    protocol: Mapping[str, object],
    attempt_index: int,
) -> dict[str, int]:
    """Reset Python, Torch CPU, available CUDA, and optional NumPy RNGs."""

    attempt = fixed_kl_pilot_attempt(protocol, attempt_index)
    cpu_seed = int(attempt["cpu_seed"])
    cuda_seed = int(attempt["cuda_seed"])
    random.seed(cpu_seed)
    torch.manual_seed(cpu_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cuda_seed)

    # NumPy is a development dependency, not a runtime requirement.  Import it
    # opportunistically so this protocol does not add a mandatory dependency.
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(cpu_seed % (1 << 32))
    return {"cpu_seed": cpu_seed, "cuda_seed": cuda_seed}


def _require_sampler(sampler: object) -> OCRQuotaSampler:
    if not isinstance(sampler, OCRQuotaSampler):
        raise TypeError("sampler must be an OCRQuotaSampler")
    return sampler


def _require_fresh_sampler(sampler: object) -> OCRQuotaSampler:
    result = _require_sampler(sampler)
    state = result.state_dict()
    buckets = state.get("buckets")
    quota_deficit = state.get("quota_deficit")
    fresh_buckets = (
        isinstance(buckets, Mapping)
        and set(buckets) == set(OCR_QUOTA_BUCKETS)
        and all(
            isinstance(value, Mapping)
            and value.get("cursor") == 0
            and value.get("epoch") == 0
            for value in buckets.values()
        )
    )
    fresh_deficit = (
        isinstance(quota_deficit, Mapping)
        and set(quota_deficit) == set(OCR_QUOTA_BUCKETS)
        and all(value == 0 for value in quota_deficit.values())
    )
    if (
        state.get("draw_counter") != 0
        or state.get("pending_global_batch") is not None
        or state.get("pending_world_size") is not None
        or not fresh_buckets
        or not fresh_deficit
    ):
        raise ValueError(
            "fixed KL pilot protocol requires a fresh initial sampler"
        )
    return result


def _validated_sample_ids(
    sample_ids: Sequence[str],
    *,
    field: str,
) -> tuple[str, ...]:
    if isinstance(sample_ids, (str, bytes)) or not isinstance(sample_ids, Sequence):
        raise ValueError(f"{field} must be a non-empty sequence")
    result = tuple(sample_ids)
    if not result:
        raise ValueError(f"{field} must be non-empty")
    if any(
        not isinstance(sample_id, str)
        or not sample_id
        or sample_id != sample_id.strip()
        for sample_id in result
    ):
        raise ValueError(f"{field} must contain stripped non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must not contain duplicate IDs")
    return result


def _validated_fixed_batch(
    value: object,
    *,
    size: int,
    field: str,
) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{field} must contain exactly {size} IDs")
    if any(
        not isinstance(sample_id, str)
        or not sample_id
        or sample_id != sample_id.strip()
        for sample_id in value
    ):
        raise ValueError(f"{field} must contain stripped non-empty strings")
    return tuple(value)


def _text_batch_for_attempt(
    registered_ids: Sequence[str],
    *,
    text_batch_size: int,
    attempt_index: int,
) -> tuple[str, ...]:
    start = attempt_index * text_batch_size
    count = len(registered_ids)
    return tuple(
        registered_ids[(start + offset) % count]
        for offset in range(text_batch_size)
    )


def _attempt_seed(base_seed: int, attempt_index: int, *, domain: str) -> int:
    material = (
        f"{FIXED_KL_PILOT_SEED_DERIVATION}\0{base_seed}\0"
        f"{attempt_index}\0{domain}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & _MAX_SEED


def _ocr_schedule_sha256(attempts: Sequence[Mapping[str, object]]) -> str:
    return canonical_json_sha256(
        [
            {
                "attempt_index": row["attempt_index"],
                "ocr_prompt_ids": row["ocr_prompt_ids"],
                "sampler_state_before_sha256": row[
                    "sampler_state_before_sha256"
                ],
                "sampler_state_after_sha256": row[
                    "sampler_state_after_sha256"
                ],
            }
            for row in attempts
        ]
    )


def _text_schedule_sha256(attempts: Sequence[Mapping[str, object]]) -> str:
    return canonical_json_sha256(
        [
            {
                "attempt_index": row["attempt_index"],
                "text_sample_ids": row["text_sample_ids"],
            }
            for row in attempts
        ]
    )


def _seed_schedule_sha256(attempts: Sequence[Mapping[str, object]]) -> str:
    return canonical_json_sha256(
        [
            {
                "attempt_index": row["attempt_index"],
                "cpu_seed": row["cpu_seed"],
                "cuda_seed": row["cuda_seed"],
            }
            for row in attempts
        ]
    )


def _exact_mapping(value: object, keys: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{field} fields differ from contract")
    return dict(value)


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be lowercase SHA-256")
    return value


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_seed(value: object, field: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > _MAX_SEED
    ):
        raise ValueError(f"{field} must be an unsigned 63-bit integer")
    return value


def _attempt_index(value: object, *, allow_terminal: bool) -> int:
    upper = (
        FIXED_KL_PILOT_ROLLOUT_ATTEMPTS
        if allow_terminal
        else FIXED_KL_PILOT_ROLLOUT_ATTEMPTS - 1
    )
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= upper:
        raise ValueError(f"attempt_index must be in [0, {upper}]")
    return value


__all__ = [
    "FIXED_KL_PILOT_PROTOCOL_KIND",
    "FIXED_KL_PILOT_ROLLOUT_ATTEMPTS",
    "FIXED_KL_PILOT_SEED_DERIVATION",
    "FIXED_KL_PILOT_TEXT_SCHEDULE",
    "advance_fixed_kl_pilot_attempt",
    "apply_attempt_seed",
    "build_fixed_kl_pilot_protocol",
    "fixed_kl_pilot_attempt",
    "next_fixed_kl_pilot_attempt",
    "rebuild_and_match_fixed_kl_pilot_protocol",
    "validate_fixed_kl_pilot_protocol",
    "validate_sampler_resume_state",
]
