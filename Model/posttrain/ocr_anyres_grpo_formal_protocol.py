# -*- coding: utf-8 -*-

"""Versioned deterministic schedule for configurable formal anyres GRPO."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import torch

from Model.ocr.tokenization import canonical_json_sha256
from Model.posttrain.ocr_quota_sampler import OCRQuotaSampler


FORMAL_GRPO_PROTOCOL_KIND = "dol_ocr_anyres_grpo_formal_protocol_v1"
FORMAL_GRPO_PROTOCOL_METADATA_KEY = "ocr_anyres_grpo_formal_protocol"
FORMAL_GRPO_PROTOCOL_SHA256_METADATA_KEY = (
    "ocr_anyres_grpo_formal_protocol_sha256"
)
_SEED_DOMAIN = "dol-ocr-anyres-formal-attempt-seed-v1"
_WRAPPER_KEYS = {"payload", "canonical_sha256"}
_PAYLOAD_KEYS = {
    "schema_version",
    "kind",
    "max_rollout_attempts",
    "max_optimizer_steps",
    "eval_every",
    "save_every",
    "early_stop_patience",
    "global_batch_size",
    "text_batch_size",
    "world_size",
    "base_seed",
    "ocr_sampler_config_sha256",
    "ocr_sample_list_sha256",
    "ocr_dataset_contract_sha256",
    "ocr_sampler_start_state_sha256",
    "text_dataset_contract_sha256",
    "text_sample_ids",
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


def build_formal_grpo_protocol(
    *,
    sampler: OCRQuotaSampler,
    ocr_dataset_contract_sha256: str,
    text_sample_ids: Sequence[str],
    text_dataset_contract_sha256: str,
    text_batch_size: int,
    base_seed: int,
    max_rollout_attempts: int,
    max_optimizer_steps: int,
    eval_every: int,
    save_every: int,
    early_stop_patience: int,
) -> dict[str, Any]:
    if not isinstance(sampler, OCRQuotaSampler):
        raise TypeError("sampler must be OCRQuotaSampler")
    for name, value in (
        ("text_batch_size", text_batch_size),
        ("max_rollout_attempts", max_rollout_attempts),
        ("max_optimizer_steps", max_optimizer_steps),
        ("eval_every", eval_every),
        ("save_every", save_every),
        ("early_stop_patience", early_stop_patience),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be positive")
    if max_optimizer_steps > max_rollout_attempts:
        raise ValueError("max_optimizer_steps cannot exceed max_rollout_attempts")
    if sampler.world_size != 1 or sampler.global_batch_size % 20:
        raise ValueError("formal protocol requires single and batch divisible by 20")
    if sampler.pending_global_batch is not None:
        raise ValueError("formal protocol requires a committed sampler boundary")
    initial_state = sampler.state_dict()
    if sampler.draw_counter != 0 or any(
        int(row["cursor"]) != 0 or int(row["epoch"]) != 0
        for row in initial_state["buckets"].values()
    ) or any(int(value) != 0 for value in initial_state["quota_deficit"].values()):
        raise ValueError("formal protocol must be built from a fresh sampler")
    if (
        isinstance(base_seed, bool)
        or not isinstance(base_seed, int)
        or not 0 <= base_seed < (1 << 63)
    ):
        raise ValueError("base_seed must be non-negative")
    ids = tuple(text_sample_ids)
    if not ids or len(ids) != len(set(ids)) or any(
        not isinstance(value, str) or not value or value != value.strip()
        for value in ids
    ):
        raise ValueError("text_sample_ids must be unique non-empty strings")
    for name, value in (
        ("ocr_dataset_contract_sha256", ocr_dataset_contract_sha256),
        ("text_dataset_contract_sha256", text_dataset_contract_sha256),
    ):
        _sha(value, name)

    live_state = deepcopy(sampler.state_dict())
    planner = deepcopy(sampler)
    start_state = canonical_json_sha256(live_state)
    attempts: list[dict[str, Any]] = []
    for index in range(max_rollout_attempts):
        before = canonical_json_sha256(planner.state_dict())
        ocr_ids = list(planner.prepare_global_batch())
        planner.commit_global_batch()
        after = canonical_json_sha256(planner.state_dict())
        text_ids = [
            ids[(index * text_batch_size + offset) % len(ids)]
            for offset in range(text_batch_size)
        ]
        attempts.append(
            {
                "attempt_index": index,
                "ocr_prompt_ids": ocr_ids,
                "text_sample_ids": text_ids,
                "cpu_seed": _seed(base_seed, index, "cpu"),
                "cuda_seed": _seed(base_seed, index, "cuda"),
                "sampler_state_before_sha256": before,
                "sampler_state_after_sha256": after,
            }
        )
    payload = {
        "schema_version": 1,
        "kind": FORMAL_GRPO_PROTOCOL_KIND,
        "max_rollout_attempts": max_rollout_attempts,
        "max_optimizer_steps": max_optimizer_steps,
        "eval_every": eval_every,
        "save_every": save_every,
        "early_stop_patience": early_stop_patience,
        "global_batch_size": sampler.global_batch_size,
        "text_batch_size": text_batch_size,
        "world_size": 1,
        "base_seed": base_seed,
        "ocr_sampler_config_sha256": sampler.config_sha256,
        "ocr_sample_list_sha256": sampler.sample_list_sha256,
        "ocr_dataset_contract_sha256": ocr_dataset_contract_sha256,
        "ocr_sampler_start_state_sha256": start_state,
        "text_dataset_contract_sha256": text_dataset_contract_sha256,
        "text_sample_ids": list(ids),
        "attempts": attempts,
        "ocr_prompt_schedule_sha256": canonical_json_sha256(
            [[row["ocr_prompt_ids"], row["sampler_state_before_sha256"], row["sampler_state_after_sha256"]] for row in attempts]
        ),
        "text_batch_schedule_sha256": canonical_json_sha256(
            [row["text_sample_ids"] for row in attempts]
        ),
        "rollout_seed_schedule_sha256": canonical_json_sha256(
            [[row["cpu_seed"], row["cuda_seed"]] for row in attempts]
        ),
    }
    wrapper = {"payload": payload, "canonical_sha256": canonical_json_sha256(payload)}
    if sampler.state_dict() != live_state:
        raise RuntimeError("formal protocol planning mutated the live sampler")
    validate_formal_grpo_protocol(wrapper)
    return wrapper


def validate_formal_grpo_protocol(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _WRAPPER_KEYS:
        raise ValueError("formal protocol wrapper fields differ")
    payload = value["payload"]
    if not isinstance(payload, Mapping) or set(payload) != _PAYLOAD_KEYS:
        raise ValueError("formal protocol payload fields differ")
    payload = dict(payload)
    if value["canonical_sha256"] != canonical_json_sha256(payload):
        raise ValueError("formal protocol canonical SHA differs")
    if payload["schema_version"] != 1 or payload["kind"] != FORMAL_GRPO_PROTOCOL_KIND:
        raise ValueError("formal protocol kind differs")
    for name in (
        "max_rollout_attempts", "max_optimizer_steps", "eval_every",
        "save_every", "early_stop_patience", "global_batch_size",
        "text_batch_size",
    ):
        if isinstance(payload[name], bool) or not isinstance(payload[name], int) or payload[name] <= 0:
            raise ValueError(f"formal protocol {name} must be positive")
    if payload["max_optimizer_steps"] > payload["max_rollout_attempts"]:
        raise ValueError("formal optimizer budget exceeds rollout budget")
    if payload["world_size"] != 1 or payload["global_batch_size"] % 20:
        raise ValueError("formal protocol distribution differs")
    if (
        isinstance(payload["base_seed"], bool)
        or not isinstance(payload["base_seed"], int)
        or not 0 <= payload["base_seed"] < (1 << 63)
    ):
        raise ValueError("formal protocol base seed differs")
    registered_ids = payload["text_sample_ids"]
    if not isinstance(registered_ids, list) or not registered_ids or len(
        registered_ids
    ) != len(set(registered_ids)) or any(
        not isinstance(item, str) or not item or item != item.strip()
        for item in registered_ids
    ):
        raise ValueError("formal protocol registered text IDs differ")
    registered_set = set(registered_ids)
    for name in (
        "ocr_sampler_config_sha256", "ocr_sample_list_sha256",
        "ocr_dataset_contract_sha256", "ocr_sampler_start_state_sha256",
        "text_dataset_contract_sha256", "ocr_prompt_schedule_sha256",
        "text_batch_schedule_sha256", "rollout_seed_schedule_sha256",
    ):
        _sha(payload[name], name)
    attempts = payload["attempts"]
    if not isinstance(attempts, list) or len(attempts) != payload["max_rollout_attempts"]:
        raise ValueError("formal protocol attempt count differs")
    previous = payload["ocr_sampler_start_state_sha256"]
    for index, raw in enumerate(attempts):
        if not isinstance(raw, Mapping) or set(raw) != _ATTEMPT_KEYS:
            raise ValueError("formal protocol attempt fields differ")
        row = dict(raw)
        if row["attempt_index"] != index:
            raise ValueError("formal protocol attempt indexes differ")
        if (
            not isinstance(row["ocr_prompt_ids"], list)
            or not isinstance(row["text_sample_ids"], list)
            or len(row["ocr_prompt_ids"]) != payload["global_batch_size"]
            or len(row["text_sample_ids"]) != payload["text_batch_size"]
            or any(
                not isinstance(item, str) or not item or item != item.strip()
                for item in [*row["ocr_prompt_ids"], *row["text_sample_ids"]]
            )
            or any(item not in registered_set for item in row["text_sample_ids"])
        ):
            raise ValueError("formal protocol batch size differs")
        if row["sampler_state_before_sha256"] != previous:
            raise ValueError("formal protocol sampler chain differs")
        _sha(row["sampler_state_before_sha256"], "sampler before SHA")
        _sha(row["sampler_state_after_sha256"], "sampler after SHA")
        previous = row["sampler_state_after_sha256"]
        for seed_name in ("cpu_seed", "cuda_seed"):
            if (
                isinstance(row[seed_name], bool)
                or not isinstance(row[seed_name], int)
                or not 0 <= row[seed_name] < (1 << 63)
            ):
                raise ValueError("formal protocol seed range differs")
        if row["cpu_seed"] != _seed(payload["base_seed"], index, "cpu") or row["cuda_seed"] != _seed(payload["base_seed"], index, "cuda"):
            raise ValueError("formal protocol seed differs")
    expected = {
        "ocr_prompt_schedule_sha256": canonical_json_sha256(
            [[row["ocr_prompt_ids"], row["sampler_state_before_sha256"], row["sampler_state_after_sha256"]] for row in attempts]
        ),
        "text_batch_schedule_sha256": canonical_json_sha256([row["text_sample_ids"] for row in attempts]),
        "rollout_seed_schedule_sha256": canonical_json_sha256([[row["cpu_seed"], row["cuda_seed"]] for row in attempts]),
    }
    if any(payload[key] != digest for key, digest in expected.items()):
        raise ValueError("formal protocol schedule SHA differs")
    return deepcopy(payload)


def rebuild_and_match_formal_protocol(
    expected: Mapping[str, Any],
    *,
    sampler: OCRQuotaSampler,
    ocr_dataset_contract_sha256: str,
    text_sample_ids: Sequence[str],
    text_dataset_contract_sha256: str,
) -> dict[str, Any]:
    payload = validate_formal_grpo_protocol(expected)
    rebuilt = build_formal_grpo_protocol(
        sampler=sampler,
        ocr_dataset_contract_sha256=ocr_dataset_contract_sha256,
        text_sample_ids=text_sample_ids,
        text_dataset_contract_sha256=text_dataset_contract_sha256,
        text_batch_size=payload["text_batch_size"],
        base_seed=payload["base_seed"],
        max_rollout_attempts=payload["max_rollout_attempts"],
        max_optimizer_steps=payload["max_optimizer_steps"],
        eval_every=payload["eval_every"],
        save_every=payload["save_every"],
        early_stop_patience=payload["early_stop_patience"],
    )
    if rebuilt != dict(expected):
        raise ValueError("formal protocol differs from fresh dataset reconstruction")
    return rebuilt


def formal_grpo_attempt(protocol: Mapping[str, Any], index: int) -> dict[str, Any]:
    payload = validate_formal_grpo_protocol(protocol)
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < payload["max_rollout_attempts"]:
        raise ValueError("formal attempt index is outside protocol")
    return deepcopy(payload["attempts"][index])


def validate_formal_sampler_boundary(protocol: Mapping[str, Any], sampler: OCRQuotaSampler, index: int) -> None:
    payload = validate_formal_grpo_protocol(protocol)
    if not isinstance(sampler, OCRQuotaSampler) or sampler.pending_global_batch is not None:
        raise ValueError("formal sampler must be committed OCRQuotaSampler")
    if sampler.config_sha256 != payload["ocr_sampler_config_sha256"] or sampler.sample_list_sha256 != payload["ocr_sample_list_sha256"]:
        raise ValueError("formal sampler identity differs")
    if not 0 <= index <= payload["max_rollout_attempts"]:
        raise ValueError("formal sampler boundary index differs")
    expected = payload["attempts"][-1]["sampler_state_after_sha256"] if index == payload["max_rollout_attempts"] else payload["attempts"][index]["sampler_state_before_sha256"]
    if canonical_json_sha256(sampler.state_dict()) != expected:
        raise ValueError("formal sampler state differs from attempt boundary")


def apply_formal_attempt_seed(protocol: Mapping[str, Any], index: int) -> None:
    row = formal_grpo_attempt(protocol, index)
    random.seed(row["cpu_seed"])
    torch.manual_seed(row["cpu_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(row["cuda_seed"])
    try:
        import numpy as np
    except ImportError:
        return
    np.random.seed(row["cpu_seed"] % (1 << 32))


def _seed(base: int, index: int, domain: str) -> int:
    data = f"{_SEED_DOMAIN}\0{base}\0{index}\0{domain}".encode("ascii")
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big") & ((1 << 63) - 1)


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


__all__ = [
    "FORMAL_GRPO_PROTOCOL_KIND",
    "FORMAL_GRPO_PROTOCOL_METADATA_KEY",
    "FORMAL_GRPO_PROTOCOL_SHA256_METADATA_KEY",
    "apply_formal_attempt_seed",
    "build_formal_grpo_protocol",
    "formal_grpo_attempt",
    "rebuild_and_match_formal_protocol",
    "validate_formal_grpo_protocol",
    "validate_formal_sampler_boundary",
]
