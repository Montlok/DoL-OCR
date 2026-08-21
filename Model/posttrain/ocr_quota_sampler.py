# -*- coding: utf-8 -*-

"""Deterministic, transactional 60/10/20/10 OCR prompt sampling."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


OCR_QUOTA_BUCKETS = (
    "print",
    "handwritten_good",
    "handwritten_medium",
    "handwritten_poor",
)
OCR_QUOTA_PER_20 = {
    "print": 12,
    "handwritten_good": 2,
    "handwritten_medium": 4,
    "handwritten_poor": 2,
}
OCR_QUOTA_SAMPLER_STATE_VERSION = 1

_QUOTA_TOTAL = sum(OCR_QUOTA_PER_20.values())
_KIND = "dol_ocr_quota_sampler"
_SHUFFLE_CONTRACT = "sha256-seed-bucket-epoch-sample-key-sort-v1"


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _quota_cycle() -> tuple[str, ...]:
    deficit = {bucket: 0 for bucket in OCR_QUOTA_BUCKETS}
    result: list[str] = []
    for _ in range(_QUOTA_TOTAL):
        for bucket in OCR_QUOTA_BUCKETS:
            deficit[bucket] += OCR_QUOTA_PER_20[bucket]
        selected = max(OCR_QUOTA_BUCKETS, key=lambda bucket: deficit[bucket])
        deficit[selected] -= _QUOTA_TOTAL
        result.append(selected)
    if deficit != {bucket: 0 for bucket in OCR_QUOTA_BUCKETS}:
        raise RuntimeError("OCR quota cycle does not close at 20 prompts")
    return tuple(result)


_QUOTA_CYCLE = _quota_cycle()


def slice_global_batch(
    sample_ids: Sequence[str],
    *,
    rank: int,
    world_size: int,
) -> tuple[str, ...]:
    """Return the deterministic strided shard of one rank0 global batch."""

    _validate_positive_int("world_size", world_size)
    _validate_nonnegative_int("rank", rank)
    if rank >= world_size:
        raise ValueError("rank must be less than world_size")
    if len(sample_ids) == 0 or len(sample_ids) % world_size != 0:
        raise ValueError(
            "global batch size must be positive and divisible by world_size"
        )
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids):
        raise ValueError("global batch sample IDs must be non-empty strings")
    return tuple(sample_ids[rank::world_size])


@dataclass
class _BucketState:
    order: list[str]
    cursor: int
    epoch: int


class OCRQuotaSampler:
    """Rank-0 sampler with explicit prepare/slice/commit batch semantics."""

    def __init__(
        self,
        sample_buckets: Mapping[str, str],
        *,
        global_batch_size: int,
        seed: int,
        world_size: int = 1,
    ) -> None:
        _validate_positive_int("global_batch_size", global_batch_size)
        _validate_positive_int("world_size", world_size)
        _validate_nonnegative_int("seed", seed)
        if global_batch_size % world_size != 0:
            raise ValueError("global_batch_size must be divisible by world_size")
        if not isinstance(sample_buckets, Mapping) or not sample_buckets:
            raise ValueError("sample_buckets must be a non-empty mapping")

        canonical: dict[str, list[str]] = {
            bucket: [] for bucket in OCR_QUOTA_BUCKETS
        }
        for sample_id, bucket in sample_buckets.items():
            if (
                not isinstance(sample_id, str)
                or not sample_id
                or sample_id != sample_id.strip()
            ):
                raise ValueError("sample IDs must be non-empty stripped strings")
            if not isinstance(bucket, str) or bucket not in OCR_QUOTA_PER_20:
                raise ValueError(
                    f"sample {sample_id!r} has unsupported quota bucket {bucket!r}"
                )
            canonical[bucket].append(sample_id)
        for bucket in OCR_QUOTA_BUCKETS:
            canonical[bucket].sort()
            if not canonical[bucket]:
                raise ValueError(f"quota bucket {bucket!r} is empty")

        self.global_batch_size = global_batch_size
        self.seed = seed
        self.world_size = world_size
        self._source_ids = {
            bucket: tuple(canonical[bucket]) for bucket in OCR_QUOTA_BUCKETS
        }
        sample_rows = [
            {"sample_id": sample_id, "bucket": bucket}
            for bucket in OCR_QUOTA_BUCKETS
            for sample_id in self._source_ids[bucket]
        ]
        self.sample_list_sha256 = _canonical_sha256(sample_rows)
        self._config = {
            "schema_version": OCR_QUOTA_SAMPLER_STATE_VERSION,
            "quota_per_20": dict(OCR_QUOTA_PER_20),
            "global_batch_size": global_batch_size,
            "seed": seed,
            "world_size": world_size,
            "shuffle_contract": _SHUFFLE_CONTRACT,
            "rank_partition": "strided_v1",
        }
        self.config_sha256 = _canonical_sha256(self._config)
        self._buckets = {
            bucket: _BucketState(
                order=self._order_for_epoch(bucket, 0),
                cursor=0,
                epoch=0,
            )
            for bucket in OCR_QUOTA_BUCKETS
        }
        self._quota_deficit = {bucket: 0 for bucket in OCR_QUOTA_BUCKETS}
        self._draw_counter = 0
        self._pending_global_batch: list[str] | None = None
        self._pending_world_size: int | None = None

    @property
    def draw_counter(self) -> int:
        return self._draw_counter

    @property
    def quota_deficit(self) -> dict[str, int]:
        return dict(self._quota_deficit)

    @property
    def pending_global_batch(self) -> tuple[str, ...] | None:
        if self._pending_global_batch is None:
            return None
        return tuple(self._pending_global_batch)

    def prepare_global_batch(self) -> tuple[str, ...]:
        """Prepare one global batch idempotently until it is committed."""

        if self._pending_global_batch is not None:
            return tuple(self._pending_global_batch)
        batch = [self._draw_one() for _ in range(self.global_batch_size)]
        self._pending_global_batch = batch
        self._pending_world_size = self.world_size
        return tuple(batch)

    def pending_batch_for_rank(self, rank: int) -> tuple[str, ...]:
        if self._pending_global_batch is None:
            raise RuntimeError("prepare_global_batch() must run before rank slicing")
        if self._pending_world_size != self.world_size:
            raise RuntimeError("pending global batch world_size is inconsistent")
        return slice_global_batch(
            self._pending_global_batch,
            rank=rank,
            world_size=self.world_size,
        )

    def commit_global_batch(self) -> None:
        if self._pending_global_batch is None:
            raise RuntimeError("there is no pending global batch to commit")
        self._pending_global_batch = None
        self._pending_world_size = None

    def _draw_one(self) -> str:
        for bucket in OCR_QUOTA_BUCKETS:
            self._quota_deficit[bucket] += OCR_QUOTA_PER_20[bucket]
        selected = max(
            OCR_QUOTA_BUCKETS,
            key=lambda bucket: self._quota_deficit[bucket],
        )
        self._quota_deficit[selected] -= _QUOTA_TOTAL
        sample_id = self._draw_from_bucket(selected)
        self._draw_counter += 1
        return sample_id

    def _draw_from_bucket(self, bucket: str) -> str:
        state = self._buckets[bucket]
        if state.cursor == len(state.order):
            state.epoch += 1
            state.order = self._order_for_epoch(bucket, state.epoch)
            state.cursor = 0
        sample_id = state.order[state.cursor]
        state.cursor += 1
        return sample_id

    def _order_for_epoch(self, bucket: str, epoch: int) -> list[str]:
        def key(sample_id: str) -> tuple[bytes, str]:
            material = (
                f"{self.seed}\0{bucket}\0{epoch}\0{sample_id}\0"
                f"{_SHUFFLE_CONTRACT}"
            ).encode("utf-8")
            return hashlib.sha256(material).digest(), sample_id

        return sorted(self._source_ids[bucket], key=key)

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": OCR_QUOTA_SAMPLER_STATE_VERSION,
            "kind": _KIND,
            "config": {
                **self._config,
                "quota_per_20": dict(OCR_QUOTA_PER_20),
            },
            "config_sha256": self.config_sha256,
            "sample_list_sha256": self.sample_list_sha256,
            "buckets": {
                bucket: {
                    "order": list(self._buckets[bucket].order),
                    "cursor": self._buckets[bucket].cursor,
                    "epoch": self._buckets[bucket].epoch,
                }
                for bucket in OCR_QUOTA_BUCKETS
            },
            "quota_deficit": dict(self._quota_deficit),
            "draw_counter": self._draw_counter,
            "replay_state": {
                "shuffle_contract": _SHUFFLE_CONTRACT,
                "seed": self.seed,
            },
            "pending_global_batch": (
                None
                if self._pending_global_batch is None
                else list(self._pending_global_batch)
            ),
            "pending_world_size": self._pending_world_size,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Validate completely, then atomically adopt a JSON-decoded state."""

        if not isinstance(state, Mapping):
            raise TypeError("OCR quota sampler state must be a mapping")
        expected_keys = {
            "schema_version",
            "kind",
            "config",
            "config_sha256",
            "sample_list_sha256",
            "buckets",
            "quota_deficit",
            "draw_counter",
            "replay_state",
            "pending_global_batch",
            "pending_world_size",
        }
        if set(state) != expected_keys:
            raise ValueError("OCR quota sampler state schema is incompatible")
        if state["schema_version"] != OCR_QUOTA_SAMPLER_STATE_VERSION:
            raise ValueError("OCR quota sampler state version is incompatible")
        if state["kind"] != _KIND:
            raise ValueError("OCR quota sampler state kind is incompatible")

        config = state["config"]
        if not isinstance(config, Mapping):
            raise ValueError("OCR quota sampler state config must be an object")
        if _canonical_sha256(config) != state["config_sha256"]:
            raise ValueError("OCR quota sampler state config hash is invalid")
        if config.get("world_size") != self.world_size:
            raise ValueError("OCR quota sampler world_size drift")
        if dict(config) != self._config or state["config_sha256"] != self.config_sha256:
            raise ValueError("OCR quota sampler config drift")
        if state["sample_list_sha256"] != self.sample_list_sha256:
            raise ValueError("OCR quota sampler sample-list hash drift")

        replay_state = state["replay_state"]
        if replay_state != {
            "shuffle_contract": _SHUFFLE_CONTRACT,
            "seed": self.seed,
        }:
            raise ValueError("OCR quota sampler replay state is incompatible")

        draw_counter = state["draw_counter"]
        _validate_nonnegative_int("draw_counter", draw_counter)
        if draw_counter % self.global_batch_size != 0:
            raise ValueError("draw_counter is not at a global-batch boundary")

        deficits = state["quota_deficit"]
        if not isinstance(deficits, Mapping) or set(deficits) != set(OCR_QUOTA_BUCKETS):
            raise ValueError("quota_deficit has incompatible buckets")
        parsed_deficits: dict[str, int] = {}
        for bucket in OCR_QUOTA_BUCKETS:
            value = deficits[bucket]
            if type(value) is not int:
                raise ValueError("quota deficits must be integers")
            parsed_deficits[bucket] = value
        if parsed_deficits != _deficits_after_draws(draw_counter):
            raise ValueError("quota_deficit does not match draw_counter")

        bucket_payload = state["buckets"]
        if not isinstance(bucket_payload, Mapping) or set(bucket_payload) != set(
            OCR_QUOTA_BUCKETS
        ):
            raise ValueError("bucket state has incompatible buckets")
        parsed_buckets: dict[str, _BucketState] = {}
        for bucket in OCR_QUOTA_BUCKETS:
            payload = bucket_payload[bucket]
            if not isinstance(payload, Mapping) or set(payload) != {
                "order",
                "cursor",
                "epoch",
            }:
                raise ValueError(f"bucket {bucket!r} state schema is incompatible")
            epoch = payload["epoch"]
            cursor = payload["cursor"]
            _validate_nonnegative_int(f"{bucket}.epoch", epoch)
            _validate_nonnegative_int(f"{bucket}.cursor", cursor)
            expected_draws = _bucket_draws_after(draw_counter, bucket)
            expected_epoch, expected_cursor = _epoch_cursor_for_draws(
                expected_draws,
                len(self._source_ids[bucket]),
            )
            if (epoch, cursor) != (expected_epoch, expected_cursor):
                raise ValueError(f"bucket {bucket!r} cursor/epoch does not match draws")
            order = payload["order"]
            if not isinstance(order, list) or any(
                not isinstance(sample_id, str) for sample_id in order
            ):
                raise ValueError(f"bucket {bucket!r} order must be a string list")
            expected_order = self._order_for_epoch(bucket, epoch)
            if order != expected_order:
                raise ValueError(f"bucket {bucket!r} order is not reproducible")
            parsed_buckets[bucket] = _BucketState(
                order=list(order),
                cursor=cursor,
                epoch=epoch,
            )

        pending = state["pending_global_batch"]
        pending_world_size = state["pending_world_size"]
        if pending is None:
            if pending_world_size is not None:
                raise ValueError("pending_world_size requires a pending batch")
            parsed_pending = None
        else:
            if not isinstance(pending, list) or len(pending) != self.global_batch_size:
                raise ValueError("pending global batch has an incompatible size")
            if any(not isinstance(sample_id, str) for sample_id in pending):
                raise ValueError("pending global batch IDs must be strings")
            if pending_world_size != self.world_size:
                raise ValueError("pending global batch world_size drift")
            expected_pending = self._expected_draw_range(
                draw_counter - self.global_batch_size,
                self.global_batch_size,
            )
            if pending != expected_pending:
                raise ValueError("pending global batch is not reproducible")
            parsed_pending = list(pending)

        self._buckets = parsed_buckets
        self._quota_deficit = parsed_deficits
        self._draw_counter = draw_counter
        self._pending_global_batch = parsed_pending
        self._pending_world_size = pending_world_size

    def _expected_draw_range(self, start: int, count: int) -> list[str]:
        if start < 0:
            raise ValueError("pending global batch starts before draw zero")
        result: list[str] = []
        for draw_index in range(start, start + count):
            bucket = _QUOTA_CYCLE[draw_index % _QUOTA_TOTAL]
            occurrence = _bucket_draws_after(draw_index, bucket)
            source = self._source_ids[bucket]
            epoch, position = divmod(occurrence, len(source))
            result.append(self._order_for_epoch(bucket, epoch)[position])
        return result


def _deficits_after_draws(draws: int) -> dict[str, int]:
    deficit = {bucket: 0 for bucket in OCR_QUOTA_BUCKETS}
    for draw_index in range(draws % _QUOTA_TOTAL):
        for bucket in OCR_QUOTA_BUCKETS:
            deficit[bucket] += OCR_QUOTA_PER_20[bucket]
        selected = _QUOTA_CYCLE[draw_index]
        deficit[selected] -= _QUOTA_TOTAL
    return deficit


def _bucket_draws_after(draws: int, bucket: str) -> int:
    cycles, remainder = divmod(draws, _QUOTA_TOTAL)
    return cycles * OCR_QUOTA_PER_20[bucket] + _QUOTA_CYCLE[:remainder].count(bucket)


def _epoch_cursor_for_draws(draws: int, bucket_size: int) -> tuple[int, int]:
    if draws == 0:
        return 0, 0
    return (draws - 1) // bucket_size, ((draws - 1) % bucket_size) + 1


def _validate_positive_int(name: str, value: object) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_nonnegative_int(name: str, value: object) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


__all__ = [
    "OCR_QUOTA_BUCKETS",
    "OCR_QUOTA_PER_20",
    "OCRQuotaSampler",
    "slice_global_batch",
]
