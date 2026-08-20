# -*- coding: utf-8 -*-

"""Deterministic, checkpointable OCR:text microbatch scheduling."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Mapping


MicrobatchKind = Literal["ocr", "text"]

OCR_MICROBATCHES_PER_CYCLE = 4
TEXT_MICROBATCHES_PER_CYCLE = 1
OCR_TEXT_SCHEDULE_VERSION = 1
_CYCLE_LENGTH = OCR_MICROBATCHES_PER_CYCLE + TEXT_MICROBATCHES_PER_CYCLE


def ocr_text_microbatch_kind(microbatches_consumed: int) -> MicrobatchKind:
    """Return the next kind in the fixed ``OCR,OCR,OCR,OCR,text`` cycle."""

    _validate_nonnegative_int("microbatches_consumed", microbatches_consumed)
    position = microbatches_consumed % _CYCLE_LENGTH
    return "ocr" if position < OCR_MICROBATCHES_PER_CYCLE else "text"


@dataclass(frozen=True)
class OCRTextMicrobatchSchedule:
    """Immutable cursor whose JSON-safe state restores the exact next kind."""

    microbatches_consumed: int = 0

    def __post_init__(self) -> None:
        _validate_nonnegative_int(
            "microbatches_consumed",
            self.microbatches_consumed,
        )

    def next_kind(self) -> MicrobatchKind:
        return ocr_text_microbatch_kind(self.microbatches_consumed)

    def advance(self, count: int = 1) -> "OCRTextMicrobatchSchedule":
        _validate_nonnegative_int("count", count)
        return replace(
            self,
            microbatches_consumed=self.microbatches_consumed + count,
        )

    def state_dict(self) -> dict[str, int]:
        return {
            "version": OCR_TEXT_SCHEDULE_VERSION,
            "ocr_microbatches_per_cycle": OCR_MICROBATCHES_PER_CYCLE,
            "text_microbatches_per_cycle": TEXT_MICROBATCHES_PER_CYCLE,
            "microbatches_consumed": self.microbatches_consumed,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> "OCRTextMicrobatchSchedule":
        if not isinstance(state, Mapping):
            raise TypeError("OCR:text schedule state must be a mapping")
        expected_keys = {
            "version",
            "ocr_microbatches_per_cycle",
            "text_microbatches_per_cycle",
            "microbatches_consumed",
        }
        if set(state) != expected_keys:
            raise ValueError("OCR:text schedule state has an incompatible schema")
        if state["version"] != OCR_TEXT_SCHEDULE_VERSION:
            raise ValueError("OCR:text schedule state version is incompatible")
        if state["ocr_microbatches_per_cycle"] != OCR_MICROBATCHES_PER_CYCLE:
            raise ValueError("OCR:text schedule OCR ratio is incompatible")
        if state["text_microbatches_per_cycle"] != TEXT_MICROBATCHES_PER_CYCLE:
            raise ValueError("OCR:text schedule text ratio is incompatible")
        consumed = state["microbatches_consumed"]
        _validate_nonnegative_int("microbatches_consumed", consumed)
        return cls(microbatches_consumed=consumed)


def _validate_nonnegative_int(name: str, value: object) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


__all__ = [
    "MicrobatchKind",
    "OCRTextMicrobatchSchedule",
    "ocr_text_microbatch_kind",
]
