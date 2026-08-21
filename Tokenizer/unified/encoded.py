# -*- coding: utf-8 -*-
"""Shared encoded-token result structures for routed tokenizers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from .dual_tokenizer import Span


@dataclass(frozen=True)
class EncodedToken:
    id: int
    token: str
    track: str
    start: int
    end: int
    surface: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class DualTrackResult:
    input_ids: list[int]
    tokens: list["EncodedToken"]
    spans: list["Span"]
    attention_mask: list[int] = field(default_factory=list)
    special_tokens_mask: list[int] = field(default_factory=list)
    token_type_ids: list[int] | None = None

    @property
    def ids(self) -> list[int]:
        return self.input_ids

    def __post_init__(self) -> None:
        if not self.attention_mask:
            self.attention_mask = [1] * len(self.input_ids)
        if not self.special_tokens_mask:
            self.special_tokens_mask = [
                1 if token.track == "special" else 0 for token in self.tokens
            ]
