# -*- coding: utf-8 -*-
"""Compatibility entry point for the generic pretraining data producer."""

from Tokenizer.pretraining.producer import (
    _BuildSummary,
    _JsonlShardWriter,
    _iter_samples,
    _nonempty_samples,
    main,
)

__all__ = ["main", "_BuildSummary", "_JsonlShardWriter", "_iter_samples", "_nonempty_samples"]

if __name__ == "__main__":
    main()
