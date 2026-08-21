# -*- coding: utf-8 -*-
"""General multilingual byte-level BPE track.

A thin wrapper around a HuggingFace ``tokenizers`` byte-level BPE model. The
*model* is trained by us (no external pretrained weights) and covers every
non-Mongolian script — Chinese, English, Japanese, Cyrillic Mongolian, digits,
punctuation, symbols. Byte-level coverage guarantees lossless round-trips and
never emits ``<unk>``.

``tokenizers`` is an optional build-time dependency (``pip install
'.[tokenizer-build]'``); it is only needed when building/training a tokenizer,
not at model train/inference time (the model consumes pre-tokenized shards).
"""

from __future__ import annotations

from typing import Iterable


def _require_tokenizers():
    try:
        import tokenizers  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without dep.
        raise ImportError(
            "the 'tokenizers' package is required for the general BPE track; "
            "install it with: pip install 'tokenizers>=0.19' "
            "(or pip install '.[tokenizer-build]')"
        ) from exc
    return tokenizers


class GeneralBPEModel:
    """Byte-level BPE model wrapper with deterministic local ids and offsets."""

    def __init__(self, tk) -> None:
        self._tk = tk

    # ---- construction ----

    @classmethod
    def minimal(cls) -> "GeneralBPEModel":
        """A merge-free byte-level model (256 byte tokens, full coverage).

        Used as a training-free fallback. It still requires the ``tokenizers``
        package (it builds a real ``tokenizers`` byte-level model), but needs no
        training corpus or merges. It is lossless but uncompressed (one token
        per byte); real runs train a model with merges.
        """
        _require_tokenizers()
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers

        alphabet = pre_tokenizers.ByteLevel.alphabet()
        vocab = {ch: i for i, ch in enumerate(sorted(alphabet))}
        tk = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token=None))
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False, use_regex=True
        )
        tk.decoder = decoders.ByteLevel()
        return cls(tk)

    @classmethod
    def load(cls, path: str) -> "GeneralBPEModel":
        _require_tokenizers()
        from tokenizers import Tokenizer

        return cls(Tokenizer.from_file(path))

    def save(self, path: str) -> None:
        self._tk.save(path)

    # ---- vocab ----

    def get_vocab(self) -> dict[str, int]:
        return self._tk.get_vocab()

    @property
    def vocab_size(self) -> int:
        return self._tk.get_vocab_size()

    # ---- encode / decode ----

    def encode_pieces(self, text: str) -> list[tuple[int, str, int, int]]:
        """Return ``(local_id, token_str, char_start, char_end)`` per piece."""
        enc = self._tk.encode(text)
        return [
            (local_id, token, offset[0], offset[1])
            for local_id, token, offset in zip(enc.ids, enc.tokens, enc.offsets)
        ]

    def decode(self, local_ids: list[int]) -> str:
        return self._tk.decode(local_ids, skip_special_tokens=False)


class GeneralBPETrainer:
    """Train a byte-level BPE ``GeneralBPEModel`` from raw text."""

    def __init__(
        self,
        vocab_size: int = 40000,
        min_frequency: int = 2,
        show_progress: bool = False,
    ) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256 (the byte alphabet)")
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.show_progress = show_progress

    def train(self, texts: Iterable[str]) -> GeneralBPEModel:
        _require_tokenizers()
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

        tk = Tokenizer(models.BPE(unk_token=None))
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False, use_regex=True
        )
        tk.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=[],
            show_progress=self.show_progress,
        )
        tk.train_from_iterator(texts, trainer=trainer)
        return GeneralBPEModel(tk)
