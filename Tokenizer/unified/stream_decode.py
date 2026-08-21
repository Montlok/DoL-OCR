# -*- coding: utf-8 -*-

"""UTF-8-safe incremental detokenization.

Byte-level fallback can split one UTF-8 code point across several tokens —
every Traditional Mongolian letter is a 3-byte sequence, so token-at-a-time
``decode`` mid-stream surfaces U+FFFD replacement characters that would
vanish once the remaining byte tokens arrive (the llama.cpp / HF streaming
bug class). ``StreamDecoder`` buffers token ids and only emits the longest
prefix of the decoded text that can no longer change.
"""

from __future__ import annotations

REPLACEMENT_CHAR = "�"


class StreamDecoder:
    """Incremental wrapper around any ``decode(list[int]) -> str`` tokenizer.

    Usage::

        sd = StreamDecoder(bundle.tokenizer)
        for token_id in generated_ids:
            emit(sd.push(token_id))
        emit(sd.flush())

    ``push`` returns the newly stable text (possibly ``""`` while a code
    point is still incomplete); ``flush`` returns whatever remains, including
    real replacement characters if the stream genuinely ended mid-sequence.
    The concatenation of all returns equals ``decode(all_ids)`` exactly.
    """

    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer
        self._ids: list[int] = []
        self._emitted = 0

    def push(self, token_id: int) -> str:
        self._ids.append(int(token_id))
        text = self._tokenizer.decode(self._ids)
        # Hold back any *trailing* replacement characters: they mark byte
        # sequences the next token may still complete. A genuine U+FFFD from
        # the source is only delayed, never lost — it is emitted as soon as
        # any non-FFFD text follows it, or by flush().
        safe_len = len(text)
        while safe_len > self._emitted and text[safe_len - 1] == REPLACEMENT_CHAR:
            safe_len -= 1
        if safe_len <= self._emitted:
            return ""
        out = text[self._emitted : safe_len]
        self._emitted = safe_len
        return out

    def flush(self) -> str:
        if not self._ids:
            return ""
        text = self._tokenizer.decode(self._ids)
        out = text[self._emitted :]
        self._ids.clear()
        self._emitted = 0
        return out
