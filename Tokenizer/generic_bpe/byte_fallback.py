# -*- coding: utf-8 -*-
"""Byte-fallback helpers shared by tokenizer tracks."""

from __future__ import annotations

from Tokenizer.unified.encoded import EncodedToken


def byte_token(byte: int) -> str:
    return f"<0x{byte:02X}>"


def is_byte_token(token: str) -> bool:
    return token.startswith("<0x") and token.endswith(">") and len(token) == 6


def decode_byte_token(token: str) -> int:
    return int(token[3:5], 16)


def encode_byte_fallback(
    text: str,
    vocab: dict[str, int],
    unk_id: int,
    base_start: int = 0,
    track: str = "misc",
) -> list[EncodedToken]:
    tokens: list[EncodedToken] = []
    for rel_pos, ch in enumerate(text):
        direct = vocab.get(ch)
        start = base_start + rel_pos
        end = start + 1
        if direct is not None:
            tokens.append(EncodedToken(direct, ch, track, start, end))
            continue

        # ``surrogatepass`` keeps encoding from raising on lone surrogates
        # (U+D800–U+DFFF), which appear in scraped/raw web text. A strict
        # ``encode`` would raise UnicodeEncodeError and abort the whole shard;
        # decode mirrors this with ``errors="replace"``.
        for byte in ch.encode("utf-8", "surrogatepass"):
            tok = byte_token(byte)
            tokens.append(EncodedToken(vocab.get(tok, unk_id), tok, track, start, end))
    return tokens


def decode_bytes(byte_tokens: list[str]) -> str:
    return bytes(decode_byte_token(token) for token in byte_tokens).decode(
        "utf-8", errors="replace"
    )
