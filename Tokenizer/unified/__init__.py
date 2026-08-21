# -*- coding: utf-8 -*-

__all__ = [
    "DualTrackResult",
    "DualTrackTokenizer",
    "EncodedToken",
    "SEGMENT",
    "SPECIAL_TOKENS",
    "Span",
    "TokenizerBundle",
    "TokenizerBundleConfig",
    "build_unified_vocab",
    "make_byte_tokens",
    "segment_by_language",
]


def __getattr__(name):
    if name in {"EncodedToken", "DualTrackResult"}:
        from . import encoded

        return getattr(encoded, name)
    if name in {"SEGMENT", "SPECIAL_TOKENS", "build_unified_vocab", "make_byte_tokens"}:
        from . import vocab

        return getattr(vocab, name)
    if name in {"TokenizerBundle", "TokenizerBundleConfig"}:
        from . import bundle

        return getattr(bundle, name)
    if name in __all__:
        from . import dual_tokenizer

        return getattr(dual_tokenizer, name)
    raise AttributeError(name)
