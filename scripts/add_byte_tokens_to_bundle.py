# -*- coding: utf-8 -*-

"""Create a NEW tokenizer bundle with the 256 ``<0xNN>`` byte-fallback tokens.

The shipped bundles (tok_build v1/v2/v3) carry no byte tokens anywhere, so the
lossless OCR-target encoder (``scripts/build_ocr_data.py``) preflight-fails on
them. This tool writes a patched COPY of a bundle (never in place):

- append ``<0x00>..<0xFF>`` to the MorphBPE vocab tail (the segment where byte
  tokens live in real bundles, per ``Tokenizer/tests/test_stream_decode.py``),
  at the next free local ids — the Mongolian segment must have >= 256 free
  slots (v3: 24007 used of 24320);
- regenerate ``vocab.json`` via the same ``build_unified_vocab`` the original
  build used;
- recompute ``manifest.json`` sha256 entries;
- verify: every PRE-EXISTING unified id is unchanged, all 256 byte tokens
  resolve, ``make_ocr_target_encoder`` preflight passes, and a round-trip
  suite (Mongolian + FVS/MVS/NNBSP + CJK/emoji/marker chars) decodes
  byte-exactly.

Append-only by construction: existing ids never move, so LM checkpoints and
previously built data are unaffected; the RDT embedding/head are sized by the
fixed ``VOCAB_SIZE`` (65536) and do not change.

Usage::

    python -m scripts.add_byte_tokens_to_bundle \
        --src /path/to/tok_build_v3/tokenizer/bundle \
        --dst /path/to/tok_build_v3b/tokenizer/bundle
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Tokenizer.unified.vocab import make_byte_tokens  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--src", required=True, help="existing bundle dir (read-only)")
    ap.add_argument("--dst", required=True, help="new bundle dir (must not exist)")
    args = ap.parse_args()

    src = Path(args.src).expanduser()
    dst = Path(args.dst).expanduser()
    if dst.exists():
        raise SystemExit(f"--dst already exists: {dst} (refusing to overwrite)")
    if not (src / "morphbpe.json").is_file():
        raise SystemExit(f"--src is not a bundle dir (no morphbpe.json): {src}")

    dst.mkdir(parents=True)
    for name in os.listdir(src):
        shutil.copy2(src / name, dst / name)

    # 1) Append byte tokens to the MorphBPE vocab tail.
    mb_path = dst / "morphbpe.json"
    mb = json.loads(mb_path.read_text(encoding="utf-8"))
    mvocab: dict[str, int] = mb["vocab"]
    byte_tokens = make_byte_tokens()
    already = [t for t in byte_tokens if t in mvocab]
    if already:
        raise SystemExit(f"src morphbpe already has {len(already)} byte tokens; nothing to do")
    next_id = max(mvocab.values()) + 1
    for i, tok in enumerate(byte_tokens):
        mvocab[tok] = next_id + i
    mb_path.write_text(json.dumps(mb, ensure_ascii=False), encoding="utf-8")

    # 2) Regenerate the unified vocab with the same builder the bundle uses.
    from Tokenizer.generic_bpe.general_bpe import GeneralBPEModel
    from Tokenizer.unified.vocab import SEGMENT, build_unified_vocab

    old_unified = json.loads((src / "vocab.json").read_text(encoding="utf-8"))
    if isinstance(old_unified, dict) and isinstance(old_unified.get("vocab"), dict):
        old_unified = old_unified["vocab"]
    general = GeneralBPEModel.load(str(dst / "general.json"))
    new_unified = build_unified_vocab(mvocab, general.get_vocab())

    mn_lo, mn_hi = SEGMENT["mongolian"]
    if max(mvocab.values()) >= mn_hi - mn_lo:
        raise SystemExit(
            f"mongolian segment overflow: local id {max(mvocab.values())} "
            f">= capacity {mn_hi - mn_lo}"
        )

    changed = {t: (old_unified[t], new_unified[t])
               for t in old_unified
               if t in new_unified and new_unified[t] != old_unified[t]}
    missing_old = [t for t in old_unified if t not in new_unified]
    if changed or missing_old:
        raise SystemExit(
            f"NOT append-only: {len(changed)} ids changed "
            f"(e.g. {list(changed.items())[:3]}), {len(missing_old)} tokens dropped"
        )
    absent = [t for t in byte_tokens if t not in new_unified]
    if absent:
        raise SystemExit(f"{len(absent)} byte tokens absent from new unified vocab")

    (dst / "vocab.json").write_text(
        json.dumps(new_unified, ensure_ascii=False), encoding="utf-8"
    )

    # 3) Manifest hashes.
    man_path = dst / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    for name in man.get("files", {}):
        man["files"][name] = _sha256(dst / name)
    man_path.write_text(json.dumps(man, ensure_ascii=False), encoding="utf-8")

    # 4) Full verification through the real loader + encoder.
    from Tokenizer.unified.bundle import TokenizerBundle
    from scripts.build_ocr_data import make_ocr_target_encoder

    bundle = TokenizerBundle.from_dir(str(dst))
    vv = bundle.tokenizer.vocab
    still_missing = [t for t in byte_tokens if t not in vv]
    if still_missing:
        raise SystemExit(f"loader does not see {len(still_missing)} byte tokens")
    drift = {t: (i, vv[t]) for t, i in old_unified.items() if vv.get(t) != i}
    if drift:
        raise SystemExit(f"loader id drift on {len(drift)} pre-existing tokens")

    enc = make_ocr_target_encoder(bundle.tokenizer)
    probes = [
        "ᠮᠣᠩᠭᠣᠯ᠎ᠠ",            # MVS
        "ᠪᠢᠴᠢᠭ᠋ ᠦᠨ",            # FVS1
        "ᠨᠡᠷ ᠡ",           # NNBSP
        "ᠲᠡᠨᠢᠭᠡᠷ ᠲᠡᠮᠡᠴᠢ ᠬᠣᠶᠠᠷ",
        "abc 123 中文 🙂 ä Ġ ▁ ◈",
    ]
    for s in probes:
        ids = enc(s)  # raises on unk / roundtrip failure
        print(f"[bundle-bytes] roundtrip ok ({len(ids):3d} ids) {s[:24]!r}")

    print(f"[bundle-bytes] wrote {dst}")
    print(f"[bundle-bytes] unified vocab {len(old_unified)} -> {len(new_unified)} "
          f"(+{len(new_unified) - len(old_unified)}); pre-existing ids unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
