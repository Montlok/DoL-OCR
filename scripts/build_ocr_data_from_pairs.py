# -*- coding: utf-8 -*-

"""Build DoL OCR training rows from pre-rendered WebDataset tar pairs.

Unlike :mod:`scripts.build_ocr_data` (which renders text to images with
Pillow), this builder consumes **already-rendered** vertical-script line
images shipped as WebDataset-style tar shards: each sample is a
``<key>.png`` (grayscale vertical strip) + ``<key>.json`` (transcription and
provenance) pair. No rendering happens here — the builder only letterboxes
the strip to a square, resizes it to the training resolution, and emits the
pre-tokenized JSONL rows that ``scripts/train_omvt_ssl.py`` and
``scripts/train_vlm_align.py`` consume directly (see :mod:`Model.ocr.data`
for the row contract).

Routing
-------
Each sample's ``src_doc`` places it in exactly one band:

    train: src_doc <  --val-src-doc-min
    val:   --val-src-doc-min <= src_doc < --test-src-doc-min
    test:  src_doc >= --test-src-doc-min   (reserved; skipped entirely)

The test band is never written anywhere by this tool — it exists so a
later, independently-built evaluation set is comparable to the CRNN OCR
line's held-out split.

Target encoding
----------------
Targets are encoded through :func:`scripts.build_ocr_data.make_ocr_target_encoder`
(byte-fallback, zero-``<unk>``, round-trip verified) — the same lossless
channel the Pillow-rendering builder uses. A round-trip failure aborts the
whole shard loudly (wrapped with shard + key context) rather than skipping
the row: a silently-corrupted OCR label is worse than a stopped build.

Letterboxing
------------
Source strips are narrow-and-tall (``~64x400-900``). Blindly square-resizing
that (what :class:`Tokenizer.multimodal.image_io.PILImageProcessor` does at
train time when handed a non-square image) would squash the glyphs
horizontally. This builder instead pastes the L-mode strip centered on a
white ``max(w,h)`` square, then LANCZOS-resizes that square down to
``--image-size``, and writes the result as the final training PNG. Because
the saved image is already exactly ``image_size x image_size``,
``PILImageProcessor``'s own resize is a no-op at train time — the letterbox
geometry is therefore fixed once, here, not re-derived per run.

Two views, one image
---------------------
Each kept train-band row contributes to two datasets from the *same* saved
PNG: an alignment row (``build_ocr_row`` — image + masked prompt + supervised
target, for ``train_vlm_align.py``) and, while its shard's SSL quota is not
yet exhausted, an OMVT SSL row (``{"images", "image_sizes", "ocr_labels"}``,
for ``train_omvt_ssl.py --data``). ``reading_order`` is omitted: the SSL
trainer's ``arange`` fallback is already correct for a single text line.

Idempotent resume
------------------
Each shard's outputs (images, jsonl rows, meta sidecar) are only considered
valid once its ``done/shard-NNNNN.json`` sentinel exists. A rerun skips any
shard with a sentinel; a shard that crashed mid-way (partial outputs, no
sentinel) has its partial outputs deleted and is rebuilt from scratch, so a
crash never leaves a shard half-written but "invisible" to a resumed run.

Usage::

    python3 -m scripts.build_ocr_data_from_pairs \\
        --shards-dir /nvme/dolocr/pairs \\
        --shard-indices "0:3479:13" \\
        --out /nvme/dolocr/data \\
        --tokenizer-bundle /nvme/dolocr/bundle \\
        --image-size 224 --n-image-tokens 256 --max-seq-len 512 \\
        --val-src-doc-min 434600 --test-src-doc-min 435200 \\
        --val-cap 200000 --ssl-rows 4500000 --workers 12
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import shutil
import sys
import tarfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Model.config import (  # noqa: E402
    BOS_ID,
    EOS_ID,
    IMAGE_END_ID,
    IMAGE_PATCH_ID,
    IMAGE_START_ID,
)
from Model.ocr.data import build_ocr_row  # noqa: E402
from scripts.build_ocr_data import make_ocr_target_encoder  # noqa: E402

# BOS + <image_start> + <image_end> + EOS, matching build_ocr_row(add_eos=True)
# with a single image (n_image_tokens counted separately by the caller). Kept
# identical to scripts/build_ocr_data.py's _PROMPT_FIXED_OVERHEAD.
_PROMPT_FIXED_OVERHEAD = 4

# Files per images/shard-NNNNN/BBB/ bucket; keeps any one directory's dirent
# count well below filesystem-unfriendly territory at 12.7M-image scale.
_IMAGES_PER_BUCKET = 4096


# ===========================================================================
# CLI-level helpers: which shards to build, in what order
# ===========================================================================


def parse_shard_indices(spec: str) -> list[int]:
    """Parse ``--shard-indices``: either ``start:stop:step`` or a comma list.

    ``"0:3479:13"`` behaves like ``range(0, 3479, 13)``. ``"3,17,42"`` is a
    literal, order-preserving index list (duplicates rejected).
    """

    spec = spec.strip()
    if not spec:
        raise ValueError("--shard-indices must not be empty")
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(
                f"--shard-indices range must be 'start:stop' or 'start:stop:step', "
                f"got {spec!r}"
            )
        start = int(parts[0])
        stop = int(parts[1])
        step = int(parts[2]) if len(parts) == 3 else 1
        if step == 0:
            raise ValueError("--shard-indices step must not be 0")
        return list(range(start, stop, step))
    indices = [int(tok) for tok in spec.split(",") if tok.strip() != ""]
    if len(indices) != len(set(indices)):
        raise ValueError(f"--shard-indices comma list has duplicates: {spec!r}")
    return indices


def shard_path(shards_dir: str | Path, index: int) -> Path:
    return Path(shards_dir) / f"shard-{index:05d}.tar"


# ===========================================================================
# Per-shard counters
# ===========================================================================


@dataclass
class ShardCounters:
    """One shard's routing/skip/output tallies (also the sidecar summary)."""

    shard_index: int
    n_members_seen: int = 0
    n_samples_seen: int = 0
    n_orphans: int = 0
    n_train: int = 0
    n_val: int = 0
    n_test_skipped: int = 0
    n_non_line_skipped: int = 0
    n_over_length_skipped: int = 0
    n_align_written: int = 0
    n_val_written: int = 0
    n_ssl_written: int = 0
    n_val_cap_skipped: int = 0
    max_row_len: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "shard_index": self.shard_index,
            "n_members_seen": self.n_members_seen,
            "n_samples_seen": self.n_samples_seen,
            "n_orphans": self.n_orphans,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test_skipped": self.n_test_skipped,
            "n_non_line_skipped": self.n_non_line_skipped,
            "n_over_length_skipped": self.n_over_length_skipped,
            "n_align_written": self.n_align_written,
            "n_val_written": self.n_val_written,
            "n_ssl_written": self.n_ssl_written,
            "n_val_cap_skipped": self.n_val_cap_skipped,
            "max_row_len": self.max_row_len,
        }


# ===========================================================================
# Tar iteration: pair up <key>.png + <key>.json, tolerating either order
# ===========================================================================


def _stem_and_suffix(name: str) -> tuple[str, str]:
    base = os.path.basename(name)
    stem, suffix = os.path.splitext(base)
    return stem, suffix.lower()


def iter_tar_pairs(tar_path: str | Path, counters: ShardCounters):
    """Yield ``(key, png_bytes, meta_dict)`` from a WebDataset-style tar.

    Streams the tar with ``tarfile.open(path, "r|")`` (sequential-only, no
    random access — required for NAS-friendly reads of very large tars).
    Members for one sample are expected adjacent (``<key>.png`` then
    ``<key>.json`` or vice versa) but either order is tolerated by buffering
    at most one pending half-pair by stem. A member whose partner never
    arrives (tar ends, or a different stem interrupts before the partner
    shows up) is counted as an orphan and dropped.
    """

    pending: dict[str, bytes] | None = None
    pending_stem: str | None = None

    def _flush_orphan() -> None:
        nonlocal pending, pending_stem
        if pending is not None:
            counters.n_orphans += 1
        pending = None
        pending_stem = None

    with tarfile.open(tar_path, "r|") as tf:
        for member in tf:
            if not member.isfile():
                continue
            counters.n_members_seen += 1
            stem, suffix = _stem_and_suffix(member.name)
            if suffix not in (".png", ".json"):
                continue
            fh = tf.extractfile(member)
            if fh is None:
                continue
            data = fh.read()

            if pending_stem is not None and stem != pending_stem:
                # A new stem showed up before the pending one's partner did:
                # the pending half is an orphan.
                _flush_orphan()

            if pending_stem is None:
                pending = {suffix: data}
                pending_stem = stem
                continue

            # Same stem as the pending half: this member is its partner
            # (regardless of order) as long as it's the other suffix.
            if suffix in pending:
                # Same suffix twice for one stem — treat the first as an
                # orphan and start fresh with this one.
                _flush_orphan()
                pending = {suffix: data}
                pending_stem = stem
                continue

            pending[suffix] = data
            png_bytes = pending.get(".png")
            json_bytes = pending.get(".json")
            pending = None
            pending_stem = None
            if png_bytes is None or json_bytes is None:
                counters.n_orphans += 1
                continue
            try:
                meta = json.loads(json_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"{tar_path}: key={stem!r}: invalid JSON sidecar: {exc}"
                ) from exc
            counters.n_samples_seen += 1
            yield stem, png_bytes, meta

        _flush_orphan()


# ===========================================================================
# Letterbox: L-mode strip -> centered white square -> LANCZOS to image_size
# ===========================================================================


def letterbox_to_square(png_bytes: bytes, image_size: int):
    """Center-paste an L-mode strip on a white square, then resize it down.

    Returns a ``PIL.Image`` in L mode, exactly ``(image_size, image_size)``.
    The blind square-resize ``PILImageProcessor`` performs at train time is
    only a no-op if the saved file already has this exact shape — this is
    the one place that geometry is decided.
    """

    from PIL import Image

    with Image.open(io.BytesIO(png_bytes)) as raw:
        raw.load()
        strip = raw.convert("L")

    w, h = strip.size
    side = max(w, h)
    canvas = Image.new("L", (side, side), 255)
    canvas.paste(strip, ((side - w) // 2, (side - h) // 2))

    resample = (
        Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    )
    if side != image_size:
        canvas = canvas.resize((image_size, image_size), resample)
    return canvas


# ===========================================================================
# Core per-shard processing (torch-free; takes an injected tokenizer so it
# is directly testable without a real TokenizerBundle on disk)
# ===========================================================================


def route_band(src_doc: int, val_min: int, test_min: int) -> str:
    if src_doc >= test_min:
        return "test"
    if src_doc >= val_min:
        return "val"
    return "train"


def process_shard(
    shard_index: int,
    tar_path: str | Path,
    out_dir: str | Path,
    encode_target: Callable[[str], list[int]],
    *,
    n_image_tokens: int,
    image_size: int,
    max_seq_len: int,
    val_src_doc_min: int,
    test_src_doc_min: int,
    val_cap_per_shard: int,
    ssl_quota_per_shard: int,
    instruction_ids: list[int] | None = None,
) -> ShardCounters:
    """Build one shard's images + JSONL rows + meta sidecar + sentinel.

    Pure function of its arguments (no CLI parsing, no bundle loading) so it
    is directly unit-testable with an injected tokenizer fixture. Writes are
    all-or-nothing at the shard granularity: on any unhandled exception the
    caller is expected to have left no sentinel, so a rerun deletes the
    partial outputs (see :func:`_reset_shard_outputs`) and rebuilds.
    """

    instruction_ids = list(instruction_ids or [])
    prompt_overhead = _PROMPT_FIXED_OVERHEAD + n_image_tokens + len(instruction_ids)

    out_dir = Path(out_dir)
    img_root = out_dir / "images" / f"shard-{shard_index:05d}"
    align_dir = out_dir / "jsonl" / "align"
    val_dir = out_dir / "jsonl" / "val"
    ssl_dir = out_dir / "jsonl" / "ssl"
    meta_dir = out_dir / "meta"
    done_dir = out_dir / "done"
    for d in (img_root, align_dir, val_dir, ssl_dir, meta_dir, done_dir):
        d.mkdir(parents=True, exist_ok=True)

    counters = ShardCounters(shard_index=shard_index)
    ssl_written = 0

    align_path = align_dir / f"shard-{shard_index:05d}.jsonl"
    val_path = val_dir / f"shard-{shard_index:05d}.jsonl"
    ssl_path = ssl_dir / f"shard-{shard_index:05d}.jsonl"
    meta_path = meta_dir / f"shard-{shard_index:05d}.jsonl"

    with align_path.open("w", encoding="utf-8") as align_fh, val_path.open(
        "w", encoding="utf-8"
    ) as val_fh, ssl_path.open("w", encoding="utf-8") as ssl_fh, meta_path.open(
        "w", encoding="utf-8"
    ) as meta_fh:
        index_within_shard = 0
        for key, png_bytes, meta in iter_tar_pairs(tar_path, counters):
            src_doc = meta.get("src_doc")
            if not isinstance(src_doc, int):
                raise ValueError(
                    f"{tar_path}: key={key!r}: missing/invalid integer 'src_doc' "
                    f"in sidecar json (got {src_doc!r})"
                )
            band = route_band(src_doc, val_src_doc_min, test_src_doc_min)
            if band == "test":
                counters.n_test_skipped += 1
                continue

            if meta.get("kind") != "line":
                counters.n_non_line_skipped += 1
                continue

            text = meta.get("text")
            if not isinstance(text, str) or not text:
                raise ValueError(
                    f"{tar_path}: key={key!r}: missing/empty 'text' in sidecar json"
                )

            try:
                target_ids = encode_target(text)
            except ValueError as exc:
                raise ValueError(
                    f"{tar_path}: shard={shard_index} key={key!r}: {exc}"
                ) from exc

            row_len = prompt_overhead + len(target_ids)
            if row_len > max_seq_len:
                counters.n_over_length_skipped += 1
                continue

            if band == "val" and counters.n_val_written >= val_cap_per_shard:
                counters.n_val_cap_skipped += 1
                counters.n_val += 1
                continue
            if band == "val":
                counters.n_val += 1
            else:
                counters.n_train += 1

            bucket = index_within_shard // _IMAGES_PER_BUCKET
            bucket_dir = img_root / f"{bucket:03d}"
            bucket_dir.mkdir(parents=True, exist_ok=True)
            img_path = bucket_dir / f"{key}.png"
            letterboxed = letterbox_to_square(png_bytes, image_size)
            letterboxed.save(img_path)
            abs_img_path = str(img_path.resolve())
            index_within_shard += 1

            row = build_ocr_row(
                target_ids,
                n_image_tokens,
                abs_img_path,
                bos_id=BOS_ID,
                image_start_id=IMAGE_START_ID,
                image_patch_id=IMAGE_PATCH_ID,
                image_end_id=IMAGE_END_ID,
                eos_id=EOS_ID,
                instruction_ids=instruction_ids,
            )
            counters.max_row_len = max(counters.max_row_len, len(row["input_ids"]))

            if band == "val":
                val_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                counters.n_val_written += 1
            else:
                align_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                counters.n_align_written += 1

                if ssl_written < ssl_quota_per_shard:
                    ssl_row = {
                        "images": [abs_img_path],
                        "image_sizes": [[image_size, image_size]],
                        "ocr_labels": [target_ids],
                    }
                    ssl_fh.write(json.dumps(ssl_row, ensure_ascii=False) + "\n")
                    ssl_written += 1
                    counters.n_ssl_written += 1

            meta_fh.write(
                json.dumps(
                    {
                        "key": key,
                        "src_doc": src_doc,
                        "font": meta.get("font"),
                        "font_px": meta.get("font_px"),
                        "bucket": meta.get("bucket"),
                        "n_target_ids": len(target_ids),
                        "band": band,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    sentinel_path = done_dir / f"shard-{shard_index:05d}.json"
    with sentinel_path.open("w", encoding="utf-8") as fh:
        json.dump(counters.as_dict(), fh, ensure_ascii=False, indent=2)
    return counters


def _reset_shard_outputs(shard_index: int, out_dir: str | Path) -> None:
    """Delete a shard's (possibly partial) outputs so it rebuilds cleanly.

    Called before (re)building a shard that has no ``done/`` sentinel, which
    covers both "never built" and "crashed mid-build" — in either case any
    files that happen to exist are stale/incomplete and must not be mistaken
    for valid output by a downstream reader.
    """

    out_dir = Path(out_dir)
    shard_name = f"shard-{shard_index:05d}"
    img_dir = out_dir / "images" / shard_name
    if img_dir.exists():
        shutil.rmtree(img_dir, ignore_errors=True)
    for sub in ("align", "val", "ssl"):
        p = out_dir / "jsonl" / sub / f"{shard_name}.jsonl"
        if p.exists():
            p.unlink()
    meta_path = out_dir / "meta" / f"{shard_name}.jsonl"
    if meta_path.exists():
        meta_path.unlink()


def build_one_shard(
    shard_index: int,
    shards_dir: str | Path,
    out_dir: str | Path,
    encode_target: Callable[[str], list[int]],
    **kwargs: Any,
) -> ShardCounters:
    """Idempotent single-shard entry point: skip if done, else reset + build."""

    out_dir = Path(out_dir)
    sentinel_path = out_dir / "done" / f"shard-{shard_index:05d}.json"
    if sentinel_path.exists():
        with sentinel_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        counters = ShardCounters(**{k: v for k, v in payload.items() if k != "errors"})
        return counters

    _reset_shard_outputs(shard_index, out_dir)
    tar_path = shard_path(shards_dir, shard_index)
    if not tar_path.exists():
        raise FileNotFoundError(f"shard tar not found: {tar_path}")
    return process_shard(
        shard_index,
        tar_path,
        out_dir,
        encode_target,
        **kwargs,
    )


# ===========================================================================
# Multiprocessing fan-out: each worker owns a disjoint list of whole shards
# ===========================================================================


def _worker_main(
    worker_shard_indices: list[int],
    shards_dir: str,
    out_dir: str,
    tokenizer_bundle: str,
    kwargs: dict[str, Any],
    result_queue: "mp.Queue",
) -> None:
    """Spawn-safe worker body: builds its own bundle/encoder, then its shards.

    Each worker process constructs its own :class:`TokenizerBundle` and
    ``encode_target`` closure rather than receiving one via IPC — under the
    ``spawn`` start method a ``DualTrackTokenizer`` (holding compiled regex /
    C-extension state) is not reliably picklable, and re-loading a bundle
    from disk per worker is cheap relative to per-shard image work anyway.
    """

    from Tokenizer.unified.bundle import TokenizerBundle

    bundle = TokenizerBundle.from_dir(tokenizer_bundle)
    encode_target = make_ocr_target_encoder(bundle.tokenizer)

    # A failing shard (e.g. a genuine encode_target round-trip failure) is
    # reported and this worker moves on to its *next* shard rather than
    # aborting outright: shards are independent, and the parent's result
    # loop waits for exactly one message per shard in shard_indices, so
    # silently dropping this worker's remaining shards would hang the
    # parent forever waiting for messages that would never arrive. The
    # failed shard itself writes no sentinel (process_shard's writes happen
    # inside the same try that raised), so a subsequent rerun retries it.
    for shard_index in worker_shard_indices:
        try:
            counters = build_one_shard(
                shard_index,
                shards_dir,
                out_dir,
                encode_target,
                **kwargs,
            )
            result_queue.put(("ok", shard_index, counters.as_dict()))
        except Exception as exc:  # noqa: BLE001 - reported to parent, not swallowed
            result_queue.put(("error", shard_index, f"{type(exc).__name__}: {exc}"))


def _partition_round_robin(items: list[int], n_workers: int) -> list[list[int]]:
    buckets: list[list[int]] = [[] for _ in range(n_workers)]
    for i, item in enumerate(items):
        buckets[i % n_workers].append(item)
    return [b for b in buckets if b]


# ===========================================================================
# CLI
# ===========================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build DoL OCR rows from pre-rendered WebDataset tar pairs "
        "(no rendering; letterbox + tokenize only)"
    )
    ap.add_argument("--shards-dir", required=True, help="dir containing shard-NNNNN.tar")
    ap.add_argument(
        "--shard-indices",
        required=True,
        help="'start:stop:step' (e.g. '0:3479:13') or a comma-separated index list",
    )
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--tokenizer-bundle", required=True)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--n-image-tokens", type=int, default=256)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--val-src-doc-min", type=int, default=434600)
    ap.add_argument("--test-src-doc-min", type=int, default=435200)
    ap.add_argument("--val-cap", type=int, default=200_000)
    ap.add_argument("--ssl-rows", type=int, default=4_500_000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument(
        "--instruction",
        default="",
        help="optional prompt text before the transcription target "
        "(kept empty for v1)",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.image_size <= 0:
        print("[build-ocr-pairs] --image-size must be positive", file=sys.stderr)
        return 2
    if args.n_image_tokens < 1:
        print("[build-ocr-pairs] --n-image-tokens must be >= 1", file=sys.stderr)
        return 2
    if args.val_src_doc_min >= args.test_src_doc_min:
        print(
            "[build-ocr-pairs] --val-src-doc-min must be < --test-src-doc-min",
            file=sys.stderr,
        )
        return 2
    if args.workers < 1:
        print("[build-ocr-pairs] --workers must be >= 1", file=sys.stderr)
        return 2

    try:
        shard_indices = parse_shard_indices(args.shard_indices)
    except ValueError as exc:
        print(f"[build-ocr-pairs] {exc}", file=sys.stderr)
        return 2
    if not shard_indices:
        print("[build-ocr-pairs] --shard-indices resolved to zero shards", file=sys.stderr)
        return 2

    n_shards = len(shard_indices)
    val_cap_per_shard = max(1, args.val_cap // n_shards)
    ssl_quota_per_shard = max(0, args.ssl_rows // n_shards)
    print(
        f"[build-ocr-pairs] {n_shards} shard(s); "
        f"val_cap_per_shard={val_cap_per_shard} (val_cap={args.val_cap} / n_shards, "
        "approximate: the true cap is enforced per-shard, not globally) "
        f"ssl_quota_per_shard={ssl_quota_per_shard}",
        flush=True,
    )

    from Tokenizer.unified.bundle import TokenizerBundle

    # Load once here only to fail fast on a broken --tokenizer-bundle /
    # --instruction before spawning workers; each worker still (re)loads its
    # own bundle (see _worker_main's docstring for why).
    bundle = TokenizerBundle.from_dir(args.tokenizer_bundle)
    instruction_ids = (
        bundle.encode(args.instruction, add_bos=False, add_eos=False)
        if args.instruction
        else []
    )
    make_ocr_target_encoder(bundle.tokenizer)  # preflight: raises on bad bundle
    del bundle

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        n_image_tokens=args.n_image_tokens,
        image_size=args.image_size,
        max_seq_len=args.max_seq_len,
        val_src_doc_min=args.val_src_doc_min,
        test_src_doc_min=args.test_src_doc_min,
        val_cap_per_shard=val_cap_per_shard,
        ssl_quota_per_shard=ssl_quota_per_shard,
        instruction_ids=instruction_ids,
    )

    n_workers = min(args.workers, n_shards)
    partitions = _partition_round_robin(shard_indices, n_workers)

    ctx = mp.get_context("spawn")
    result_queue: "mp.Queue" = ctx.Queue()
    procs = []
    for partition in partitions:
        proc = ctx.Process(
            target=_worker_main,
            args=(
                partition,
                args.shards_dir,
                str(out_dir),
                args.tokenizer_bundle,
                kwargs,
                result_queue,
            ),
        )
        proc.start()
        procs.append(proc)

    total = Counter()
    max_row_len = 0
    n_ok = 0
    n_error = 0
    errors: list[str] = []
    t0 = time.time()
    expected = n_shards
    received = 0
    while received < expected:
        try:
            status, shard_index, payload = result_queue.get(timeout=60)
        except Exception:  # queue.Empty
            # A worker killed by the OS (OOM, signal) never puts its result;
            # without this check the parent would wait forever. Fail loudly
            # instead: the missing shards have no sentinel, so a rerun
            # retries exactly them.
            if not any(proc.is_alive() for proc in procs):
                missing = expected - received
                n_error += missing
                errors.append(
                    f"{missing} shard result(s) never arrived: worker process(es) "
                    "died without reporting (check dmesg for OOM kills)"
                )
                print(
                    f"[build-ocr-pairs] ERROR: all workers exited but only "
                    f"{received}/{expected} shard results arrived",
                    file=sys.stderr,
                    flush=True,
                )
                break
            continue
        received += 1
        if status == "error":
            n_error += 1
            errors.append(f"shard-{shard_index:05d}: {payload}")
            print(
                f"[build-ocr-pairs] ERROR shard-{shard_index:05d}: {payload}",
                file=sys.stderr,
                flush=True,
            )
            continue
        n_ok += 1
        for k, v in payload.items():
            if k == "max_row_len":
                max_row_len = max(max_row_len, v)
            elif k != "shard_index" and isinstance(v, int):
                total[k] += v
        if n_ok % 10 == 0 or received == expected:
            print(
                f"[build-ocr-pairs] {received}/{expected} shards done "
                f"({time.time() - t0:.0f}s elapsed)",
                flush=True,
            )

    for proc in procs:
        proc.join()

    if n_error > 0:
        print(
            f"[build-ocr-pairs] {n_error} shard(s) failed; a rerun will retry "
            "them (no sentinel was written for a failed shard). Errors:",
            file=sys.stderr,
        )
        for line in errors:
            print(f"  {line}", file=sys.stderr)

    n_samples = total["n_samples_seen"]
    print(
        "[build-ocr-pairs] summary: "
        f"shards_ok={n_ok} shards_failed={n_error} "
        f"members_seen={total['n_members_seen']} samples_seen={n_samples} "
        f"orphans={total['n_orphans']} "
        f"train={total['n_train']} val={total['n_val']} test_skipped={total['n_test_skipped']} "
        f"non_line_skipped={total['n_non_line_skipped']} "
        f"over_length_skipped={total['n_over_length_skipped']} "
        f"align_written={total['n_align_written']} val_written={total['n_val_written']} "
        f"ssl_written={total['n_ssl_written']} val_cap_skipped={total['n_val_cap_skipped']} "
        f"max_row_len={max_row_len}",
        flush=True,
    )

    def _pct(part: int, whole: int) -> float:
        return 100.0 * part / whole if whole else 0.0

    if n_samples > 0:
        for label, count in (
            ("orphans", total["n_orphans"]),
            ("non_line_skipped", total["n_non_line_skipped"]),
            ("over_length_skipped", total["n_over_length_skipped"]),
        ):
            pct = _pct(count, n_samples)
            if pct > 1.0:
                print(
                    f"[build-ocr-pairs] WARNING: {label}={count} "
                    f"({pct:.2f}% of samples_seen) exceeds the 1% budget",
                    flush=True,
                )

    return 1 if n_error > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
