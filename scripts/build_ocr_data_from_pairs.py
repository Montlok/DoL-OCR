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

Usage (tar mode)::

    python3 -m scripts.build_ocr_data_from_pairs \\
        --shards-dir /nvme/dolocr/pairs \\
        --shard-indices "0:3479:13" \\
        --out /nvme/dolocr/data \\
        --tokenizer-bundle /nvme/dolocr/bundle \\
        --image-size 224 --n-image-tokens 256 --max-seq-len 512 \\
        --val-src-doc-min 434600 --test-src-doc-min 435200 \\
        --val-cap 200000 --ssl-rows 4500000 --workers 12

Hanshi mode
-----------
A second, mutually-exclusive input mode for the hanshi handwriting corpus:
one flat meta file (JSONL, one row per line image: ``doc_id``, ``kind``,
``text``, ``src_doc``, ``bucket``, ``font``, ``font_px``) plus a
``<pages-root>/<bucket>/<doc_id>.png`` image tree, instead of WebDataset
tar shards. Enabled by passing **both** ``--hanshi-meta`` and
``--hanshi-pages``; passing either of those together with ``--shards-dir``
or ``--shard-indices`` is an argument error (the two modes never mix).

The meta file is streamed once, top to bottom, line by line -- it is tens of
millions of lines, so it is never loaded whole. ``--hanshi-stride K`` keeps
every K-th line (the subsetting mechanism for building a smaller dev slice
without a separate meta file). Kept rows are grouped, in stream order, into
fixed-size "virtual shards" of ``--hanshi-shard-size`` rows each; virtual
shard ``i`` writes to the same ``images/shard-NNNNN/`` /
``jsonl/{align,val,ssl}/shard-NNNNN.jsonl`` / ``meta/shard-NNNNN.jsonl`` /
``done/shard-NNNNN.json`` layout tar mode uses, with
``NNNNN = --shard-offset + i`` so hanshi output can never collide with a
tar-mode shard number (tar shards run ``0..4053``; the default offset,
``10000``, leaves a wide gap). Each virtual shard is routed, letterboxed,
target-encoded, and val-capped/ssl-quota'd through the exact same per-sample
logic as a tar shard (:func:`_process_one_sample`); the only differences are
the image source (a file at ``<pages-root>/<bucket>/<doc_id>.png`` instead of
a tar member -- a **missing file is an orphan, not fatal**) and that
``--val-cap``/``--ssl-rows`` are interpreted as **per-virtual-shard** caps
directly (not divided across shards), because the total row count is not
known without a full scan of the meta file -- see ``--help`` for both flags.

Parallelism in hanshi mode is a single reader process (the only thing that
streams the multi-GB meta file) feeding a bounded work queue of
``(virtual_shard_index, rows)`` batches to a pool of letterbox/encode worker
processes, each of which owns and fully builds whole virtual shards -- never
more than one meta-file read total, regardless of ``--workers``. Idempotency
is identical to tar mode: the reader skips enqueuing a virtual shard whose
``done/`` sentinel already exists, so resuming a partially-built hanshi run
still streams the whole meta file (cheap -- text only) but does no image
work for shards already done.

Usage (hanshi mode)::

    python3 -m scripts.build_ocr_data_from_pairs \\
        --hanshi-meta /nas/hanshi/meta.jsonl \\
        --hanshi-pages /nas/hanshi/pages \\
        --out /nvme/dolocr/data \\
        --tokenizer-bundle /nvme/dolocr/bundle \\
        --image-size 224 --n-image-tokens 256 --max-seq-len 512 \\
        --val-src-doc-min 434600 --test-src-doc-min 435200 \\
        --hanshi-shard-size 100000 --shard-offset 10000 \\
        --val-cap 5000 --ssl-rows 90000 --workers 12
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
# Hanshi meta streaming: flat JSONL -> (kept_idx, doc_id, meta) with stride
# ===========================================================================


def iter_hanshi_meta_rows(meta_path: str | Path, stride: int):
    """Stream ``--hanshi-meta``, yielding ``(kept_idx, doc_id, meta_dict)``.

    Reads the file once, sequentially, line by line -- never loaded whole
    (the real corpus is 83.5M lines). ``stride`` keeps every ``stride``-th
    *raw* line (0-indexed: line 0 is always kept); ``kept_idx`` is a separate
    0-based counter over only the rows that survive the stride, i.e. the
    sequence virtual-shard grouping (:func:`hanshi_virtual_shard_index`) and
    the worker modulo filter both key off. A blank line is skipped without
    consuming a stride slot or a ``kept_idx``. A line that is present but not
    valid JSON, or valid JSON missing a required field, raises immediately
    (loud failure, matching this module's target-encoding contract: a
    silently-skipped/corrupted meta row is worse than a stopped build).
    """

    meta_path = Path(meta_path)
    kept_idx = 0
    with meta_path.open("r", encoding="utf-8") as fh:
        for raw_line_idx, raw in enumerate(fh):
            line = raw.strip()
            if not line:
                continue
            if raw_line_idx % stride != 0:
                continue
            try:
                meta = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{meta_path}:{raw_line_idx + 1}: invalid JSON: {exc}"
                ) from exc
            doc_id = meta.get("doc_id")
            if not isinstance(doc_id, str) or not doc_id:
                raise ValueError(
                    f"{meta_path}:{raw_line_idx + 1}: missing/invalid "
                    f"'doc_id' (got {doc_id!r})"
                )
            yield kept_idx, doc_id, meta
            kept_idx += 1


def hanshi_virtual_shard_index(kept_idx: int, hanshi_shard_size: int) -> int:
    return kept_idx // hanshi_shard_size


def hanshi_output_shard_number(virtual_index: int, shard_offset: int) -> int:
    """Map a 0-based virtual shard index to its output ``shard-NNNNN`` number.

    ``shard_offset`` must be large enough that no hanshi output number ever
    collides with a tar-mode shard number (tar shards run ``0..4053``; the
    CLI default offset ``10000`` leaves a wide gap and is not itself
    validated against a live tar corpus size -- callers pointing both modes
    at the same ``--out`` are responsible for choosing a non-colliding
    offset).
    """

    return shard_offset + virtual_index


def hanshi_image_path(pages_root: str | Path, bucket: str, doc_id: str) -> Path:
    return Path(pages_root) / bucket / f"{doc_id}.png"


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


@dataclass
class _SampleContext:
    """Immutable per-call config shared by every sample in one shard build.

    Bundles the arguments that are constant across an entire
    :func:`process_shard` / hanshi virtual-shard call so the per-sample
    helper (:func:`_process_one_sample`) takes one object instead of
    threading ~10 loose keyword arguments through — the two call sites (tar
    mode, hanshi mode) construct one of these each and otherwise share the
    exact same per-sample body.
    """

    shard_index: int
    encode_target: Callable[[str], list[int]]
    n_image_tokens: int
    image_size: int
    max_seq_len: int
    val_src_doc_min: int
    test_src_doc_min: int
    val_cap_per_shard: int
    ssl_quota_per_shard: int
    instruction_ids: list[int]
    prompt_overhead: int
    source_label: str  # tar path or meta path, for error-message context only


def _process_one_sample(
    key: str,
    load_image_bytes: Callable[[], bytes | None],
    meta: dict[str, Any],
    ctx: _SampleContext,
    *,
    img_root: Path,
    index_within_shard: int,
    align_fh,
    val_fh,
    ssl_fh,
    meta_fh,
    counters: ShardCounters,
    ssl_written: int,
) -> tuple[int, int]:
    """Route, letterbox, encode, and write exactly one sample.

    Shared by the tar-mode loop (:func:`process_shard`) and the hanshi-mode
    virtual-shard loop (:func:`process_hanshi_virtual_shard`) so routing,
    letterboxing, target encoding, row construction, and val-cap/ssl-quota
    bookkeeping cannot drift between the two input modes. ``load_image_bytes``
    is called only once a sample has cleared every skip check (test band,
    non-"line" kind, over-length) — a lazy load so a missing file never costs
    more than the checks that would have skipped the row anyway had it been
    present. Returns the (possibly incremented) ``(index_within_shard,
    ssl_written)`` pair, since Python closures can't mutate caller-scope ints.
    """

    src_doc = meta.get("src_doc")
    if not isinstance(src_doc, int):
        raise ValueError(
            f"{ctx.source_label}: key={key!r}: missing/invalid integer 'src_doc' "
            f"in sidecar json (got {src_doc!r})"
        )
    band = route_band(src_doc, ctx.val_src_doc_min, ctx.test_src_doc_min)
    if band == "test":
        counters.n_test_skipped += 1
        return index_within_shard, ssl_written

    if meta.get("kind") != "line":
        counters.n_non_line_skipped += 1
        return index_within_shard, ssl_written

    text = meta.get("text")
    if not isinstance(text, str) or not text:
        raise ValueError(
            f"{ctx.source_label}: key={key!r}: missing/empty 'text' in sidecar json"
        )

    try:
        target_ids = ctx.encode_target(text)
    except ValueError as exc:
        raise ValueError(
            f"{ctx.source_label}: shard={ctx.shard_index} key={key!r}: {exc}"
        ) from exc

    row_len = ctx.prompt_overhead + len(target_ids)
    if row_len > ctx.max_seq_len:
        counters.n_over_length_skipped += 1
        return index_within_shard, ssl_written

    # Image bytes are only loaded once the row has cleared every band/kind/
    # length check, and a miss (hanshi mode: file not on disk) is an orphan,
    # not a fatal error -- mirrors iter_tar_pairs' unpaired-member handling.
    image_bytes = load_image_bytes()
    if image_bytes is None:
        counters.n_orphans += 1
        return index_within_shard, ssl_written

    if band == "val" and counters.n_val_written >= ctx.val_cap_per_shard:
        counters.n_val_cap_skipped += 1
        counters.n_val += 1
        return index_within_shard, ssl_written
    if band == "val":
        counters.n_val += 1
    else:
        counters.n_train += 1

    bucket = index_within_shard // _IMAGES_PER_BUCKET
    bucket_dir = img_root / f"{bucket:03d}"
    bucket_dir.mkdir(parents=True, exist_ok=True)
    img_path = bucket_dir / f"{key}.png"
    letterboxed = letterbox_to_square(image_bytes, ctx.image_size)
    letterboxed.save(img_path)
    abs_img_path = str(img_path.resolve())
    index_within_shard += 1

    row = build_ocr_row(
        target_ids,
        ctx.n_image_tokens,
        abs_img_path,
        bos_id=BOS_ID,
        image_start_id=IMAGE_START_ID,
        image_patch_id=IMAGE_PATCH_ID,
        image_end_id=IMAGE_END_ID,
        eos_id=EOS_ID,
        instruction_ids=ctx.instruction_ids,
    )
    counters.max_row_len = max(counters.max_row_len, len(row["input_ids"]))

    if band == "val":
        val_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        counters.n_val_written += 1
    else:
        align_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        counters.n_align_written += 1

        if ssl_written < ctx.ssl_quota_per_shard:
            ssl_row = {
                "images": [abs_img_path],
                "image_sizes": [[ctx.image_size, ctx.image_size]],
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
    return index_within_shard, ssl_written


def _open_shard_output_dirs(
    shard_index: int, out_dir: str | Path
) -> tuple[Path, Path, Path, Path, Path, Path]:
    """Create + return ``(img_root, align_dir, val_dir, ssl_dir, meta_dir, done_dir)``."""

    out_dir = Path(out_dir)
    img_root = out_dir / "images" / f"shard-{shard_index:05d}"
    align_dir = out_dir / "jsonl" / "align"
    val_dir = out_dir / "jsonl" / "val"
    ssl_dir = out_dir / "jsonl" / "ssl"
    meta_dir = out_dir / "meta"
    done_dir = out_dir / "done"
    for d in (img_root, align_dir, val_dir, ssl_dir, meta_dir, done_dir):
        d.mkdir(parents=True, exist_ok=True)
    return img_root, align_dir, val_dir, ssl_dir, meta_dir, done_dir


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

    img_root, align_dir, val_dir, ssl_dir, meta_dir, done_dir = _open_shard_output_dirs(
        shard_index, out_dir
    )

    counters = ShardCounters(shard_index=shard_index)
    ssl_written = 0

    align_path = align_dir / f"shard-{shard_index:05d}.jsonl"
    val_path = val_dir / f"shard-{shard_index:05d}.jsonl"
    ssl_path = ssl_dir / f"shard-{shard_index:05d}.jsonl"
    meta_path = meta_dir / f"shard-{shard_index:05d}.jsonl"

    ctx = _SampleContext(
        shard_index=shard_index,
        encode_target=encode_target,
        n_image_tokens=n_image_tokens,
        image_size=image_size,
        max_seq_len=max_seq_len,
        val_src_doc_min=val_src_doc_min,
        test_src_doc_min=test_src_doc_min,
        val_cap_per_shard=val_cap_per_shard,
        ssl_quota_per_shard=ssl_quota_per_shard,
        instruction_ids=instruction_ids,
        prompt_overhead=prompt_overhead,
        source_label=str(tar_path),
    )

    with align_path.open("w", encoding="utf-8") as align_fh, val_path.open(
        "w", encoding="utf-8"
    ) as val_fh, ssl_path.open("w", encoding="utf-8") as ssl_fh, meta_path.open(
        "w", encoding="utf-8"
    ) as meta_fh:
        index_within_shard = 0
        for key, png_bytes, meta in iter_tar_pairs(tar_path, counters):
            index_within_shard, ssl_written = _process_one_sample(
                key,
                lambda b=png_bytes: b,
                meta,
                ctx,
                img_root=img_root,
                index_within_shard=index_within_shard,
                align_fh=align_fh,
                val_fh=val_fh,
                ssl_fh=ssl_fh,
                meta_fh=meta_fh,
                counters=counters,
                ssl_written=ssl_written,
            )

    sentinel_path = done_dir / f"shard-{shard_index:05d}.json"
    with sentinel_path.open("w", encoding="utf-8") as fh:
        json.dump(counters.as_dict(), fh, ensure_ascii=False, indent=2)
    return counters


# ===========================================================================
# Hanshi mode: same per-sample core as process_shard, fed from an in-memory
# list of (doc_id, meta) rows instead of a tar, images loaded from disk.
# ===========================================================================


def process_hanshi_virtual_shard(
    output_shard_number: int,
    rows: list[tuple[str, dict[str, Any]]],
    pages_root: str | Path,
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
    """Build one hanshi virtual shard's outputs, sharing tar mode's per-sample core.

    ``rows`` is ``[(doc_id, meta_dict), ...]`` for exactly the kept meta rows
    assigned to this virtual shard (already stride-filtered and grouped by
    the caller -- see :func:`iter_hanshi_meta_rows` /
    :func:`hanshi_virtual_shard_index`). Writes to the same
    ``images/shard-NNNNN/`` / ``jsonl/*/shard-NNNNN.jsonl`` /
    ``meta/shard-NNNNN.jsonl`` / ``done/shard-NNNNN.json`` layout as
    :func:`process_shard`, keyed by ``output_shard_number`` (the caller has
    already applied ``--shard-offset``). Each row's image is loaded lazily
    from ``<pages_root>/<bucket>/<doc_id>.png``; a missing file is an orphan
    (counted, skipped) rather than a fatal error, since the pages tree is
    only ~1:1 with the meta file, not guaranteed 1:1.

    ``val_cap_per_shard``/``ssl_quota_per_shard`` here are the raw CLI
    ``--val-cap``/``--ssl-rows`` values, unlike tar mode where the caller
    divides them by shard count first: the hanshi caller cannot know the
    total kept-row count (and therefore the true virtual-shard count)
    without a full scan, so these are documented (see ``--help`` and the
    module docstring) as per-virtual-shard caps directly.
    """

    instruction_ids = list(instruction_ids or [])
    prompt_overhead = _PROMPT_FIXED_OVERHEAD + n_image_tokens + len(instruction_ids)

    img_root, align_dir, val_dir, ssl_dir, meta_dir, done_dir = _open_shard_output_dirs(
        output_shard_number, out_dir
    )

    counters = ShardCounters(shard_index=output_shard_number)
    ssl_written = 0

    align_path = align_dir / f"shard-{output_shard_number:05d}.jsonl"
    val_path = val_dir / f"shard-{output_shard_number:05d}.jsonl"
    ssl_path = ssl_dir / f"shard-{output_shard_number:05d}.jsonl"
    meta_path = meta_dir / f"shard-{output_shard_number:05d}.jsonl"

    ctx = _SampleContext(
        shard_index=output_shard_number,
        encode_target=encode_target,
        n_image_tokens=n_image_tokens,
        image_size=image_size,
        max_seq_len=max_seq_len,
        val_src_doc_min=val_src_doc_min,
        test_src_doc_min=test_src_doc_min,
        val_cap_per_shard=val_cap_per_shard,
        ssl_quota_per_shard=ssl_quota_per_shard,
        instruction_ids=instruction_ids,
        prompt_overhead=prompt_overhead,
        source_label=str(pages_root),
    )

    with align_path.open("w", encoding="utf-8") as align_fh, val_path.open(
        "w", encoding="utf-8"
    ) as val_fh, ssl_path.open("w", encoding="utf-8") as ssl_fh, meta_path.open(
        "w", encoding="utf-8"
    ) as meta_fh:
        index_within_shard = 0
        counters.n_samples_seen = len(rows)
        for doc_id, meta in rows:
            bucket = meta.get("bucket")
            if not isinstance(bucket, str) or not bucket:
                raise ValueError(
                    f"{pages_root}: doc_id={doc_id!r}: missing/invalid 'bucket' "
                    f"in meta row (got {bucket!r})"
                )
            img_file = hanshi_image_path(pages_root, bucket, doc_id)

            def _load(path: Path = img_file) -> bytes | None:
                try:
                    return path.read_bytes()
                except FileNotFoundError:
                    return None

            index_within_shard, ssl_written = _process_one_sample(
                doc_id,
                _load,
                meta,
                ctx,
                img_root=img_root,
                index_within_shard=index_within_shard,
                align_fh=align_fh,
                val_fh=val_fh,
                ssl_fh=ssl_fh,
                meta_fh=meta_fh,
                counters=counters,
                ssl_written=ssl_written,
            )

    sentinel_path = done_dir / f"shard-{output_shard_number:05d}.json"
    with sentinel_path.open("w", encoding="utf-8") as fh:
        json.dump(counters.as_dict(), fh, ensure_ascii=False, indent=2)
    return counters


def build_one_hanshi_virtual_shard(
    output_shard_number: int,
    rows: list[tuple[str, dict[str, Any]]],
    pages_root: str | Path,
    out_dir: str | Path,
    encode_target: Callable[[str], list[int]],
    **kwargs: Any,
) -> ShardCounters:
    """Idempotent hanshi virtual-shard entry point: skip if done, else reset + build.

    Mirrors :func:`build_one_shard` exactly (same sentinel check, same
    ``_reset_shard_outputs`` reuse) except it takes ``rows`` directly instead
    of resolving a tar path, since hanshi rows arrive pre-batched from the
    meta-stream reader rather than being re-read from a shard file.
    """

    out_dir = Path(out_dir)
    sentinel_path = out_dir / "done" / f"shard-{output_shard_number:05d}.json"
    if sentinel_path.exists():
        with sentinel_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        counters = ShardCounters(**{k: v for k, v in payload.items() if k != "errors"})
        return counters

    _reset_shard_outputs(output_shard_number, out_dir)
    return process_hanshi_virtual_shard(
        output_shard_number,
        rows,
        pages_root,
        out_dir,
        encode_target,
        **kwargs,
    )


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


@dataclass
class _CollectedResults:
    total: "Counter[str]"
    max_row_len: int
    n_ok: int
    n_error: int
    errors: list[str]


def _collect_shard_results(
    result_queue: "mp.Queue",
    procs: list,
    expected: int,
    *,
    log_prefix: str,
) -> _CollectedResults:
    """Drain ``expected`` ``("ok"|"error", shard_number, payload)`` results.

    Shared tail of both CLI modes' fan-out: tar mode's ``_worker_main`` and
    hanshi mode's per-virtual-shard workers put messages in this exact shape.
    The liveness guard (bail out once every worker process has exited but
    fewer than ``expected`` results have arrived -- an OS-level kill, e.g.
    OOM, never puts a result) is the load-bearing part and must not drift
    between call sites, hence factored here rather than duplicated.
    """

    total: Counter = Counter()
    max_row_len = 0
    n_ok = 0
    n_error = 0
    errors: list[str] = []
    t0 = time.time()
    received = 0
    while received < expected:
        try:
            status, shard_number, payload = result_queue.get(timeout=60)
        except Exception:  # queue.Empty
            if not any(proc.is_alive() for proc in procs):
                missing = expected - received
                n_error += missing
                errors.append(
                    f"{missing} shard result(s) never arrived: worker process(es) "
                    "died without reporting (check dmesg for OOM kills)"
                )
                print(
                    f"{log_prefix} ERROR: all workers exited but only "
                    f"{received}/{expected} shard results arrived",
                    file=sys.stderr,
                    flush=True,
                )
                break
            continue
        received += 1
        if status == "error":
            n_error += 1
            errors.append(f"shard-{shard_number:05d}: {payload}")
            print(
                f"{log_prefix} ERROR shard-{shard_number:05d}: {payload}",
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
                f"{log_prefix} {received}/{expected} shards done "
                f"({time.time() - t0:.0f}s elapsed)",
                flush=True,
            )

    return _CollectedResults(total, max_row_len, n_ok, n_error, errors)


def _print_summary(results: _CollectedResults, *, log_prefix: str) -> None:
    if results.n_error > 0:
        print(
            f"{log_prefix} {results.n_error} shard(s) failed; a rerun will "
            "retry them (no sentinel was written for a failed shard). Errors:",
            file=sys.stderr,
        )
        for line in results.errors:
            print(f"  {line}", file=sys.stderr)

    total = results.total
    n_samples = total["n_samples_seen"]
    print(
        f"{log_prefix} summary: "
        f"shards_ok={results.n_ok} shards_failed={results.n_error} "
        f"members_seen={total['n_members_seen']} samples_seen={n_samples} "
        f"orphans={total['n_orphans']} "
        f"train={total['n_train']} val={total['n_val']} test_skipped={total['n_test_skipped']} "
        f"non_line_skipped={total['n_non_line_skipped']} "
        f"over_length_skipped={total['n_over_length_skipped']} "
        f"align_written={total['n_align_written']} val_written={total['n_val_written']} "
        f"ssl_written={total['n_ssl_written']} val_cap_skipped={total['n_val_cap_skipped']} "
        f"max_row_len={results.max_row_len}",
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
                    f"{log_prefix} WARNING: {label}={count} "
                    f"({pct:.2f}% of samples_seen) exceeds the 1% budget",
                    flush=True,
                )


# ===========================================================================
# Hanshi mode fan-out: one reader process streams the meta file exactly once
# and feeds whole-virtual-shard work items to a pool of letterbox workers.
# ===========================================================================

# Bounds the reader-to-worker queue: at most this many virtual shards' worth
# of (text-only) rows sit in memory waiting for a free worker, so a slow
# worker pool cannot make the reader buffer unboundedly far ahead.
_HANSHI_QUEUE_MAXSIZE = 4

# Sentinel put on the *result* queue once the reader has finished streaming
# the meta file and knows the true total number of virtual shards it
# enqueued (a count not known upfront, unlike tar mode's shard_indices).
_HANSHI_READER_DONE = "__hanshi_reader_done__"


def _hanshi_reader_main(
    meta_path: str,
    pages_root: str,
    out_dir: str,
    stride: int,
    hanshi_shard_size: int,
    shard_offset: int,
    work_queue: "mp.Queue",
    result_queue: "mp.Queue",
) -> None:
    """Single process: stream the meta file once, enqueue virtual-shard batches.

    The only process that reads ``meta_path`` -- regardless of ``--workers``,
    the multi-GB meta file is read exactly once. Rows are buffered per
    virtual shard as they stream in; a virtual shard's batch is enqueued the
    moment its last row (by ``--hanshi-shard-size`` grouping) is seen, or at
    end-of-file for a final partial batch. A virtual shard whose ``done/``
    sentinel already exists is never enqueued at all (cheap check, no image
    work), which is how a resumed hanshi run skips finished shards while
    still paying the (cheap, text-only) cost of streaming past their rows.
    """

    out_dir_path = Path(out_dir)
    current_virtual_index: int | None = None
    current_rows: list[tuple[str, dict[str, Any]]] = []
    n_enqueued = 0

    def _flush() -> None:
        nonlocal current_virtual_index, current_rows, n_enqueued
        if current_virtual_index is None or not current_rows:
            current_virtual_index = None
            current_rows = []
            return
        output_shard_number = hanshi_output_shard_number(
            current_virtual_index, shard_offset
        )
        sentinel_path = out_dir_path / "done" / f"shard-{output_shard_number:05d}.json"
        if not sentinel_path.exists():
            work_queue.put((output_shard_number, current_rows))
            n_enqueued += 1
        current_virtual_index = None
        current_rows = []

    for kept_idx, doc_id, meta in iter_hanshi_meta_rows(meta_path, stride):
        virtual_index = hanshi_virtual_shard_index(kept_idx, hanshi_shard_size)
        if current_virtual_index is not None and virtual_index != current_virtual_index:
            _flush()
        current_virtual_index = virtual_index
        current_rows.append((doc_id, meta))
    _flush()

    result_queue.put((_HANSHI_READER_DONE, n_enqueued, None))


def _hanshi_worker_main(
    pages_root: str,
    out_dir: str,
    tokenizer_bundle: str,
    kwargs: dict[str, Any],
    work_queue: "mp.Queue",
    result_queue: "mp.Queue",
) -> None:
    """Spawn-safe hanshi worker: pulls whole virtual shards off the work queue.

    Loads its own :class:`TokenizerBundle` for the same reason
    :func:`_worker_main` does (spawn-safety; see that docstring). Exits when
    it pulls the ``None`` poison pill the parent sends once the reader has
    finished and every enqueued item has been claimed.
    """

    from Tokenizer.unified.bundle import TokenizerBundle

    bundle = TokenizerBundle.from_dir(tokenizer_bundle)
    encode_target = make_ocr_target_encoder(bundle.tokenizer)

    while True:
        item = work_queue.get()
        if item is None:
            return
        output_shard_number, rows = item
        try:
            counters = build_one_hanshi_virtual_shard(
                output_shard_number,
                rows,
                pages_root,
                out_dir,
                encode_target,
                **kwargs,
            )
            result_queue.put(("ok", output_shard_number, counters.as_dict()))
        except Exception as exc:  # noqa: BLE001 - reported to parent, not swallowed
            result_queue.put(
                ("error", output_shard_number, f"{type(exc).__name__}: {exc}")
            )


def _run_hanshi_mode(args: argparse.Namespace, out_dir: Path) -> int:
    """Drive hanshi mode end to end: reader process + worker pool + summary.

    ``expected`` (total virtual shard count) is unknown until the reader has
    streamed the whole meta file, unlike tar mode where ``--shard-indices``
    fixes it upfront -- so this loop first drains results until it sees the
    reader's ``_HANSHI_READER_DONE`` sentinel (which carries the true count),
    then keeps draining (reusing :func:`_collect_shard_results`'s liveness
    guard) until that many ``ok``/``error`` results have also arrived.
    """

    from Tokenizer.unified.bundle import TokenizerBundle

    bundle = TokenizerBundle.from_dir(args.tokenizer_bundle)
    instruction_ids = (
        bundle.encode(args.instruction, add_bos=False, add_eos=False)
        if args.instruction
        else []
    )
    make_ocr_target_encoder(bundle.tokenizer)  # preflight: raises on bad bundle
    del bundle

    kwargs = dict(
        n_image_tokens=args.n_image_tokens,
        image_size=args.image_size,
        max_seq_len=args.max_seq_len,
        val_src_doc_min=args.val_src_doc_min,
        test_src_doc_min=args.test_src_doc_min,
        val_cap_per_shard=args.val_cap,
        ssl_quota_per_shard=args.ssl_rows,
        instruction_ids=instruction_ids,
    )
    print(
        "[build-ocr-pairs] hanshi mode: hanshi_shard_size="
        f"{args.hanshi_shard_size} shard_offset={args.shard_offset} "
        f"stride={args.hanshi_stride} val_cap_per_shard={args.val_cap} "
        "(per-virtual-shard, NOT divided across shards -- see --help) "
        f"ssl_quota_per_shard={args.ssl_rows} workers={args.workers}",
        flush=True,
    )

    ctx = mp.get_context("spawn")
    work_queue: "mp.Queue" = ctx.Queue(maxsize=_HANSHI_QUEUE_MAXSIZE)
    result_queue: "mp.Queue" = ctx.Queue()

    reader_proc = ctx.Process(
        target=_hanshi_reader_main,
        args=(
            args.hanshi_meta,
            args.hanshi_pages,
            str(out_dir),
            args.hanshi_stride,
            args.hanshi_shard_size,
            args.shard_offset,
            work_queue,
            result_queue,
        ),
    )
    reader_proc.start()

    worker_procs = []
    for _ in range(args.workers):
        proc = ctx.Process(
            target=_hanshi_worker_main,
            args=(
                args.hanshi_pages,
                str(out_dir),
                args.tokenizer_bundle,
                kwargs,
                work_queue,
                result_queue,
            ),
        )
        proc.start()
        worker_procs.append(proc)

    log_prefix = "[build-ocr-pairs]"
    all_procs = [reader_proc, *worker_procs]

    # Phase 1: drain the result queue until the reader's done-sentinel shows
    # up, learning the true expected count. Workers may also report ok/error
    # results interleaved with this wait, so those are folded in eagerly
    # rather than discarded.
    total: Counter = Counter()
    max_row_len = 0
    n_ok = 0
    n_error = 0
    errors: list[str] = []
    expected: int | None = None
    received = 0
    t0 = time.time()
    while expected is None:
        try:
            status, payload_a, payload_b = result_queue.get(timeout=60)
        except Exception:  # queue.Empty
            if not reader_proc.is_alive():
                print(
                    f"{log_prefix} ERROR: reader process exited without "
                    "reporting how many virtual shards it enqueued",
                    file=sys.stderr,
                    flush=True,
                )
                expected = received  # stop waiting; report what we have
                break
            continue
        if status == _HANSHI_READER_DONE:
            expected = payload_a
            print(
                f"{log_prefix} reader done: {expected} virtual shard(s) "
                f"enqueued ({time.time() - t0:.0f}s)",
                flush=True,
            )
            continue
        received += 1
        if status == "error":
            n_error += 1
            errors.append(f"shard-{payload_a:05d}: {payload_b}")
            print(
                f"{log_prefix} ERROR shard-{payload_a:05d}: {payload_b}",
                file=sys.stderr,
                flush=True,
            )
        else:
            n_ok += 1
            for k, v in payload_b.items():
                if k == "max_row_len":
                    max_row_len = max(max_row_len, v)
                elif k != "shard_index" and isinstance(v, int):
                    total[k] += v

    # Tell every worker to stop once no more work is coming; harmless if a
    # worker is still mid-shard, since it drains the queue before exiting.
    for _ in worker_procs:
        work_queue.put(None)

    # Phase 2: drain any remaining ok/error results (reusing the shared
    # liveness-guarded loop) for however many are still outstanding.
    remaining = expected - received
    if remaining > 0:
        rest = _collect_shard_results(
            result_queue, all_procs, remaining, log_prefix=log_prefix
        )
        total.update(rest.total)
        max_row_len = max(max_row_len, rest.max_row_len)
        n_ok += rest.n_ok
        n_error += rest.n_error
        errors.extend(rest.errors)

    for proc in all_procs:
        proc.join()

    results = _CollectedResults(total, max_row_len, n_ok, n_error, errors)
    _print_summary(results, log_prefix=log_prefix)
    return 1 if n_error > 0 else 0


# ===========================================================================
# CLI
# ===========================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build DoL OCR rows from pre-rendered pairs -- either "
        "WebDataset tar shards (--shards-dir/--shard-indices) or the hanshi "
        "handwriting corpus's flat meta file (--hanshi-meta/--hanshi-pages). "
        "Exactly one mode's flags must be given; no rendering happens in "
        "either mode (letterbox + tokenize only)."
    )
    ap.add_argument(
        "--shards-dir",
        default=None,
        help="[tar mode] dir containing shard-NNNNN.tar; mutually exclusive "
        "with --hanshi-meta/--hanshi-pages",
    )
    ap.add_argument(
        "--shard-indices",
        default=None,
        help="[tar mode] 'start:stop:step' (e.g. '0:3479:13') or a "
        "comma-separated index list; required iff --shards-dir is given",
    )
    ap.add_argument(
        "--hanshi-meta",
        default=None,
        help="[hanshi mode] path to the hanshi corpus meta JSONL (one row "
        "per line image: doc_id/kind/text/src_doc/bucket/font/font_px); "
        "mutually exclusive with --shards-dir/--shard-indices",
    )
    ap.add_argument(
        "--hanshi-pages",
        default=None,
        help="[hanshi mode] root dir of <bucket>/<doc_id>.png page images; "
        "required iff --hanshi-meta is given",
    )
    ap.add_argument(
        "--hanshi-stride",
        type=int,
        default=1,
        help="[hanshi mode] keep every K-th meta line (0-indexed, line 0 "
        "always kept); the subsetting mechanism for a smaller dev slice "
        "without a separate meta file (default: 1, i.e. keep every line)",
    )
    ap.add_argument(
        "--hanshi-shard-size",
        type=int,
        default=100_000,
        help="[hanshi mode] number of consecutive *kept* meta rows per "
        "virtual shard (default: 100000)",
    )
    ap.add_argument(
        "--shard-offset",
        type=int,
        default=10_000,
        help="[hanshi mode] virtual shard i writes outputs named "
        "shard-(offset+i), so hanshi output never collides with tar-mode "
        "shard numbers (tar shards run 0..4053; default offset 10000 "
        "leaves a wide gap)",
    )
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--tokenizer-bundle", required=True)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--n-image-tokens", type=int, default=256)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--val-src-doc-min", type=int, default=434600)
    ap.add_argument("--test-src-doc-min", type=int, default=435200)
    ap.add_argument(
        "--val-cap",
        type=int,
        default=200_000,
        help="[tar mode] total val-row cap, divided across shards "
        "(val_cap // n_shards each). [hanshi mode] this same value is used "
        "*directly* as the per-virtual-shard cap, NOT divided -- the total "
        "kept-row count (and therefore virtual shard count) is not known "
        "without a full scan of --hanshi-meta, so hanshi mode cannot derive "
        "a global cap the way tar mode does. Pass a per-shard-sized number "
        "in hanshi mode, not a corpus-wide total.",
    )
    ap.add_argument(
        "--ssl-rows",
        type=int,
        default=4_500_000,
        help="[tar mode] total SSL-row quota, divided across shards "
        "(ssl_rows // n_shards each). [hanshi mode] used directly as the "
        "per-virtual-shard quota, NOT divided -- same reasoning as --val-cap "
        "above; pass a per-shard-sized number in hanshi mode.",
    )
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument(
        "--instruction",
        default="",
        help="optional prompt text before the transcription target "
        "(kept empty for v1)",
    )
    args = ap.parse_args(argv)

    tar_flags_given = args.shards_dir is not None or args.shard_indices is not None
    hanshi_flags_given = args.hanshi_meta is not None or args.hanshi_pages is not None
    if tar_flags_given and hanshi_flags_given:
        ap.error(
            "--shards-dir/--shard-indices and --hanshi-meta/--hanshi-pages "
            "are mutually exclusive (pick exactly one input mode)"
        )
    if not tar_flags_given and not hanshi_flags_given:
        ap.error(
            "must pass either --shards-dir+--shard-indices (tar mode) or "
            "--hanshi-meta+--hanshi-pages (hanshi mode)"
        )
    if tar_flags_given and (args.shards_dir is None or args.shard_indices is None):
        ap.error("tar mode requires both --shards-dir and --shard-indices")
    if hanshi_flags_given and (args.hanshi_meta is None or args.hanshi_pages is None):
        ap.error("hanshi mode requires both --hanshi-meta and --hanshi-pages")
    if hanshi_flags_given and args.hanshi_stride < 1:
        ap.error("--hanshi-stride must be >= 1")
    if hanshi_flags_given and args.hanshi_shard_size < 1:
        ap.error("--hanshi-shard-size must be >= 1")

    return args


def _run_tar_mode(args: argparse.Namespace, out_dir: Path) -> int:
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

    results = _collect_shard_results(
        result_queue, procs, n_shards, log_prefix="[build-ocr-pairs]"
    )
    for proc in procs:
        proc.join()

    _print_summary(results, log_prefix="[build-ocr-pairs]")
    return 1 if results.n_error > 0 else 0


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

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.hanshi_meta is not None:
        return _run_hanshi_mode(args, out_dir)
    return _run_tar_mode(args, out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
