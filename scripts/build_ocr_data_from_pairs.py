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
Targets use the shared native OCR tokenizer contract: the exact span-aware
representation learned during language pretraining, including persisted
morphology features. ``<unk>``, Mongolian-to-general fallback, or a round-trip
change aborts the shard rather than changing frozen-LM supervision.

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
Each shard is only complete when its sentinel matches the current tokenizer
contract, producer code, build parameters, source identity, and exact
JSONL/meta hashes. Legacy, partial, corrupt, or stale sentinels cause a clean
rebuild. Image digests live in each row and are verified lazily when consumed.

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
more than one meta-file read total, regardless of ``--workers``. The reader
enqueues current rows for every virtual shard; workers validate the full
sentinel before deciding whether image work can be skipped.

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
import hashlib
import io
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
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
from Model.ocr.alignment_contract import (  # noqa: E402
    build_ocr_alignment_data_contract,
    file_sha256,
    write_json_atomically,
    write_ocr_alignment_data_contract,
)
from Model.ocr.data import build_ocr_row  # noqa: E402
from Model.ocr.image_preprocess import (  # noqa: E402
    letterbox_grayscale_to_square,
)
from Model.ocr.pair_shards import ShardCounters, iter_tar_pairs  # noqa: E402
from Model.ocr.shard_contract import (  # noqa: E402
    DigestingTextWriter,
    ImageBindingAccumulator,
    build_shard_sentinel,
    producer_algorithm_contract,
    update_canonical_manifest,
    validate_sha256,
    validate_shard_sentinel,
    validate_shard_sentinel_payload,
)
from Model.ocr.tokenization import (  # noqa: E402
    canonical_json_sha256,
    encode_lm_text_features,
    make_ocr_target_encoder,
    native_tokenization_contract,
)

# BOS + <image_start> + <image_end> + EOS, matching build_ocr_row(add_eos=True)
# with a single image (n_image_tokens counted separately by the caller). Kept
# identical to scripts/build_ocr_data.py's _PROMPT_FIXED_OVERHEAD.
_PROMPT_FIXED_OVERHEAD = 4

# Files per images/shard-NNNNN/BBB/ bucket; keeps any one directory's dirent
# count well below filesystem-unfriendly territory at 12.7M-image scale.
_IMAGES_PER_BUCKET = 4096
_SEMANTIC_BUILD_PARAMETER_KEYS = (
    "n_image_tokens",
    "image_size",
    "max_seq_len",
    "val_src_doc_min",
    "test_src_doc_min",
    "instruction_ids",
    "instruction_track_ids",
)


def _shard_build_parameters(
    *,
    n_image_tokens: int,
    image_size: int,
    max_seq_len: int,
    val_src_doc_min: int,
    test_src_doc_min: int,
    val_cap_per_shard: int,
    ssl_quota_per_shard: int,
    instruction_ids: list[int] | None,
    instruction_track_ids: list[int] | None,
) -> dict[str, Any]:
    """Canonical set of every option that changes emitted shard bytes."""

    return {
        "n_image_tokens": int(n_image_tokens),
        "image_size": int(image_size),
        "max_seq_len": int(max_seq_len),
        "val_src_doc_min": int(val_src_doc_min),
        "test_src_doc_min": int(test_src_doc_min),
        "val_cap_per_shard": int(val_cap_per_shard),
        "ssl_quota_per_shard": int(ssl_quota_per_shard),
        "instruction_ids": [int(value) for value in instruction_ids or []],
        "instruction_track_ids": [
            int(value) for value in instruction_track_ids or []
        ],
    }


def _build_parameters_from_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return _shard_build_parameters(
        n_image_tokens=kwargs["n_image_tokens"],
        image_size=kwargs["image_size"],
        max_seq_len=kwargs["max_seq_len"],
        val_src_doc_min=kwargs["val_src_doc_min"],
        test_src_doc_min=kwargs["test_src_doc_min"],
        val_cap_per_shard=kwargs["val_cap_per_shard"],
        ssl_quota_per_shard=kwargs["ssl_quota_per_shard"],
        instruction_ids=kwargs.get("instruction_ids"),
        instruction_track_ids=kwargs.get("instruction_track_ids"),
    )


def _tar_source_identity(tar_path: Path) -> dict[str, Any]:
    stat = tar_path.stat()
    return {
        "kind": "tar_stat_v1",
        "path": str(tar_path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }


def _hanshi_source_identity(
    rows: list[tuple[str, dict[str, Any]]],
    pages_root: str | Path,
) -> dict[str, Any]:
    row_manifest = hashlib.sha256()
    for doc_id, meta in rows:
        update_canonical_manifest(row_manifest, [doc_id, meta])
    return {
        "kind": "hanshi_rows_v1",
        "pages_root": str(Path(pages_root).resolve()),
        "row_count": len(rows),
        "rows_sha256": row_manifest.hexdigest(),
    }


@lru_cache(maxsize=1)
def _producer_algorithm_contract() -> dict[str, Any]:
    """Fingerprint code that turns source pairs into emitted OCR rows."""

    return producer_algorithm_contract(
        (
            ("scripts/build_ocr_data_from_pairs.py", Path(__file__)),
            ("Model/ocr/data.py", Path(_REPO_ROOT) / "Model" / "ocr" / "data.py"),
            (
                "Model/ocr/image_preprocess.py",
                Path(_REPO_ROOT) / "Model" / "ocr" / "image_preprocess.py",
            ),
            (
                "Model/ocr/pair_shards.py",
                Path(_REPO_ROOT) / "Model" / "ocr" / "pair_shards.py",
            ),
        )
    )


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


_COUNTER_KEYS = frozenset(ShardCounters(0).as_dict())


def _build_shard_completion(
    *,
    mode: str,
    shard_index: int,
    tokenization_contract_sha256: str,
    build_parameters: dict[str, Any],
    source_identity: dict[str, Any],
    source_manifest_sha256: str,
    source_record_count: int,
    output_artifacts: list[dict[str, Any]],
    image_manifest: ImageBindingAccumulator,
    counters: ShardCounters,
) -> dict[str, Any]:
    return build_shard_sentinel(
        mode=mode,
        shard_index=shard_index,
        tokenization_contract_sha256=tokenization_contract_sha256,
        producer_algorithm=_producer_algorithm_contract(),
        build_parameters=build_parameters,
        source_identity=source_identity,
        source_manifest_sha256=source_manifest_sha256,
        source_record_count=source_record_count,
        output_artifacts=output_artifacts,
        image_manifest=image_manifest,
        counters=counters.as_dict(),
    )


def _load_valid_shard_sentinel(
    sentinel_path: Path,
    out_dir: Path,
    *,
    mode: str,
    shard_index: int,
    tokenization_contract_sha256: str,
    build_parameters: dict[str, Any],
    source_identity: dict[str, Any],
) -> ShardCounters | None:
    if not sentinel_path.is_file():
        return None
    try:
        counters = validate_shard_sentinel(
            sentinel_path,
            out_dir,
            mode=mode,
            shard_index=shard_index,
            tokenization_contract_sha256=tokenization_contract_sha256,
            producer_algorithm=_producer_algorithm_contract(),
            build_parameters=build_parameters,
            source_identity=source_identity,
            counter_keys=_COUNTER_KEYS,
        )
        return ShardCounters(**counters)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(
            f"[build-ocr-pairs] invalid sentinel {sentinel_path}: {exc}; "
            "rebuilding shard",
            flush=True,
        )
        return None


def _validated_shard_completion_manifest(
    data_dir: Path,
    out_dir: Path,
    *,
    tokenization_contract_sha256: str,
    expected_semantics: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prove every JSONL included in an aggregate receipt has a current sentinel."""

    data_files = sorted(data_dir.glob("shard-*.jsonl"))
    if not data_files:
        raise ValueError(f"no shard JSONL files found in {data_dir}")
    producer_algorithm = _producer_algorithm_contract()
    sentinel_entries: list[dict[str, str]] = []
    semantics = expected_semantics
    for data_path in data_files:
        shard_suffix = data_path.stem.removeprefix("shard-")
        if not shard_suffix.isdigit():
            raise ValueError(f"invalid OCR shard filename: {data_path}")
        shard_index = int(shard_suffix)
        sentinel_path = out_dir / "done" / f"shard-{shard_index:05d}.json"
        if not sentinel_path.is_file():
            raise ValueError(
                f"{data_path} has no completion sentinel; refusing to re-sign "
                "unproven shard output"
            )
        payload = validate_shard_sentinel_payload(
            sentinel_path,
            out_dir,
            counter_keys=_COUNTER_KEYS,
        )
        if payload["shard_index"] != shard_index:
            raise ValueError(
                f"{sentinel_path} claims shard {payload['shard_index']}, "
                f"expected {shard_index}"
            )
        if (
            payload["tokenization_contract_sha256"]
            != tokenization_contract_sha256
        ):
            raise ValueError(
                f"{sentinel_path} was built with a different tokenizer contract"
            )
        if payload["producer_algorithm"] != producer_algorithm:
            raise ValueError(
                f"{sentinel_path} was built by a different producer algorithm"
            )
        build_parameters = payload["build_parameters"]
        missing = [
            key
            for key in _SEMANTIC_BUILD_PARAMETER_KEYS
            if key not in build_parameters
        ]
        if missing:
            raise ValueError(
                f"{sentinel_path} omits semantic build parameters: "
                + ", ".join(missing)
            )
        current_semantics = {
            key: build_parameters[key]
            for key in _SEMANTIC_BUILD_PARAMETER_KEYS
        }
        if semantics is None:
            semantics = current_semantics
        elif current_semantics != semantics:
            raise ValueError(
                f"{sentinel_path} has incompatible OCR semantic build parameters"
            )
        expected_output = data_path.relative_to(out_dir).as_posix()
        output_paths = {
            entry["path"] for entry in payload["outputs"]
        }
        if expected_output not in output_paths:
            raise ValueError(
                f"{sentinel_path} does not bind included shard {data_path}"
            )
        sentinel_entries.append(
            {
                "name": sentinel_path.name,
                "sha256": file_sha256(sentinel_path),
            }
        )
    assert semantics is not None
    return (
        {
            "mode": "validated_shard_sentinels_v2",
            "sentinel_count": len(sentinel_entries),
            "sentinels": sentinel_entries,
            "sentinel_manifest_sha256": canonical_json_sha256(
                sentinel_entries
            ),
        },
        semantics,
    )


def letterbox_to_square(png_bytes: bytes, image_size: int):
    """Compatibility wrapper for the shared legacy OCR letterbox contract."""

    return letterbox_grayscale_to_square(png_bytes, image_size)


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
    instruction_track_ids: list[int]
    prompt_overhead: int
    image_manifest: ImageBindingAccumulator
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
        encode_with_features = getattr(
            ctx.encode_target,
            "encode_with_features",
            None,
        )
        if not callable(encode_with_features):
            raise TypeError(
                "strict OCR shard building requires an encoder that emits "
                "pretraining morphology features"
            )
        target_features = encode_with_features(text)
        target_ids = target_features.input_ids
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
    encoded_image = io.BytesIO()
    letterboxed.save(encoded_image, format="PNG")
    output_image_bytes = encoded_image.getvalue()
    img_path.write_bytes(output_image_bytes)
    abs_img_path = str(img_path.resolve())
    image_size_bytes = len(output_image_bytes)
    image_sha256 = hashlib.sha256(output_image_bytes).hexdigest()
    ctx.image_manifest.add(abs_img_path, image_sha256, image_size_bytes)
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
        instruction_track_ids=ctx.instruction_track_ids,
        target_track_ids=target_features.morphology_track_ids,
        image_sha256=image_sha256,
        image_size_bytes=image_size_bytes,
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
    tokenization_contract_sha256: str,
    instruction_ids: list[int] | None = None,
    instruction_track_ids: list[int] | None = None,
) -> ShardCounters:
    """Build one shard's images + JSONL rows + meta sidecar + sentinel.

    Pure function of its arguments (no CLI parsing, no bundle loading) so it
    is directly unit-testable with an injected tokenizer fixture. Writes are
    all-or-nothing at the shard granularity: on any unhandled exception the
    caller is expected to have left no sentinel, so a rerun deletes the
    partial outputs (see :func:`_reset_shard_outputs`) and rebuilds.
    """

    instruction_ids = list(instruction_ids or [])
    instruction_track_ids = list(instruction_track_ids or [])
    if len(instruction_track_ids) != len(instruction_ids):
        raise ValueError("instruction_track_ids must align with instruction_ids")
    prompt_overhead = _PROMPT_FIXED_OVERHEAD + n_image_tokens + len(instruction_ids)
    validate_sha256(
        tokenization_contract_sha256,
        field_name="tokenization_contract_sha256",
    )
    build_parameters = _shard_build_parameters(
        n_image_tokens=n_image_tokens,
        image_size=image_size,
        max_seq_len=max_seq_len,
        val_src_doc_min=val_src_doc_min,
        test_src_doc_min=test_src_doc_min,
        val_cap_per_shard=val_cap_per_shard,
        ssl_quota_per_shard=ssl_quota_per_shard,
        instruction_ids=instruction_ids,
        instruction_track_ids=instruction_track_ids,
    )

    img_root, align_dir, val_dir, ssl_dir, meta_dir, done_dir = _open_shard_output_dirs(
        shard_index, out_dir
    )

    counters = ShardCounters(shard_index=shard_index)
    ssl_written = 0
    image_manifest = ImageBindingAccumulator()
    source_manifest = hashlib.sha256()

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
        instruction_track_ids=instruction_track_ids,
        prompt_overhead=prompt_overhead,
        image_manifest=image_manifest,
        source_label=str(tar_path),
    )

    with align_path.open("w", encoding="utf-8") as align_handle, val_path.open(
        "w", encoding="utf-8"
    ) as val_handle, ssl_path.open("w", encoding="utf-8") as ssl_handle, meta_path.open(
        "w", encoding="utf-8"
    ) as meta_handle:
        align_fh = DigestingTextWriter(align_handle, align_path)
        val_fh = DigestingTextWriter(val_handle, val_path)
        ssl_fh = DigestingTextWriter(ssl_handle, ssl_path)
        meta_fh = DigestingTextWriter(meta_handle, meta_path)
        index_within_shard = 0
        for key, png_bytes, meta in iter_tar_pairs(tar_path, counters):
            update_canonical_manifest(
                source_manifest,
                [
                    key,
                    hashlib.sha256(png_bytes).hexdigest(),
                    canonical_json_sha256(meta),
                ],
            )
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

    output_artifacts = [
        writer.artifact(Path(out_dir))
        for writer in (align_fh, val_fh, ssl_fh, meta_fh)
    ]
    sentinel_path = done_dir / f"shard-{shard_index:05d}.json"
    sentinel = _build_shard_completion(
        mode="tar",
        shard_index=shard_index,
        tokenization_contract_sha256=tokenization_contract_sha256,
        build_parameters=build_parameters,
        source_identity=_tar_source_identity(Path(tar_path)),
        source_manifest_sha256=source_manifest.hexdigest(),
        source_record_count=counters.n_samples_seen,
        output_artifacts=output_artifacts,
        image_manifest=image_manifest,
        counters=counters,
    )
    write_json_atomically(sentinel_path, sentinel)
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
    tokenization_contract_sha256: str,
    instruction_ids: list[int] | None = None,
    instruction_track_ids: list[int] | None = None,
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
    instruction_track_ids = list(instruction_track_ids or [])
    if len(instruction_track_ids) != len(instruction_ids):
        raise ValueError("instruction_track_ids must align with instruction_ids")
    prompt_overhead = _PROMPT_FIXED_OVERHEAD + n_image_tokens + len(instruction_ids)
    if not rows:
        raise ValueError("hanshi virtual shards must contain at least one row")
    validate_sha256(
        tokenization_contract_sha256,
        field_name="tokenization_contract_sha256",
    )
    build_parameters = _shard_build_parameters(
        n_image_tokens=n_image_tokens,
        image_size=image_size,
        max_seq_len=max_seq_len,
        val_src_doc_min=val_src_doc_min,
        test_src_doc_min=test_src_doc_min,
        val_cap_per_shard=val_cap_per_shard,
        ssl_quota_per_shard=ssl_quota_per_shard,
        instruction_ids=instruction_ids,
        instruction_track_ids=instruction_track_ids,
    )
    source_identity = _hanshi_source_identity(rows, pages_root)

    img_root, align_dir, val_dir, ssl_dir, meta_dir, done_dir = _open_shard_output_dirs(
        output_shard_number, out_dir
    )

    counters = ShardCounters(shard_index=output_shard_number)
    ssl_written = 0
    image_manifest = ImageBindingAccumulator()

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
        instruction_track_ids=instruction_track_ids,
        prompt_overhead=prompt_overhead,
        image_manifest=image_manifest,
        source_label=str(pages_root),
    )

    with align_path.open("w", encoding="utf-8") as align_handle, val_path.open(
        "w", encoding="utf-8"
    ) as val_handle, ssl_path.open("w", encoding="utf-8") as ssl_handle, meta_path.open(
        "w", encoding="utf-8"
    ) as meta_handle:
        align_fh = DigestingTextWriter(align_handle, align_path)
        val_fh = DigestingTextWriter(val_handle, val_path)
        ssl_fh = DigestingTextWriter(ssl_handle, ssl_path)
        meta_fh = DigestingTextWriter(meta_handle, meta_path)
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

    output_artifacts = [
        writer.artifact(Path(out_dir))
        for writer in (align_fh, val_fh, ssl_fh, meta_fh)
    ]
    sentinel_path = done_dir / f"shard-{output_shard_number:05d}.json"
    sentinel = _build_shard_completion(
        mode="hanshi",
        shard_index=output_shard_number,
        tokenization_contract_sha256=tokenization_contract_sha256,
        build_parameters=build_parameters,
        source_identity=source_identity,
        source_manifest_sha256=source_identity["rows_sha256"],
        source_record_count=len(rows),
        output_artifacts=output_artifacts,
        image_manifest=image_manifest,
        counters=counters,
    )
    write_json_atomically(sentinel_path, sentinel)
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
    build_parameters = _build_parameters_from_kwargs(kwargs)
    tokenization_contract_sha256 = kwargs["tokenization_contract_sha256"]
    source_identity = _hanshi_source_identity(rows, pages_root)
    sentinel_path = out_dir / "done" / f"shard-{output_shard_number:05d}.json"
    counters = _load_valid_shard_sentinel(
        sentinel_path,
        out_dir,
        mode="hanshi",
        shard_index=output_shard_number,
        tokenization_contract_sha256=tokenization_contract_sha256,
        build_parameters=build_parameters,
        source_identity=source_identity,
    )
    if counters is not None:
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
    sentinel_path = out_dir / "done" / f"{shard_name}.json"
    if sentinel_path.exists():
        sentinel_path.unlink()


def build_one_shard(
    shard_index: int,
    shards_dir: str | Path,
    out_dir: str | Path,
    encode_target: Callable[[str], list[int]],
    **kwargs: Any,
) -> ShardCounters:
    """Idempotent single-shard entry point: skip if done, else reset + build."""

    out_dir = Path(out_dir)
    tar_path = shard_path(shards_dir, shard_index)
    if not tar_path.exists():
        raise FileNotFoundError(f"shard tar not found: {tar_path}")
    build_parameters = _build_parameters_from_kwargs(kwargs)
    tokenization_contract_sha256 = kwargs["tokenization_contract_sha256"]
    sentinel_path = out_dir / "done" / f"shard-{shard_index:05d}.json"
    counters = _load_valid_shard_sentinel(
        sentinel_path,
        out_dir,
        mode="tar",
        shard_index=shard_index,
        tokenization_contract_sha256=tokenization_contract_sha256,
        build_parameters=build_parameters,
        source_identity=_tar_source_identity(tar_path),
    )
    if counters is not None:
        return counters

    _reset_shard_outputs(shard_index, out_dir)
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
    end-of-file for a final partial batch. Workers validate completion
    sentinels against these current row bytes, the tokenizer contract, build
    parameters, and output hashes. The reader must therefore enqueue every
    batch; trusting sentinel existence here would bypass that proof.
    """

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
    instruction_features = (
        encode_lm_text_features(
            bundle.tokenizer,
            args.instruction,
            interpret_special_tokens=True,
        )
        if args.instruction
        else None
    )
    instruction_ids = (
        instruction_features.input_ids if instruction_features is not None else []
    )
    instruction_track_ids = (
        instruction_features.morphology_track_ids
        if instruction_features is not None
        else []
    )
    make_ocr_target_encoder(bundle.tokenizer)  # preflight: raises on bad bundle
    tokenization_contract_sha256 = canonical_json_sha256(
        native_tokenization_contract(
            bundle.tokenizer,
            args.tokenizer_bundle,
        )
    )
    del bundle

    kwargs = dict(
        n_image_tokens=args.n_image_tokens,
        image_size=args.image_size,
        max_seq_len=args.max_seq_len,
        val_src_doc_min=args.val_src_doc_min,
        test_src_doc_min=args.test_src_doc_min,
        val_cap_per_shard=args.val_cap,
        ssl_quota_per_shard=args.ssl_rows,
        tokenization_contract_sha256=tokenization_contract_sha256,
        instruction_ids=instruction_ids,
        instruction_track_ids=instruction_track_ids,
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
    instruction_features = (
        encode_lm_text_features(
            bundle.tokenizer,
            args.instruction,
            interpret_special_tokens=True,
        )
        if args.instruction
        else None
    )
    instruction_ids = (
        instruction_features.input_ids if instruction_features is not None else []
    )
    instruction_track_ids = (
        instruction_features.morphology_track_ids
        if instruction_features is not None
        else []
    )
    make_ocr_target_encoder(bundle.tokenizer)  # preflight: raises on bad bundle
    tokenization_contract_sha256 = canonical_json_sha256(
        native_tokenization_contract(
            bundle.tokenizer,
            args.tokenizer_bundle,
        )
    )
    del bundle

    kwargs = dict(
        n_image_tokens=args.n_image_tokens,
        image_size=args.image_size,
        max_seq_len=args.max_seq_len,
        val_src_doc_min=args.val_src_doc_min,
        test_src_doc_min=args.test_src_doc_min,
        val_cap_per_shard=val_cap_per_shard,
        ssl_quota_per_shard=ssl_quota_per_shard,
        tokenization_contract_sha256=tokenization_contract_sha256,
        instruction_ids=instruction_ids,
        instruction_track_ids=instruction_track_ids,
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
        result = _run_hanshi_mode(args, out_dir)
    else:
        result = _run_tar_mode(args, out_dir)
    if result != 0:
        return result

    from Tokenizer.unified.bundle import TokenizerBundle

    bundle = TokenizerBundle.from_dir(args.tokenizer_bundle)
    issues = bundle.validate()
    if issues:
        raise ValueError(
            "invalid tokenizer bundle after shard build:\n  - "
            + "\n  - ".join(issues)
        )
    token_contract = native_tokenization_contract(
        bundle.tokenizer,
        args.tokenizer_bundle,
    )
    token_contract_sha256 = canonical_json_sha256(token_contract)
    semantic_parameters: dict[str, Any] | None = None
    for split in ("align", "val"):
        data_dir = out_dir / "jsonl" / split
        if split == "val" and not any(
            path.stat().st_size > 0 for path in data_dir.glob("*.jsonl")
        ):
            print(
                "[build-ocr-pairs] no val rows; no val receipt emitted",
                flush=True,
            )
            continue
        shard_completion, semantic_parameters = (
            _validated_shard_completion_manifest(
                data_dir,
                out_dir,
                tokenization_contract_sha256=token_contract_sha256,
                expected_semantics=semantic_parameters,
            )
        )
        contract = build_ocr_alignment_data_contract(
            data_dir,
            token_contract,
            shard_completion=shard_completion,
        )
        contract_path = data_dir / "ocr_data_contract.json"
        write_ocr_alignment_data_contract(contract_path, contract)
        print(
            f"[build-ocr-pairs] immutable {split} receipt -> {contract_path}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
