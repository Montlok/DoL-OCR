# -*- coding: utf-8 -*-

"""JSONL datasets, sequence packing, and collators for pretraining."""

from __future__ import annotations

import glob
import json
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from Model.config import IGNORE_INDEX, PAD_ID, TrainingConfig
from Model.ocr.alignment_contract import (
    read_verified_ocr_image_bytes,
    validate_ocr_image_binding_fields,
)
from Model.ocr.position_contract import (
    BOUNDARY_V1,
    LEGACY_SEQUENTIAL_V0,
    validate_ocr_position_contract,
)
from Tokenizer.pretraining import derive_morph_info_from_offsets


class JsonlPretrainingDataset(Dataset):
    """Eager JSONL dataset for rows emitted by ``build_pretraining_data``."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


class StreamingJsonlDataset(IterableDataset):
    """Sharded, rank-aware streaming JSONL reader.

    Files in ``paths`` are statically assigned to ranks (round-robin); within
    a rank, dataloader workers further partition the assigned files. A small
    in-memory shuffle buffer can be enabled.
    """

    def __init__(
        self,
        paths: Sequence[str | Path],
        *,
        world_size: int = 1,
        rank: int = 0,
        shuffle_buffer: int = 0,
        seed: int = 0,
        infinite: bool = True,
    ) -> None:
        if not paths:
            raise ValueError("StreamingJsonlDataset requires at least one path")
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if not (0 <= rank < world_size):
            raise ValueError("rank must be in [0, world_size)")
        self.paths = [Path(p) for p in paths]
        self.world_size = world_size
        self.rank = rank
        self.shuffle_buffer = max(0, int(shuffle_buffer))
        self.seed = int(seed)
        self.infinite = bool(infinite)

    def _files_for_worker(self) -> list[Path]:
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        # When the shard count is smaller than ``world_size`` strict
        # round-robin partitioning would leave some ranks with zero
        # shards. Returning an empty iterator on those ranks deadlocks
        # collective ops (the active ranks still march through
        # ``forward``/``backward``). Fall back to a **replicated** stream
        # so every rank sees every shard; rank-aware RNG already
        # decorrelates the shuffle order per rank.
        if 0 < len(self.paths) < self.world_size:
            rank_shards = list(self.paths)
        else:
            rank_shards = self.paths[self.rank :: self.world_size]
        if not rank_shards:
            return []
        return rank_shards[worker_id::num_workers]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        files = self._files_for_worker()
        if not files:
            return

        # Seed the shuffle from (base seed, rank, worker) only — NOT os.getpid().
        # The PID made the shuffle order non-reproducible across runs/processes
        # (breaking resumable/deterministic training); rank+worker already
        # decorrelate the per-shard buffer order deterministically.
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        base_seed = self.seed + 1000 * self.rank + worker_id
        buffer: list[dict[str, Any]] = []

        def _emit(item: dict[str, Any]) -> Iterator[dict[str, Any]]:
            if self.shuffle_buffer <= 0:
                yield item
                return
            buffer.append(item)
            if len(buffer) >= self.shuffle_buffer:
                idx = rng.randrange(len(buffer))
                buffer[idx], buffer[-1] = buffer[-1], buffer[idx]
                yield buffer.pop()

        pass_idx = 0
        while pass_idx == 0 or self.infinite:
            # Re-derive the RNG every pass: a pass-stable seed would shuffle
            # each epoch into the same order, and — worse — a resumed run
            # rebuilds the iterator from row 0 with the original order, so
            # epoch 2 after a resume would replay epoch 1's exact stream.
            # (The multiplier just spaces the per-pass seeds apart; any odd
            # constant larger than seed+1000*rank+worker collisions works.)
            rng = random.Random(base_seed + 7_368_787 * pass_idx)
            pass_idx += 1
            rows_seen = 0
            for path in files:
                with path.open("r", encoding="utf-8") as f:
                    for line_no, line in enumerate(f, start=1):
                        line = line.rstrip("\n")
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                f"invalid JSON in shard {path}:{line_no}: {exc}"
                            ) from exc
                        rows_seen += 1
                        yield from _emit(row)

            if rows_seen == 0:
                raise ValueError(
                    f"no JSONL rows available for rank={self.rank} from "
                    f"{[str(path) for path in files]}"
                )

            rng.shuffle(buffer)
            while buffer:
                yield buffer.pop()


@dataclass
class PretrainingCollator:
    pad_id: int = PAD_ID
    ignore_index: int = IGNORE_INDEX
    pad_to_multiple_of: int | None = None
    include_metadata: bool = False
    max_seq_len: int | None = None
    require_precomputed_morphology: bool = False
    require_verified_images: bool = False
    # Multimodal: when both ``image_processor`` and ``omvt_cfg`` are set,
    # rows that carry an ``images`` field are turned into a stacked OMVT
    # multi-scale ``pixel_values`` batch dict. Rows without images still
    # work — they just emit no pixel batch. We require **either** every
    # row to have images **or** none, to keep the batch shape uniform.
    image_processor: Any = None
    omvt_cfg: Any = None
    position_contract: str | None = None

    def __post_init__(self) -> None:
        if self.position_contract is not None:
            self.position_contract = validate_ocr_position_contract(
                self.position_contract
            )
        if self.position_contract is not None and self.require_precomputed_morphology:
            raise ValueError(
                "an explicit OCR position contract cannot consume precomputed "
                "morphology fields"
            )

    def __call__(self, rows: Sequence[Any]) -> dict[str, Any]:
        if not rows:
            raise ValueError("rows cannot be empty")

        normalized = [
            _normalize_row(
                row,
                max_seq_len=self.max_seq_len,
                require_precomputed_morphology=self.require_precomputed_morphology,
                require_verified_images=self.require_verified_images,
                position_contract=self.position_contract,
            )
            for row in rows
        ]
        max_len = max(len(row["input_ids"]) for row in normalized)
        if max_len <= 0:
            raise ValueError("rows cannot contain empty input_ids")
        if self.pad_to_multiple_of is not None:
            if self.pad_to_multiple_of <= 0:
                raise ValueError("pad_to_multiple_of must be positive")
            remainder = max_len % self.pad_to_multiple_of
            if remainder:
                max_len += self.pad_to_multiple_of - remainder

        batch: dict[str, list[Any]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        include_positions = self.position_contract != BOUNDARY_V1
        if include_positions:
            batch["word_pos"] = []
            batch["morph_depth"] = []

        for row in normalized:
            pad_count = max_len - len(row["input_ids"])
            batch["input_ids"].append(row["input_ids"] + [self.pad_id] * pad_count)
            batch["attention_mask"].append(row["attention_mask"] + [0] * pad_count)
            batch["labels"].append(row["labels"] + [self.ignore_index] * pad_count)
            if include_positions:
                batch["word_pos"].append(row["word_pos"] + [0] * pad_count)
                batch["morph_depth"].append(
                    row["morph_depth"] + [0] * pad_count
                )

        tensors: dict[str, Any] = {
            key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()
        }
        if self.position_contract is not None:
            tensors["position_contract"] = self.position_contract
        if self.include_metadata:
            tensors["metadata"] = [row.get("metadata", {}) for row in normalized]
            tensors["modality_spans"] = [
                row.get("modality_spans", {}) for row in normalized
            ]

        pixel_values = self._build_pixel_batch(normalized)
        if pixel_values is not None:
            tensors["pixel_values"] = pixel_values

        # Pass-through optional SSL labels (one list per row, ragged).
        if any(row.get("ocr_labels") for row in normalized):
            tensors["ocr_labels"] = [row.get("ocr_labels", []) for row in normalized]
        if any(row.get("reading_order") for row in normalized):
            tensors["reading_order"] = [
                row.get("reading_order", []) for row in normalized
            ]
        return tensors

    def _build_pixel_batch(self, normalized: list[dict[str, Any]]) -> Any:
        if self.image_processor is None or self.omvt_cfg is None:
            return None
        # Current OMVT injector assumes exactly **one image per row**: it
        # treats the leading axis of every pixel tensor as the text-batch
        # axis and replaces ``<image_patch>`` slots from a contiguous
        # ``[B, compress_to, D]`` feature block. Multi-image-per-row would
        # require either reshaping features back to ``[B, N*compress_to,
        # D]`` *and* a matching N*compress_to <image_patch> count, or a
        # bucketed dataloader. Neither is wired yet — we reject the batch
        # explicitly rather than producing a silently-misaligned forward.
        per_row_images = [list(row.get("images") or []) for row in normalized]
        n_images_per_row = {len(items) for items in per_row_images}
        if n_images_per_row == {0}:
            return None
        if n_images_per_row != {1}:
            raise ValueError(
                "PretrainingCollator currently supports exactly one image per row "
                f"(got {sorted(n_images_per_row)}). Split mixed-cardinality data via "
                "bucketed dataloaders, or extend _build_pixel_batch + VisionInjector "
                "to handle N>1 images per row."
            )
        flat: list[Any] = []
        for row, row_items in zip(normalized, per_row_images):
            image_ref = row_items[0]
            if self.require_verified_images or "image_sha256" in row:
                image_bytes = read_verified_ocr_image_bytes(row)
                # PILImageProcessor accepts bytes, so verification does not
                # cause a second filesystem read on the hot path.
                flat.append(image_bytes)
            else:
                flat.append(image_ref)
        images = self.image_processor(flat)
        # images: [B, C, H, W]. OMVT will turn each row into compress_to
        # visual tokens and the injector replaces the row's
        # <image_patch> slots one-for-one. Lazy import keeps the
        # Tokenizer → Model edge cold for text-only setups.
        from Model.omvt.patcher import collate_omvt_batch

        return dict(collate_omvt_batch(images, self.omvt_cfg))


def _normalize_row(
    row: Any,
    *,
    max_seq_len: int | None,
    require_precomputed_morphology: bool = False,
    require_verified_images: bool = False,
    position_contract: str | None = None,
) -> dict[str, Any]:
    if not isinstance(row, dict):
        row = row.__dict__

    required = ("input_ids", "attention_mask", "labels")
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"missing pretraining fields: {', '.join(missing)}")

    out: dict[str, Any] = {
        "input_ids": [int(value) for value in row["input_ids"]],
        "attention_mask": [int(value) for value in row["attention_mask"]],
        "labels": [int(value) for value in row["labels"]],
        "metadata": dict(row.get("metadata") or {}),
        "modality_spans": dict(row.get("modality_spans") or {}),
    }
    n = len(out["input_ids"])
    if len(out["attention_mask"]) != n or len(out["labels"]) != n:
        raise ValueError("input_ids, attention_mask, and labels must align")

    word_pos = row.get("word_pos")
    morph_depth = row.get("morph_depth")
    token_offsets = row.get("token_offsets")
    if position_contract == BOUNDARY_V1:
        if word_pos is not None or morph_depth is not None or token_offsets is not None:
            raise ValueError(
                "boundary_v1 OCR rows must omit word_pos, morph_depth, and "
                "token_offsets so the model derives the checkpoint-compatible "
                "position contract"
            )
    else:
        if require_precomputed_morphology and (
            word_pos is None or morph_depth is None
        ):
            raise ValueError(
                "strict receipt-backed training rows must persist word_pos and "
                "morph_depth from the pretraining-compatible tokenizer route"
            )
        if position_contract == LEGACY_SEQUENTIAL_V0:
            if word_pos is not None or morph_depth is not None or token_offsets is not None:
                raise ValueError(
                    "legacy_sequential_v0 OCR rows must omit precomputed position fields"
                )
            word_pos = list(range(n))
            morph_depth = [0] * n
        elif word_pos is None or morph_depth is None:
            if token_offsets is None:
                word_pos = list(range(n))
                morph_depth = [0] * n
            else:
                word_pos, morph_depth = derive_morph_info_from_offsets(token_offsets)

        out["word_pos"] = [int(value) for value in word_pos]
        out["morph_depth"] = [int(value) for value in morph_depth]
        if len(out["word_pos"]) != n or len(out["morph_depth"]) != n:
            raise ValueError("word_pos and morph_depth must align with input_ids")

    has_image_binding = (
        "image_sha256" in row or "image_size_bytes" in row
    )
    if require_verified_images or has_image_binding:
        image_ref, image_sha256, image_size_bytes = (
            validate_ocr_image_binding_fields(row)
        )
        out["images"] = [image_ref]
        out["image_sha256"] = image_sha256
        out["image_size_bytes"] = image_size_bytes

    if max_seq_len is not None and n > max_seq_len:
        # Multimodal rows carry a fixed number of ``<image_patch>`` slots that
        # must stay one-for-one with the ``images`` payload (the OMVT injector
        # asserts this at forward). Blindly slicing ``input_ids`` here could
        # drop image-patch slots while the ``images`` list stays intact,
        # producing a silently-misaligned batch. Refuse loudly instead and tell
        # the operator to size ``seq_len`` to the builder's max length.
        if row.get("images") or row.get("videos"):
            raise ValueError(
                "multimodal row length "
                f"({n}) exceeds seq_len ({max_seq_len}); model-side truncation "
                "is not span-aware and would desync <image_patch> slots from the "
                "image payload. Set TrainingConfig.seq_len >= the builder's "
                "max_length so multimodal rows are not truncated."
            )
        truncation_keys = ["input_ids", "attention_mask", "labels"]
        if position_contract != BOUNDARY_V1:
            truncation_keys.extend(("word_pos", "morph_depth"))
        for key in truncation_keys:
            out[key] = out[key][:max_seq_len]

    # Multimodal pass-through fields (opaque to the text-side normalizer).
    for key in ("images", "image_sizes", "videos", "video_sizes"):
        if row.get(key):
            out[key] = list(row[key])
    for key in ("ocr_labels", "reading_order"):
        if row.get(key):
            out[key] = [list(item) for item in row[key]]
    return out


def _resolve_shards(spec: str | Sequence[str | Path]) -> list[Path]:
    if isinstance(spec, (str, Path)):
        path = Path(spec)
        if path.is_dir():
            return sorted(path.glob("*.jsonl"))
        if any(ch in str(spec) for ch in "*?["):
            # ``Path().glob`` cannot accept absolute patterns and rejects
            # ``**`` outside ``rglob``; ``glob.glob`` handles both natively
            # and is the right tool for user-supplied shard specs that may
            # be absolute mount paths like ``/mnt/corpus/*.jsonl``.
            return sorted(Path(p) for p in glob.glob(str(spec), recursive=True))
        return [path] if path.is_file() else []
    return [Path(p) for p in spec]


def build_dataloader(
    spec: str | Sequence[str | Path],
    cfg: TrainingConfig,
    *,
    world_size: int = 1,
    rank: int = 0,
    seed: int | None = None,
    pad_id: int = PAD_ID,
    ignore_index: int = IGNORE_INDEX,
    infinite: bool = True,
    drop_last: bool = True,
    image_processor: Any = None,
    omvt_cfg: Any = None,
    require_precomputed_morphology: bool = False,
    require_verified_images: bool = False,
    position_contract: str | None = None,
) -> DataLoader:
    """Construct a rank-aware streaming dataloader.

    Pass ``image_processor`` + ``omvt_cfg`` together to enable the
    multimodal path: rows that carry an ``images`` column will be turned
    into a stacked OMVT multi-scale ``pixel_values`` batch dict that the
    model's ``VisionInjector`` consumes directly.
    """

    paths = _resolve_shards(spec)
    if not paths:
        raise FileNotFoundError(f"no shards resolved from spec: {spec!r}")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "resolved shard paths do not exist: "
            + ", ".join(str(path) for path in missing[:5])
        )

    dataset = StreamingJsonlDataset(
        paths,
        world_size=world_size,
        rank=rank,
        shuffle_buffer=cfg.shuffle_buffer,
        seed=seed if seed is not None else cfg.seed,
        infinite=infinite,
    )

    collator = PretrainingCollator(
        pad_id=pad_id,
        ignore_index=ignore_index,
        pad_to_multiple_of=None,
        max_seq_len=cfg.seq_len,
        image_processor=image_processor,
        omvt_cfg=omvt_cfg,
        require_precomputed_morphology=require_precomputed_morphology,
        require_verified_images=require_verified_images,
        position_contract=position_contract,
    )

    return DataLoader(
        dataset,
        batch_size=cfg.micro_batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collator,
        drop_last=drop_last,
        persistent_workers=cfg.num_workers > 0,
    )


def estimate_tokens_per_step(cfg: TrainingConfig, world_size: int = 1) -> int:
    return cfg.micro_batch_size * cfg.grad_accum_steps * cfg.seq_len * world_size


__all__ = [
    "JsonlPretrainingDataset",
    "PretrainingCollator",
    "StreamingJsonlDataset",
    "build_dataloader",
    "estimate_tokens_per_step",
]
