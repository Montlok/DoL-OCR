# -*- coding: utf-8 -*-

"""Distributed setup, DDP/FSDP wrappers."""

from __future__ import annotations

import functools
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from Model.config import TrainingConfig


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed(backend: str | None = None) -> tuple[int, int, int]:
    """Initialize torch.distributed from env vars.

    Returns ``(rank, world_size, local_rank)``. If env vars are missing,
    runs in single-process mode and returns ``(0, 1, 0)``.
    ``backend`` defaults to ``nccl`` (auto-falls back to ``gloo`` when CUDA
    is unavailable).
    """

    if backend is None:
        backend = "nccl"

    if dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return dist.get_rank(), dist.get_world_size(), local_rank

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if backend == "nccl" and not torch.cuda.is_available():
        backend = "gloo"

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available() and backend == "nccl":
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def destroy_distributed() -> None:
    """Tear down the torch.distributed process group if this process owns one."""

    if is_distributed():
        dist.destroy_process_group()


def _transformer_block_classes() -> set[type]:
    from Model.blocks import (
        AttnSubLayer,
        MambaSubLayer,
        RecurrentBlock,
        StandardBlock,
    )

    return {StandardBlock, RecurrentBlock, AttnSubLayer, MambaSubLayer}


def wrap_ddp(model: nn.Module, local_rank: int, find_unused_parameters: bool = True) -> nn.Module:
    device_ids = [local_rank] if torch.cuda.is_available() else None
    return DDP(
        model,
        device_ids=device_ids,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=True,
    )


def wrap_fsdp(
    model: nn.Module,
    cfg: TrainingConfig,
) -> nn.Module:
    """Wrap with FSDP at transformer-block granularity.

    Wraps ``StandardBlock`` / ``RecurrentBlock`` / ``AttnSubLayer`` /
    ``MambaSubLayer`` so recurrent steps don't share a single shard.
    """

    try:
        from torch.distributed.fsdp import (
            CPUOffload,
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
        )
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("FSDP not available in this torch build") from exc

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    mp_dtype = dtype_map.get(cfg.fsdp_mixed_precision, torch.bfloat16)
    mp_policy = MixedPrecision(
        param_dtype=mp_dtype,
        reduce_dtype=mp_dtype,
        buffer_dtype=mp_dtype,
    )

    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=_transformer_block_classes(),
    )

    return FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mp_policy,
        cpu_offload=CPUOffload(offload_params=True) if cfg.fsdp_cpu_offload else None,
        device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
        use_orig_params=True,
    )


def apply_parallelism(
    model: nn.Module,
    cfg: TrainingConfig,
    local_rank: int,
) -> nn.Module:
    """Apply DDP, FSDP, or single-process wrapping based on ``cfg.parallel``."""

    if cfg.parallel == "single":
        return model
    if not is_distributed():
        # Allow single-process "ddp" smoke runs to no-op rather than crash.
        return model
    if cfg.parallel == "ddp":
        return wrap_ddp(model, local_rank)
    if cfg.parallel == "fsdp":
        return wrap_fsdp(model, cfg)
    raise ValueError(f"unknown parallel mode: {cfg.parallel}")


__all__ = [
    "apply_parallelism",
    "destroy_distributed",
    "get_rank",
    "get_world_size",
    "init_distributed",
    "is_distributed",
    "is_main_process",
    "wrap_ddp",
    "wrap_fsdp",
]
