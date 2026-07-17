# -*- coding: utf-8 -*-

"""Checkpoint save / load (DDP and FSDP aware)."""

from __future__ import annotations

import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from Model.training.dist import is_distributed, is_main_process


@dataclass
class CheckpointPayload:
    step: int
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any] | None
    scheduler_state: dict[str, Any] | None
    rng_state: dict[str, Any]
    metadata: dict[str, Any]
    scaler_state: dict[str, Any] | None = None


def _unwrap(model: nn.Module) -> nn.Module:
    if hasattr(model, "module"):
        return model.module
    return model


def _is_fsdp(model: nn.Module) -> bool:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        return isinstance(model, FSDP)
    except ImportError:  # pragma: no cover
        return False


def _get_model_state(model: nn.Module) -> dict[str, Any]:
    if _is_fsdp(model):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            return model.state_dict()
    return _unwrap(model).state_dict()


def _get_optimizer_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    if _is_fsdp(model):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if hasattr(FSDP, "full_optim_state_dict"):
            return FSDP.full_optim_state_dict(
                model, optimizer, rank0_only=True
            )

        from torch.distributed.fsdp import (
            FullOptimStateDictConfig,
            StateDictType,
        )

        optim_cfg = FullOptimStateDictConfig(
            offload_to_cpu=True,
            rank0_only=True,
        )
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            optim_state_dict_config=optim_cfg,
        ):
            return FSDP.optim_state_dict(model, optimizer)
    return optimizer.state_dict()


def _load_model_state(model: nn.Module, state: dict[str, Any]) -> None:
    if _is_fsdp(model):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            model.load_state_dict(state)
    else:
        _unwrap(model).load_state_dict(state)


def _load_optimizer_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    state: dict[str, Any],
) -> None:
    if _is_fsdp(model):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if hasattr(FSDP, "optim_state_dict_to_load"):
            state = FSDP.optim_state_dict_to_load(model, optimizer, state)
        else:
            state = FSDP.shard_full_optim_state_dict(state, model)
    optimizer.load_state_dict(state)


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "cpu": torch.get_rng_state(),
        "python": random.getstate(),
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:  # pragma: no cover - numpy is an optional dep
        pass
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    if "cpu" in state:
        torch.set_rng_state(state["cpu"])
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except ImportError:  # pragma: no cover - numpy is an optional dep
            pass
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    output_dir: str | Path,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    metadata: dict[str, Any] | None = None,
    keep_last_n: int = 0,
    scaler: Any = None,
) -> Path | None:
    """Save a step checkpoint under ``{output_dir}/step_{step:08d}/``."""

    out = Path(output_dir)
    step_dir = out / f"step_{step:08d}"

    model_state = _get_model_state(model)
    optimizer_state = (
        _get_optimizer_state(model, optimizer)
        if optimizer is not None
        else None
    )
    if not is_main_process():
        return None

    step_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model_state, step_dir / "model.pt")
    if optimizer_state is not None:
        torch.save(optimizer_state, step_dir / "optimizer.pt")
    if scheduler is not None:
        torch.save(scheduler.state_dict(), step_dir / "scheduler.pt")
    torch.save(_rng_state(), step_dir / "rng.pt")
    if scaler is not None:
        torch.save(scaler.state_dict(), step_dir / "scaler.pt")
    torch.save(
        {"step": step, "metadata": metadata or {}},
        step_dir / "meta.pt",
    )

    if keep_last_n > 0:
        ckpts = sorted(out.glob("step_*"))
        for old in ckpts[:-keep_last_n]:
            shutil.rmtree(old, ignore_errors=True)

    latest = out / "latest"
    if latest.exists() or latest.is_symlink():
        try:
            latest.unlink()
        except OSError:
            shutil.rmtree(latest, ignore_errors=True)
    try:
        os.symlink(step_dir.name, latest)
    except OSError:
        # symlinks may not be available on some filesystems
        pass
    return step_dir


def resolve_checkpoint_dir(path: str | Path) -> Path:
    """Resolve an output root/``latest`` link/model file to its step dir.

    Unlike :func:`load_checkpoint`, this helper performs no tensor loads.  It
    is used by model builders that must read ``meta.pt`` before allocating a
    model, and by initialization paths that only need ``model.pt`` (loading a
    multi-gigabyte optimizer state there would waste both time and RAM).
    """

    p = Path(path)
    if p.is_file():
        if p.name == "model.pt":
            return p.parent
        raise ValueError(f"checkpoint file must be named model.pt: {p}")
    if p.name == "latest":
        if not p.exists():
            raise FileNotFoundError(f"checkpoint not found: {p}")
        p = p.resolve()
    elif p.is_dir() and not (p / "model.pt").exists():
        if (p / "latest").exists():
            p = (p / "latest").resolve()
        else:
            steps = sorted(d for d in p.glob("step_*") if (d / "model.pt").exists())
            if steps:
                p = steps[-1]
    elif not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    if not (p / "model.pt").exists():
        raise FileNotFoundError(f"checkpoint model not found: {p / 'model.pt'}")
    return p


def load_checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    """Load only a checkpoint's user metadata, without loading model weights."""

    p = resolve_checkpoint_dir(path)
    meta_path = p / "meta.pt"
    if not meta_path.exists():
        return {}
    meta = torch.load(meta_path, map_location="cpu", weights_only=False)
    if not isinstance(meta, dict):
        raise TypeError(f"checkpoint metadata must be a dict: {meta_path}")
    payload = meta.get("metadata", meta)
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint metadata payload must be a dict: {meta_path}")
    return payload


def load_checkpoint(path: str | Path) -> CheckpointPayload:
    p = resolve_checkpoint_dir(path)

    model_state = torch.load(p / "model.pt", map_location="cpu", weights_only=False)
    opt_path = p / "optimizer.pt"
    sched_path = p / "scheduler.pt"
    rng_path = p / "rng.pt"
    meta_path = p / "meta.pt"
    scaler_path = p / "scaler.pt"

    opt_state = (
        torch.load(opt_path, map_location="cpu", weights_only=False)
        if opt_path.exists()
        else None
    )
    sched_state = (
        torch.load(sched_path, map_location="cpu", weights_only=False)
        if sched_path.exists()
        else None
    )
    rng_state = (
        torch.load(rng_path, map_location="cpu", weights_only=False)
        if rng_path.exists()
        else {}
    )
    meta = (
        torch.load(meta_path, map_location="cpu", weights_only=False)
        if meta_path.exists()
        else {"step": 0, "metadata": {}}
    )
    scaler_state = (
        torch.load(scaler_path, map_location="cpu", weights_only=False)
        if scaler_path.exists()
        else None
    )

    return CheckpointPayload(
        step=int(meta.get("step", 0)),
        model_state=model_state,
        optimizer_state=opt_state,
        scheduler_state=sched_state,
        rng_state=rng_state,
        metadata=meta.get("metadata", {}),
        scaler_state=scaler_state,
    )


def resume_state(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    state: Any = None,
) -> int:
    """Load a checkpoint into the given model/optimizer/scheduler.

    If ``state`` (a ``TrainState``) is provided and the checkpoint stored a
    fp16 ``GradScaler`` state, it is stashed in ``state.extra`` so the lazily
    created scaler in ``train_one_step`` restores its dynamic loss scale,
    keeping overflow/scheduler decisions consistent with an uninterrupted run.
    """

    payload = load_checkpoint(path)
    _load_model_state(model, payload.model_state)
    if optimizer is not None and payload.optimizer_state is not None:
        _load_optimizer_state(model, optimizer, payload.optimizer_state)
    if scheduler is not None and payload.scheduler_state is not None:
        scheduler.load_state_dict(payload.scheduler_state)
    if payload.rng_state:
        _restore_rng(payload.rng_state)
    if state is not None and payload.scaler_state is not None:
        state.extra["grad_scaler_state"] = payload.scaler_state
    if is_distributed():
        torch.distributed.barrier()
    return payload.step


__all__ = [
    "CheckpointPayload",
    "load_checkpoint",
    "load_checkpoint_metadata",
    "resolve_checkpoint_dir",
    "resume_state",
    "save_checkpoint",
]
