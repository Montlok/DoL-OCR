# -*- coding: utf-8 -*-

"""Checkpoint save / load (DDP and FSDP aware)."""

from __future__ import annotations

import os
import random
import shutil
import tempfile
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
    if "per_rank" in state:
        per_rank = state["per_rank"]
        if not isinstance(per_rank, list) or not per_rank:
            raise ValueError("checkpoint per-rank RNG state is malformed")
        current_world = torch.distributed.get_world_size() if is_distributed() else 1
        saved_world = int(state.get("world_size", len(per_rank)))
        if current_world != saved_world:
            raise ValueError(
                "checkpoint RNG world-size mismatch: "
                f"checkpoint={saved_world} current={current_world}"
            )
        rank = torch.distributed.get_rank() if is_distributed() else 0
        if rank >= len(per_rank):
            raise ValueError(
                f"checkpoint has RNG for {len(per_rank)} ranks, cannot restore rank {rank}"
            )
        state = per_rank[rank]
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


def _checkpoint_rng_state() -> dict[str, Any]:
    """Capture one RNG stream per rank so resumed sampling does not collapse."""

    local = _rng_state()
    if not is_distributed():
        return local
    world_size = torch.distributed.get_world_size()
    gathered: list[dict[str, Any] | None] = [None] * world_size
    torch.distributed.all_gather_object(gathered, local)
    if any(item is None for item in gathered):
        raise RuntimeError("failed to gather checkpoint RNG state from every rank")
    return {"per_rank": gathered, "world_size": world_size}


def _durable_torch_save(payload: Any, path: Path) -> None:
    """Write one checkpoint member and force its bytes to stable storage."""

    torch.save(payload, path)
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    """Persist directory-entry updates where the platform supports it."""

    try:
        directory_fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        # Some filesystems/platforms do not expose directory fsync. Individual
        # checkpoint members are still fsynced before the atomic rename.
        pass


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
    rng_state = _checkpoint_rng_state()
    if not is_main_process():
        return None

    out.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{step_dir.name}.tmp-", dir=out)
    )
    backup = out / f".{step_dir.name}.backup"
    swapped_old = False
    try:
        _durable_torch_save(model_state, staging / "model.pt")
        if optimizer_state is not None:
            _durable_torch_save(optimizer_state, staging / "optimizer.pt")
        if scheduler is not None:
            _durable_torch_save(scheduler.state_dict(), staging / "scheduler.pt")
        _durable_torch_save(rng_state, staging / "rng.pt")
        if scaler is not None:
            _durable_torch_save(scaler.state_dict(), staging / "scaler.pt")
        _durable_torch_save(
            {"step": step, "metadata": metadata or {}},
            staging / "meta.pt",
        )
        # This marker is written last.  Staging directories are hidden from
        # normal resolution, but the marker also makes manual recovery
        # unambiguous after a machine-level interruption.
        marker = staging / "COMPLETE"
        with marker.open("w", encoding="ascii") as handle:
            handle.write(f"step={step}\n")
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(staging)

        if backup.exists():
            shutil.rmtree(backup)
        if step_dir.exists():
            os.replace(step_dir, backup)
            swapped_old = True
        os.replace(staging, step_dir)
        _fsync_directory(out)
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if swapped_old and not step_dir.exists() and backup.exists():
            os.replace(backup, step_dir)
        raise

    latest = out / "latest"
    try:
        latest_tmp = out / f".latest.tmp-{os.getpid()}"
        if latest_tmp.exists() or latest_tmp.is_symlink():
            latest_tmp.unlink()
        os.symlink(step_dir.name, latest_tmp)
        if latest.exists() and latest.is_dir() and not latest.is_symlink():
            shutil.rmtree(latest)
        os.replace(latest_tmp, latest)
        _fsync_directory(out)
    except OSError:
        # symlinks may not be available on some filesystems
        if latest_tmp.exists() or latest_tmp.is_symlink():
            latest_tmp.unlink()

    # Prune only after ``latest`` durably points at the new step. Otherwise an
    # outage between deleting the old target and swapping the symlink leaves a
    # broken run root despite a complete new checkpoint being present.
    if keep_last_n > 0:
        ckpts = sorted(out.glob("step_*"))
        for old in ckpts[:-keep_last_n]:
            shutil.rmtree(old, ignore_errors=True)
        _fsync_directory(out)
    return step_dir


def _interrupted_backup(path: Path) -> Path | None:
    """Return the last complete directory left by an interrupted atomic swap."""

    backup = path.parent / f".{path.name}.backup"
    if (backup / "model.pt").exists() and (backup / "COMPLETE").exists():
        return backup
    return None


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
            if p.is_symlink():
                target = p.parent / os.readlink(p)
                backup = _interrupted_backup(target)
                if backup is not None:
                    return backup
                complete_steps = sorted(
                    step
                    for step in p.parent.glob("step_*")
                    if (step / "model.pt").exists()
                    and (step / "COMPLETE").exists()
                )
                if complete_steps:
                    return complete_steps[-1]
            raise FileNotFoundError(f"checkpoint not found: {p}")
        p = p.resolve()
    elif p.is_dir() and not (p / "model.pt").exists():
        latest = p / "latest"
        if latest.exists() or latest.is_symlink():
            return resolve_checkpoint_dir(latest)
        else:
            steps = sorted(d for d in p.glob("step_*") if (d / "model.pt").exists())
            if steps:
                p = steps[-1]
    elif not p.exists():
        backup = _interrupted_backup(p)
        if backup is not None:
            return backup
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
