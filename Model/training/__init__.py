# -*- coding: utf-8 -*-

"""RDT training utilities.

This package exposes the data-loading primitives that the previous
flat ``Model/training.py`` module provided, plus the new optimizer,
distributed, checkpoint, and training-loop helpers required for
formal pretraining.

The legacy import paths are preserved:

    from Model.training import JsonlPretrainingDataset, PretrainingCollator
"""

from Model.training.checkpoint import (
    clear_no_update_progress,
    load_checkpoint,
    load_checkpoint_metadata,
    load_no_update_progress,
    resolve_checkpoint_dir,
    restore_rng_state,
    resume_state,
    save_no_update_progress,
    save_checkpoint,
    validate_resumable_checkpoint,
)
from Model.training.data import (
    JsonlPretrainingDataset,
    PretrainingCollator,
    StreamingJsonlDataset,
    build_dataloader,
)
from Model.training.dist import (
    apply_parallelism,
    destroy_distributed,
    init_distributed,
    is_main_process,
    wrap_ddp,
    wrap_fsdp,
)
from Model.training.early_stopping import (
    EarlyStoppingConfig,
    LossPlateauStopper,
)
from Model.training.logging import RankZeroLogger, throughput_str
from Model.training.loop import (
    TrainState,
    clip_or_check_grad_norm,
    evaluate,
    train_one_step,
)
from Model.training.multimodal_cli import (
    add_multimodal_args,
    build_image_processor,
    build_omvt_cfg,
)
from Model.training.optim import (
    build_optimizer,
    build_scheduler,
    param_groups_with_no_decay,
)

__all__ = [
    "EarlyStoppingConfig",
    "JsonlPretrainingDataset",
    "LossPlateauStopper",
    "PretrainingCollator",
    "RankZeroLogger",
    "StreamingJsonlDataset",
    "TrainState",
    "add_multimodal_args",
    "apply_parallelism",
    "build_dataloader",
    "build_image_processor",
    "build_omvt_cfg",
    "build_optimizer",
    "build_scheduler",
    "clip_or_check_grad_norm",
    "clear_no_update_progress",
    "destroy_distributed",
    "evaluate",
    "init_distributed",
    "is_main_process",
    "load_checkpoint",
    "load_checkpoint_metadata",
    "load_no_update_progress",
    "param_groups_with_no_decay",
    "resolve_checkpoint_dir",
    "restore_rng_state",
    "resume_state",
    "save_no_update_progress",
    "save_checkpoint",
    "throughput_str",
    "train_one_step",
    "validate_resumable_checkpoint",
    "wrap_ddp",
    "wrap_fsdp",
]
