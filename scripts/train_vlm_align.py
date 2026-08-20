# -*- coding: utf-8 -*-

"""VLM alignment: OMVT vision tower → projector → RDT.

Runs a synthetic end-to-end forward/backward where the RDT LM consumes
``<image_patch>`` slots filled by the OMVT compressed tokens.  The LM head
is fine-tuned by default; pass ``--freeze-rdt`` to train only the
projector/tower.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Iterator

import torch

from Model.config import (
    BOS_ID,
    EOS_ID,
    IMAGE_END_ID,
    IMAGE_PATCH_ID,
    IMAGE_START_ID,
    OMVTConfig,
    PAD_ID,
    RDTConfig,
    TrainingConfig,
)
from Model.ocr.data import build_ocr_row
from Model.ocr.position_contract import (
    BOUNDARY_V1,
    OCR_POSITION_CONTRACT_CHOICES,
    OCR_POSITION_CONTRACT_METADATA_VERSION,
    resolve_checkpoint_ocr_position_contract,
)
from Model.model import RDTForCausalLM
from Model.ocr.alignment_contract import (
    default_ocr_alignment_contract_path,
    load_and_validate_ocr_alignment_data_contract,
)
from Model.ocr.tokenization import native_tokenization_contract
from Model.omvt import OMVTInjector
from Model.omvt.patcher import collate_omvt_batch
from Model.training import (
    RankZeroLogger,
    TrainState,
    build_dataloader,
    build_optimizer,
    build_scheduler,
    clip_or_check_grad_norm,
    load_checkpoint_metadata,
    resolve_checkpoint_dir,
    resume_state,
    save_checkpoint,
    train_one_step,
    validate_resumable_checkpoint,
)
from Model.training.data import PretrainingCollator
from Model.training.multimodal_cli import make_omvt_cfg
from Model.training.omvt_checkpoint import (
    load_omvt_payload,
    tower_state_from_payload,
)
from Tokenizer.multimodal import PILImageProcessor
from scripts.train_rdt import CONFIG_CHOICES, _resolve_mamba_backend

OCR_TARGET_ENCODING_CHOICES = ("native", "native_fallback", "byte_fallback")
OCR_TOKENIZATION_CONTRACT_VERSION = 2


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=list(CONFIG_CHOICES), default="tiny")
    p.add_argument(
        "--mamba",
        choices=["auto", "official", "naive"],
        default="auto",
        help=(
            "Mamba backend for the RDT side. auto uses official CUDA Mamba on "
            "CUDA/Linux and NaiveSSM on macOS/CPU."
        ),
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="hard safety ceiling; plateau early stopping may finish sooner",
    )
    p.add_argument(
        "--steps",
        dest="legacy_steps",
        type=int,
        default=None,
        help="deprecated alias for --max-steps",
    )
    p.add_argument(
        "--min-steps",
        type=int,
        default=0,
        help="minimum optimizer steps before loss-plateau patience is counted",
    )
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="smoothed-loss observations without improvement (0 disables)",
    )
    p.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="minimum absolute smoothed-loss decrease considered an improvement",
    )
    p.add_argument(
        "--early-stop-mode",
        choices=("ema", "window"),
        default="ema",
        help="loss smoother used for plateau detection",
    )
    p.add_argument(
        "--early-stop-ema-alpha",
        type=float,
        default=0.01,
        help="new-observation weight for EMA smoothing",
    )
    p.add_argument(
        "--early-stop-window",
        type=int,
        default=100,
        help="rolling mean width when --early-stop-mode=window",
    )
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--image-size", type=int, default=56)
    p.add_argument("--seq-len", type=int, default=24)
    p.add_argument("--n-image-tokens", type=int, default=None)
    p.add_argument(
        "--recurrent-steps",
        type=int,
        default=None,
        help="fixed RDT refinement depth; defaults to the init/resume "
        "checkpoint value (or the selected config when no checkpoint exists)",
    )
    p.add_argument("--freeze-rdt", action="store_true")
    p.add_argument(
        "--frozen-vision",
        action="store_true",
        help="freeze the OMVT tower as well (useful for projector-only ablations)",
    )
    p.add_argument(
        "--data",
        default="",
        help="JSONL spec for real multimodal pretraining (rows must carry an 'images' field)",
    )
    p.add_argument(
        "--ocr-native-targets",
        action="store_true",
        help=(
            "mark this as frozen-LM OCR alignment and require a builder receipt "
            "that proves native tokenizer targets"
        ),
    )
    p.add_argument(
        "--ocr-tokenizer-bundle",
        default="",
        help="TokenizerBundle used by the OCR data builder",
    )
    p.add_argument(
        "--ocr-data-contract",
        default="",
        help="defaults to ocr_data_contract.json beside --data",
    )
    p.add_argument("--stream-wds-dir", default="")
    p.add_argument("--stream-hanshi-meta", default="")
    p.add_argument("--stream-hanshi-pages", default="")
    p.add_argument("--stream-tokenizer-bundle", default="")
    p.add_argument(
        "--ocr-target-encoding",
        choices=OCR_TARGET_ENCODING_CHOICES,
        default="native",
        help=(
            "OCR label representation for direct corpus streaming. Production "
            "frozen-language alignment requires 'native', which uses the same "
            "MorphBPE/general route as RDT pretraining and fails closed on any "
            "unknown or non-round-tripping target."
        ),
    )
    p.add_argument(
        "--ocr-position-contract",
        choices=OCR_POSITION_CONTRACT_CHOICES,
        default=BOUNDARY_V1,
        help=(
            "versioned OCR train/inference position semantics; boundary_v1 is "
            "required for new runs, while legacy_sequential_v0 is explicit "
            "compatibility for historical checkpoints"
        ),
    )
    p.add_argument(
        "--stream-exclude-shard-id",
        action="append",
        type=int,
        default=None,
        help="WDS shard id to exclude (repeatable; default: known-corrupt 2303)",
    )
    p.add_argument(
        "--stream-max-wds-shards",
        type=int,
        default=0,
        help="smoke-only cap after numeric discovery; 0 uses every usable shard",
    )
    p.add_argument(
        "--stream-prefetch-batches",
        type=int,
        default=4,
        help="bounded read-ahead; the consumed cursor travels with every batch",
    )
    p.add_argument("--val-src-doc-min", type=int, default=434600)
    p.add_argument("--test-src-doc-min", type=int, default=435200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--output", default="outputs/vlm_align")
    p.add_argument("--init-rdt-checkpoint", default="")
    p.add_argument("--init-omvt-checkpoint", default="")
    p.add_argument(
        "--use-ema-tower",
        action="store_true",
        help="when --init-omvt-checkpoint carries 'tower_ema', overlay the EMA "
        "weights on the tower state before loading",
    )
    p.add_argument("--resume", default="")
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument(
        "--keep-last-n",
        type=int,
        default=0,
        help="delete older step checkpoints as new ones are saved (0 = keep all)",
    )
    p.add_argument(
        "--no-resume-skip-data",
        action="store_true",
        help="do not fast-forward the deterministic data stream on resume",
    )
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--precision",
        choices=("auto", "fp32", "bf16"),
        default="fp32",
        help="'auto' = bf16 on cuda, fp32 elsewhere; default keeps the legacy "
        "fp32 smoke behavior",
    )
    p.add_argument("--warmup-steps", type=int, default=1)
    p.add_argument(
        "--d-vision",
        type=int,
        default=64,
        help="OMVT tower width for from-scratch towers; ignored when "
        "--init-omvt-checkpoint provides its own omvt_config",
    )
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="'auto' = cuda if available else cpu (legacy behavior)",
    )
    p.add_argument(
        "--patch-preset",
        choices=("derived", "prod"),
        default="derived",
        help="for from-scratch towers: 'derived' keeps the legacy smoke "
        "geometry (patches scaled from --image-size); 'prod' uses the "
        "OMVTConfig dataclass multi-scale defaults (32x8 / 8x32 / 16x16 / "
        "56x56). Ignored when --init-omvt-checkpoint provides omvt_config.",
    )
    p.add_argument(
        "--grad-ckpt",
        action="store_true",
        help="enable RDT gradient checkpointing (grad_ckpt_recurrent + "
        "grad_ckpt_prelude_coda) to fit long sequences on small GPUs",
    )
    args = p.parse_args(argv)
    if args.max_steps is not None and args.legacy_steps is not None:
        p.error("--max-steps and deprecated --steps cannot be used together")
    args.max_steps = (
        args.max_steps
        if args.max_steps is not None
        else (args.legacy_steps if args.legacy_steps is not None else 4)
    )
    if args.max_steps <= 0:
        p.error("--max-steps must be positive")
    if args.min_steps < 0 or args.min_steps > args.max_steps:
        p.error("--min-steps must be between 0 and --max-steps")
    if args.early_stop_patience < 0:
        p.error("--early-stop-patience must be non-negative")
    if not math.isfinite(args.early_stop_min_delta) or args.early_stop_min_delta < 0:
        p.error("--early-stop-min-delta must be finite and non-negative")
    if (
        not math.isfinite(args.early_stop_ema_alpha)
        or not 0.0 < args.early_stop_ema_alpha <= 1.0
    ):
        p.error("--early-stop-ema-alpha must be in (0, 1]")
    if args.early_stop_window <= 0:
        p.error("--early-stop-window must be positive")
    if args.stream_max_wds_shards < 0:
        p.error("--stream-max-wds-shards must be non-negative")
    if args.stream_prefetch_batches < 0:
        p.error("--stream-prefetch-batches must be non-negative")
    if args.val_src_doc_min < 0 or args.test_src_doc_min <= args.val_src_doc_min:
        p.error("stream split boundaries must satisfy 0 <= val < test")
    # Retain the historical attribute for callers that inspect parse_args().
    args.steps = args.max_steps
    return args


@dataclass
class LossPlateauEarlyStop:
    """Serializable smoothed training-loss plateau detector.

    Training loss is noisy enough that raw consecutive-step comparisons are
    unsafe.  This detector supports either an EMA or a fixed rolling mean and
    persists all continuation-critical state in checkpoint metadata.
    """

    mode: str
    min_steps: int
    patience: int
    min_delta: float
    ema_alpha: float
    window_size: int
    best: float | None = None
    smoothed: float | None = None
    bad_steps: int = 0
    observations: int = 0
    window_values: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mode not in {"ema", "window"}:
            raise ValueError(f"unsupported early-stop mode: {self.mode!r}")
        if self.min_steps < 0:
            raise ValueError("early-stop min_steps must be non-negative")
        if self.patience < 0:
            raise ValueError("early-stop patience must be non-negative")
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("early-stop min_delta must be finite and non-negative")
        if not math.isfinite(self.ema_alpha) or not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("early-stop ema_alpha must be in (0, 1]")
        if self.window_size <= 0:
            raise ValueError("early-stop window_size must be positive")
        self._validate_state()

    def _validate_state(self) -> None:
        for name, value in (("best", self.best), ("smoothed", self.smoothed)):
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"early-stop {name} must be finite when present")
        if self.bad_steps < 0 or self.observations < 0:
            raise ValueError("early-stop counters must be non-negative")
        if self.bad_steps > self.observations:
            raise ValueError("early-stop bad_steps cannot exceed observations")
        if len(self.window_values) > self.window_size:
            raise ValueError("early-stop rolling window is larger than window_size")
        if any(not math.isfinite(float(value)) for value in self.window_values):
            raise ValueError("early-stop rolling window contains a non-finite loss")

    @property
    def enabled(self) -> bool:
        return self.patience > 0

    @classmethod
    def from_args(
        cls,
        args,
        checkpoint_metadata: dict | None = None,
    ) -> "LossPlateauEarlyStop":
        tracker = cls(
            mode=args.early_stop_mode,
            min_steps=args.min_steps,
            patience=args.early_stop_patience,
            min_delta=args.early_stop_min_delta,
            ema_alpha=args.early_stop_ema_alpha,
            window_size=args.early_stop_window,
        )
        saved = (checkpoint_metadata or {}).get("early_stop")
        if not isinstance(saved, dict):
            return tracker
        state = saved.get("state")
        if not isinstance(state, dict):
            return tracker
        best = state.get("best")
        smoothed = state.get("smoothed")
        tracker.best = None if best is None else float(best)
        tracker.smoothed = None if smoothed is None else float(smoothed)
        tracker.bad_steps = int(state.get("bad_steps", 0))
        tracker.observations = int(state.get("observations", 0))
        raw_window = state.get("window_values", [])
        if isinstance(raw_window, list):
            tracker.window_values = [float(value) for value in raw_window]
        else:
            raise ValueError("checkpoint early-stop window_values must be a list")
        tracker._validate_state()
        return tracker

    def config_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "min_steps": self.min_steps,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "ema_alpha": self.ema_alpha,
            "window_size": self.window_size,
        }

    def state_dict(self) -> dict:
        return {
            "best": self.best,
            "smoothed": self.smoothed,
            "bad_steps": self.bad_steps,
            "observations": self.observations,
            # Required for bit-for-bit continuation in rolling-window mode.
            "window_values": list(self.window_values),
        }

    def metadata_dict(self) -> dict:
        return {"config": self.config_dict(), "state": self.state_dict()}

    def observe(self, loss: float, step: int) -> bool:
        loss = float(loss)
        if not math.isfinite(loss):
            raise FloatingPointError(
                f"non-finite loss passed to early stopping at step {step}"
            )
        self.observations += 1
        ready = True
        if self.mode == "ema":
            self.smoothed = (
                loss
                if self.smoothed is None
                else self.ema_alpha * loss + (1.0 - self.ema_alpha) * self.smoothed
            )
        else:
            self.window_values.append(loss)
            if len(self.window_values) > self.window_size:
                del self.window_values[0]
            self.smoothed = sum(self.window_values) / len(self.window_values)
            ready = len(self.window_values) == self.window_size

        if not ready:
            self.bad_steps = 0
            return False
        assert self.smoothed is not None
        if step < self.min_steps:
            # Burn-in follows the current loss level instead of retaining an
            # anomalously low early value that could consume patience later.
            self.best = self.smoothed
            self.bad_steps = 0
            return False

        improved = self.best is None or self.smoothed < self.best - self.min_delta
        if improved:
            self.best = self.smoothed
            self.bad_steps = 0
        else:
            self.bad_steps += 1
        return (
            self.enabled and step >= self.min_steps and self.bad_steps >= self.patience
        )


def _build_omvt_cfg(args, checkpoint_metadata: dict | None = None) -> OMVTConfig:
    if args.init_omvt_checkpoint:
        # The checkpoint's own omvt_config is the only authoritative source of
        # tower geometry: building a CLI-derived config here and loading a
        # prod-geometry tower (e.g. d_vision=512, dataclass patch shapes) into
        # it would fail on shape mismatch.
        payload = load_omvt_payload(args.init_omvt_checkpoint, weights_only=False)
        if isinstance(payload, dict) and "omvt_config" in payload:
            cfg = OMVTConfig(**payload["omvt_config"])
            if args.image_size != cfg.image_size:
                print(
                    f"[init] --image-size {args.image_size} -> {cfg.image_size} "
                    "(from OMVT checkpoint)",
                )
            return cfg
    checkpoint_metadata = checkpoint_metadata or {}
    if isinstance(checkpoint_metadata.get("omvt_config"), dict):
        cfg = OMVTConfig(**checkpoint_metadata["omvt_config"])
        if args.image_size != cfg.image_size:
            print(
                f"[init] --image-size {args.image_size} -> {cfg.image_size} "
                "(from RDT/VLM checkpoint)",
            )
        return cfg
    return make_omvt_cfg(
        args.image_size,
        args.d_vision,
        args.n_image_tokens,
        preset=getattr(args, "patch_preset", "derived"),
    )


def _make_text_batch(args, vocab_floor=300, vocab_ceil=320):
    B, L, N = args.batch_size, args.seq_len, args.n_image_tokens
    rng = torch.Generator().manual_seed(args.seed)
    # layout: [BOS] <image_patch>*N <random text...> [EOS]
    text_len = L - N - 2
    if text_len <= 0:
        raise ValueError("seq_len must be greater than 2 + n_image_tokens")
    text_ids = torch.randint(vocab_floor, vocab_ceil, (B, text_len), generator=rng)
    input_ids = torch.full((B, L), 0, dtype=torch.long)
    input_ids[:, 0] = BOS_ID
    input_ids[:, 1 : 1 + N] = IMAGE_PATCH_ID
    input_ids[:, 1 + N : 1 + N + text_len] = text_ids
    input_ids[:, -1] = EOS_ID
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    return input_ids, attention_mask, labels


def _checkpoint_metadata(path: str) -> dict:
    if not path:
        return {}
    p = Path(path)
    # Legacy callers may pass an arbitrary standalone state-dict file.  Such a
    # file has no neighbouring checkpoint contract, so fall back to CLI config.
    if p.is_file() and p.name != "model.pt":
        return {}
    return load_checkpoint_metadata(path)


def _build_rdt_cfg(args, metadata: dict, device: torch.device) -> RDTConfig:
    raw = metadata.get("rdt_config")
    if isinstance(raw, dict):
        rdt_cfg = RDTConfig(**raw)
        print("[init] using RDTConfig from checkpoint metadata")
    else:
        rdt_cfg = CONFIG_CHOICES[args.config]()
    rdt_cfg = replace(rdt_cfg, max_seq_len=args.seq_len)
    if args.recurrent_steps is not None:
        rdt_cfg = replace(rdt_cfg, recurrent_steps=args.recurrent_steps)
    if args.grad_ckpt:
        rdt_cfg = replace(
            rdt_cfg, grad_ckpt_recurrent=True, grad_ckpt_prelude_coda=True
        )
    return _resolve_mamba_backend(
        rdt_cfg,
        args.mamba,
        device=device,
        context="scripts.train_vlm_align",
    )


def _resolve_model_state(path: str):
    p = Path(path)
    if p.is_file():
        state = torch.load(p, map_location="cpu", weights_only=False)
    else:
        ckpt_dir = resolve_checkpoint_dir(p)
        # Initialization only needs model weights.  Avoid loading the optimizer
        # state, which is substantially larger than the frozen language model.
        state = torch.load(
            ckpt_dir / "model.pt", map_location="cpu", weights_only=False
        )
    if isinstance(state, dict) and "model" in state and "embed.weight" not in state:
        state = state["model"]
    return state


def _load_rdt_init(model: RDTForCausalLM, path: str) -> None:
    if not path:
        return
    state = _resolve_model_state(path)
    checkpoint_has_omvt = any(k.startswith("vision.omvt.") for k in state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = (
        [k for k in missing if k.startswith("vision.omvt.")]
        if not checkpoint_has_omvt
        else []
    )
    bad_missing = [k for k in missing if k not in allowed_missing]
    if bad_missing or unexpected:
        detail = f"missing={bad_missing[:8]} unexpected={list(unexpected)[:8]}"
        raise RuntimeError(
            f"RDT checkpoint {path} does not exactly match the language model; "
            f"refusing to freeze partially loaded weights ({detail})"
        )
    if allowed_missing:
        print(
            f"[init] loaded text-only RDT checkpoint {path}; "
            f"initialized {len(allowed_missing)} vision.omvt tensors separately"
        )
    else:
        print(f"[init] loaded full RDT/VLM checkpoint {path} (exact key match)")


def _configure_trainable_modules(model: RDTForCausalLM, args) -> list[str]:
    if args.freeze_rdt:
        for param in model.parameters():
            param.requires_grad_(False)
        for param in model.vision.omvt.parameters():
            param.requires_grad_(True)
    if args.frozen_vision:
        for param in model.vision.omvt.tower.parameters():
            param.requires_grad_(False)

    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    if args.freeze_rdt:
        leaked = [
            name for name in trainable_names if not name.startswith("vision.omvt.")
        ]
        if leaked:
            raise RuntimeError(
                "--freeze-rdt left non-OMVT parameters trainable: "
                + ", ".join(leaked[:8])
            )
    if not trainable_names:
        raise ValueError("model has no trainable parameters")
    return trainable_names


def _alignment_metadata(
    args,
    rdt_cfg: RDTConfig,
    omvt_cfg: OMVTConfig,
    train_cfg: TrainingConfig,
    *,
    inherited: dict | None = None,
    early_stopper: LossPlateauEarlyStop | None = None,
    streaming: dict[str, Any] | None = None,
    stop_reason: str = "",
    final: bool = False,
    ocr_data_contract: dict | None = None,
) -> dict:
    inherited = inherited or {}
    metadata = {
        "phase": "vlm_align",
        "config": inherited.get("config", args.config) if args.resume else args.config,
        "rdt_config": asdict(rdt_cfg),
        "omvt_config": asdict(omvt_cfg),
        "training_config": asdict(train_cfg),
        "freeze_rdt": bool(args.freeze_rdt),
        "frozen_vision": bool(args.frozen_vision),
        "ocr_target_encoding": str(
            getattr(args, "ocr_target_encoding", OCR_TARGET_ENCODING_CHOICES[0])
        ),
        "ocr_tokenization_contract_version": OCR_TOKENIZATION_CONTRACT_VERSION,
        "ocr_position_contract": str(
            getattr(args, "ocr_position_contract", BOUNDARY_V1)
        ),
        "ocr_position_contract_version": OCR_POSITION_CONTRACT_METADATA_VERSION,
        "recurrent_steps": int(rdt_cfg.recurrent_steps),
        "mamba_backend": "official" if rdt_cfg.use_official_mamba else "naive",
        "source_rdt_checkpoint": (
            args.init_rdt_checkpoint or inherited.get("source_rdt_checkpoint", "")
        ),
        "source_omvt_checkpoint": (
            args.init_omvt_checkpoint or inherited.get("source_omvt_checkpoint", "")
        ),
        "use_ema_tower": bool(
            args.use_ema_tower or inherited.get("use_ema_tower", False)
        ),
        "early_stop": (
            early_stopper.metadata_dict() if early_stopper is not None else {}
        ),
        "streaming": copy.deepcopy(
            streaming if streaming is not None else inherited.get("streaming", {})
        ),
        "stop_reason": stop_reason,
        "final": bool(final),
    }
    if ocr_data_contract is not None:
        token_contract = ocr_data_contract["ocr_tokenization_contract"]
        source_tokenizer = inherited.get("source_rdt_tokenizer_bundle")
        if not isinstance(source_tokenizer, dict) or not source_tokenizer:
            source_tokenizer = inherited.get("tokenizer_bundle")
        source_tokenizer_algorithm = inherited.get(
            "source_rdt_tokenizer_algorithm"
        )
        if (
            not isinstance(source_tokenizer_algorithm, dict)
            or not source_tokenizer_algorithm
        ):
            source_tokenizer_algorithm = inherited.get("tokenizer_algorithm")
        metadata.update(
            {
                "ocr_target_encoding": token_contract["target_encoding"],
                "ocr_tokenization_contract_version": token_contract[
                    "tokenization_contract_version"
                ],
                "ocr_data_contract": ocr_data_contract,
                "source_rdt_tokenizer_bundle": source_tokenizer,
                "source_rdt_tokenizer_algorithm": (
                    source_tokenizer_algorithm
                ),
            }
        )
    return metadata


def _validate_frozen_rdt_tokenizer_lineage(
    source_metadata: dict,
    token_contract: dict,
) -> tuple[dict, dict]:
    """Require exact RDT and OCR tokenizer identities across phase changes."""

    source_bundle = source_metadata.get("source_rdt_tokenizer_bundle")
    if not isinstance(source_bundle, dict) or not source_bundle:
        source_bundle = source_metadata.get("tokenizer_bundle")
    expected_bundle = token_contract.get("tokenizer_bundle")
    if (
        not isinstance(expected_bundle, dict)
        or not expected_bundle
        or source_bundle != expected_bundle
    ):
        raise ValueError(
            "frozen source RDT tokenizer bundle contract differs from the "
            "OCR data/tokenizer contract"
        )

    source_algorithm = source_metadata.get("source_rdt_tokenizer_algorithm")
    if not isinstance(source_algorithm, dict) or not source_algorithm:
        source_algorithm = source_metadata.get("tokenizer_algorithm")
    expected_algorithm = token_contract.get("pretraining_tokenizer_algorithm")
    if (
        not isinstance(expected_algorithm, dict)
        or not expected_algorithm
        or source_algorithm != expected_algorithm
    ):
        raise ValueError(
            "frozen source RDT tokenizer algorithm differs from the "
            "OCR runtime contract"
        )
    return source_bundle, source_algorithm


def _early_stop_resume_conflicts(
    args,
    checkpoint_metadata: dict,
) -> list[str]:
    current = LossPlateauEarlyStop.from_args(args).config_dict()
    saved = checkpoint_metadata.get("early_stop")
    if not isinstance(saved, dict):
        return ["checkpoint has no early_stop metadata"] if current["enabled"] else []
    saved_config = saved.get("config")
    if not isinstance(saved_config, dict):
        return ["checkpoint has no early_stop config"] if current["enabled"] else []
    conflicts = []
    for key, current_value in current.items():
        if key not in saved_config:
            conflicts.append(f"{key}: missing from checkpoint metadata")
        elif saved_config[key] != current_value:
            conflicts.append(
                f"{key}: checkpoint={saved_config[key]!r} current={current_value!r}"
            )
    if current["enabled"] and not isinstance(saved.get("state"), dict):
        conflicts.append("checkpoint has no early_stop state")
    return conflicts


_RESUME_MUTABLE_TRAINING_FIELDS = {
    "output_dir",
    "save_every",
    "keep_last_n",
    "resume",
    "resume_skip_data",
    "log_every",
    "eval_every",
    "eval_max_batches",
    "tensorboard",
    "wandb_project",
}


def _resume_training_conflicts(
    train_cfg: TrainingConfig,
    checkpoint_metadata: dict,
) -> list[str]:
    """Return continuation-critical config differences for a resume.

    Optimizer and scheduler state are only meaningful under the configuration
    that produced them.  Operational settings such as checkpoint retention and
    logging may change; data geometry, batch shape, optimizer, LR schedule,
    precision, recurrence curriculum, distributed mode, and seed may not.
    """

    saved = checkpoint_metadata.get("training_config")
    if not isinstance(saved, dict):
        return [
            "checkpoint has no training_config metadata; use it as "
            "--init-rdt-checkpoint for a new optimizer run instead of --resume"
        ]
    current = asdict(train_cfg)
    conflicts: list[str] = []
    for config_field in fields(TrainingConfig):
        name = config_field.name
        if name in _RESUME_MUTABLE_TRAINING_FIELDS:
            continue
        if name not in saved:
            conflicts.append(f"{name}: missing from checkpoint metadata")
        elif saved[name] != current[name]:
            conflicts.append(
                f"{name}: checkpoint={saved[name]!r} current={current[name]!r}"
            )
    return conflicts


def _fast_forward_stream(
    batch_iter, resumed_step: int, train_cfg: TrainingConfig
) -> None:
    skip = resumed_step * train_cfg.grad_accum_steps
    if skip <= 0:
        return
    t0 = time.time()
    for done in range(skip):
        next(batch_iter)
        if (done + 1) % 5000 == 0:
            print(
                f"scripts.train_vlm_align: resume fast-forward "
                f"{done + 1}/{skip} batches ({time.time() - t0:.0f}s elapsed)",
                flush=True,
            )
    print(
        f"scripts.train_vlm_align: resume fast-forwarded {skip} batches in "
        f"{time.time() - t0:.0f}s",
        flush=True,
    )


def _resolve_omvt_state(path: str, use_ema: bool = False):
    payload = load_omvt_payload(path, weights_only=False)
    if isinstance(payload, dict) and "tower_state" in payload:
        print("[init] using joint byte-CTC tower_state")
        return payload["tower_state"]
    state = tower_state_from_payload(payload, use_ema=use_ema)
    if use_ema and isinstance(payload, dict) and payload.get("tower_ema"):
        print("[init] using EMA tower weights")
    return state


def _load_omvt_init(model: RDTForCausalLM, path: str, use_ema: bool = False) -> None:
    if not path:
        return
    if model.vision.omvt is None:
        raise ValueError("OMVT injector must be installed before loading tower weights")
    model.vision.omvt.tower.load_state_dict(_resolve_omvt_state(path, use_ema=use_ema))


@dataclass(frozen=True)
class StreamingCorpusSpec:
    paths: tuple[Path, ...]
    manifest: dict[str, Any]
    manifest_sha256: str
    tokenizer_manifest_sha256: str
    tokenizer_bundle: Any
    resume_cursor: dict[str, Any] | None


class CursorTrackingIterator(Iterator[dict[str, Any]]):
    """Commit only the data cursor consumed by a completed optimizer step."""

    def __init__(
        self,
        iterable,
        *,
        initial_cursor: dict[str, Any],
    ) -> None:
        self._iterator = iter(iterable)
        self.pending_cursor = copy.deepcopy(initial_cursor)
        self.committed_cursor = copy.deepcopy(initial_cursor)

    def __iter__(self) -> "CursorTrackingIterator":
        return self

    def __next__(self) -> dict[str, Any]:
        batch = next(self._iterator)
        cursor = batch.get("corpus_cursor")
        if not isinstance(cursor, dict):
            raise RuntimeError("streaming batch is missing its exact corpus cursor")
        self.pending_cursor = copy.deepcopy(cursor)
        return batch

    def commit(self) -> None:
        self.committed_cursor = copy.deepcopy(self.pending_cursor)


def _canonical_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prepare_streaming_corpus(
    args,
    checkpoint_metadata: dict[str, Any],
) -> StreamingCorpusSpec:
    """Validate every source identity before allocating the training model."""

    from Model.ocr.streaming_corpus import (
        corpus_manifest,
        discover_wds_shards,
        validate_cursor,
    )
    from Tokenizer.unified.bundle import TokenizerBundle, read_manifest
    from scripts.build_ocr_data import make_ocr_target_encoder

    excluded_ids = args.stream_exclude_shard_id or [2303]
    paths = discover_wds_shards(
        args.stream_wds_dir,
        exclude_ids=excluded_ids,
        limit=args.stream_max_wds_shards,
    )
    manifest, manifest_sha = corpus_manifest(
        paths,
        hanshi_meta=args.stream_hanshi_meta,
        hanshi_pages=args.stream_hanshi_pages,
        excluded_shard_ids=excluded_ids,
        val_src_doc_min=args.val_src_doc_min,
        test_src_doc_min=args.test_src_doc_min,
        seed=args.seed,
    )

    bundle = TokenizerBundle.from_dir(args.stream_tokenizer_bundle)
    issues = bundle.validate()
    if issues:
        detail = "; ".join(issues[:8])
        raise ValueError(f"invalid streaming tokenizer bundle: {detail}")
    tokenizer_manifest = read_manifest(args.stream_tokenizer_bundle)
    if not tokenizer_manifest:
        raise ValueError("streaming tokenizer bundle has no manifest.json")
    tokenizer_sha = _canonical_sha256(tokenizer_manifest)

    # Exercise the exact production label route before model/GPU allocation.
    # These probes cover MorphBPE words, ordinary word boundaries, and the
    # Mongolian control characters whose loss would change glyph identity.
    encode_target = make_ocr_target_encoder(
        bundle.tokenizer,
        mode=args.ocr_target_encoding,
    )
    tokenization_probes = (
        "ᠮᠣᠩᠭᠣᠯ ᠪᠢᠴᠢᠭ",
        "ᠪᠢᠴᠢᠭ᠋ ᠦᠨ",
        "ᠮᠣᠩᠭᠣᠯ᠎ᠠ",
        "ᠨᠡᠷ ᠡ",
        "︱ 9/7 ︱ 9/9/9",
        "中文 Кирилл 🙂",
        "a b",
        "▁◈<bos><image_patch>",
    )
    probe_lengths = [len(encode_target(text)) for text in tokenization_probes]
    if encode_target.stats["byte_fallback"]:
        raise ValueError(
            "production OCR tokenization preflight used byte fallback; "
            "frozen-language visual training requires native tokens only"
        )
    print(
        "[tokenization] contract=native-pretraining-v2 "
        f"mode={args.ocr_target_encoding} probes={len(tokenization_probes)} "
        f"token_lengths={probe_lengths} unk=0 fallback=0 "
        f"canonicalized={encode_target.stats['canonicalized']} "
        f"tokenizer_manifest_sha256={tokenizer_sha}",
        flush=True,
    )

    resume_cursor = None
    if args.resume:
        saved = checkpoint_metadata.get("streaming")
        if not isinstance(saved, dict):
            raise ValueError("streaming resume checkpoint has no streaming metadata")
        expected = {
            "corpus_manifest_sha256": manifest_sha,
            "tokenizer_manifest_sha256": tokenizer_sha,
            "ocr_target_encoding": args.ocr_target_encoding,
            "ocr_tokenization_contract_version": OCR_TOKENIZATION_CONTRACT_VERSION,
        }
        for key, current in expected.items():
            previous = saved.get(key)
            if previous != current:
                raise ValueError(
                    f"streaming resume {key} mismatch: "
                    f"checkpoint={previous!r} current={current!r}"
                )
        raw_cursor = saved.get("corpus_cursor")
        if not isinstance(raw_cursor, dict):
            raise ValueError("streaming resume checkpoint has no corpus_cursor")
        resume_cursor = validate_cursor(raw_cursor)

    return StreamingCorpusSpec(
        paths=tuple(paths),
        manifest=manifest,
        manifest_sha256=manifest_sha,
        tokenizer_manifest_sha256=tokenizer_sha,
        tokenizer_bundle=bundle,
        resume_cursor=resume_cursor,
    )


def _stream_vlm_batches(
    args,
    omvt_cfg: OMVTConfig,
    spec: StreamingCorpusSpec,
) -> CursorTrackingIterator:
    """Build frozen-RDT OCR batches directly from x2 WDS + Hanshi NAS data."""

    from Model.ocr.streaming_corpus import MixedOCRCorpus
    from scripts.build_ocr_data import make_ocr_target_encoder
    from scripts.train_ctc_head import _prefetch_batches

    encode_target = make_ocr_target_encoder(
        spec.tokenizer_bundle.tokenizer,
        mode=args.ocr_target_encoding,
    )
    max_target_len = args.seq_len - omvt_cfg.compress_to - 4
    if max_target_len <= 0:
        raise ValueError("stream seq_len leaves no room for an OCR target")
    corpus = MixedOCRCorpus(
        spec.paths,
        hanshi_meta=args.stream_hanshi_meta,
        hanshi_pages=args.stream_hanshi_pages,
        image_size=omvt_cfg.image_size,
        seed=args.seed,
        val_src_doc_min=args.val_src_doc_min,
        max_target_len=max_target_len,
        cursor=spec.resume_cursor,
    )
    collator = PretrainingCollator(
        pad_id=PAD_ID,
        max_seq_len=args.seq_len,
        position_contract=args.ocr_position_contract,
    )

    def _iter():
        for visual in corpus.batches(args.batch_size):
            lengths = visual["target_lengths"].tolist()
            flat = visual["targets"].tolist()
            rows = []
            offset = 0
            keys = list(visual.get("keys") or [])
            sources = list(visual.get("sources") or [])
            for row_index, length in enumerate(lengths):
                raw = bytes(flat[offset : offset + length])
                offset += length
                text = raw.decode("utf-8", errors="strict")
                try:
                    target_ids = encode_target(text)
                except ValueError as exc:
                    source = (
                        sources[row_index]
                        if row_index < len(sources)
                        else "unknown"
                    )
                    key = keys[row_index] if row_index < len(keys) else "unknown"
                    raise ValueError(
                        "strict native OCR target encoding failed for "
                        f"source={source!r} key={key!r}: {exc}"
                    ) from exc
                row = build_ocr_row(
                    target_ids,
                    omvt_cfg.compress_to,
                    None,
                    bos_id=BOS_ID,
                    image_start_id=IMAGE_START_ID,
                    image_patch_id=IMAGE_PATCH_ID,
                    image_end_id=IMAGE_END_ID,
                    eos_id=EOS_ID,
                )
                row.pop("images", None)
                rows.append(row)
            batch = collator(rows)
            batch["pixel_values"] = dict(collate_omvt_batch(visual["pixels"], omvt_cfg))
            batch["corpus_cursor"] = visual["corpus_cursor"]
            yield batch

    print(
        f"[stream] WDS shards={len(spec.paths)} + Hanshi; mix=2:1; "
        f"batch={args.batch_size}; full_corpus={args.stream_max_wds_shards == 0}; "
        f"ocr_target_encoding={args.ocr_target_encoding}; "
        f"manifest_sha256={spec.manifest_sha256}",
        flush=True,
    )
    prefetched = _prefetch_batches(_iter(), args.stream_prefetch_batches)
    return CursorTrackingIterator(prefetched, initial_cursor=corpus.state_dict())


def main(argv=None):
    args = parse_args(argv)
    stream_fields = (
        args.stream_wds_dir,
        args.stream_hanshi_meta,
        args.stream_hanshi_pages,
        args.stream_tokenizer_bundle,
    )
    stream_mode = any(bool(value) for value in stream_fields)
    if stream_mode and not all(bool(value) for value in stream_fields):
        print(
            "scripts.train_vlm_align: streaming requires WDS, Hanshi meta/pages, "
            "and tokenizer bundle together",
            file=sys.stderr,
        )
        return 2
    if stream_mode and args.data:
        print(
            "scripts.train_vlm_align: --data and direct corpus streaming are exclusive",
            file=sys.stderr,
        )
        return 2
    if stream_mode and not args.freeze_rdt:
        print(
            "scripts.train_vlm_align: full-corpus visual training requires "
            "--freeze-rdt so no language parameter can update",
            file=sys.stderr,
        )
        return 2
    if stream_mode and args.ocr_target_encoding != "native":
        print(
            "scripts.train_vlm_align: full-corpus frozen-language training "
            "requires --ocr-target-encoding native; fallback representations "
            "are incompatible with the pretrained RDT token distribution",
            file=sys.stderr,
        )
        return 2
    if stream_mode and args.stream_max_wds_shards and not args.smoke:
        print(
            "scripts.train_vlm_align: --stream-max-wds-shards is restricted to "
            "--smoke runs; production streaming must expose every usable shard",
            file=sys.stderr,
        )
        return 2
    # Fast-fail validation **before** any device alloc / model construction.
    # Mirrors the train_rdt CLI pattern: misconfigured runs should not pay the
    # cost of building the model only to crash inside the first step.
    if args.image_size <= 0 or args.image_size % 4 != 0:
        print(
            "scripts/train_vlm_align: --image-size must be a positive multiple of 4",
            file=sys.stderr,
        )
        return 2
    if args.resume and (args.init_rdt_checkpoint or args.init_omvt_checkpoint):
        print(
            "scripts/train_vlm_align: --resume cannot be combined with "
            "--init-rdt-checkpoint or --init-omvt-checkpoint",
            file=sys.stderr,
        )
        return 2
    if args.recurrent_steps is not None and args.recurrent_steps <= 0:
        print(
            "scripts/train_vlm_align: --recurrent-steps must be positive",
            file=sys.stderr,
        )
        return 2
    if args.keep_last_n < 0:
        print(
            "scripts/train_vlm_align: --keep-last-n must be non-negative",
            file=sys.stderr,
        )
        return 2
    if args.ocr_native_targets and (
        not args.data or not args.freeze_rdt or not args.ocr_tokenizer_bundle
    ):
        print(
            "scripts/train_vlm_align: --ocr-native-targets requires --data, "
            "--freeze-rdt, and --ocr-tokenizer-bundle",
            file=sys.stderr,
        )
        return 2
    if not args.ocr_native_targets and (
        args.ocr_tokenizer_bundle or args.ocr_data_contract
    ):
        print(
            "scripts/train_vlm_align: OCR contract arguments require "
            "--ocr-native-targets",
            file=sys.stderr,
        )
        return 2

    if getattr(args, "device", "auto") == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    precision = (
        "bf16"
        if args.precision == "auto" and device.type == "cuda"
        else "fp32"
        if args.precision == "auto"
        else args.precision
    )
    if args.resume:
        try:
            validate_resumable_checkpoint(
                args.resume,
                require_scaler=(precision == "fp16" and device.type == "cuda"),
                context="VLM --resume",
            )
        except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
            print(f"scripts/train_vlm_align: {exc}", file=sys.stderr)
            return 2

    source_path = args.resume or args.init_rdt_checkpoint
    stream_spec: StreamingCorpusSpec | None = None
    try:
        source_metadata = _checkpoint_metadata(source_path)
        source_is_vlm = source_metadata.get("phase") == "vlm_align"
        if args.resume or source_is_vlm:
            args.ocr_position_contract = resolve_checkpoint_ocr_position_contract(
                source_metadata, args.ocr_position_contract
            )
        elif args.ocr_position_contract != BOUNDARY_V1:
            raise ValueError(
                "new OCR training runs must use --ocr-position-contract "
                "boundary_v1; legacy_sequential_v0 is compatibility-only"
            )
        if stream_mode:
            stream_spec = _prepare_streaming_corpus(args, source_metadata)
        omvt_cfg = _build_omvt_cfg(args, source_metadata)
    except (OSError, TypeError, ValueError) as exc:
        print(f"scripts/train_vlm_align: {exc}", file=sys.stderr)
        return 2
    saved_ocr_mode = source_metadata.get("ocr_target_encoding")
    if args.resume and bool(saved_ocr_mode) != bool(args.ocr_native_targets):
        print(
            "scripts/train_vlm_align: --ocr-native-targets conflicts with "
            f"resume checkpoint ocr_target_encoding={saved_ocr_mode!r}",
            file=sys.stderr,
        )
        return 2

    ocr_data_contract = None
    if args.ocr_native_targets:
        try:
            from Tokenizer.unified.bundle import TokenizerBundle

            bundle = TokenizerBundle.from_dir(args.ocr_tokenizer_bundle)
            bundle_issues = bundle.validate()
            if bundle_issues:
                raise ValueError(
                    "invalid tokenizer bundle:\n  - "
                    + "\n  - ".join(bundle_issues)
                )
            token_contract = native_tokenization_contract(
                bundle.tokenizer,
                args.ocr_tokenizer_bundle,
            )
            data_path = args.data
            contract_path = (
                Path(args.ocr_data_contract)
                if args.ocr_data_contract
                else default_ocr_alignment_contract_path(data_path)
            )
            ocr_data_contract = load_and_validate_ocr_alignment_data_contract(
                contract_path,
                data_path,
                token_contract,
            )
            _validate_frozen_rdt_tokenizer_lineage(
                source_metadata,
                token_contract,
            )
            if args.resume:
                saved_contract = source_metadata.get("ocr_data_contract")
                if saved_contract != ocr_data_contract:
                    raise ValueError(
                        "resume OCR data contract differs from the checkpoint; "
                        "refusing to mix corpus bytes or LM representations "
                        "under an existing optimizer/data cursor"
                    )
        except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
            print(f"scripts/train_vlm_align: {exc}", file=sys.stderr)
            return 2
    if args.n_image_tokens is None:
        args.n_image_tokens = omvt_cfg.compress_to
    elif args.n_image_tokens != omvt_cfg.compress_to:
        print(
            "scripts/train_vlm_align: --n-image-tokens does not match checkpoint "
            f"OMVT config ({args.n_image_tokens} != {omvt_cfg.compress_to})",
            file=sys.stderr,
        )
        return 2
    args.image_size = omvt_cfg.image_size
    if args.seq_len <= args.n_image_tokens + 2:
        print(
            "scripts/train_vlm_align: --seq-len must be > --n-image-tokens + 2 "
            f"(got seq_len={args.seq_len}, n_image_tokens={args.n_image_tokens})",
            file=sys.stderr,
        )
        return 2
    if args.resume and source_metadata:
        for key in ("freeze_rdt", "frozen_vision"):
            if key in source_metadata and bool(source_metadata[key]) != bool(
                getattr(args, key)
            ):
                print(
                    f"scripts/train_vlm_align: resume {key}={getattr(args, key)} "
                    f"conflicts with checkpoint value {source_metadata[key]}",
                    file=sys.stderr,
                )
                return 2
        saved_rdt = source_metadata.get("rdt_config")
        if (
            args.recurrent_steps is not None
            and isinstance(saved_rdt, dict)
            and int(saved_rdt.get("recurrent_steps", args.recurrent_steps))
            != args.recurrent_steps
        ):
            print(
                "scripts/train_vlm_align: --recurrent-steps conflicts with the "
                "resume checkpoint",
                file=sys.stderr,
            )
            return 2

    torch.manual_seed(args.seed)

    try:
        rdt_cfg = _build_rdt_cfg(args, source_metadata, device)
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    train_cfg = TrainingConfig(
        train_data=args.data or ("nas-stream" if stream_mode else ""),
        seq_len=args.seq_len,
        micro_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.05,
        max_steps=args.max_steps,
        warmup_steps=max(1, args.warmup_steps),
        precision=precision,
        output_dir=args.output,
        save_every=args.save_every,
        resume=args.resume,
        keep_last_n=args.keep_last_n,
        resume_skip_data=not args.no_resume_skip_data,
    )
    if args.resume:
        conflicts = _resume_training_conflicts(train_cfg, source_metadata)
        if conflicts:
            print(
                "scripts/train_vlm_align: resume training config conflicts with "
                "the checkpoint; refusing mixed optimizer/scheduler semantics:",
                file=sys.stderr,
            )
            for conflict in conflicts[:12]:
                print(f"  - {conflict}", file=sys.stderr)
            if len(conflicts) > 12:
                print(f"  - ... and {len(conflicts) - 12} more", file=sys.stderr)
            return 2
        early_stop_conflicts = _early_stop_resume_conflicts(args, source_metadata)
        if early_stop_conflicts:
            print(
                "scripts/train_vlm_align: resume early-stop state conflicts with "
                "the checkpoint; refusing to reset plateau history:",
                file=sys.stderr,
            )
            for conflict in early_stop_conflicts:
                print(f"  - {conflict}", file=sys.stderr)
            return 2
        if source_metadata.get("stop_reason") == "loss_plateau":
            print(
                "scripts/train_vlm_align: checkpoint already completed because "
                "loss plateaued; use it as an initialization checkpoint for a "
                "deliberately new run",
                file=sys.stderr,
            )
            return 2

    try:
        early_stopper = LossPlateauEarlyStop.from_args(
            args,
            source_metadata if args.resume else None,
        )
    except (TypeError, ValueError) as exc:
        print(
            f"scripts/train_vlm_align: invalid early-stop state: {exc}", file=sys.stderr
        )
        return 2

    model = RDTForCausalLM(rdt_cfg).to(device)
    # plug in matching-size OMVT injector (otherwise dispatcher would build
    # a default-sized one on first forward and fail on tiny synthetic inputs).
    model.vision._omvt_cfg = omvt_cfg
    model.vision.omvt = OMVTInjector(rdt_cfg, omvt_cfg).to(device)
    _load_rdt_init(model, args.init_rdt_checkpoint)
    _load_omvt_init(model, args.init_omvt_checkpoint, use_ema=args.use_ema_tower)

    trainable_names = _configure_trainable_modules(model, args)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"[train] recurrent_steps={rdt_cfg.recurrent_steps} "
        f"trainable={n_trainable:,}/{n_total:,} tensors={len(trainable_names)}"
    )

    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)
    state = TrainState()
    if args.resume:
        state.step = resume_state(
            args.resume,
            model,
            optimizer,
            scheduler,
            state=state,
        )

    Path(args.output).mkdir(parents=True, exist_ok=True)
    logger = RankZeroLogger(args.output, enable_tensorboard=False)

    t0 = time.time()
    completed = False
    stop_reason = ""
    stream_batch_iter: CursorTrackingIterator | None = None

    def _streaming_checkpoint_metadata() -> dict[str, Any] | None:
        if stream_spec is None:
            return None
        if stream_batch_iter is None:
            cursor = stream_spec.resume_cursor
        else:
            cursor = stream_batch_iter.committed_cursor
        return {
            "corpus_manifest": stream_spec.manifest,
            "corpus_manifest_sha256": stream_spec.manifest_sha256,
            "tokenizer_manifest_sha256": stream_spec.tokenizer_manifest_sha256,
            "ocr_target_encoding": args.ocr_target_encoding,
            "ocr_tokenization_contract_version": OCR_TOKENIZATION_CONTRACT_VERSION,
            "corpus_cursor": copy.deepcopy(cursor),
            "corpus_complete": stop_reason == "corpus_exhausted",
        }

    def _save(*, final: bool = False) -> None:
        save_checkpoint(
            args.output,
            state.step,
            model,
            optimizer,
            scheduler,
            metadata=_alignment_metadata(
                args,
                rdt_cfg,
                omvt_cfg,
                train_cfg,
                inherited=source_metadata,
                early_stopper=early_stopper,
                streaming=_streaming_checkpoint_metadata(),
                stop_reason=stop_reason,
                final=final,
                ocr_data_contract=ocr_data_contract,
            ),
            keep_last_n=args.keep_last_n,
            scaler=state.extra.get("grad_scaler"),
        )

    def _record_completed_step(metrics: dict[str, float]) -> bool:
        nonlocal stop_reason
        if stream_batch_iter is not None:
            stream_batch_iter.commit()
        should_stop = early_stopper.observe(metrics["loss"], state.step)
        logged = {"loss": float(metrics["loss"])}
        for key in ("grad_norm", "lr", "tokens"):
            value = metrics.get(key)
            if value is not None and math.isfinite(float(value)):
                logged[key] = float(value)
        if early_stopper.enabled:
            if early_stopper.smoothed is not None:
                logged["early_stop_smoothed_loss"] = early_stopper.smoothed
            if early_stopper.best is not None:
                logged["early_stop_best_loss"] = early_stopper.best
            logged["early_stop_bad_steps"] = float(early_stopper.bad_steps)
        logger.log(state.step, logged)
        if should_stop:
            stop_reason = "loss_plateau"
            print(
                f"[early-stop] loss plateau at step {state.step}: "
                f"smoothed={early_stopper.smoothed:.8f} "
                f"best={early_stopper.best:.8f} "
                f"bad_steps={early_stopper.bad_steps}/"
                f"{early_stopper.patience}",
                flush=True,
            )
        if args.save_every and state.step % args.save_every == 0:
            _save()
        return should_stop

    try:
        if stream_mode:
            assert stream_spec is not None
            stream_batch_iter = _stream_vlm_batches(args, omvt_cfg, stream_spec)
            while state.step < args.max_steps:
                try:
                    metrics = train_one_step(
                        model,
                        stream_batch_iter,
                        optimizer,
                        scheduler,
                        train_cfg,
                        state,
                        device=device,
                    )
                except StopIteration:
                    stop_reason = "corpus_exhausted"
                    print(
                        "[stream] corpus exhausted: every eligible source row "
                        "was exposed once",
                        flush=True,
                    )
                    break
                if _record_completed_step(metrics):
                    break
        elif args.data:
            # Real-data path: pull pixel-aware batches from the streaming
            # JSONL dataloader and reuse the canonical train_one_step so
            # CLI behaviour matches train_rdt.
            dataloader = build_dataloader(
                args.data,
                train_cfg,
                world_size=1,
                rank=0,
                pad_id=PAD_ID,
                image_processor=PILImageProcessor(image_size=args.image_size),
                omvt_cfg=omvt_cfg,
                require_precomputed_morphology=False,
                require_verified_images=args.ocr_native_targets,
                position_contract=args.ocr_position_contract,
            )
            batch_iter = iter(dataloader)
            if args.resume and train_cfg.resume_skip_data and state.step > 0:
                _fast_forward_stream(batch_iter, state.step, train_cfg)
            while state.step < args.max_steps:
                metrics = train_one_step(
                    model,
                    batch_iter,
                    optimizer,
                    scheduler,
                    train_cfg,
                    state,
                    device=device,
                )
                if _record_completed_step(metrics):
                    break
        else:
            while state.step < args.max_steps:
                step = state.step + 1
                input_ids, attention_mask, labels = _make_text_batch(args)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                images = torch.randn(
                    args.batch_size,
                    omvt_cfg.in_channels,
                    omvt_cfg.image_size,
                    omvt_cfg.image_size,
                    device=device,
                )
                batch = collate_omvt_batch(images, omvt_cfg)

                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    pixel_values=dict(batch),
                    steps=rdt_cfg.recurrent_steps,
                )
                loss = out["loss"]
                if not bool(torch.isfinite(loss.detach())):
                    raise FloatingPointError(
                        f"non-finite VLM align loss at step {state.step}"
                    )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = clip_or_check_grad_norm(model, 1.0, step=step)
                optimizer.step()
                scheduler.step()
                state.step = step

                metrics = {
                    "loss": float(loss.detach()),
                    "grad_norm": grad_norm,
                    "lr": float(scheduler.get_last_lr()[0]),
                }
                if _record_completed_step(metrics):
                    break
        if not stop_reason:
            stop_reason = "max_steps"
        completed = True
    finally:
        logger.close()
        if completed and not args.smoke:
            _save(final=True)
    mode = "nas-stream" if stream_mode else ("real-data" if args.data else "smoke")
    print(
        f"VLM align {mode} run OK in {time.time() - t0:.1f}s "
        f"(step={state.step}, stop_reason={stop_reason})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
