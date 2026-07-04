# -*- coding: utf-8 -*-

"""CTC-head trainer on top of a FROZEN OMVT vision tower.

Insurance deliverable: if the generative RDT decoder path misses its
deadline, this trains a much smaller, purely discriminative recognizer on
the same pretrained vision tower. The tower (from ``scripts/train_omvt_ssl``)
is never updated here; only a small BiLSTM + CTC head is trained on top of
its ``compress_to`` (=256) compressed visual tokens.

Targets are raw UTF-8 bytes (256 classes) plus one CTC blank class (index
256), decoded from the same pre-tokenized OCR JSONL rows the generative path
consumes (:mod:`Model.ocr.data`, ``build_ocr_row``/``split_ocr_row``): each
row's supervised token tail is decoded back to text via a
:class:`~Tokenizer.unified.bundle.TokenizerBundle`, then UTF-8-encoded. This
keeps the two paths' ground truth identical and their CER numbers directly
comparable (:mod:`scripts.eval_ctc_head` scores with the same
:func:`Model.ocr.metrics.ocr_report` yardstick as :mod:`scripts.eval_vlm_ocr`).

Usage::

    python3 -m scripts.train_ctc_head --smoke   # CPU self-check, no artifacts

    PYTHONPATH=. python3 -m scripts.train_ctc_head \\
        --omvt-checkpoint ~/dolocr/runs/omvt_ssl_v1/latest \\
        --data ~/dolocr/data_v1/jsonl/align \\
        --tokenizer-bundle ~/dolocr/bundle_v3b \\
        --output ~/dolocr/runs/ctc_head_v1 \\
        --steps 20000 --batch-size 32 --save-every 1000 --probe-every 200
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Model.config import EOS_ID, OMVTConfig, TrainingConfig  # noqa: E402
from Model.ocr.data import split_ocr_row  # noqa: E402
from Model.omvt import OMVTVisionTower  # noqa: E402
from Model.training import RankZeroLogger, build_optimizer, build_scheduler  # noqa: E402
from Model.training.loop import clip_or_check_grad_norm  # noqa: E402
from Model.training.omvt_checkpoint import (  # noqa: E402
    load_omvt_payload,
    tower_state_from_payload,
)
from Tokenizer.multimodal import PILImageProcessor  # noqa: E402

N_BYTE_CLASSES = 256
BLANK_ID = 256
NUM_CLASSES = N_BYTE_CLASSES + 1
_CHECKPOINT_NAME = "ctc_head.pt"


# --------------------------------------------------------------------------
# Model: frozen tower + trainable BiLSTM/CTC head.
# --------------------------------------------------------------------------


class CTCHead(nn.Module):
    """2-layer BiLSTM(512->384, bidirectional) + Linear(768->257).

    Consumes the tower's ``[B, T, d_vision]`` compressed visual tokens and
    emits per-frame class logits over 256 UTF-8 byte values + 1 CTC blank
    (index :data:`BLANK_ID`). ~5M trainable params at the spec'd sizes.
    """

    def __init__(self, d_vision: int = 512, hidden: int = 384) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_vision,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(hidden * 2, NUM_CLASSES)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """``[B, T, d_vision] -> [B, T, NUM_CLASSES]`` raw logits."""
        out, _ = self.lstm(features)
        return self.proj(out)


def num_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def build_tower_from_checkpoint(
    path: str, *, use_ema: bool = True, device: torch.device | None = None
) -> tuple[OMVTVisionTower, OMVTConfig]:
    """Load a frozen :class:`OMVTVisionTower` from a ``train_omvt_ssl`` checkpoint.

    Mirrors the pattern in ``scripts/train_vlm_align.py`` /
    ``scripts/eval_omvt_ssl.py``: the checkpoint's own ``omvt_config`` is the
    only authoritative source of tower geometry (building a CLI-derived
    config here could mismatch a prod-geometry tower on shape). All tower
    parameters are frozen — this trainer never updates the vision backbone.
    """

    payload = load_omvt_payload(path, weights_only=False)
    if not (isinstance(payload, dict) and "omvt_config" in payload):
        raise ValueError(
            f"OMVT checkpoint at {path!r} carries no 'omvt_config'; "
            "cannot reconstruct tower geometry"
        )
    cfg = OMVTConfig(**payload["omvt_config"])
    tower = OMVTVisionTower(cfg)
    if device is not None:
        tower = tower.to(device)
    state = tower_state_from_payload(payload, use_ema=use_ema)
    if use_ema and isinstance(payload, dict) and payload.get("tower_ema"):
        print("[init] using EMA tower weights")
    missing, unexpected = tower.load_state_dict(state, strict=True)
    assert not missing and not unexpected  # strict=True already raises otherwise
    for p in tower.parameters():
        p.requires_grad_(False)
    tower.eval()
    return tower, cfg


# --------------------------------------------------------------------------
# Data: lean streaming dataset over build_ocr_row-shaped JSONL rows.
# --------------------------------------------------------------------------


def _resolve_shards(spec: str) -> list[Path]:
    path = Path(spec)
    if path.is_dir():
        return sorted(path.glob("*.jsonl"))
    if any(ch in spec for ch in "*?["):
        return sorted(Path(p) for p in glob.glob(spec, recursive=True))
    return [path] if path.is_file() else []


def row_to_byte_target(row: dict[str, Any], decode) -> tuple[Any, list[int]]:
    """Recover ``(image_ref, byte_target_ids)`` from one ``build_ocr_row`` row.

    Uses the already-tested :func:`Model.ocr.data.split_ocr_row` to split off
    the supervised tail (rather than hand-rolling ``labels[labels !=
    -100][:-1]``): it strips the trailing EOS and validates the row shape,
    then ``decode`` turns the recovered token ids back into ground-truth text,
    which is UTF-8 byte-encoded to build the CTC target alphabet.
    """

    _, target_ids, image_ref = split_ocr_row(row, eos_id=EOS_ID)
    text = decode(target_ids)
    byte_ids = list(text.encode("utf-8"))
    return image_ref, byte_ids


class CTCOcrDataset(IterableDataset):
    """Streams ``build_ocr_row`` JSONL rows into ``(pixels, byte_target)`` pairs.

    Not the LM-batch :mod:`Model.training.data` loader (that pads to a common
    ``input_ids`` length and builds ``pixel_values`` via ``PretrainingCollator``
    for a *causal LM* consumer) — this is a lean, torch ``IterableDataset`` that
    decodes each row's target text exactly once at iteration time and yields
    raw tensors for a CTC head. A simple in-memory shuffle buffer mirrors
    :class:`~Model.training.data.StreamingJsonlDataset`'s approach.

    Rows whose byte target exceeds ``max_target_len`` (the tower's fixed
    frame count, since CTC requires ``target_len <= input_len``) are skipped
    and counted in ``self.n_skipped_long`` (best-effort: IterableDataset
    workers each keep their own counter).
    """

    def __init__(
        self,
        spec: str,
        *,
        tokenizer_bundle,
        image_size: int,
        max_target_len: int = 256,
        shuffle_buffer: int = 0,
        seed: int = 0,
        infinite: bool = True,
    ) -> None:
        paths = _resolve_shards(spec)
        if not paths:
            raise FileNotFoundError(f"no JSONL shards resolved from: {spec!r}")
        self.paths = paths
        self.bundle = tokenizer_bundle
        self.image_size = int(image_size)
        self.max_target_len = int(max_target_len)
        self.shuffle_buffer = max(0, int(shuffle_buffer))
        self.seed = int(seed)
        self.infinite = bool(infinite)
        self.n_skipped_long = 0

    def _files_for_worker(self) -> list[Path]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return self.paths
        return self.paths[worker_info.id :: worker_info.num_workers]

    def __iter__(self) -> Iterator[tuple[torch.Tensor, list[int]]]:
        files = self._files_for_worker()
        if not files:
            return
        # PILImageProcessor is constructed lazily inside the worker process:
        # it holds no torch state that needs sharing, and building it here
        # keeps this dataset usable with num_workers > 0 (fork-safety).
        processor = PILImageProcessor(image_size=self.image_size)
        decode = self.bundle.tokenizer.decode

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rng = random.Random(self.seed + 1000 * worker_id)
        buffer: list[tuple[torch.Tensor, list[int]]] = []

        def _emit(item):
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
            rows_seen = 0
            for path in files:
                with path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        rows_seen += 1
                        image_ref, byte_ids = row_to_byte_target(row, decode)
                        if image_ref is None:
                            continue
                        if len(byte_ids) > self.max_target_len:
                            self.n_skipped_long += 1
                            continue
                        pixels = processor([image_ref])[0]
                        yield from _emit((pixels, byte_ids))
            if rows_seen == 0:
                raise ValueError(
                    f"no JSONL rows available from {[str(p) for p in files]}"
                )
            pass_idx += 1
            rng.shuffle(buffer)
            while buffer:
                yield buffer.pop()


def ctc_collate(
    batch: list[tuple[torch.Tensor, list[int]]],
) -> dict[str, torch.Tensor]:
    """Stack pixels; flatten byte targets (CTC's concatenated-targets form)."""

    pixels = torch.stack([b[0] for b in batch], dim=0)
    target_lengths = torch.tensor([len(b[1]) for b in batch], dtype=torch.long)
    flat_targets = torch.tensor(
        [t for b in batch for t in b[1]], dtype=torch.long
    )
    return {
        "pixels": pixels,
        "targets": flat_targets,
        "target_lengths": target_lengths,
    }


# --------------------------------------------------------------------------
# Greedy CTC decode (shared by the training probe and scripts/eval_ctc_head.py).
# --------------------------------------------------------------------------


def greedy_ctc_decode(logits: torch.Tensor, blank: int = BLANK_ID) -> list[list[int]]:
    """Greedy CTC decode: argmax per frame, collapse repeats, drop blank.

    ``logits``: ``[B, T, C]`` (raw logits or log-probs; only argmax matters).
    Returns one list of byte-class ids per row (blank-free, repeat-collapsed).
    """

    ids = logits.argmax(dim=-1)  # [B, T]
    out: list[list[int]] = []
    for row in ids.tolist():
        decoded: list[int] = []
        prev = None
        for tok in row:
            if tok != prev and tok != blank:
                decoded.append(tok)
            prev = tok
        out.append(decoded)
    return out


def bytes_to_text(byte_ids: list[int]) -> str:
    return bytes(byte_ids).decode("utf-8", errors="replace")


# --------------------------------------------------------------------------
# Checkpointing.
# --------------------------------------------------------------------------


def save_checkpoint(
    output: str | Path,
    step: int,
    head: CTCHead,
    optimizer: torch.optim.Optimizer,
    omvt_cfg: OMVTConfig,
    *,
    d_vision: int,
    hidden: int,
) -> Path:
    out = Path(output)
    step_dir = out / f"step_{step:08d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "head": head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "omvt_config": asdict(omvt_cfg),
        "byte_vocab": {
            "n_byte_classes": N_BYTE_CLASSES,
            "blank_id": BLANK_ID,
            "num_classes": NUM_CLASSES,
        },
        "head_config": {"d_vision": d_vision, "hidden": hidden},
    }
    torch.save(payload, step_dir / _CHECKPOINT_NAME)
    latest = out / "latest"
    if latest.exists() or latest.is_symlink():
        try:
            latest.unlink()
        except OSError:
            import shutil

            shutil.rmtree(latest, ignore_errors=True)
    try:
        os.symlink(step_dir.name, latest)
    except OSError:
        pass
    return step_dir


def resolve_ctc_checkpoint_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    for candidate in (p / _CHECKPOINT_NAME, p / "latest" / _CHECKPOINT_NAME):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"CTC head checkpoint not found: {path}")


def load_ctc_payload(path: str | Path) -> dict[str, Any]:
    return torch.load(
        resolve_ctc_checkpoint_path(path), map_location="cpu", weights_only=False
    )


def load_checkpoint_into(
    path: str | Path,
    head: CTCHead,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    payload = load_ctc_payload(path)
    head.load_state_dict(payload["head"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return int(payload.get("step", 0))


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--omvt-checkpoint", default="", help="train_omvt_ssl checkpoint (dir or .pt)")
    p.add_argument("--data", default="", help="dir of JSONL shards (build_ocr_row rows) or a glob")
    p.add_argument("--tokenizer-bundle", default="", help="unified tokenizer bundle dir")
    p.add_argument(
        "--use-ema-tower",
        type=lambda s: s.lower() not in ("0", "false", "no"),
        default=True,
        help="prefer 'tower_ema' from the checkpoint when present (default True)",
    )
    p.add_argument("--output", default="outputs/ctc_head")
    p.add_argument("--resume", default="")

    p.add_argument("--hidden", type=int, default=384, help="BiLSTM hidden size (per direction)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--shuffle-buffer", type=int, default=1024)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument(
        "--probe-every", type=int, default=200,
        help="every N steps, greedy-decode 4 training samples and print pred-vs-GT (0 disables)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto",
        help="'auto' = cuda if available else cpu",
    )
    p.add_argument(
        "--precision", choices=("auto", "fp32", "bf16"), default="auto",
        help="tower forward precision; 'auto' = bf16 autocast on cuda, fp32 elsewhere. "
        "The head + CTC loss always run fp32 regardless of this flag.",
    )
    p.add_argument("--smoke", action="store_true", help="tiny synthetic CPU self-check")
    return p.parse_args(argv)


def _build_scheduler_cfg(args) -> TrainingConfig:
    return TrainingConfig(
        train_data=args.data or "smoke",
        seq_len=256,
        micro_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_steps=args.steps,
        warmup_steps=max(1, args.warmup_steps),
        precision="fp32",
    )


def _run_probe(
    head: CTCHead,
    features: torch.Tensor,
    byte_targets: list[list[int]],
    step: int,
    n: int = 4,
) -> None:
    with torch.no_grad():
        logits = head(features[: min(n, features.shape[0])].float())
    preds = greedy_ctc_decode(logits)
    print(f"[probe] step={step}")
    for i, (pred, gt) in enumerate(zip(preds, byte_targets[: len(preds)])):
        print(f"  [{i}] pred={bytes_to_text(pred)!r}")
        print(f"       gt  ={bytes_to_text(gt)!r}")


def _train_step(
    tower: OMVTVisionTower,
    head: CTCHead,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    use_bf16: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One forward: frozen-tower features (no grad) -> head -> CTC loss.

    Returns ``(loss, logits, features)`` — ``features`` and ``logits`` are
    reused by the probe printer so it does not re-run the tower forward.
    """

    pixels = batch["pixels"].to(device)
    with torch.no_grad():
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
        ):
            features = tower(pixels)["compressed"]  # [B, T, d_vision]
    features = features.float()

    logits = head(features)  # [B, T, NUM_CLASSES], fp32
    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # [T, B, C]

    bsz, t_len = pixels.shape[0], features.shape[1]
    input_lengths = torch.full((bsz,), t_len, dtype=torch.long, device=device)
    targets = batch["targets"].to(device)
    target_lengths = batch["target_lengths"].to(device)

    loss = F.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=BLANK_ID,
        reduction="mean",
        zero_infinity=True,
    )
    return loss, logits, features


def _unflatten_targets(
    flat: torch.Tensor, lengths: torch.Tensor
) -> list[list[int]]:
    out: list[list[int]] = []
    offset = 0
    for length in lengths.tolist():
        out.append(flat[offset : offset + length].tolist())
        offset += length
    return out


def _smoke(args) -> int:
    """CPU self-check: tiny tower + synthetic in-memory rows, no filesystem I/O."""

    torch.manual_seed(args.seed)
    from Model.tests.test_ctc_head import (  # local import: test-only fixture
        _tiny_omvt_config,
    )

    device = torch.device("cpu")
    omvt_cfg = _tiny_omvt_config()
    tower = OMVTVisionTower(omvt_cfg).to(device).eval()
    for p in tower.parameters():
        p.requires_grad_(False)

    head = CTCHead(d_vision=omvt_cfg.d_vision, hidden=32).to(device)
    print(f"[ctc-head] trainable params: {num_trainable_params(head):,}")

    # Size warmup to the smoke run itself (not the CLI's production default of
    # 500, which would still be inside warmup after only 20 steps and make
    # the loss curve look nearly flat) unless the caller explicitly asked for
    # a specific --warmup-steps.
    steps = min(args.steps, 20)
    smoke_args = args
    if args.warmup_steps == parse_args([]).warmup_steps:
        smoke_args = argparse.Namespace(**{**vars(args), "warmup_steps": 2, "steps": steps})
    train_cfg = _build_scheduler_cfg(smoke_args)
    optimizer = build_optimizer(head, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)

    rng = torch.Generator().manual_seed(args.seed)
    bsz = 4
    pixels = torch.rand(
        bsz, omvt_cfg.in_channels, omvt_cfg.image_size, omvt_cfg.image_size,
        generator=rng,
    )
    texts = ["ab", "cd", "e", "fgh"]
    byte_targets = [list(t.encode("utf-8")) for t in texts]

    logger = RankZeroLogger(args.output, enable_tensorboard=False)
    for step in range(1, steps + 1):
        batch = ctc_collate(list(zip(pixels, byte_targets)))
        loss, logits, features = _train_step(tower, head, batch, device, use_bf16=False)
        if not bool(torch.isfinite(loss.detach())):
            raise FloatingPointError(f"non-finite CTC loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_or_check_grad_norm(head, args.grad_clip, step=step)
        optimizer.step()
        scheduler.step()
        logger.log(step, {"loss": float(loss.detach()), "lr": float(scheduler.get_last_lr()[0])})
        if args.probe_every and step % args.probe_every == 0:
            _run_probe(head, features, byte_targets, step)
    logger.close()
    print("CTC head smoke run OK")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.smoke:
        return _smoke(args)

    for flag in ("omvt_checkpoint", "data", "tokenizer_bundle"):
        if not getattr(args, flag):
            print(
                f"scripts/train_ctc_head: --{flag.replace('_', '-')} is required "
                "(or use --smoke)",
                file=sys.stderr,
            )
            return 2

    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_bf16 = args.precision == "bf16" or (
        args.precision == "auto" and device.type == "cuda"
    )

    from Tokenizer.unified.bundle import TokenizerBundle

    bundle = TokenizerBundle.from_dir(args.tokenizer_bundle)

    tower, omvt_cfg = build_tower_from_checkpoint(
        args.omvt_checkpoint, use_ema=args.use_ema_tower, device=device
    )
    print(
        f"[ctc-head] tower loaded: d_vision={omvt_cfg.d_vision} "
        f"compress_to={omvt_cfg.compress_to} image_size={omvt_cfg.image_size}"
    )

    head = CTCHead(d_vision=omvt_cfg.d_vision, hidden=args.hidden).to(device)
    print(f"[ctc-head] trainable params: {num_trainable_params(head):,}")

    dataset = CTCOcrDataset(
        args.data,
        tokenizer_bundle=bundle,
        image_size=omvt_cfg.image_size,
        max_target_len=omvt_cfg.compress_to,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed,
        infinite=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=ctc_collate,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )
    batch_iter = iter(loader)

    train_cfg = _build_scheduler_cfg(args)
    optimizer = build_optimizer(head, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint_into(args.resume, head, optimizer)
        for _ in range(start_step):
            scheduler.step()
        print(f"[ctc-head] resumed from step {start_step}")

    Path(args.output).mkdir(parents=True, exist_ok=True)
    logger = RankZeroLogger(args.output, enable_tensorboard=False)

    t0 = time.time()
    last_step = start_step
    for step in range(start_step + 1, args.steps + 1):
        last_step = step
        batch = next(batch_iter)
        loss, logits, features = _train_step(tower, head, batch, device, use_bf16)
        if not bool(torch.isfinite(loss.detach())):
            raise FloatingPointError(f"non-finite CTC loss at step {step}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = clip_or_check_grad_norm(head, args.grad_clip, step=step)
        optimizer.step()
        scheduler.step()

        if step % args.log_every == 0 or step == start_step + 1:
            logger.log(step, {
                "loss": float(loss.detach()),
                "grad_norm": float(grad_norm),
                "lr": float(scheduler.get_last_lr()[0]),
            })
        if args.probe_every and step % args.probe_every == 0:
            byte_targets = _unflatten_targets(batch["targets"], batch["target_lengths"])
            _run_probe(head, features, byte_targets, step)
        if args.save_every and step % args.save_every == 0:
            save_checkpoint(
                args.output, step, head, optimizer, omvt_cfg,
                d_vision=omvt_cfg.d_vision, hidden=args.hidden,
            )
            if dataset.n_skipped_long:
                print(f"[ctc-head] rows skipped (target > {omvt_cfg.compress_to} bytes): "
                      f"{dataset.n_skipped_long}")

    save_checkpoint(
        args.output, last_step, head, optimizer, omvt_cfg,
        d_vision=omvt_cfg.d_vision, hidden=args.hidden,
    )
    logger.close()
    dt = time.time() - t0
    print(f"CTC head training OK in {dt:.1f}s ({last_step} steps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
