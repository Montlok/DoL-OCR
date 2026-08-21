# -*- coding: utf-8 -*-

"""Minimal rank-zero logger with optional tensorboard hook."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from Model.training.dist import is_main_process


class RankZeroLogger:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        enable_tensorboard: bool = False,
        stream=sys.stdout,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.stream = stream
        self._tb = None
        if enable_tensorboard and is_main_process():
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.output_dir.mkdir(parents=True, exist_ok=True)
                self._tb = SummaryWriter(log_dir=str(self.output_dir / "tb"))
            except ImportError:  # pragma: no cover - optional dep
                self._tb = None

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        if not is_main_process():
            return
        parts = [f"step={step}"] + [
            f"{k}={_format(v)}" for k, v in metrics.items()
        ]
        self.stream.write(" ".join(parts) + "\n")
        self.stream.flush()
        if self._tb is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and v == v:  # not NaN
                    self._tb.add_scalar(k, float(v), step)

    def close(self) -> None:
        if self._tb is not None:
            self._tb.close()


def _format(v: Any) -> str:
    if isinstance(v, float):
        if abs(v) < 1e-3 or abs(v) > 1e4:
            return f"{v:.3e}"
        return f"{v:.4f}"
    return str(v)


def throughput_str(tokens: int, dt: float) -> str:
    if dt <= 0:
        return "n/a"
    tps = tokens / dt
    if tps >= 1e6:
        return f"{tps / 1e6:.2f}M tok/s"
    if tps >= 1e3:
        return f"{tps / 1e3:.2f}K tok/s"
    return f"{tps:.1f} tok/s"


def now() -> float:
    return time.time()


__all__ = ["RankZeroLogger", "now", "throughput_str"]
