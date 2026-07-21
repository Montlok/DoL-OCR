# -*- coding: utf-8 -*-

"""Serializable training-loss plateau detection.

The detector is deliberately independent of a particular trainer or corpus.
Call :meth:`LossPlateauStopper.observe` after a completed optimizer step and
store :meth:`LossPlateauStopper.metadata_dict` with the checkpoint.  Resuming
through :meth:`LossPlateauStopper.from_metadata` restores both the smoother
and patience counters instead of silently starting a new plateau window.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


_STATE_VERSION = 1


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Configuration for smoothed training-loss plateau detection.

    ``patience=0`` disables stopping while retaining a valid, serializable
    configuration.  ``min_steps`` is an optimizer-step boundary: observations
    before it update the baseline but do not consume patience.
    """

    patience: int = 0
    min_steps: int = 0
    min_delta: float = 0.0
    smoothing: str = "ema"
    ema_alpha: float = 0.01
    window_size: int = 100

    def __post_init__(self) -> None:
        if self.patience < 0:
            raise ValueError("early-stop patience must be non-negative")
        if self.min_steps < 0:
            raise ValueError("early-stop min_steps must be non-negative")
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("early-stop min_delta must be finite and non-negative")
        if self.smoothing not in {"ema", "window"}:
            raise ValueError(
                f"unsupported early-stop smoothing mode: {self.smoothing!r}"
            )
        if not math.isfinite(self.ema_alpha) or not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("early-stop ema_alpha must be in (0, 1]")
        if self.window_size <= 0:
            raise ValueError("early-stop window_size must be positive")

    @property
    def enabled(self) -> bool:
        return self.patience > 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LossPlateauStopper:
    """Stateful, exactly resumable loss-plateau detector."""

    config: EarlyStoppingConfig
    best_loss: float | None = None
    smoothed_loss: float | None = None
    bad_steps: int = 0
    observations: int = 0
    last_step: int = -1
    window_values: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._validate_state()

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def _validate_state(self) -> None:
        for name, value in (
            ("best_loss", self.best_loss),
            ("smoothed_loss", self.smoothed_loss),
        ):
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"early-stop {name} must be finite when present")
        if self.bad_steps < 0 or self.observations < 0:
            raise ValueError("early-stop counters must be non-negative")
        if self.bad_steps > self.observations:
            raise ValueError("early-stop bad_steps cannot exceed observations")
        if self.last_step < -1:
            raise ValueError("early-stop last_step must be at least -1")
        if self.observations == 0 and self.last_step != -1:
            raise ValueError("early-stop empty state must have last_step=-1")
        if self.observations > 0 and self.last_step < 0:
            raise ValueError("early-stop observed state must have a non-negative step")
        if len(self.window_values) > self.config.window_size:
            raise ValueError("early-stop rolling window exceeds window_size")
        if any(not math.isfinite(float(value)) for value in self.window_values):
            raise ValueError("early-stop rolling window contains a non-finite loss")
        if self.config.smoothing == "ema" and self.window_values:
            raise ValueError("EMA early-stop state cannot contain a rolling window")
        if len(self.window_values) > self.observations:
            raise ValueError("early-stop window cannot exceed observation count")

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": _STATE_VERSION,
            "best_loss": self.best_loss,
            "smoothed_loss": self.smoothed_loss,
            "bad_steps": self.bad_steps,
            "observations": self.observations,
            "last_step": self.last_step,
            # The complete window is continuation-critical in rolling mode.
            "window_values": list(self.window_values),
        }

    def metadata_dict(self) -> dict[str, Any]:
        return {"config": self.config.to_dict(), "state": self.state_dict()}

    @classmethod
    def from_state_dict(
        cls,
        config: EarlyStoppingConfig,
        state: Mapping[str, Any],
    ) -> "LossPlateauStopper":
        version = int(state.get("version", -1))
        if version != _STATE_VERSION:
            raise ValueError(
                f"unsupported early-stop state version {version}; "
                f"expected {_STATE_VERSION}"
            )
        raw_window = state.get("window_values", [])
        if not isinstance(raw_window, list):
            raise ValueError("early-stop window_values must be a list")
        best = state.get("best_loss")
        smoothed = state.get("smoothed_loss")
        return cls(
            config=config,
            best_loss=None if best is None else float(best),
            smoothed_loss=None if smoothed is None else float(smoothed),
            bad_steps=int(state.get("bad_steps", 0)),
            observations=int(state.get("observations", 0)),
            last_step=int(state.get("last_step", -1)),
            window_values=[float(value) for value in raw_window],
        )

    @classmethod
    def from_metadata(
        cls,
        config: EarlyStoppingConfig,
        metadata: Mapping[str, Any] | None,
        *,
        require_state: bool = False,
    ) -> "LossPlateauStopper":
        """Restore a detector and reject continuation-semantic drift.

        Set ``require_state=True`` for checkpoint resume.  A legacy checkpoint
        may omit early-stop metadata only when stopping remains disabled.
        """

        if not isinstance(metadata, Mapping):
            if require_state and config.enabled:
                raise ValueError(
                    "checkpoint has no early-stop metadata; initialize a new "
                    "run to enable plateau stopping"
                )
            return cls(config)

        saved_config = metadata.get("config")
        if not isinstance(saved_config, Mapping):
            raise ValueError("checkpoint early-stop config is missing or invalid")
        expected = config.to_dict()
        conflicts = [
            f"{key}: checkpoint={saved_config.get(key)!r} current={value!r}"
            for key, value in expected.items()
            if saved_config.get(key) != value
        ]
        conflicts.extend(
            f"{key}: present only in checkpoint"
            for key in saved_config
            if key not in expected
        )
        if conflicts:
            raise ValueError(
                "early-stop configuration differs from checkpoint: "
                + "; ".join(conflicts)
            )

        state = metadata.get("state")
        if not isinstance(state, Mapping):
            if require_state:
                raise ValueError("checkpoint early-stop state is missing or invalid")
            return cls(config)
        return cls.from_state_dict(config, state)

    def observe(self, loss: float, step: int) -> bool:
        """Record one completed optimizer step and return whether to stop."""

        loss = float(loss)
        step = int(step)
        if not math.isfinite(loss):
            raise FloatingPointError(
                f"non-finite loss passed to early stopping at step {step}"
            )
        if step < 0:
            raise ValueError("early-stop step must be non-negative")
        if step <= self.last_step:
            raise ValueError(
                f"early-stop steps must increase ({step} <= {self.last_step})"
            )

        self.observations += 1
        self.last_step = step
        ready = True
        if self.config.smoothing == "ema":
            self.smoothed_loss = (
                loss
                if self.smoothed_loss is None
                else self.config.ema_alpha * loss
                + (1.0 - self.config.ema_alpha) * self.smoothed_loss
            )
        else:
            self.window_values.append(loss)
            if len(self.window_values) > self.config.window_size:
                del self.window_values[0]
            self.smoothed_loss = math.fsum(self.window_values) / len(
                self.window_values
            )
            ready = len(self.window_values) == self.config.window_size

        if not ready:
            self.bad_steps = 0
            return False
        assert self.smoothed_loss is not None
        if step < self.config.min_steps:
            # Follow the burn-in level so a lucky early outlier cannot consume
            # patience after the minimum-step boundary.
            self.best_loss = self.smoothed_loss
            self.bad_steps = 0
            return False

        improved = (
            self.best_loss is None
            or self.smoothed_loss < self.best_loss - self.config.min_delta
        )
        if improved:
            self.best_loss = self.smoothed_loss
            self.bad_steps = 0
        else:
            self.bad_steps += 1
        return self.enabled and self.bad_steps >= self.config.patience


__all__ = ["EarlyStoppingConfig", "LossPlateauStopper"]
