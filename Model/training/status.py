# -*- coding: utf-8 -*-

"""Training status reporting + file-based control plane.

Rank-0 writes ``monitor/status.json`` (atomic replace) and appends to
``monitor/history.jsonl``; tools and agents read those files directly or
through ``scripts/rdt_monitor.py``. Control is a single JSON file
``monitor/control.json``: any process writes ``{"commands": ["save", ...]}``,
training pops the file each step and acts on it.

Everything here is best-effort: a slow disk or a malformed file may delay a
control command, but it never crashes the training loop.

Layout::

    <output_dir>/monitor/
        status.json    # latest snapshot
        history.jsonl  # ring of recent steps
        control.json   # pending commands (consumed by trainer)
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from pathlib import Path

VALID_COMMANDS = ("save", "eval", "stop")


class StatusReporter:
    def __init__(
        self,
        output_dir: str | Path,
        run_metadata: dict | None = None,
        max_steps: int | None = None,
        history_keep: int = 2048,
    ) -> None:
        if history_keep <= 0:
            raise ValueError("history_keep must be positive")

        self.dir = Path(output_dir) / "monitor"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.dir / "status.json"
        self.history_path = self.dir / "history.jsonl"
        self.control_path = self.dir / "control.json"
        self.max_steps = max_steps
        self.history_keep = history_keep
        self.run_metadata = run_metadata or {}
        self.started_at = time.time()
        self._history_lines = self._count_history_lines()
        self._last_step = 0
        self._last_metrics: dict = {}

        self._write_status(step=0, metrics={}, state="starting")

    def update(
        self,
        step: int,
        metrics: dict,
        state: str = "running",
        extra: dict | None = None,
    ) -> None:
        try:
            self._last_step = int(step)
            self._last_metrics = dict(metrics)
            self._write_status(step, metrics, state, extra)
            self._append_history(step, metrics)
        except OSError:
            # Status writes are best-effort; a full or slow disk must never
            # crash the training loop.
            pass

    def finish(self, state: str = "finished", step: int | None = None) -> None:
        try:
            self._write_status(
                step=self._last_step if step is None else step,
                metrics=self._last_metrics,
                state=state,
            )
        except OSError:
            # Final status write is best-effort; never mask the original
            # exit path (success or exception) of the training run.
            pass

    def poll_control(self) -> list[str]:
        """Consume pending commands; unknown command names are dropped."""

        try:
            raw = self.control_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        except OSError:
            return []

        try:
            self.control_path.unlink()
        except OSError:
            # Commands were already read into memory; failing to delete the
            # file only means they may be consumed again next poll.
            pass

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []

        commands = data.get("commands", [])
        if not isinstance(commands, list):
            return []
        seen: list[str] = []
        for cmd in commands:
            if cmd in VALID_COMMANDS and cmd not in seen:
                seen.append(cmd)
        return seen

    @staticmethod
    def request(output_dir: str | Path, *commands: str) -> Path:
        """Queue control commands for a run rooted at ``output_dir``."""

        bad = [c for c in commands if c not in VALID_COMMANDS]
        if bad:
            raise ValueError(f"unknown commands: {bad}; valid: {VALID_COMMANDS}")
        path = Path(output_dir) / "monitor" / "control.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        pending: list[str] = []
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            pending = [c for c in old.get("commands", []) if c in VALID_COMMANDS]
        except (OSError, json.JSONDecodeError):
            # An unreadable or malformed control file is treated as having no
            # pending commands; enqueueing stays best-effort and non-fatal.
            pass
        for cmd in commands:
            if cmd not in pending:
                pending.append(cmd)
        _atomic_write(path, json.dumps({"commands": pending}))
        return path

    def _write_status(
        self,
        step: int,
        metrics: dict,
        state: str,
        extra: dict | None = None,
    ) -> None:
        now = time.time()
        payload = {
            "state": state,
            "step": int(step),
            "max_steps": self.max_steps,
            "progress": (step / self.max_steps) if self.max_steps else None,
            "metrics": _jsonable(metrics),
            "started_at": self.started_at,
            "updated_at": now,
            "elapsed_s": now - self.started_at,
            "pid": os.getpid(),
            "run": self.run_metadata,
        }
        if extra:
            payload.update(_jsonable(extra))
        _atomic_write(self.status_path, json.dumps(payload))

    def _count_history_lines(self) -> int:
        """Count lines in a pre-existing history file (resumed runs) so the
        ring-buffer trim threshold stays accurate across restarts."""
        try:
            with self.history_path.open("rb") as fh:
                return sum(1 for _ in fh)
        except OSError:
            return 0

    def _append_history(self, step: int, metrics: dict) -> None:
        line = json.dumps({"step": int(step), "t": time.time(), **_jsonable(metrics)})
        with self.history_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        self._history_lines += 1

        if self._history_lines >= self.history_keep * 2:
            # Stream through the file keeping only the newest lines so the
            # trim never materializes a huge file in memory.
            with self.history_path.open("r", encoding="utf-8") as fh:
                keep = deque(fh, maxlen=self.history_keep)
            _atomic_write(self.history_path, "".join(keep))
            self._history_lines = len(keep)


def read_status(output_dir: str | Path) -> dict | None:
    path = Path(output_dir) / "monitor" / "status.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_history(output_dir: str | Path, last_n: int = 200) -> list[dict]:
    path = Path(output_dir) / "monitor" / "history.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[dict] = []
    for line in lines[-max(last_n, 0):]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float):
        # NaN/Inf are not valid JSON; strict parsers (and the HTTP API
        # consumers) would choke on them, so map to null.
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return _jsonable(item())
        except (TypeError, ValueError, RuntimeError):
            pass
    return str(obj)


__all__ = ["StatusReporter", "read_status", "read_history", "VALID_COMMANDS"]
