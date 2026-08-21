# -*- coding: utf-8 -*-

"""Training monitor CLI for RDT runs.

Three modes against the ``monitor/`` directory that ``train_rdt.py`` keeps:

    # one-shot status (machine readable: --json)
    python -m scripts.rdt_monitor status --run runs/pretrain

    # HTTP API for agents (pure stdlib server)
    python -m scripts.rdt_monitor serve --run runs/pretrain --port 8787

    # interactive dashboard (requires `rich`)
    python -m scripts.rdt_monitor tui --run runs/pretrain

    # queue a control command without UI (also works while serve/tui run)
    python -m scripts.rdt_monitor control save --run runs/pretrain

HTTP API:

    GET  /api/health   -> {"ok": true}
    GET  /api/status   -> latest snapshot of training (status.json)
    GET  /api/history?n=200 -> recent step metrics
    POST /api/control  {"command": "save" | "eval" | "stop"}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Model.training.status import (  # noqa: E402
    VALID_COMMANDS,
    StatusReporter,
    read_history,
    read_status,
)


# ----------------------------------------------------------------- HTTP API


def _make_handler(run_dir: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "rdt-monitor/1.0"

        def log_message(self, fmt, *args):  # noqa: D102 - silence stderr spam
            pass

        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            url = urlparse(self.path)
            if url.path == "/api/health":
                self._send(200, {"ok": True, "run": run_dir})
            elif url.path == "/api/status":
                status = read_status(run_dir)
                if status is None:
                    self._send(404, {"error": "no status.json yet"})
                else:
                    self._send(200, status)
            elif url.path == "/api/history":
                n = 200
                qs = parse_qs(url.query)
                if "n" in qs:
                    try:
                        n = max(1, min(int(qs["n"][0]), 10000))
                    except ValueError:
                        self._send(400, {"error": "n must be an integer"})
                        return
                self._send(200, {"history": read_history(run_dir, last_n=n)})
            else:
                self._send(404, {"error": f"unknown path {url.path}"})

        def do_POST(self):  # noqa: N802
            if urlparse(self.path).path != "/api/control":
                self._send(404, {"error": "unknown path"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                data = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send(400, {"error": "invalid JSON"})
                return
            command = data.get("command")
            if command not in VALID_COMMANDS:
                self._send(
                    400,
                    {"error": f"command must be one of {list(VALID_COMMANDS)}"},
                )
                return
            StatusReporter.request(run_dir, command)
            self._send(200, {"ok": True, "queued": command})

    return Handler


def serve(run_dir: str, host: str, port: int) -> int:
    httpd = ThreadingHTTPServer((host, port), _make_handler(run_dir))
    print(f"[rdt-monitor] api on http://{host}:{httpd.server_address[1]} run={run_dir}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        # Ctrl+C is the normal way to stop the server; shut down quietly.
        pass
    finally:
        httpd.server_close()
    return 0


# ----------------------------------------------------------------- one-shot


def _fmt_eta(status: dict) -> str:
    step = status.get("step") or 0
    max_steps = status.get("max_steps")
    elapsed = status.get("elapsed_s") or 0
    if not max_steps or step <= 0:
        return "?"
    rate = elapsed / step
    return _fmt_dur(rate * (max_steps - step))


def _fmt_dur(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h >= 24:
        d, h = divmod(h, 24)
        return f"{d}d{h:02d}h"
    return f"{h:02d}:{m:02d}:{s:02d}"


def one_shot(run_dir: str, as_json: bool) -> int:
    status = read_status(run_dir)
    if status is None:
        print(f"no status under {run_dir}/monitor", file=sys.stderr)
        return 1
    if as_json:
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return 0
    metrics = status.get("metrics") or {}
    stale = time.time() - (status.get("updated_at") or 0)
    print(f"state     {status.get('state')} (updated {stale:.0f}s ago)")
    print(f"step      {status.get('step')} / {status.get('max_steps') or '?'}  eta {_fmt_eta(status)}")
    for key in ("loss", "lr", "grad_norm", "tokens", "rec_steps", "throughput"):
        if key in metrics:
            print(f"{key:<9} {metrics[key]}")
    return 0


def control(run_dir: str, command: str) -> int:
    StatusReporter.request(run_dir, command)
    print(f"queued '{command}' for {run_dir}")
    return 0


# ----------------------------------------------------------------- rich TUI


def tui(run_dir: str, refresh: float) -> int:
    try:
        from rich.console import Console, Group
        from rich.live import Live
        from rich.panel import Panel
        from rich.progress import BarColumn, Progress, TextColumn
        from rich.table import Table
        from rich.text import Text
    except ImportError:
        print("tui mode needs `pip install rich`", file=sys.stderr)
        return 1

    console = Console()
    keys = _KeyReader()
    queued: list[tuple[float, str]] = []

    def render():
        status = read_status(run_dir) or {}
        history = read_history(run_dir, last_n=120)
        metrics = status.get("metrics") or {}
        state = status.get("state", "unknown")
        step = status.get("step") or 0
        max_steps = status.get("max_steps")
        stale = time.time() - (status.get("updated_at") or time.time())

        prog = Progress(
            TextColumn("[bold]step"),
            BarColumn(bar_width=40),
            TextColumn("{task.completed}/{task.total}  eta " + _fmt_eta(status)),
        )
        prog.add_task("", total=max_steps or max(step, 1), completed=step)

        table = Table.grid(padding=(0, 2))
        table.add_column(style="cyan", justify="right")
        table.add_column()
        state_style = {"running": "green", "finished": "blue"}.get(state, "yellow")
        table.add_row("state", f"[{state_style}]{state}[/] (updated {stale:.0f}s ago)")
        for key in ("loss", "lr", "grad_norm", "tokens", "rec_steps", "throughput"):
            if key in metrics:
                table.add_row(key, str(metrics[key]))
        table.add_row("run", str(run_dir))

        losses = [h["loss"] for h in history if isinstance(h.get("loss"), (int, float))]
        spark = Text(_sparkline(losses), style="magenta")

        lines = [prog, table, Text("loss "), spark]
        for ts, cmd in queued[-3:]:
            lines.append(Text(f"queued '{cmd}' at {time.strftime('%H:%M:%S', time.localtime(ts))}", style="dim"))
        lines.append(Text("[s]ave  [e]val  [x] stop  [q]uit", style="bold dim"))
        return Panel(Group(*lines), title="RDT pretraining", border_style="bright_black")

    try:
        with keys, Live(render(), console=console, refresh_per_second=4) as live:
            while True:
                key = keys.poll(refresh)
                if key == "q":
                    break
                if key in ("s", "e", "x"):
                    cmd = {"s": "save", "e": "eval", "x": "stop"}[key]
                    StatusReporter.request(run_dir, cmd)
                    queued.append((time.time(), cmd))
                live.update(render())
    except KeyboardInterrupt:
        # Ctrl+C exits the TUI like 'q'; treat it as a clean shutdown.
        pass
    return 0


def _sparkline(values: list[float], width: int = 60) -> str:
    if not values:
        return "(no data)"
    values = values[-width:]
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return "▄" * len(values) + f"  {hi:.4f}"
    blocks = "▁▂▃▄▅▆▇█"
    out = "".join(blocks[int((v - lo) / (hi - lo) * (len(blocks) - 1))] for v in values)
    return out + f"  [{lo:.4f}, {hi:.4f}]"


class _KeyReader:
    """Raw single-key reader with timeout; degrades to sleep without a TTY."""

    def __enter__(self):
        self._fd = None
        if sys.stdin.isatty():
            import termios
            import tty

            self._fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc):
        if self._fd is not None:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
        return False

    def poll(self, timeout: float) -> str | None:
        if self._fd is None:
            time.sleep(timeout)
            return None
        import select

        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        return sys.stdin.read(1) if ready else None


# ----------------------------------------------------------------- entry


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    def add_run(p):
        p.add_argument("--run", required=True, help="training output_dir")

    p_status = sub.add_parser("status", help="print one snapshot and exit")
    add_run(p_status)
    p_status.add_argument("--json", action="store_true")

    p_serve = sub.add_parser("serve", help="stdlib HTTP API for agents")
    add_run(p_serve)
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8787)

    p_tui = sub.add_parser("tui", help="interactive rich dashboard")
    add_run(p_tui)
    p_tui.add_argument("--refresh", type=float, default=1.0)

    p_ctl = sub.add_parser("control", help="queue save/eval/stop")
    p_ctl.add_argument("command", choices=list(VALID_COMMANDS))
    add_run(p_ctl)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "status":
        return one_shot(args.run, args.json)
    if args.mode == "serve":
        return serve(args.run, args.host, args.port)
    if args.mode == "tui":
        return tui(args.run, args.refresh)
    if args.mode == "control":
        return control(args.run, args.command)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
