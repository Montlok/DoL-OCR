# -*- coding: utf-8 -*-
"""Local server for the annotation UI: serves annotate.html and an inline
normalization + zero-<unk> validation API, so the whole labeling flow is
visual (no command-line steps between typing and export).

Usage (Mac, from the repo root):
  PYTHONPATH=. python3 scripts/annotate_server.py --bundle ~/Desktop/bundle_v3b
Then the browser opens http://127.0.0.1:8765 automatically.

POST /validate  {"rows": [{"id": ..., "text": ...}, ...]}
  -> {"results": {id: {"ok": bool, "norm": str, "n_unk": int}}, "summary": ...}

Batching: all texts go through ONE normalize() subprocess call (newline-
separated; newlines cannot occur inside a row -- the UI strips them).
"""
import argparse
import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from Tokenizer.tools.normalize_mongolian import normalize  # noqa: E402
from Tokenizer.unified import TokenizerBundle  # noqa: E402

HTML_PATH = REPO / "scripts" / "annotate.html"


def make_handler(bundle: TokenizerBundle):
    unk = bundle.tokenizer.unk_id

    class Handler(BaseHTTPRequestHandler):
        server_version = "dol-annotate/1.0"

        def log_message(self, fmt, *args):
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, HTML_PATH.read_bytes(), "text/html; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/validate":
                self._send(404, b"not found", "text/plain")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                rows = [
                    r for r in payload.get("rows", [])
                    if isinstance(r.get("text"), str) and r["text"].strip()
                ]
                texts = [r["text"].replace("\n", " ").strip() for r in rows]
                normed = (
                    normalize("\n".join(texts) + "\n", nominal=True).split("\n")
                    if texts
                    else []
                )
                results = {}
                n_bad = 0
                for r, norm in zip(rows, normed):
                    ids = bundle.encode(norm)
                    n_unk = sum(1 for t in ids if t == unk)
                    ok = n_unk == 0 and bool(norm.strip())
                    if not ok:
                        n_bad += 1
                    results[r["id"]] = {"ok": ok, "norm": norm, "n_unk": n_unk}
                body = json.dumps(
                    {
                        "results": results,
                        "summary": f"checked={len(rows)} problems={n_bad}",
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                self._send(200, body, "application/json; charset=utf-8")
            except Exception as exc:  # surface as a readable UI error
                body = json.dumps({"error": str(exc)}, ensure_ascii=False).encode()
                self._send(500, body, "application/json; charset=utf-8")

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True, help="tokenizer bundle dir")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    bundle = TokenizerBundle.from_dir(args.bundle)
    # Warm the cargo build once so the first UI click is not a compile wait.
    normalize("ᠮᠣᠩᠭᠣᠯ\n", nominal=True)
    print(f"[annotate] bundle loaded, normalizer warm; http://127.0.0.1:{args.port}")

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(bundle))
    webbrowser.open(f"http://127.0.0.1:{args.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[annotate] bye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
