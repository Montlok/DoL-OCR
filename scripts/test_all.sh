#!/usr/bin/env bash
# Run all Python and Rust test suites.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> python -m unittest discover Tokenizer"
python3 -m unittest discover Tokenizer

echo "==> python -m unittest discover Model"
python3 -m unittest discover Model

echo "==> cargo test (Encoding Mapping)"
( cd "Encoding Mapping" && cargo test --quiet )

echo "==> cargo fmt --check"
( cd "Encoding Mapping" && cargo fmt --check )

echo "All tests passed."
