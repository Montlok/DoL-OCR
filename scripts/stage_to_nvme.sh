#!/usr/bin/env bash
# Stage the tokenizer bundle onto local NVMe so training/build jobs never
# read it over the network mount. Idempotent: reruns are a fast rsync no-op
# once the bundle is already staged.
#
# Usage:
#   NVME=/mnt/nvme scripts/stage_to_nvme.sh /path/to/tokenizer/bundle
#
# Env:
#   NVME   NVMe mount root (required; the bundle lands at $NVME/dolocr/bundle)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ "$#" -ne 1 ]; then
    echo "usage: NVME=/mnt/nvme $0 <tokenizer-bundle-dir>" >&2
    exit 2
fi
SRC_BUNDLE="$1"

if [ -z "${NVME:-}" ]; then
    echo "stage_to_nvme: NVME env var must be set (e.g. NVME=/mnt/nvme)" >&2
    exit 2
fi
if [ ! -d "$SRC_BUNDLE" ]; then
    echo "stage_to_nvme: source bundle dir does not exist: $SRC_BUNDLE" >&2
    exit 2
fi
if [ ! -d "$NVME" ]; then
    echo "stage_to_nvme: NVME mount does not exist: $NVME" >&2
    exit 2
fi

DEST_BUNDLE="$NVME/dolocr/bundle"
mkdir -p "$DEST_BUNDLE"

echo "==> capacity check before staging"
df -h "$NVME"
df -i "$NVME"

echo "==> staging bundle: $SRC_BUNDLE -> $DEST_BUNDLE"
if command -v rsync >/dev/null 2>&1; then
    rsync -a --info=progress2 "$SRC_BUNDLE"/ "$DEST_BUNDLE"/
else
    echo "stage_to_nvme: rsync not found, falling back to cp -a" >&2
    cp -a "$SRC_BUNDLE"/. "$DEST_BUNDLE"/
fi

echo "==> verifying required bundle files"
for name in config.json vocab.json manifest.json; do
    if [ ! -s "$DEST_BUNDLE/$name" ]; then
        echo "stage_to_nvme: missing or empty required file after staging: $name" >&2
        exit 1
    fi
done

echo "==> capacity check after staging"
df -h "$NVME"
df -i "$NVME"

echo "stage_to_nvme: OK -> $DEST_BUNDLE"
