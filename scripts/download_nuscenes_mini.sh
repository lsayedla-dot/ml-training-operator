#!/bin/bash
# Download nuScenes mini split (~4GB) from AWS open data
set -euo pipefail

DATA_DIR="${1:-data/nuscenes-mini}"
NUSCENES_URL="https://www.nuscenes.org/data/v1.0-mini.tgz"

echo "==> Downloading nuScenes mini split..."
echo "    Target directory: ${DATA_DIR}"

mkdir -p "${DATA_DIR}"

if [ -f "${DATA_DIR}/v1.0-mini/scene.json" ]; then
    echo "==> nuScenes mini already downloaded. Skipping."
    exit 0
fi

echo "==> Downloading from ${NUSCENES_URL}..."
curl -L -o /tmp/nuscenes-mini.tgz "${NUSCENES_URL}"

echo "==> Extracting..."
tar -xzf /tmp/nuscenes-mini.tgz -C "${DATA_DIR}" --strip-components=1

echo "==> Cleaning up..."
rm -f /tmp/nuscenes-mini.tgz

echo "==> Done! nuScenes mini data is in ${DATA_DIR}"
echo "    Scenes: $(ls ${DATA_DIR}/v1.0-mini/ | wc -l) metadata files"
