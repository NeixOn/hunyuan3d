#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

export HY3D_SERVER_DATA_DIR="${HY3D_SERVER_DATA_DIR:-$(pwd)/server_data}"
export HY3D_USE_SAFETENSORS="${HY3D_USE_SAFETENSORS:-0}"
export HY3D_STEPS="${HY3D_STEPS:-30}"
export HY3D_OCTREE_RESOLUTION="${HY3D_OCTREE_RESOLUTION:-256}"

python -m server.worker
