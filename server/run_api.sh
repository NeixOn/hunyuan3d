#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

export HY3D_SERVER_DATA_DIR="${HY3D_SERVER_DATA_DIR:-$(pwd)/server_data}"
export HY3D_SERVER_API_KEY="${HY3D_SERVER_API_KEY:-change-me}"

uvicorn server.app:app --host "${HY3D_API_HOST:-0.0.0.0}" --port "${HY3D_API_PORT:-1111}"
