#!/usr/bin/env bash
# One-command bootstrap for the CloudCompute/Vast-style GPU server.
#
# Intended usage on a fresh server:
#   cd /root
#   bash bootstrap_cloudcompute_server.sh
#
# Or without downloading the script first:
#   curl -fsSL https://raw.githubusercontent.com/NeixOn/hunyuan3d/main/bootstrap_cloudcompute_server.sh | bash

set -Eeuo pipefail

REPO_URL="${HY3D_BOOTSTRAP_REPO_URL:-https://github.com/NeixOn/hunyuan3d.git}"
INSTALL_DIR="${HY3D_BOOTSTRAP_DIR:-/root/hunyuan3d}"
API_KEY="${HY3D_SERVER_API_KEY:-change-me}"
API_PORT="${HY3D_API_PORT:-1111}"
STEPS="${HY3D_STEPS:-30}"
OCTREE_RESOLUTION="${HY3D_OCTREE_RESOLUTION:-256}"

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

log "Bootstrap Hunyuan3D server"
log "Repository: ${REPO_URL}"
log "Install dir: ${INSTALL_DIR}"
log "Internal API port: ${API_PORT}"

if command -v nvidia-smi >/dev/null 2>&1; then
  run nvidia-smi
else
  log "WARNING: nvidia-smi not found. GPU may be unavailable."
fi

if [[ ! -d "${INSTALL_DIR}/.git" ]]; then
  log "Cloning project"
  mkdir -p "$(dirname "${INSTALL_DIR}")"
  run git clone "${REPO_URL}" "${INSTALL_DIR}"
else
  log "Project already exists, pulling latest changes"
  run git -C "${INSTALL_DIR}" pull --ff-only
fi

cd "${INSTALL_DIR}"

log "Installing Hunyuan3D and GPU dependencies"
run bash server_install_hunyuan3d_deps.sh

log "Installing API dependencies"
run .venv/bin/python -m pip install --no-cache-dir -r server/requirements_server_api.txt

log "Stopping old services if present"
if [[ -f server/stop_services.sh ]]; then
  bash server/stop_services.sh || true
fi

log "Starting API and worker"
export HY3D_SERVER_API_KEY="${API_KEY}"
export HY3D_API_PORT="${API_PORT}"
export HY3D_USE_SAFETENSORS=0
export HY3D_STEPS="${STEPS}"
export HY3D_OCTREE_RESOLUTION="${OCTREE_RESOLUTION}"
run bash server/start_services.sh

log "Service status"
run bash server/status_services.sh

log "Local health check"
run curl -fsS "http://127.0.0.1:${API_PORT}/health"

cat <<EOF

DONE.

Server is running locally on:
  http://127.0.0.1:${API_PORT}

For CloudCompute external access, use the external address mapped to internal
port ${API_PORT}. Example from your panel:
  1111/tcp -> 159.48.242.11:27745

Then client API address is:
  http://159.48.242.11:27745

API key:
  ${API_KEY}

Logs:
  tail -f ${INSTALL_DIR}/server_logs/api.log
  tail -f ${INSTALL_DIR}/server_logs/worker.log

Stop:
  cd ${INSTALL_DIR}
  bash server/stop_services.sh

EOF
