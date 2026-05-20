#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "Missing .venv. Run: bash server_install_hunyuan3d_deps.sh" >&2
  exit 1
fi

source .venv/bin/activate

export HY3D_SERVER_DATA_DIR="${HY3D_SERVER_DATA_DIR:-$(pwd)/server_data}"
export HY3D_SERVER_API_KEY="${HY3D_SERVER_API_KEY:-change-me}"
export HY3D_USE_SAFETENSORS="${HY3D_USE_SAFETENSORS:-0}"
export HY3D_STEPS="${HY3D_STEPS:-30}"
export HY3D_OCTREE_RESOLUTION="${HY3D_OCTREE_RESOLUTION:-256}"
export HY3D_API_HOST="${HY3D_API_HOST:-0.0.0.0}"
export HY3D_API_PORT="${HY3D_API_PORT:-1111}"

LOG_DIR="${HY3D_SERVER_LOG_DIR:-$(pwd)/server_logs}"
PID_DIR="${HY3D_SERVER_PID_DIR:-$(pwd)/server_pids}"
mkdir -p "${LOG_DIR}" "${PID_DIR}" "${HY3D_SERVER_DATA_DIR}"

is_running() {
  local pid_file="$1"
  [[ -f "${pid_file}" ]] || return 1
  local pid
  pid="$(cat "${pid_file}")"
  [[ -n "${pid}" ]] || return 1
  kill -0 "${pid}" >/dev/null 2>&1
}

start_api() {
  if is_running "${PID_DIR}/api.pid"; then
    echo "API already running: PID $(cat "${PID_DIR}/api.pid")"
    return
  fi

  nohup uvicorn server.app:app \
    --host "${HY3D_API_HOST}" \
    --port "${HY3D_API_PORT}" \
    > "${LOG_DIR}/api.log" 2>&1 &
  echo "$!" > "${PID_DIR}/api.pid"
  echo "Started API: PID $(cat "${PID_DIR}/api.pid"), log ${LOG_DIR}/api.log"
}

start_worker() {
  if is_running "${PID_DIR}/worker.pid"; then
    echo "Worker already running: PID $(cat "${PID_DIR}/worker.pid")"
    return
  fi

  nohup python -m server.worker > "${LOG_DIR}/worker.log" 2>&1 &
  echo "$!" > "${PID_DIR}/worker.pid"
  echo "Started worker: PID $(cat "${PID_DIR}/worker.pid"), log ${LOG_DIR}/worker.log"
}

start_api
start_worker

echo
echo "Health check:"
echo "  curl http://127.0.0.1:${HY3D_API_PORT}/health"
echo
echo "Logs:"
echo "  tail -f ${LOG_DIR}/api.log"
echo "  tail -f ${LOG_DIR}/worker.log"
