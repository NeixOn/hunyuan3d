#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

PID_DIR="${HY3D_SERVER_PID_DIR:-$(pwd)/server_pids}"

stop_one() {
  local name="$1"
  local pid_file="${PID_DIR}/${name}.pid"
  if [[ ! -f "${pid_file}" ]]; then
    echo "${name}: no pid file"
    return
  fi

  local pid
  pid="$(cat "${pid_file}")"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
    kill "${pid}"
    echo "${name}: stopped PID ${pid}"
  else
    echo "${name}: not running"
  fi
  rm -f "${pid_file}"
}

stop_one api
stop_one worker
