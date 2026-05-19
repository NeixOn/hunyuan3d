#!/usr/bin/env bash
# Install Hunyuan3D shape-inference dependencies on a plain GPU server.
#
# Intended usage on the server:
#   cd /root/hunyuan3d
#   bash server_install_hunyuan3d_deps.sh
#
# Then run the existing generator without reinstalling deps:
#   source .venv/bin/activate
#   export HY3D_SKIP_DEP_INSTALL=1
#   export HY3D_USE_SAFETENSORS=0
#   python3 kaggle_hunyuan3d_airplane_smoke_test.py

set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${HY3D_REPO_DIR:-${PROJECT_DIR}/Hunyuan3D-2.1}"
REPO_URL="${HY3D_REPO_URL:-https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git}"
SYSTEM_PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHON_BIN="${SYSTEM_PYTHON_BIN}"
USE_VENV="${HY3D_USE_VENV:-1}"
VENV_DIR="${HY3D_VENV_DIR:-${PROJECT_DIR}/.venv}"
REQUIREMENTS_OUT="${HY3D_SERVER_REQUIREMENTS:-${PROJECT_DIR}/hy3d_requirements_server_shape.txt}"

INSTALL_TORCH="${HY3D_INSTALL_TORCH:-1}"
TORCH_INDEX_URL="${HY3D_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
TORCH_VERSION="${HY3D_TORCH_VERSION:-2.4.1}"
TORCHVISION_VERSION="${HY3D_TORCHVISION_VERSION:-0.19.1}"
TORCHAUDIO_VERSION="${HY3D_TORCHAUDIO_VERSION:-2.4.1}"
SKIP_APT="${HY3D_SKIP_APT:-0}"

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
export PIP_RETRIES="${PIP_RETRIES:-5}"
export PIP_BREAK_SYSTEM_PACKAGES=1
PIP_COMMON_ARGS=(
  --no-cache-dir
  --ignore-installed
  --break-system-packages
  --retries "${PIP_RETRIES}"
  --default-timeout "${PIP_DEFAULT_TIMEOUT}"
)

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

log "Project dir: ${PROJECT_DIR}"
log "Repo dir: ${REPO_DIR}"
log "System Python: $(${SYSTEM_PYTHON_BIN} --version 2>&1)"

if command -v nvidia-smi >/dev/null 2>&1; then
  run nvidia-smi
else
  log "WARNING: nvidia-smi not found. Continue anyway."
fi

if [[ "${SKIP_APT}" != "1" ]] && command -v apt-get >/dev/null 2>&1; then
  log "Installing system packages needed by OpenCV/Open3D/GL loaders"
  run apt-get update
  run apt-get install -y --no-install-recommends \
    git \
    python3-pip \
    python3-venv \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libxrender1 \
    libxext6 \
    libsm6
else
  log "Skipping apt package installation."
fi

if [[ "${USE_VENV}" == "1" ]]; then
  log "Creating/updating virtual environment: ${VENV_DIR}"
  run "${SYSTEM_PYTHON_BIN}" -m venv "${VENV_DIR}"
  PYTHON_BIN="${VENV_DIR}/bin/python"
  log "Venv Python: $(${PYTHON_BIN} --version 2>&1)"
  log "Upgrading pip tooling inside venv"
  run "${PYTHON_BIN}" -m pip install --upgrade --no-cache-dir pip setuptools wheel packaging
  PIP_COMMON_ARGS=(
    --no-cache-dir
    --retries "${PIP_RETRIES}"
    --default-timeout "${PIP_DEFAULT_TIMEOUT}"
  )
else
  log "Using system Python because HY3D_USE_VENV=${USE_VENV}"
fi

log "Checking Python packaging helpers"
run "${PYTHON_BIN}" - <<'PY'
import importlib.util

for name in ("setuptools", "wheel", "packaging"):
    status = "OK" if importlib.util.find_spec(name) else "MISSING"
    print(f"{name}: {status}")
PY

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  log "Cloning Hunyuan3D repository"
  run git clone "${REPO_URL}" "${REPO_DIR}"
else
  log "Using existing Hunyuan3D repository"
fi

if [[ "${INSTALL_TORCH}" == "1" ]]; then
  log "Installing PyTorch wheels from ${TORCH_INDEX_URL}"
  run "${PYTHON_BIN}" -m pip uninstall -y torch torchvision torchaudio
  run "${PYTHON_BIN}" -m pip install "${PIP_COMMON_ARGS[@]}" \
    --index-url "${TORCH_INDEX_URL}" \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}"
else
  log "Skipping PyTorch install because HY3D_INSTALL_TORCH=${INSTALL_TORCH}"
fi

log "Building server shape-only requirements file"
"${PYTHON_BIN}" - "${REPO_DIR}/requirements.txt" "${REQUIREMENTS_OUT}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])

if not source.exists():
    raise SystemExit(f"requirements.txt not found: {source}")

py312_replacements = {
    "numpy==1.24.4": "numpy==1.26.4",
    "pymeshlab==2022.2.post3": "pymeshlab==2023.12.post3",
    "open3d==0.18.0": "open3d==0.19.0",
    "onnxruntime==1.16.3": "onnxruntime==1.18.0",
}

excluded_packages = {
    # Installed explicitly or provided by the CUDA wheel index above.
    "torch",
    "torchvision",
    "torchaudio",
    # CUDA / training / texture / demo packages not needed for shape-only inference.
    "cupy-cuda12x",
    "deepspeed",
    "bpy",
    "onnxruntime",
    "realesrgan",
    "basicsr",
    "tb-nightly",
    "gradio",
    "fastapi",
    "uvicorn",
    "pythreejs",
}


def requirement_name(line: str) -> str:
    token = line.strip()
    for marker in ("==", ">=", "<=", "~=", "!=", ">", "<"):
        if marker in token:
            token = token.split(marker, 1)[0]
            break
    extras_start = token.find("[")
    if extras_start != -1:
        token = token[:extras_start]
    return token.strip().lower().replace("_", "-")


patched: list[str] = []
for line in source.read_text(encoding="utf-8").splitlines():
    stripped = line.strip()
    if stripped.startswith(("-i ", "--index-url", "--extra-index-url")):
        patched.append(f"# skipped unstable package index on server: {line}")
        continue

    if stripped and not stripped.startswith(("#", "--")):
        name = requirement_name(stripped)
        if name in excluded_packages:
            patched.append(f"# skipped for server shape-only inference: {line}")
            continue

    replacement = py312_replacements.get(stripped)
    if sys.version_info >= (3, 12) and replacement:
        patched.append(replacement)
    else:
        patched.append(line)

extra = [
    "",
    "# Keep compiled scientific wheels ABI-compatible on Python 3.12.",
    "numpy==1.26.4",
    "scipy==1.14.1",
    "scikit-learn==1.6.1",
    "# Extra utilities used by our quality/post-processing scripts.",
    "trimesh==4.4.7",
    "pymeshlab==2023.12.post3",
]

text = "\n".join(patched + extra).rstrip() + "\n"
target.write_text(text, encoding="utf-8")
print(target)
PY

log "Installing compatible NumPy/SciPy stack first"
run "${PYTHON_BIN}" -m pip install --force-reinstall --no-cache-dir \
  --retries "${PIP_RETRIES}" \
  --default-timeout "${PIP_DEFAULT_TIMEOUT}" \
  numpy==1.26.4 \
  scipy==1.14.1 \
  scikit-learn==1.6.1

log "Installing Hunyuan3D shape dependencies"
run "${PYTHON_BIN}" -m pip install "${PIP_COMMON_ARGS[@]}" \
  -r "${REQUIREMENTS_OUT}"

export PYTHONPATH="${REPO_DIR}/hy3dshape:${REPO_DIR}:${PYTHONPATH:-}"

log "Verifying key imports"
"${PYTHON_BIN}" - <<'PY'
import importlib
import sys

modules = [
    "torch",
    "torchvision",
    "transformers",
    "diffusers",
    "trimesh",
    "pymeshlab",
    "open3d",
    "hy3dshape",
]

for name in modules:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        print(f"OK {name}: {version}")
    except Exception as exc:
        print(f"FAIL {name}: {exc}")
        raise

import torch
print(f"torch cuda available: {torch.cuda.is_available()}")
print(f"torch cuda version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"gpu: {torch.cuda.get_device_name(0)}")
PY

log "DONE"
cat <<EOF

Next run:
  source .venv/bin/activate
  export HY3D_SKIP_DEP_INSTALL=1
  export HY3D_USE_SAFETENSORS=0
  export HY3D_STEPS=30
  export HY3D_OCTREE_RESOLUTION=256
  ${PYTHON_BIN} kaggle_hunyuan3d_airplane_smoke_test.py

If PyTorch CUDA import fails on this server, rerun with another wheel index, for example:
  HY3D_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \\
  HY3D_TORCH_VERSION=2.5.1 \\
  HY3D_TORCHVISION_VERSION=0.20.1 \\
  HY3D_TORCHAUDIO_VERSION=2.5.1 \\
  bash server_install_hunyuan3d_deps.sh

EOF
