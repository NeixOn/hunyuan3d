from __future__ import annotations

import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("HY3D_SERVER_DATA_DIR", PROJECT_DIR / "server_data")).resolve()
JOBS_DIR = DATA_DIR / "jobs"

REPO_DIR = Path(os.environ.get("HY3D_REPO_DIR", PROJECT_DIR / "Hunyuan3D-2.1")).resolve()
MODEL_ID = os.environ.get("HY3D_MODEL_ID", "tencent/Hunyuan3D-2.1")
SUBFOLDER = os.environ.get("HY3D_SUBFOLDER", "hunyuan3d-dit-v2-1")
USE_SAFETENSORS_ENV = os.environ.get("HY3D_USE_SAFETENSORS", "0").strip().lower()
USE_SAFETENSORS = USE_SAFETENSORS_ENV in {"1", "true", "yes", "on"}
STEPS = int(os.environ.get("HY3D_STEPS", "30"))
OCTREE_RESOLUTION = int(os.environ.get("HY3D_OCTREE_RESOLUTION", "256"))

API_KEY = os.environ.get("HY3D_SERVER_API_KEY", "")
MAX_UPLOAD_MB = int(os.environ.get("HY3D_SERVER_MAX_UPLOAD_MB", "25"))
WORKER_POLL_SECONDS = float(os.environ.get("HY3D_WORKER_POLL_SECONDS", "2"))
RUN_METRICS = os.environ.get("HY3D_SERVER_RUN_METRICS", "1") == "1"
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("HY3D_SERVER_CORS_ORIGINS", "*").split(",")
    if origin.strip()
]

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "application/octet-stream",
}
