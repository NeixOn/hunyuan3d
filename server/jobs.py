from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from .config import JOBS_DIR


def now() -> float:
    return time.time()


def ensure_dirs() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)


def job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def status_path(job_id: str) -> Path:
    return job_dir(job_id) / "status.json"


def input_path(job_id: str, suffix: str) -> Path:
    return job_dir(job_id) / f"input{suffix.lower()}"


def result_path(job_id: str) -> Path:
    return job_dir(job_id) / "result.glb"


def metrics_path(job_id: str) -> Path:
    return job_dir(job_id) / "metrics.json"


def log_path(job_id: str) -> Path:
    return job_dir(job_id) / "log.txt"


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(path)


def read_status(job_id: str) -> dict[str, Any]:
    path = status_path(job_id)
    if not path.exists():
        raise FileNotFoundError(f"Job does not exist: {job_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def update_status(job_id: str, **updates: Any) -> dict[str, Any]:
    data = read_status(job_id)
    data.update(updates)
    data["updated_at"] = now()
    write_json_atomic(status_path(job_id), data)
    return data


def create_job(original_filename: str, suffix: str, status: str = "queued") -> dict[str, Any]:
    ensure_dirs()
    job_id = uuid.uuid4().hex
    directory = job_dir(job_id)
    directory.mkdir(parents=True, exist_ok=False)
    data = {
        "job_id": job_id,
        "status": status,
        "original_filename": original_filename,
        "input_path": str(input_path(job_id, suffix)),
        "result_path": str(result_path(job_id)),
        "metrics_path": str(metrics_path(job_id)),
        "log_path": str(log_path(job_id)),
        "created_at": now(),
        "updated_at": now(),
        "started_at": None,
        "finished_at": None,
        "error": None,
    }
    write_json_atomic(status_path(job_id), data)
    return data


def list_jobs(limit: int = 50) -> list[dict[str, Any]]:
    ensure_dirs()
    statuses = []
    for path in sorted(JOBS_DIR.glob("*/status.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            statuses.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
        if len(statuses) >= limit:
            break
    return statuses


def next_queued_job() -> dict[str, Any] | None:
    ensure_dirs()
    for path in sorted(JOBS_DIR.glob("*/status.json"), key=lambda item: item.stat().st_mtime):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("status") == "queued":
            return data
    return None
