from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .config import PROJECT_DIR, RUN_METRICS, WORKER_POLL_SECONDS
from .hy3d_runtime import runtime
from .jobs import log_path, metrics_path, next_queued_job, result_path, update_status


def append_log(job_id: str, text: str) -> None:
    path = log_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(text)
        if text and not text.endswith("\n"):
            file.write("\n")


def run_metrics(job_id: str, glb_path: Path) -> None:
    if not RUN_METRICS:
        return
    metrics_script = PROJECT_DIR / "mesh_quality_metrics.py"
    if not metrics_script.exists():
        append_log(job_id, "metrics skipped: mesh_quality_metrics.py not found")
        return
    cmd = [
        sys.executable,
        str(metrics_script),
        str(glb_path),
        "--json-out",
        str(metrics_path(job_id)),
    ]
    append_log(job_id, "+ " + " ".join(cmd))
    completed = subprocess.run(cmd, text=True, capture_output=True)
    append_log(job_id, completed.stdout)
    append_log(job_id, completed.stderr)
    if completed.returncode != 0:
        append_log(job_id, f"metrics failed with code {completed.returncode}")


def process_job(job: dict) -> None:
    job_id = job["job_id"]
    input_image = Path(job["input_path"])
    output_glb = result_path(job_id)

    update_status(job_id, status="running", started_at=time.time(), error=None)
    append_log(job_id, f"started job {job_id}")

    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            runtime.generate(input_image, output_glb)
        append_log(job_id, buffer.getvalue())
        run_metrics(job_id, output_glb)
        update_status(job_id, status="done", finished_at=time.time(), result_path=str(output_glb))
        append_log(job_id, f"finished job {job_id}")
    except Exception as exc:
        append_log(job_id, buffer.getvalue())
        append_log(job_id, traceback.format_exc())
        update_status(job_id, status="failed", finished_at=time.time(), error=str(exc))
        append_log(job_id, f"failed job {job_id}: {exc}")


def main() -> None:
    print("Hunyuan3D worker started.", flush=True)
    runtime.load()
    while True:
        job = next_queued_job()
        if job is None:
            time.sleep(WORKER_POLL_SECONDS)
            continue
        process_job(job)


if __name__ == "__main__":
    main()
