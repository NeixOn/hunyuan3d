from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .config import (
    ALLOWED_CONTENT_TYPES,
    ALLOWED_EXTENSIONS,
    API_KEY,
    CORS_ORIGINS,
    JOBS_DIR,
    MAX_UPLOAD_MB,
)
from .jobs import create_job, input_path, list_jobs, read_status, result_path, update_status


app = FastAPI(title="Hunyuan3D Generation API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "jobs_dir": str(JOBS_DIR)}


@app.post("/jobs", dependencies=[Depends(require_api_key)])
async def create_generation_job(image: UploadFile = File(...)) -> dict:
    suffix = Path(image.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file extension: {suffix}")
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported content type: {image.content_type}")

    content = await image.read()
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File is larger than {MAX_UPLOAD_MB} MB")
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    job = create_job(original_filename=image.filename or f"input{suffix}", suffix=suffix, status="uploading")
    path = input_path(job["job_id"], suffix)
    path.write_bytes(content)
    job = update_status(job["job_id"], status="queued")
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "status_url": f"/jobs/{job['job_id']}",
        "result_url": f"/jobs/{job['job_id']}/result",
    }


@app.get("/jobs", dependencies=[Depends(require_api_key)])
def get_jobs(limit: int = 50) -> dict:
    return {"jobs": list_jobs(limit=limit)}


@app.get("/jobs/{job_id}", dependencies=[Depends(require_api_key)])
def get_job(job_id: str) -> dict:
    try:
        return read_status(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found") from None


@app.get("/jobs/{job_id}/result", dependencies=[Depends(require_api_key)])
def get_result(job_id: str):
    try:
        status = read_status(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found") from None

    if status.get("status") != "done":
        raise HTTPException(status_code=409, detail=f"Job is not done: {status.get('status')}")

    path = result_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result file not found")
    return FileResponse(path, media_type="model/gltf-binary", filename=f"{job_id}.glb")


@app.get("/jobs/{job_id}/metrics", dependencies=[Depends(require_api_key)])
def get_metrics(job_id: str):
    try:
        status = read_status(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found") from None

    metrics = Path(status["metrics_path"])
    if not metrics.exists():
        raise HTTPException(status_code=404, detail="Metrics file not found")
    return FileResponse(metrics, media_type="application/json", filename=f"{job_id}_metrics.json")


@app.get("/jobs/{job_id}/log", dependencies=[Depends(require_api_key)])
def get_log(job_id: str):
    try:
        status = read_status(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found") from None

    log_file = Path(status["log_path"])
    if not log_file.exists():
        raise HTTPException(status_code=404, detail="Log file not found")
    return FileResponse(log_file, media_type="text/plain", filename=f"{job_id}.log")
