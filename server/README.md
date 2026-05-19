# Hunyuan3D Server

Minimal server-side pipeline for image-to-3D generation on a GPU server.

The working Kaggle scripts are not required to change. This server uses its own
API, file queue, and long-running worker.

## Architecture

```text
Client app
  -> POST /jobs with an image
  -> GET /jobs/{job_id} until status is done
  -> GET /jobs/{job_id}/result to download result.glb

FastAPI server
  -> saves uploaded images and job status files

GPU worker
  -> loads Hunyuan3D once
  -> processes queued jobs one by one
  -> writes result.glb, metrics.json, log.txt
```

## Install API Dependencies

Run this after `server_install_hunyuan3d_deps.sh` has completed:

```bash
cd /root/hunyuan3d
source .venv/bin/activate
pip install -r server/requirements_server_api.txt
```

## Start

Recommended background start:

```bash
cd /root/hunyuan3d
source .venv/bin/activate
pip install -r server/requirements_server_api.txt

export HY3D_SERVER_API_KEY=change-me
bash server/start_services.sh
```

Check services:

```bash
bash server/status_services.sh
tail -f server_logs/api.log
tail -f server_logs/worker.log
```

Stop services:

```bash
bash server/stop_services.sh
```

Manual start, terminal 1, API:

```bash
cd /root/hunyuan3d
source .venv/bin/activate

export HY3D_SERVER_API_KEY=change-me
export HY3D_SERVER_DATA_DIR=/root/hunyuan3d/server_data

uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Terminal 2, GPU worker:

```bash
cd /root/hunyuan3d
source .venv/bin/activate

export HY3D_SERVER_API_KEY=change-me
export HY3D_SERVER_DATA_DIR=/root/hunyuan3d/server_data
export HY3D_USE_SAFETENSORS=0
export HY3D_STEPS=30
export HY3D_OCTREE_RESOLUTION=256

python -m server.worker
```

## Test From Another Machine

Use the public server IP, not `127.0.0.1`.

```bash
curl -X POST http://SERVER_PUBLIC_IP:8000/jobs \
  -H "X-API-Key: change-me" \
  -F "image=@image/airplan.png"
```

Then poll:

```bash
curl -H "X-API-Key: change-me" http://SERVER_PUBLIC_IP:8000/jobs/JOB_ID
```

Download:

```bash
curl -L -H "X-API-Key: change-me" \
  http://SERVER_PUBLIC_IP:8000/jobs/JOB_ID/result \
  -o result.glb
```

More examples, including Windows PowerShell and SSH tunnel usage, are in
`server/remote_request_examples.md`.

## Status Values

- `queued`: uploaded and waiting.
- `running`: worker is generating the model.
- `done`: `result.glb` is ready.
- `failed`: generation failed; see `error` and `log.txt`.

## Notes

- Run only one worker per GPU for now.
- The API does not run generation directly, so long jobs do not block HTTP requests.
- Set `HY3D_SERVER_API_KEY` in production. If it is empty, API key checks are disabled.
