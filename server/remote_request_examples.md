# Remote Requests

Use the public server IP, not `127.0.0.1`, from your local computer.

Set variables locally:

```bash
SERVER_URL=http://SERVER_PUBLIC_IP:8000
API_KEY=change-me
IMAGE_PATH=./airplan.png
```

Create a job:

```bash
curl -X POST "$SERVER_URL/jobs" \
  -H "X-API-Key: $API_KEY" \
  -F "image=@$IMAGE_PATH"
```

The response contains `job_id`.

Poll status:

```bash
curl -H "X-API-Key: $API_KEY" "$SERVER_URL/jobs/JOB_ID"
```

Download result:

```bash
curl -L -H "X-API-Key: $API_KEY" \
  "$SERVER_URL/jobs/JOB_ID/result" \
  -o result.glb
```

Download metrics:

```bash
curl -L -H "X-API-Key: $API_KEY" \
  "$SERVER_URL/jobs/JOB_ID/metrics" \
  -o metrics.json
```

## Windows PowerShell

```powershell
$ServerUrl = "http://SERVER_PUBLIC_IP:8000"
$ApiKey = "change-me"
$ImagePath = "C:\path\to\airplan.png"

curl.exe -X POST "$ServerUrl/jobs" `
  -H "X-API-Key: $ApiKey" `
  -F "image=@$ImagePath"
```

Then:

```powershell
curl.exe -H "X-API-Key: $ApiKey" "$ServerUrl/jobs/JOB_ID"

curl.exe -L -H "X-API-Key: $ApiKey" `
  "$ServerUrl/jobs/JOB_ID/result" `
  -o result.glb
```

## If Remote Access Fails

Check that API listens on all interfaces:

```bash
curl http://127.0.0.1:8000/health
ss -ltnp | grep 8000
```

Open the firewall/security group for TCP port `8000`, or use an SSH tunnel:

```bash
ssh -L 8000:127.0.0.1:8000 root@SERVER_PUBLIC_IP
```

Then send requests locally to:

```bash
http://127.0.0.1:8000
```
