# AppAgentX Backend Worker

RunPod Serverless worker for [AppAgentX](https://github.com/Westlake-AGI-Lab/AppAgentX) — handles screen parsing and image feature extraction in a single endpoint.

## Architecture

This repo is **independent from AppAgentX**. It only contains the RunPod worker code. AppAgentX itself is unchanged.

```
AppAgentX (your local machine)
    │
    │  HTTP multipart POST → localhost:8000 / localhost:8001
    ▼
proxy.py  ← runs locally, bridges the API format gap
    │
    │  JSON API call → RunPod Serverless
    ▼
This Docker image (RunPod GPU)
    └─ OmniParser   (service: "omni")
    └─ ImageEmbedding (service: "embed")
```

**AppAgentX never calls RunPod directly.** It still talks to `localhost:8000` and `localhost:8001` as always. The proxy is the only local piece that is RunPod-aware — and it is only needed when using this worker. If you run the backend a different way, the proxy is not needed.

## Services

| Service | `"service"` field | Description |
|---|---|---|
| OmniParser | `"omni"` | Parses Android screenshots into labeled UI elements |
| ImageEmbedding | `"embed"` | Extracts feature vectors from images |

Model weights (OmniParser YOLO + Florence2) are baked into the Docker image at build time.

## CI/CD

Pushing to `main` triggers a GitHub Actions build that pushes the image to:
```
ghcr.io/<org>/runpod-app-agent-x:latest
ghcr.io/<org>/runpod-app-agent-x:<sha>
```

Required repository secret: `HF_TOKEN` (HuggingFace token to download weights during build).

## Setup

### 1. Create a RunPod Serverless endpoint

Point it to `ghcr.io/<org>/runpod-app-agent-x:latest`.

### 2. Run the local proxy

The proxy lives in the AppAgentX repo at `backend/proxy.py`. It translates AppAgentX's HTTP multipart calls into RunPod JSON API calls.

```sh
cd AppAgentX
RUNPOD_API_KEY=your_key ENDPOINT_ID=your_endpoint_id python backend/proxy.py
```

### 3. Configure AppAgentX

`config.py` already defaults to localhost — no changes needed:
```python
Omni_URI    = "http://127.0.0.1:8000"
Feature_URI = "http://127.0.0.1:8001"
```

### 4. Run AppAgentX normally

```sh
python demo.py
```

## Input format

### OmniParser — parse a screenshot

```json
{
  "input": {
    "service": "omni",
    "image": "<base64-encoded PNG>",
    "box_threshold": 0.05,
    "iou_threshold": 0.1,
    "imgsz": 640
  }
}
```

**Response:**
```json
{
  "status": "success",
  "parsed_content": [{ "ID": 0, "type": "text", "bbox": [...], "content": "..." }],
  "labeled_image": "<base64-encoded PNG>",
  "e_time": 1.23
}
```

### ImageEmbedding — single image

```json
{
  "input": {
    "service": "embed",
    "image": "<base64-encoded PNG>",
    "model_name": "resnet50"
  }
}
```

### ImageEmbedding — batch

```json
{
  "input": {
    "service": "embed",
    "images": ["<base64>", "<base64>"],
    "model_name": "resnet50"
  }
}
```

**Response:**
```json
{
  "features": [[0.12, 0.34, ...]],
  "time_taken": 0.45,
  "shape": [1, 2048],
  "model_name": "resnet50"
}
```

Available models: `resnet50`, `vit_base_patch16_224`, `efficientnet_b0`, `efficientnet_b4`, `swin_base_patch4_window7_224`, `convnext_base`, `eva02_base_patch14_448`.
