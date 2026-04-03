# AppAgentX — RunPod Serverless Worker
# OmniParser screen parsing + image feature extraction
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# System dependencies
RUN apt-get update && apt-get install -y \
    wget libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && apt-get autoremove -y && rm -rf /var/lib/apt/lists/* && apt-get clean -y

# PaddlePaddle GPU + PaddleOCR
RUN pip install --no-cache-dir "paddlepaddle-gpu==2.6.1.post1" "paddleocr==2.7.0.3"

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download OmniParser weights from HuggingFace
RUN --mount=type=secret,id=HF_TOKEN \
    HF_TOKEN_VALUE=$(cat /run/secrets/HF_TOKEN 2>/dev/null || echo "") && \
    mkdir -p /weights/icon_detect_v1_5 /weights/icon_caption_florence && \
    echo "Downloading weights..." && \
    AUTH_HEADER="" && \
    if [ -n "$HF_TOKEN_VALUE" ]; then \
        AUTH_HEADER="Authorization: Bearer $HF_TOKEN_VALUE"; \
    fi && \
    BASE="https://huggingface.co/microsoft/OmniParser/resolve/main" && \
    \
    echo "  [1/4] YOLO detection model..." && \
    wget -q ${AUTH_HEADER:+--header="$AUTH_HEADER"} \
        -O /weights/icon_detect_v1_5/best.pt \
        "$BASE/icon_detect_v1_5/model_v1_5.pt" || exit 1 && \
    \
    echo "  [2/4] Florence2 config..." && \
    wget -q ${AUTH_HEADER:+--header="$AUTH_HEADER"} \
        -O /weights/icon_caption_florence/config.json \
        "$BASE/icon_caption_florence/config.json" || exit 1 && \
    \
    echo "  [3/4] Florence2 generation config..." && \
    wget -q ${AUTH_HEADER:+--header="$AUTH_HEADER"} \
        -O /weights/icon_caption_florence/generation_config.json \
        "$BASE/icon_caption_florence/generation_config.json" || exit 1 && \
    \
    echo "  [4/4] Florence2 model weights (~900MB)..." && \
    wget -q ${AUTH_HEADER:+--header="$AUTH_HEADER"} \
        -O /weights/icon_caption_florence/model.safetensors \
        "$BASE/icon_caption_florence/model.safetensors" || exit 1 && \
    \
    echo "Verifying..." && \
    test -f /weights/icon_detect_v1_5/best.pt || (echo "ERROR: best.pt missing" && exit 1) && \
    test -f /weights/icon_caption_florence/model.safetensors || (echo "ERROR: model.safetensors missing" && exit 1) && \
    ls -lh /weights/icon_detect_v1_5/ && \
    ls -lh /weights/icon_caption_florence/ && \
    echo "Weights ready."

# Copy handler and OmniParser utilities
COPY src/ /src/

ENV WEIGHTS_DIR=/weights
ENV PYTHONPATH=/src

RUN chmod +x /src/start.sh

CMD ["/src/start.sh"]
