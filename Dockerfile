# Voice Clone App - GPU image (CUDA 12.4, PyTorch 2.6, Python 3.11)
#
# Build:  docker compose build
# Run:    docker compose up -d
#
# The PyTorch runtime image already contains torch/torchaudio 2.6.0 with CUDA
# 12.4, which is exactly what chatterbox-tts pins, so no second torch download.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/data/hf-cache \
    DATA_DIR=/data

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so code changes do not invalidate the (large) install layer.
COPY requirements.txt requirements-engine.txt ./
RUN pip install -r requirements.txt \
    && pip install -r requirements-engine.txt

COPY app ./app
COPY scripts ./scripts

VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=5 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"

CMD ["python", "-m", "uvicorn", "app.main:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips=*"]
