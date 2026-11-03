# Voice Clone App - GPU images (CUDA 12.4, PyTorch 2.6, Python 3.11)
#
# Two targets share one base layer (PyTorch + Chatterbox):
#   app      the web app and generation worker   (docker compose service "app")
#   trainer  the fine-tuning service             (docker compose service "trainer")
#
# Build:  docker compose build
# Run:    docker compose up -d
#
# The PyTorch runtime image already contains torch/torchaudio 2.6.0 with CUDA
# 12.4, which is exactly what chatterbox-tts pins, so no second torch download.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/data/hf-cache \
    DATA_DIR=/data

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git libsndfile1 sox \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Engine dependencies follow TTS_ENGINE (docker compose passes it from .env):
# requirements-engine-qwen.txt for qwen (the default), requirements-engine.txt
# otherwise.  They cannot share an image: the two engines pin incompatible
# transformers versions.  The trainer target fine-tunes Chatterbox: build it with
# TTS_ENGINE=chatterbox (docker compose does).
ARG TTS_ENGINE=qwen

# Dependencies first so code changes do not invalidate the (large) install layer.
COPY requirements.txt requirements-engine*.txt ./
RUN if [ "${TTS_ENGINE}" = "qwen" ]; then req=requirements-engine-qwen.txt; else req=requirements-engine.txt; fi \
    && pip install -r requirements.txt \
    && pip install -r "${req}" \
    && echo "${req}" > /app/engine-requirements.txt

VOLUME ["/data"]

# ---------------------------------------------------------------------------
# Build stamp: the git commit the image was built from, shown in the UI.  Only
# this small stage sees .git; without it (a source archive) the commit is unknown.
FROM base AS buildinfo
COPY . /src
RUN git config --global --add safe.directory /src \
    && (git -C /src log -1 --format='{"commit": "%H", "date": "%cI"}' 2>/dev/null || echo '{}') > /build-info.json

# ---------------------------------------------------------------------------
# Fine-tuning service: same base plus the training/evaluation extras.  It reads
# queued runs from the app's database on /data and uses the GPU only while the
# app has released it.
FROM base AS trainer

COPY requirements-finetune.txt ./
RUN pip install -r requirements-finetune.txt

COPY app ./app
COPY scripts ./scripts
COPY trainer ./trainer

CMD ["python", "-m", "trainer"]

# ---------------------------------------------------------------------------
# Web app (last stage: the default target of a plain "docker build").
FROM base AS app

COPY app ./app
COPY scripts ./scripts
COPY --from=buildinfo /build-info.json ./app/build-info.json

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=5 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"

CMD ["python", "-m", "uvicorn", "app.main:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips=*"]
