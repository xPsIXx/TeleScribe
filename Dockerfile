FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_SYNC=1 \
    TELESCRIBE_DATA_DIR=/data \
    TALKSCRIBE_CONFIG_PATH=/data/config.yaml \
    MOONSHINE_VOICE_CACHE=/data/models/moonshine \
    HF_HOME=/data/models/huggingface \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgomp1 \
    libsndfile1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml ./
COPY src/ ./src/

# Do not use `uv run` at container start — it re-syncs without extras and
# silently uninstalls sherpa-onnx / moonshine-voice.
RUN uv sync --no-dev --no-editable

# sherpa-onnx 1.13 wheels need libonnxruntime.so with ELF version VERS_1.28.2.
# The pip onnxruntime 1.30 wheel does not export that. Install Microsoft ORT 1.28.2.
RUN curl -fsSL -o /tmp/ort.tgz \
      https://github.com/microsoft/onnxruntime/releases/download/v1.28.2/onnxruntime-linux-x64-1.28.2.tgz \
 && mkdir -p /tmp/ort \
 && tar --no-same-owner -xzf /tmp/ort.tgz -C /tmp/ort \
 && dest=/app/.venv/lib/python3.12/site-packages/sherpa_onnx/lib \
 && mkdir -p "$dest" \
 && cp -a /tmp/ort/*/lib/libonnxruntime.so* "$dest/" \
 && ls -l "$dest"/libonnxruntime.so* \
 && rm -rf /tmp/ort /tmp/ort.tgz

# Fail the image if the dashboard or ASR engines did not actually install.
RUN python -c "\
from pathlib import Path; \
import faster_whisper, moonshine_voice, sherpa_onnx; \
from telescribe.web.app import _templates_dir; \
p = _templates_dir() / 'dashboard.html'; \
assert p.is_file(), p; \
print('ok dashboard', p); \
print('ok faster_whisper', faster_whisper.__file__); \
print('ok moonshine', moonshine_voice.__file__); \
print('ok sherpa_onnx', sherpa_onnx.__file__)"

VOLUME ["/data"]
EXPOSE 8180

CMD ["telescribe"]
