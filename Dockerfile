FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    TELESCRIBE_DATA_DIR=/data \
    TALKSCRIBE_CONFIG_PATH=/data/config.yaml \
    MOONSHINE_VOICE_CACHE=/data/models/moonshine \
    HF_HOME=/data/models/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy project
COPY pyproject.toml ./
COPY src/ ./src/
COPY .gitignore ./

# Install core dependencies
RUN uv sync --no-dev --no-editable

# Install optional engine dependencies (best-effort, may fail)
RUN uv pip install moonshine-voice 2>/dev/null || true
RUN uv pip install sherpa-onnx 2>/dev/null || true

VOLUME ["/data"]
EXPOSE 8180

# Run both bot and web dashboard
CMD ["uv", "run", "telescribe"]