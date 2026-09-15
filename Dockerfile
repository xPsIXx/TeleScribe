FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    TELESCRIBE_DATA_DIR=/data \
    TALKSCRIBE_CONFIG_PATH=/data/config.yaml \
    MOONSHINE_VOICE_CACHE=/data/models/moonshine \
    HF_HOME=/data/models/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml ./
COPY src/ ./src/

# Core package (src layout) plus Moonshine + Parakeet extras
RUN uv sync --no-dev --no-editable --extra moonshine --extra parakeet

VOLUME ["/data"]
EXPOSE 8180

CMD ["uv", "run", "telescribe"]
