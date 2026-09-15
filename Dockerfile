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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml ./
COPY src/ ./src/

# Do not use `uv run` at container start — it re-syncs without extras and
# silently uninstalls sherpa-onnx / moonshine-voice.
RUN uv sync --no-dev --no-editable
# Fail the image if the dashboard template did not ship in the wheel.
RUN python -c "from pathlib import Path; from telescribe.web.app import _templates_dir; p = _templates_dir() / 'dashboard.html'; assert p.is_file(), p"

VOLUME ["/data"]
EXPOSE 8180

CMD ["telescribe"]
