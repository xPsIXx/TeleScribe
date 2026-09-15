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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml ./
COPY src/ ./src/

# Do not use `uv run` at container start — it re-syncs without extras and
# silently uninstalls sherpa-onnx / moonshine-voice.
RUN uv sync --no-dev --no-editable

# sherpa-onnx 1.13 wheels link libonnxruntime.so but do not bundle it.
# Point $ORIGIN (sherpa_onnx/lib) at the .so from the onnxruntime wheel.
RUN python - <<'PY'
from pathlib import Path
import onnxruntime

capi = Path(onnxruntime.__file__).resolve().parent / "capi"
libs = sorted(capi.glob("libonnxruntime.so*"))
print("onnxruntime capi libs:", [str(p) for p in libs])
if not libs:
    raise SystemExit("onnxruntime wheel has no libonnxruntime.so*")
src = next((p for p in libs if p.name == "libonnxruntime.so"), libs[0])
dest_dir = Path("/app/.venv/lib/python3.12/site-packages/sherpa_onnx/lib")
dest_dir.mkdir(parents=True, exist_ok=True)
dest = dest_dir / "libonnxruntime.so"
if dest.exists() or dest.is_symlink():
    dest.unlink()
dest.symlink_to(src)
print("linked", dest, "->", src)
PY

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
