"""Web dashboard for TeleScribe — single-page config editor with model management."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from telescribe import __version__
from telescribe.config import AppConfig
from telescribe.logger import get_logger, get_log_file_path

logger = get_logger("web")

app = FastAPI(title="TeleScribe Dashboard")

_web_dir = Path(__file__).parent
templates = Jinja2Templates(directory=str(_web_dir / "templates"))

_config: AppConfig | None = None
_bot_reload_requested = False

# Model download tracking: {model_name: {"status": "pending"|"downloading"|"done"|"error", "progress": 0-100}}
_downloads: dict[str, dict] = {}

# Per-engine model definitions
ENGINE_MODELS = {
    "local": [
        ("tiny", "Tiny (39M params)"),
        ("base", "Base (74M params)"),
        ("small", "Small (244M params)"),
        ("medium", "Medium (769M params)"),
        ("large-v3", "Large v3 (1.55B params)"),
        ("distil-small.en", "Distil Small EN (6x faster)"),
        ("distil-medium.en", "Distil Medium EN (6x faster) — Recommended"),
        ("distil-large-v3", "Distil Large v3 (6x faster)"),
    ],
    "moonshine": [
        ("en", "English"),
        ("es", "Spanish"),
        ("zh", "Chinese"),
        ("ja", "Japanese"),
        ("ko", "Korean"),
        ("vi", "Vietnamese"),
        ("ar", "Arabic"),
        ("uk", "Ukrainian"),
    ],
    "parakeet": [
        ("sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000-int8", "Parakeet TDT 110M EN — best accuracy"),
    ],
}

# Device support per engine
ENGINE_DEVICE = {
    "local": ["cpu", "cuda"],
    "moonshine": ["cpu"],
    "parakeet": ["cpu"],
}


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = AppConfig.load()
    return _config


def _check_model_downloaded(engine: str, model: str) -> bool:
    """Check if a model file exists on disk."""
    data_dir = get_config().data_dir
    home = Path.home()

    if engine == "local":
        # Check /data/models/ first (persistent volume), then ~/.cache
        # HF_HOME is set to /data/models/huggingface, so HF hub cache is /data/models/huggingface/hub/
        # Check direct model name match (e.g. /data/models/distil-medium.en/)
        model_dir = Path(data_dir) / "models" / model
        if model_dir.exists() and any(model_dir.rglob("*.bin")):
            return True
        # Check HF_HOME-based cache (models downloaded via huggingface hub)
        # HF hub stores models as models--ORG--MODELNAME, e.g. models--Systran--faster-distil-whisper-medium.en
        hf_dir = Path(data_dir) / "models" / "huggingface" / "hub"
        if hf_dir.exists():
            # Try all possible naming patterns
            for p in hf_dir.rglob(f"*{model}*"):
                if p.is_dir() and any(p.rglob("*.bin")):
                    return True
            # Also check direct org-model patterns
            for p in hf_dir.iterdir():
                if p.is_dir() and p.name.startswith("models--"):
                    if any(p.rglob("*.bin")):
                        # Check if this is the right model by name
                        if model.replace("-", "_") in p.name or model in p.name:
                            return True
        model_dir = home / ".cache" / "faster-whisper" / model
        if model_dir.exists():
            return True
        hf_cache = home / ".cache" / "huggingface" / "hub"
        if hf_cache.exists():
            for p in hf_cache.rglob(f"*{model}*"):
                if p.is_dir():
                    return True
        return False

    if engine == "moonshine":
        # Check /data/models/ first, then home cache
        # Moonshine stores models per-language, e.g. medium-streaming-en, medium-streaming-es
        # Check for language-specific model directory
        cache = Path(data_dir) / "models" / "moonshine"
        if cache.exists():
            for p in cache.rglob(f"*{model}*/encoder*.ort"):
                if p.stat().st_size > 500_000:
                    return True
        cache = Path(os.environ.get("MOONSHINE_VOICE_CACHE", ""))
        if not cache.exists():
            cache = home / ".cache" / "moonshine_voice"
        if not cache.exists():
            cache = home / ".cache" / "moonshine"
        if cache.exists():
            # Check for language-specific model directory, e.g. .../medium-streaming-en/.../encoder.ort
            for p in cache.rglob(f"*{model}*/encoder*.ort"):
                if p.stat().st_size > 500_000:
                    return True
            # Also check if model name is in the directory path
            for p in cache.rglob(f"*{model}*"):
                if p.is_dir() and any(p.rglob("encoder*.ort")):
                    for f in p.rglob("encoder*.ort"):
                        if f.stat().st_size > 500_000:
                            return True
        return False

    if engine == "parakeet":
        model_dir = Path(data_dir) / "models" / model
        return model_dir.exists() and (model_dir / "tokens.txt").exists()

    return True


def _get_model_paths(engine: str, model: str) -> list[Path]:
    """Return list of file/dir paths that would be deleted for a given model."""
    data_dir = get_config().data_dir
    home = Path.home()
    paths = []

    if engine == "local":
        # Check /data/models/ first, then ~/.cache
        model_dir = Path(data_dir) / "models" / model
        if model_dir.exists():
            paths.append(model_dir)
        model_dir = home / ".cache" / "faster-whisper" / model
        if model_dir.exists():
            paths.append(model_dir)
        hf_cache = home / ".cache" / "huggingface" / "hub"
        if hf_cache.exists():
            for p in hf_cache.rglob(f"*{model}*"):
                if p.is_dir():
                    paths.append(p)
    elif engine == "moonshine":
        # Check /data/models/moonshine/ first, then home cache
        cache = Path(data_dir) / "models" / "moonshine"
        if cache.exists():
            paths.append(cache)
        cache = Path(os.environ.get("MOONSHINE_VOICE_CACHE", ""))
        if not cache.exists():
            cache = home / ".cache" / "moonshine_voice"
        if not cache.exists():
            cache = home / ".cache" / "moonshine"
        if cache.exists():
            paths.append(cache)
    elif engine == "parakeet":
        model_dir = Path(data_dir) / "models" / model
        if model_dir.exists():
            paths.append(model_dir)

    # Deduplicate
    seen = set()
    unique = []
    for p in paths:
        s = str(p.resolve())
        if s not in seen:
            seen.add(s)
            unique.append(p)
    return unique


def _check_model_valid(engine: str, model: str) -> bool:
    """Verify model files are valid (non-zero size, not corrupt)."""
    data_dir = get_config().data_dir
    home = Path.home()

    if engine == "local":
        # Check /data/models/ first, then ~/.cache
        model_dir = Path(data_dir) / "models" / model
        if model_dir.exists():
            bins = list(model_dir.rglob("*.bin"))
            if bins and max(f.stat().st_size for f in bins) > 1_000_000:
                return True
        model_dir = home / ".cache" / "faster-whisper" / model
        if model_dir.exists():
            bins = list(model_dir.rglob("*.bin"))
            if bins and max(f.stat().st_size for f in bins) > 1_000_000:
                return True
        hf_cache = home / ".cache" / "huggingface" / "hub"
        if hf_cache.exists():
            for p in hf_cache.rglob(f"*{model}*"):
                if p.is_dir():
                    bins = list(p.rglob("*.bin"))
                    if bins and max(f.stat().st_size for f in bins) > 1_000_000:
                        return True
        return False

    if engine == "moonshine":
        # Check /data/models/moonshine/ first, then home cache
        cache = Path(data_dir) / "models" / "moonshine"
        if cache.exists():
            for p in cache.rglob("*.ort"):
                if p.stat().st_size > 500_000:
                    return True
        cache = Path(os.environ.get("MOONSHINE_VOICE_CACHE", ""))
        if not cache.exists():
            cache = home / ".cache" / "moonshine_voice"
        if not cache.exists():
            cache = home / ".cache" / "moonshine"
        if cache.exists():
            for p in cache.rglob("*.ort"):
                if p.stat().st_size > 500_000:
                    return True
        return False

    if engine == "parakeet":
        model_dir = Path(data_dir) / "models" / model
        if model_dir.exists() and (model_dir / "tokens.txt").exists():
            tok_size = (model_dir / "tokens.txt").stat().st_size
            # Check for any large .onnx model file (not just encoder.onnx)
            model_files = list(model_dir.glob("model*.onnx")) + list(model_dir.glob("*.onnx"))
            if tok_size > 100 and model_files:
                largest = max(f.stat().st_size for f in model_files if f.is_file())
                if largest > 1_000_000:
                    return True
        return False

    return True


def _get_model_size_str(engine: str, model: str) -> str:
    """Get human-readable total size of downloaded model files."""
    paths = _get_model_paths(engine, model)
    if not paths:
        return ""
    total = 0
    for p in paths:
        if p.is_file():
            total += p.stat().st_size
        elif p.is_dir():
            for f in p.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
    if total < 1024:
        return f"{total}B"
    elif total < 1024 * 1024:
        return f"{total/1024:.0f}KB"
    else:
        return f"{total/1024/1024:.1f}MB"


# ---- API Endpoints ----

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    cfg = get_config()
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={"request": request, "config": cfg.model_dump(), "version": __version__},
    )


class ConfigUpdate(BaseModel):
    engine: str
    model: str
    device: str
    compute_type: str
    vad_filter: bool
    language: str
    llm_url: str
    llm_model: str
    llm_temp: float
    llm_max_tokens: int
    privacy_mode: bool
    show_transcribing_feedback: bool
    show_transcription_header: bool
    authorized_users: list[int]
    auth_required_summarize: bool
    auth_required_reply: bool
    summary_prompt: str
    reply_prompt: str
    retention_days: int
    reply_retention_days: int
    summarize_retention_days: int
    max_chats: int


@app.post("/api/config")
async def update_config(update: ConfigUpdate):
    cfg = get_config()
    old_engine = cfg.transcription.engine
    cfg.transcription.engine = update.engine  # type: ignore
    cfg.transcription.model = update.model
    cfg.transcription.device = update.device  # type: ignore
    cfg.transcription.compute_type = update.compute_type
    cfg.transcription.vad_filter = update.vad_filter
    cfg.transcription.language = update.language or None
    cfg.llm.base_url = update.llm_url
    cfg.llm.model = update.llm_model
    cfg.llm.temperature = update.llm_temp
    cfg.llm.max_tokens = update.llm_max_tokens
    cfg.bot.privacy_mode = update.privacy_mode
    cfg.bot.show_transcribing_feedback = update.show_transcribing_feedback
    cfg.bot.show_transcription_header = update.show_transcription_header
    cfg.bot.authorized_users = update.authorized_users
    cfg.bot.auth_required_summarize = update.auth_required_summarize
    cfg.bot.auth_required_reply = update.auth_required_reply
    cfg.bot.prompts.summary = update.summary_prompt
    cfg.bot.prompts.reply = update.reply_prompt
    cfg.history.retention_days = update.retention_days
    cfg.history.reply_retention_days = update.reply_retention_days
    cfg.history.summarize_retention_days = update.summarize_retention_days
    cfg.history.max_chats = update.max_chats
    cfg.save()
    logger.info("Config updated: engine=%s->%s, model=%s", old_engine, update.engine, update.model)

    # Auto-trigger bot reload
    global _bot_reload_requested
    _bot_reload_requested = True
    import telescribe.bot
    telescribe.bot._reload_requested = True
    logger.info("Auto-reload triggered after config save")

    return {"status": "ok", "reload_triggered": True}


@app.get("/api/config")
async def get_config_api():
    cfg = get_config()
    return {
        "engine": cfg.transcription.engine,
        "model": cfg.transcription.model,
        "device": cfg.transcription.device,
        "compute_type": cfg.transcription.compute_type,
        "vad_filter": cfg.transcription.vad_filter,
        "language": cfg.transcription.language or "",
        "llm_url": cfg.llm.base_url,
        "llm_model": cfg.llm.model,
        "llm_temp": cfg.llm.temperature,
        "llm_max_tokens": cfg.llm.max_tokens,
        "privacy_mode": cfg.bot.privacy_mode,
        "show_transcribing_feedback": cfg.bot.show_transcribing_feedback,
        "show_transcription_header": cfg.bot.show_transcription_header,
        "authorized_users": cfg.bot.authorized_users,
        "auth_required_summarize": cfg.bot.auth_required_summarize,
        "auth_required_reply": cfg.bot.auth_required_reply,
        "summary_prompt": cfg.bot.prompts.summary,
        "reply_prompt": cfg.bot.prompts.reply,
        "retention_days": cfg.history.retention_days,
        "reply_retention_days": cfg.history.reply_retention_days,
        "summarize_retention_days": cfg.history.summarize_retention_days,
        "max_chats": cfg.history.max_chats,
    }


@app.get("/api/models")
async def get_models():
    """Return all available models per engine with download status."""
    result = {}
    for engine, models in ENGINE_MODELS.items():
        engine_models = []
        for model_id, label in models:
            downloaded = _check_model_downloaded(engine, model_id)
            status = _downloads.get(model_id, {}).get("status", "done" if downloaded else "none")
            progress = _downloads.get(model_id, {}).get("progress", 100 if downloaded else 0)
            engine_models.append({
                "id": model_id,
                "label": label,
                "downloaded": downloaded,
                "status": status,
                "progress": progress,
                "valid": _check_model_valid(engine, model_id) if downloaded else False,
                "size": _get_model_size_str(engine, model_id) if downloaded else "",
            })
        result[engine] = engine_models
    return {
        "engines": list(ENGINE_MODELS.keys()),
        "engine_models": result,
        "engine_devices": ENGINE_DEVICE,
    }


@app.post("/api/download-model")
async def download_model(data: dict):
    """Trigger model download in background thread with progress tracking."""
    model_id = data.get("model_id", "")
    engine = data.get("engine", "")
    if not model_id or not engine:
        return {"status": "error", "message": "model_id and engine required"}

    if model_id in _downloads and _downloads[model_id]["status"] == "downloading":
        return {"status": "error", "message": "Already downloading"}

    _downloads[model_id] = {"status": "downloading", "progress": 0, "error": ""}

    def _progress(model_id, bytes_downloaded, bytes_total):
        if bytes_total > 0:
            pct = int((bytes_downloaded / bytes_total) * 100)
            _downloads[model_id]["progress"] = min(pct, 99)
        else:
            # Approximate by chunk
            _downloads[model_id]["progress"] = min(
                _downloads[model_id].get("progress", 0) + 5, 90
            )

    def _download():
        try:
            _downloads[model_id]["progress"] = 5
            if engine == "local":
                # Check if model is already cached
                if _check_model_downloaded("local", model_id):
                    _downloads[model_id]["progress"] = 100
                    _downloads[model_id]["status"] = "done"
                    _downloads[model_id]["status_text"] = "Already cached"
                    return
                try:
                    from faster_whisper import WhisperModel
                except ImportError:
                    raise ImportError("faster-whisper is not installed. Run: pip install faster-whisper")
                _downloads[model_id]["progress"] = 30
                _downloads[model_id]["status_text"] = "Downloading model weights..."
                model = WhisperModel(model_id, device="cpu", compute_type="int8")
                _downloads[model_id]["progress"] = 90
                del model
            elif engine == "moonshine":
                try:
                    from moonshine_voice import get_model_for_language
                except ImportError:
                    raise ImportError("moonshine-voice is not installed. Run: pip install moonshine-voice")
                _downloads[model_id]["progress"] = 30
                _downloads[model_id]["status_text"] = "Downloading Moonshine model via CDN..."
                model_path, model_arch = get_model_for_language(model_id)
                _downloads[model_id]["progress"] = 90
            elif engine == "parakeet":
                try:
                    import sherpa_onnx
                except ImportError:
                    raise ImportError("sherpa-onnx is not installed. Run: pip install sherpa-onnx")
                from telescribe.transcriber import ParakeetTranscriber
                cfg = AppConfig.load()
                t = ParakeetTranscriber(cfg)

                # Override _get_model_dir to track download progress
                import urllib.request
                import tarfile

                cache_dir = Path(cfg.data_dir) / "models" / t.MODEL_NAME
                if cache_dir.exists() and (cache_dir / "tokens.txt").exists():
                    _downloads[model_id]["progress"] = 100
                else:
                    _downloads[model_id]["progress"] = 10
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    archive = cache_dir.with_suffix(".tar.bz2")
                    url = f"{t.MODEL_URL}/{t.MODEL_NAME}.tar.bz2"

                    _downloads[model_id]["status_text"] = f"Downloading {t.MODEL_NAME}..."
                    urllib.request.urlretrieve(
                        url, archive,
                        reporthook=lambda b, bs, t: _progress(model_id, b * bs, t),
                    )
                    _downloads[model_id]["progress"] = 80
                    _downloads[model_id]["status_text"] = "Extracting archive..."
                    with tarfile.open(archive, "r:bz2") as tar:
                        tar.extractall(path=cache_dir.parent)
                    archive.unlink()
                    _downloads[model_id]["progress"] = 95

            _downloads[model_id]["status"] = "done"
            _downloads[model_id]["progress"] = 100
            _downloads[model_id]["status_text"] = "Download complete"
        except Exception as e:
            _downloads[model_id]["status"] = "error"
            _downloads[model_id]["error"] = str(e)
            _downloads[model_id]["status_text"] = f"Failed: {str(e)[:60]}"
            logger.exception("Download failed for model %s", model_id)

    thread = threading.Thread(target=_download, daemon=True)
    thread.start()
    return {"status": "started"}


@app.post("/api/clear-model")
async def clear_model(data: dict):
    """Delete downloaded model files from disk."""
    model_id = data.get("model_id", "")
    engine = data.get("engine", "")
    if not model_id or not engine:
        return {"status": "error", "message": "model_id and engine required"}

    paths = _get_model_paths(engine, model_id)
    if not paths:
        return {"status": "error", "message": "No model files found"}

    deleted = []
    for p in paths:
        try:
            if p.is_dir():
                import shutil
                shutil.rmtree(p)
            else:
                p.unlink()
            deleted.append(str(p))
        except Exception as e:
            logger.error("Failed to delete %s: %s", p, e)

    # Reset download tracking
    if model_id in _downloads:
        del _downloads[model_id]

    logger.info("Cleared model %s (%s): %d paths", model_id, engine, len(deleted))
    return {"status": "ok", "deleted": deleted, "count": len(deleted)}


@app.get("/api/download-status")
async def get_download_status():
    """Return current download status for all models."""
    return {"downloads": _downloads}


@app.post("/api/reload")
async def reload_bot():
    """Signal the bot to hot-reload the transcriber."""
    global _bot_reload_requested
    _bot_reload_requested = True
    import telescribe.bot
    telescribe.bot._reload_requested = True
    logger.info("Bot reload requested via dashboard")
    return {"status": "reload_requested"}


@app.get("/api/reload-status")
async def reload_status():
    """Check if reload was requested."""
    global _bot_reload_requested
    return {"reload_requested": _bot_reload_requested}


@app.get("/api/stats")
async def get_stats():
    """Get bot stats including DB size and message count."""
    from pathlib import Path
    db_path = Path(f"{get_config().data_dir}/telescribe.db")
    stats = {"db_size": 0, "message_count": 0}
    if db_path.exists():
        stats["db_size"] = db_path.stat().st_size
        try:
            import aiosqlite
            async with aiosqlite.connect(str(db_path)) as conn:
                cursor = await conn.execute("SELECT COUNT(*) as cnt FROM messages")
                row = await cursor.fetchone()
                stats["message_count"] = row[0] if row else 0
        except Exception:
            pass
    return stats


@app.post("/api/history/prune")
async def prune_history():
    """Delete messages older than retention period."""
    cfg = get_config()
    from telescribe.history import MessageStore
    store = MessageStore(f"{cfg.data_dir}/telescribe.db")
    try:
        deleted = await store.prune_old_messages(cfg.history.retention_days)
        return {"status": "ok", "deleted": deleted}
    except Exception as e:
        logger.exception("Prune failed")
        return {"status": "error", "message": str(e)}
    finally:
        await store.close()


@app.get("/api/logs")
async def get_logs(offset: int = 0, limit: int = 100, level: str = "", search: str = ""):
    """Return log lines from the log file.
    
    - offset: line number to start from (0 = newest, -1 = end)
    - limit: max lines to return (default 100, max 500)
    - level: filter by level (INFO, WARN, ERROR, DEBUG) — empty = all
    - search: substring filter
    """
    log_path = get_log_file_path()
    if not log_path or not Path(log_path).exists():
        return {"lines": [], "total": 0, "next_offset": 0}

    with open(log_path, encoding="utf-8", errors="replace") as f:
        all_lines = f.readlines()

    # Filter
    filtered = []
    for line in all_lines:
        stripped = line.rstrip("\n")
        if level and level.upper() not in stripped:
            continue
        if search and search.lower() not in stripped.lower():
            continue
        filtered.append(stripped)

    total = len(filtered)

    # Paginate: offset 0 = newest, -1 = end
    if offset < 0:
        offset = max(0, total - abs(offset))
    start = max(0, total - offset - limit)
    end = max(0, total - offset)
    if start >= end:
        start = max(0, total - limit)
        end = total

    page = filtered[start:end]
    page.reverse()  # newest first

    next_offset = min(offset + limit, total)

    return {
        "lines": page,
        "total": total,
        "next_offset": next_offset,
    }


# ---- CLI ----

def run() -> None:
    cfg = get_config()
    logger.info("Starting web dashboard on %s:%s", cfg.web.host, cfg.web.port)
    uvicorn.run(
        "telescribe.web.app:app",
        host=cfg.web.host,
        port=cfg.web.port,
        log_level="info",
    )
