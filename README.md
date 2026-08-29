# TeleScribe 🎙️

**Telegram bot with ASR transcription, group message history, authorization, and a web dashboard.**

TeleScribe is a self-hosted Telegram bot that transcribes voice messages, stores group chat history, and provides AI-powered summaries — all running on your own hardware.

## Features

- **3 Transcription Engines** — Select via the dashboard:
  - **Local** (faster-whisper) — CPU or GPU, private, no API costs
  - **Moonshine** — Edge-optimized ASR, ~100ms latency, ~60M params, CPU
  - **Parakeet** — NVIDIA NeMo TDT, 6.3% WER, CPU
- **Public Transcription** — Replies publicly to the original voice message, visible to everyone in the group
- **Authorization System** — Lock `/summarize`, `/reply`, and the "💬 Reply to this" button to specific users
- **Per-function Auth Toggles** — Independently control which features require authorization
- **Group Message History** — All messages stored in SQLite for summarization
- **Commands:**
  - `/summarize 2d` — Summarize the last N hours/days (ephemeral, private reply)
  - `/reply <text>` — AI-generated reply with conversation context (ephemeral, private reply)
- **Ephemeral Callback Replies** — Clicking "💬 Reply to this" sends the reply privately to you only
- **Customizable Prompts** — Edit summary and reply prompts via dashboard
- **Web Dashboard** — Single-page dark-mode UI. No login required. Manage all settings at a glance:
  - Engine selection with clickable cards
  - Dynamic model list (per-engine, with download status)
  - Model download with progress bar
  - Device and compute type controls
  - LLM config (URL, model, temperature, max tokens)
  - Authorization (user IDs + per-function toggles)
  - Feedback toggles (transcribing status, transcription header)
  - History retention settings
  - **Live log viewer** — See bot logs in real-time
  - **Save & Reload** — One button saves config and hot-reloads the transcriber
- **Comprehensive Logging** — Every command, voice, and error logged to stdout for `docker logs` + file-based log viewer
- **Self-Hosted** — Runs in Docker on any Linux server or Unraid

## Quick Start

### Docker Compose

```bash
# Clone and configure
git clone https://github.com/xPsIXx/TeleScribe.git
cd TeleScribe
cp .env.example .env

# Edit .env with your bot token
nano .env

# Start (default: faster-whisper on CPU)
docker compose up -d

# View logs
docker compose logs -f
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TELEGRAM_BOT_TOKEN` | ✅ | — | Your bot token from @BotFather |
| `ASR_ENGINE` | ❌ | `local` | `local`, `moonshine`, or `parakeet` |
| `ASR_MODEL` | ❌ | `distil-medium.en` | Model name (for local engine) |
| `ASR_DEVICE` | ❌ | `cpu` | `cpu` or `cuda` (for local engine) |
| `ASR_COMPUTE_TYPE` | ❌ | `int8` | `int8`, `float16`, `float32` (for local engine) |
| `ASR_LANGUAGE` | ❌ | (auto) | Language code for transcription (e.g. `en`, `es`) |
| `LLM_BASE_URL` | ❌ | `http://localhost:8088/v1` | OpenAI-compatible LLM endpoint |
| `LLM_API_KEY` | ❌ | — | API key for LLM endpoint |
| `LLM_MODEL` | ❌ | `qwen3.5-9b` | LLM model for summaries/replies |
| `PRIVACY_MODE` | ❌ | `true` | Use ephemeral messages for private replies |
| `LOG_LEVEL` | ❌ | `INFO` | Set to `DEBUG` for verbose logging |
| `DATA_DIR` | ❌ | `/data` | Persistent data directory |

## Unraid Deployment

### Docker Setup

1. **Docker → Add Container**
2. **Repository:** `ghcr.io/xPsIXx/TeleScribe:latest`
3. **Web UI:** `http://[Unraid-IP]:8180`

### Traefik Reverse Proxy (recommended)

```yaml
services:
  telescribe:
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.telescribe.rule=Host(`telescribe.example.com`)"
      - "traefik.http.routers.telescribe.entrypoints=websecure"
      - "traefik.http.routers.telescribe.tls.certresolver=letsencrypt"
      - "traefik.http.services.telescribe.loadbalancer.server.port=8180"
```

### Environment Variables (Unraid GUI)

| Variable | Value |
|----------|-------|
| `TELEGRAM_BOT_TOKEN` | `123456:ABC-def_123` |
| `ASR_ENGINE` | `local` |
| `LLM_BASE_URL` | `http://192.168.1.2:8088/v1` |
| `LLM_MODEL` | `qwen3.5-9b` |
| `LOG_LEVEL` | `INFO` |

### Volumes

| Container Path | Host Path (Unraid) |
|---------------|-------------------|
| `/data` | `/mnt/user/appdata/telescribe/data/` |

Model files are stored under `/data/models/` on the persistent volume.

### Port

| Container | Host |
|-----------|------|
| `8180` | `127.0.0.1:8180` (behind Traefik) |

## Web Dashboard

The dashboard runs on **port 8180** with no login required. Everything is on one page:

- **Engine Cards** — Clickable cards for faster-whisper, Moonshine, Parakeet
- **Model Dropdown** — Shows models relevant to the selected engine, with download status (✅ downloaded / ❌ not downloaded / ⏳ downloading)
- **Download / Clear** — Download model files or clear corrupt downloads
- **Device** — Auto-hides for engines that don't support it
- **LLM Config** — Endpoint, model, temperature slider, max tokens
- **Custom Prompts** — Summary and reply prompts
- **Authorization** — User IDs list + per-function checkboxes
- **Bot Behavior** — Privacy mode, feedback toggles
- **Message History** — Retention days, max chats, prune button
- **Live Logs** — Real-time log viewer with level filter and search
- **Save & Reload** — One button saves config and hot-reloads the transcriber

## Authorization

By default, the authorized users list is **empty** — meaning no one can use `/summarize`, `/reply`, or the "💬 Reply to this" button until you add your Telegram user ID.

1. Open the dashboard at `http://[your-server]:8180`
2. Find your Telegram user ID (use @userinfobot)
3. Enter it in the **Authorized User IDs** field
4. Check which functions to restrict
5. Click **Save & Reload**

Voice transcription is always available to everyone.

## Logging

All output goes to stdout — view with `docker logs`:

```
$ docker logs telescribe
2026-08-28 10:00:00 - telescribe.main - INFO - === TeleScribe v0.1.24 starting ===
2026-08-28 10:00:00 - telescribe.main - INFO - Config:
2026-08-28 10:00:00 - telescribe.main - INFO -   ASR engine:  local
2026-08-28 10:00:00 - telescribe.main - INFO -   ASR model:   distil-medium.en
2026-08-28 10:00:00 - telescribe.main - INFO -   LLM endpoint: http://localhost:8088/v1
2026-08-28 10:00:00 - telescribe.bot - INFO - Bot instance created
2026-08-28 10:00:00 - telescribe.bot - INFO - Starting bot polling (Telegram API)...
2026-08-28 10:00:15 - telescribe.bot - INFO - Voice msg from user 123456: duration=12s
2026-08-28 10:00:18 - telescribe.bot - INFO - Transcription OK: 145 chars, 3.2s total (lang=en)
2026-08-28 10:00:20 - telescribe.bot - INFO - Command /summarize 48h from user 123456 in chat -789
2026-08-28 10:00:22 - telescribe.bot - INFO - Summary generated: 42 msgs, 892 chars, 1.4s
```

The dashboard also includes a **Live Logs** section with real-time auto-refresh, level filtering, and search.

## Architecture

```
                    ┌─────────────┐
                    │  Telegram    │
                    │  Bot API     │
                    └──────┬──────┘
                           │ polling
               ┌───────────┴───────────┐
               │   TeleScribe Bot      │
               │  (python-telegram-bot) │
               └───────┬───────┬───────┘
                       │       │
               ┌───────┘       └───────┐
               │                       │
        ┌──────▼──────┐       ┌──────▼──────┐
        │ Transcriber  │       │  Message    │
        │ (pluggable)  │       │  Store      │
        │              │       │  (SQLite)   │
        └──────┬───────┘       └──────┬──────┘
               │                      │
        ┌──────▼──────┐       ┌──────▼──────┐
        │ faster-     │       │   LLM API   │
        │ whisper /   │       │ (OpenAI-    │
        │ Moonshine / │       │ compatible) │
        │ Parakeet    │       └─────────────┘
        └──────┬───────┘
               │
        ┌──────▼──────┐
        │ Web Dashboard│
        │ (FastAPI +   │
        │  Tailwind)   │
        │ + Live Logs  │
        └─────────────┘
```

## Development

```bash
# Clone and install
git clone https://github.com/xPsIXx/TeleScribe.git
cd TeleScribe
uv sync

# Run bot
uv run telescribe

# Install optional engines
uv pip install moonshine-voice   # Moonshine support
uv pip install sherpa-onnx       # Parakeet support
```

## Changelog

### v0.1.24
- Callback reply now sends ephemeral message to clicking user only (no longer edits the original transcription)
- All users' text messages stored in history for complete summaries
- Models download to `/data/models/` via `MOONSHINE_VOICE_CACHE` and `HF_HOME` env vars
- Version now dynamic from `__version__`

### v0.1.23
- Fix callback crash (invalid `ephemeral_message_parameters`)
- Model detection checks `/data/models/` first
- Save button auto-triggers reload (no separate Reload button)

### v0.1.22
- Transcription is public reply (visible to everyone)
- Button only attached for authorized users

### v0.1.21
- Remove Parakeet V2 engine
- Fix faster-whisper download status

## License

MIT