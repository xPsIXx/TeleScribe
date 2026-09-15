# TeleScribe 🎙️

**Telegram bot with on-device ASR, group history, and a web dashboard.**

Self-hosted. Voice notes are transcribed in the group; `/summarize` and `/reply` talk to your own OpenAI-compatible LLM.

## Features

- **3 transcription engines** (dashboard cards):
  - **faster-whisper** — CPU or CUDA. Default `distil-large-v3`. Previous-text conditioning off so long notes don't stop after a pause. VAD **off** by default (full clip, including silence).
  - **Moonshine** — CPU, ~100ms, 8 languages (en/es/zh/ja/ko/vi/ar/uk). Long audio is chunked.
  - **Parakeet** — NVIDIA **TDT 0.6B v2 INT8** via sherpa-onnx (not the old 110M CTC). English, CPU.
- **Public transcription** — replies to the voice message in the group. Long text is split under Telegram's 4096-character limit (not truncated).
- **Commands** (locked to authorized user IDs unless you turn the toggles off):
  - `/summarize` — unsummarized transcriptions from *other* people in this chat
  - `/summarize_all` / `/summarize all` — last day (or `2d` / `12h`) per person, all chats
  - `/reply` — draft replies to unreplied transcriptions from others
  - `/transcribe` — retry voice notes that were stored but never transcribed
- **Dashboard** on port **8180**, **no login**. Engine cards, models, download, LLM URL/model/temp/tokens, auth, live logs with a **Failures** filter, Save & Reload.
- **Logging** — stdout + rotating `/data/telescribe.log` (kept across restarts). Search `FAILURE` for crashes.

## Quick Start

```bash
git clone https://github.com/xPsIXx/TeleScribe.git
cd TeleScribe
cp .env.example .env
# set TELEGRAM_BOT_TOKEN (and LLM_BASE_URL if the LLM is not on localhost:8088)
docker compose up -d --build
docker compose logs -f
```

Dashboard: `http://127.0.0.1:8180`

1. Pick an engine → **Download** the model if needed → **Save & Reload**.
2. Put your Telegram user ID in **Authorized User IDs** so `/summarize` / `/reply` / `/transcribe` work.
3. Set the LLM endpoint to a running OpenAI-compatible server.

Voice transcription works for everyone without auth.

## Environment

| Variable | Required | After first boot | Description |
|----------|----------|------------------|-------------|
| `TELEGRAM_BOT_TOKEN` | ✅ | always | @BotFather token |
| `LLM_API_KEY` | ❌ | always | LLM secret. YAML never overrides this. |
| `LLM_BASE_URL` | ❌ | **first run only** | Seeds `config.yaml`. Change later in the dashboard. |
| `LLM_MODEL` | ❌ | **first run only** | Same — dashboard wins after Save. |
| `ASR_ENGINE` | ❌ | first run only | `local`, `moonshine`, `parakeet` |
| `ASR_MODEL` | ❌ | first run only | Default `distil-large-v3` |
| `ASR_DEVICE` | ❌ | first run only | `cpu` or `cuda` |
| `ASR_COMPUTE_TYPE` | ❌ | first run only | `int8`, `float16`, `float32` |
| `LOG_LEVEL` | ❌ | always | `INFO` or `DEBUG` |
| `DATA_DIR` | ❌ | always | Default `/data` |

Compose still *passes* `LLM_BASE_URL` into the container; TeleScribe only uses it to seed a missing `config.yaml`. After you Save in the dashboard, that YAML is the source of truth for URL, model, temperature, and max tokens.

## Engines

| Engine | Dashboard model | Device | Notes |
|--------|-----------------|--------|--------|
| faster-whisper | tiny … large-v3, distil-* | CPU / CUDA | Distil models force previous-text conditioning **off**. Leave VAD unchecked unless you want silence stripped. |
| Moonshine | language code (`en`, `es`, …) | CPU | Language *is* the model dropdown. |
| Parakeet | `sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8` | CPU | English. Old 110M id is auto-migrated. |

Switching engines in the UI resets the model to one that engine can actually load. Save coerces leftover ids (e.g. `engine=moonshine` + `distil-large-v3` → `en`).

## LLM

`/summarize` and `/reply` call `POST {base_url}/chat/completions` with the dashboard model, temperature, and max tokens (60s timeout). Point `LLM_BASE_URL` at anything OpenAI-compatible (llama-swap, Ollama with `/v1`, vLLM, OpenRouter, …).

Empty or refused responses are logged as warnings; connection errors as `FAILURE llm endpoint=… model=…`.

## Unraid

Images publish to GHCR on every `main` push and on `v*` tags: `ghcr.io/xpsixx/telescribe:latest`

1. Docker → Add Container → `ghcr.io/xpsixx/telescribe:latest`
2. Web UI: `http://[Unraid-IP]:8180`
3. Volume: `/data` → `/mnt/user/appdata/telescribe/data/`
4. Env: `TELEGRAM_BOT_TOKEN`, optional `LLM_API_KEY`, `LOG_LEVEL`.

Traefik:

```yaml
labels:
  - "traefik.enable=true"
  - "traefik.http.routers.telescribe.rule=Host(`telescribe.example.com`)"
  - "traefik.http.routers.telescribe.entrypoints=websecure"
  - "traefik.http.routers.telescribe.tls.certresolver=letsencrypt"
  - "traefik.http.services.telescribe.loadbalancer.server.port=8180"
```

## Logging

```
docker logs -f telescribe
```

File: `/data/telescribe.log` (5MB × 5, not wiped on restart). Dashboard **Failures** chip shows `ERROR` / `FAILURE` lines.

Typical ASR line: `ASR job queued chat=… engine=local model=distil-large-v3`. Typical crash: `FAILURE transcription chat=… err=RuntimeError: …`.

## Architecture

```
Telegram Bot API ──polling──► TeleScribe
                                 ├─ Transcriber (faster-whisper / Moonshine / Parakeet)
                                 ├─ SQLite history
                                 ├─ LLM (OpenAI-compatible)
                                 └─ Dashboard :8180
```

## Development

```bash
uv sync
uv run telescribe
# optional extras are already in the Docker image:
# uv pip install moonshine-voice sherpa-onnx
```

## Changelog

### v0.3.4
- Dashboard LLM URL/model/temp/max tokens survive Save & Reload (env no longer overwrites YAML)
- `/summarize` and `/reply` use the saved LLM settings; failures log `FAILURE llm`
- README matches current engines, commands, and logging

### v0.3.3
- Whisper cutoff fix: default `distil-large-v3`, previous-text conditioning off, VAD off
- Parakeet switched to `sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8` (TDT 0.6B)
- Moonshine long-audio chunking; engine switch no longer saves a stale model
- Rotating logs, `FAILURE` lines, dashboard Failures filter
- No dashboard login; Telegram long-reply splitting

## License

MIT
