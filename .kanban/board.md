# TeleScribe Kanban Board

## Meta
project_id: telescribe
board_version: 1.2
updated: 2026-08-27 15:58
lane_model: custom

## Lanes
- backlog
- ready
- in-progress
- blocked
- review
- done

## Cards

| id | title | state | priority | owner | due | depends_on | updated |
|----|-------|-------|----------|-------|-----|------------|---------|
| KB-001 | Core bot framework (python-telegram-bot) | done | P0 | agent | - | - | 2026-08-27 |
| KB-002 | 6 ASR engine backends (local, moonshine, parakeet, openai, groq, custom) | done | P0 | agent | - | - | 2026-08-27 |
| KB-003 | Ephemeral privacy mode (Bot API 10.3+) | done | P0 | agent | - | - | 2026-08-27 |
| KB-004 | Group message history (SQLite) | done | P1 | agent | - | - | 2026-08-27 |
| KB-005 | /summarize command with LLM | done | P1 | agent | - | - | 2026-08-27 |
| KB-006 | /reply command + "Reply to this" button | done | P1 | agent | - | - | 2026-08-27 |
| KB-007 | Web dashboard (single-page, no login) | done | P1 | agent | - | - | 2026-08-27 |
| KB-008 | Dynamic model list per engine | done | P1 | agent | - | - | 2026-08-27 |
| KB-009 | Model download with progress bar | done | P1 | agent | - | - | 2026-08-27 |
| KB-010 | Hot-reload transcriber without container restart | done | P1 | agent | - | - | 2026-08-27 |
| KB-011 | Engine switch env-var priority bug fix | done | P0 | agent | - | - | 2026-08-27 |
| KB-012 | Authorization system (public voice, gated functions) | done | P1 | agent | - | - | 2026-08-27 |
| KB-013 | Moonshine transcription quality fix (Python API rewrite) | done | P1 | agent | - | - | 2026-08-27 |
| KB-014 | Callback_data 64-byte limit fix | done | P1 | agent | - | - | 2026-08-27 |
| KB-015 | Feedback toggles (transcribing msg, header) | done | P2 | agent | - | - | 2026-08-27 |
| KB-016 | Real-time progress bar for model downloads | done | P2 | agent | - | - | 2026-08-27 |
| KB-017 | Moonshine & Parakeet model detection fix | done | P2 | agent | - | - | 2026-08-27 |
| KB-018 | Dashboard redesign (engine cards, sliders, history section) | done | P2 | agent | - | - | 2026-08-27 |
| KB-019 | Prune old messages endpoint | done | P2 | agent | - | - | 2026-08-27 |
| KB-020 | Moonshine quality verified in production | ready | P1 | user | - | KB-013 | 2026-08-27 |
| KB-021 | Parakeet download button verified | ready | P2 | user | - | - | 2026-08-27 |
| KB-022 | Webhook mode (replace polling) | backlog | P2 | unassigned | - | - | 2026-08-27 |
| KB-023 | Recording messages across chats | backlog | P2 | unassigned | - | - | 2026-08-27 |
| KB-024 | Multi-language transcription support | backlog | P3 | unassigned | - | - | 2026-08-27 |
| KB-025 | Live dashboard logs (stream bot logs to web UI) | backlog | P3 | unassigned | - | - | 2026-08-27 |
| KB-026 | Export history command | backlog | P3 | unassigned | - | - | 2026-08-27 |
| KB-027 | Docker image size optimization (multi-stage, ~450MB) | backlog | P3 | unassigned | - | - | 2026-08-27 |

## WIP Limits
- in-progress: 2
- review: 3

## Notes
- Moonshine and Parakeet need real-world testing to confirm stability
- User prefers to control when pushes happen — no auto-push
- Dashboard is open access (no auth), authorization is at the bot function level