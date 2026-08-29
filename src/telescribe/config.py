"""Pydantic-based configuration with YAML file + environment variable overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CONFIG_PATH = Path(os.getenv("TALKSCRIBE_CONFIG_PATH", "/opt/data/projects/telescribe/data/config.yaml"))


class PromptsConfig(BaseSettings):
    summary: str = (
        "Summarize the following group chat messages concisely. "
        "Focus on key decisions, action items, and important information. "
        "Format as bullet points."
    )
    reply: str = "You are a helpful assistant in a Telegram group chat. Reply naturally and conversationally."
    transcription_context: str = "The following text was transcribed from a voice message. Please process it as if the user typed it directly."


class TranscriptionConfig(BaseSettings):
    engine: Literal["local", "moonshine", "parakeet"] = "local"
    model: str = "distil-medium.en"
    device: Literal["cpu", "cuda"] = "cpu"
    compute_type: str = "int8"
    beam_size: int = 5
    vad_filter: bool = True
    language: Optional[str] = None


class LLMConfig(BaseSettings):
    base_url: str = "http://localhost:8088/v1"
    api_key: str = ""
    model: str = "qwen3.5-9b"
    temperature: float = 0.7
    max_tokens: int = 4096
    stream: bool = True


class WebConfig(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8180
    username: str = "admin"
    password: str = "changeme"


class HistoryConfig(BaseSettings):
    retention_days: int = 90
    reply_retention_days: int = 30
    summarize_retention_days: int = 30
    max_chats: int = 100


class BotConfig(BaseSettings):
    privacy_mode: bool = True
    admin_user_ids: list[int] = Field(default_factory=list)
    authorized_users: list[int] = Field(default_factory=list)
    prompts: PromptsConfig = PromptsConfig()
    show_transcribing_feedback: bool = True
    show_transcription_header: bool = True
    auth_required_summarize: bool = True
    auth_required_reply: bool = True


class AppConfig(BaseSettings):
    """Root configuration — loads from YAML, overridable by env vars."""

    model_config = SettingsConfigDict(env_prefix="TALKSCRIBE_", env_nested_delimiter="__")

    bot: BotConfig = BotConfig()
    transcription: TranscriptionConfig = TranscriptionConfig()
    llm: LLMConfig = LLMConfig()
    web: WebConfig = WebConfig()
    history: HistoryConfig = HistoryConfig()

    telegram_bot_token: str = ""
    data_dir: str = "/data"

    # Internal: set to True when save() is called by the dashboard
    # Prevents env vars from overriding dashboard-managed settings on reload
    _dashboard_managed: bool = False

    @classmethod
    def load(cls, path: str | Path | None = None) -> AppConfig:
        """Load from YAML file, then overlay env vars."""
        path = Path(path) if path else DEFAULT_CONFIG_PATH

        dashboard_managed = False
        if path.exists():
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
            dashboard_managed = raw.pop("_dashboard_managed", False)
            # Migrate old format: authorized_users was a string "*" but is now a list
            if isinstance(raw.get("bot", {}).get("authorized_users"), str):
                raw["bot"]["authorized_users"] = []
            # Strip removed fields that may exist in old config files
            raw.get("bot", {}).pop("auth_required_callback", None)
            base = cls.model_validate(raw)
        else:
            # Create default config if it doesn't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            base = cls()
            base.save(path)

        # Overlay env var overrides
        # NOTE: when the config was saved by the dashboard (_dashboard_managed = True),
        # skip env var overrides for transcription/engine/model/device since those
        # are managed through the UI.
        token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        if token:
            base.telegram_bot_token = token

        if not dashboard_managed:
            asr_engine = os.getenv("ASR_ENGINE", "")
            if asr_engine:
                base.transcription.engine = asr_engine  # type: ignore

            asr_model = os.getenv("ASR_MODEL", "")
            if asr_model:
                base.transcription.model = asr_model

            asr_device = os.getenv("ASR_DEVICE", "")
            if asr_device:
                base.transcription.device = asr_device  # type: ignore

            asr_compute = os.getenv("ASR_COMPUTE_TYPE", "")
            if asr_compute:
                base.transcription.compute_type = asr_compute

            asr_beam = os.getenv("ASR_BEAM_SIZE", "")
            if asr_beam:
                base.transcription.beam_size = int(asr_beam)

            asr_lang = os.getenv("ASR_LANGUAGE", "")
            if asr_lang:
                base.transcription.language = asr_lang

        # These env var overrides always apply (they can't be set via dashboard)
        llm_url = os.getenv("LLM_BASE_URL", "")
        if llm_url:
            base.llm.base_url = llm_url

        llm_api_key = os.getenv("LLM_API_KEY", "")
        if llm_api_key:
            base.llm.api_key = llm_api_key

        llm_model = os.getenv("LLM_MODEL", "")
        if llm_model:
            base.llm.model = llm_model

        data_dir = os.getenv("DATA_DIR", "")
        if data_dir:
            base.data_dir = data_dir

        privacy = os.getenv("PRIVACY_MODE", "")
        if privacy:
            base.bot.privacy_mode = privacy.lower() in ("true", "1", "yes")

        return base

    def save(self, path: str | Path | None = None) -> None:
        """Serialize back to YAML, marking it as dashboard-managed."""
        path = Path(path) if path else DEFAULT_CONFIG_PATH

        data = {
            "_dashboard_managed": True,
            "bot": {
                "privacy_mode": self.bot.privacy_mode,
                "admin_user_ids": self.bot.admin_user_ids,
                "authorized_users": self.bot.authorized_users,
                "show_transcribing_feedback": self.bot.show_transcribing_feedback,
                "show_transcription_header": self.bot.show_transcription_header,
                "auth_required_summarize": self.bot.auth_required_summarize,
                "auth_required_reply": self.bot.auth_required_reply,
                "prompts": {
                    "summary": self.bot.prompts.summary,
                    "reply": self.bot.prompts.reply,
                    "transcription_context": self.bot.prompts.transcription_context,
                },
            },
            "transcription": {
                "engine": self.transcription.engine,
                "model": self.transcription.model,
                "device": self.transcription.device,
                "compute_type": self.transcription.compute_type,
                "beam_size": self.transcription.beam_size,
                "vad_filter": self.transcription.vad_filter,
                "language": self.transcription.language,
            },
            "llm": {
                "base_url": self.llm.base_url,
                "api_key": self.llm.api_key,
                "model": self.llm.model,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
                "stream": self.llm.stream,
            },
            "web": {
                "host": self.web.host,
                "port": self.web.port,
                "username": self.web.username,
                "password": self.web.password,
            },
            "history": {
                "retention_days": self.history.retention_days,
                "reply_retention_days": self.history.reply_retention_days,
                "summarize_retention_days": self.history.summarize_retention_days,
                "max_chats": self.history.max_chats,
            },
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)