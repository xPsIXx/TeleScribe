"""Pydantic-based configuration with YAML file + environment variable overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

def _default_config_path() -> Path:
    for key in ("TALKSCRIBE_CONFIG_PATH", "TELESCRIBE_CONFIG_PATH"):
        value = os.getenv(key)
        if value:
            return Path(value)
    data = (
        os.getenv("DATA_DIR")
        or os.getenv("TELESCRIBE_DATA_DIR")
        or os.getenv("TALKSCRIBE_DATA_DIR")
        or "data"
    )
    return Path(data) / "config.yaml"


DEFAULT_CONFIG_PATH = _default_config_path()

# Old Parakeet 110M CTC model id — auto-migrated to TDT 0.6B v2.
_LEGACY_PARAKEET_MODELS = {
    "sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000-int8",
}

PARAKEET_TDT_V2 = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"


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
    model: str = "distil-large-v3"
    device: Literal["cpu", "cuda"] = "cpu"
    compute_type: str = "int8"
    beam_size: int = 5
    temperature: float = 0.0
    best_of: int = 5
    patience: float = 1.0
    length_penalty: float = 1.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    suppress_blank: bool = True
    condition_on_previous_text: bool = False
    vad_filter: bool = False
    vad_min_silence_ms: int = 700
    vad_speech_pad_ms: int = 400
    vad_threshold: float = 0.45
    hallucination_silence_threshold: float = 2.0
    no_speech_threshold: Optional[float] = None
    log_prob_threshold: Optional[float] = None
    compression_ratio_threshold: Optional[float] = None
    initial_prompt: Optional[str] = None
    language: Optional[str] = None
    # Bumped once when we disabled previous-text conditioning. Do not re-apply
    # that migration on later loads — the user may turn the option back on.
    asr_defaults_version: int = 2


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
    auth_required_transcribe: bool = True


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

    @classmethod
    def load(cls, path: str | Path | None = None) -> AppConfig:
        """Load YAML, then apply env overrides.

        Always from env: TELEGRAM_BOT_TOKEN, LLM_API_KEY, DATA_DIR.
        First run only (no YAML yet): ASR_*, LLM_BASE_URL, LLM_MODEL, PRIVACY_MODE.
        After config.yaml exists, dashboard Save is the source of truth for those.
        """
        path = Path(path) if path else DEFAULT_CONFIG_PATH

        yaml_exists = path.exists()
        migrated = False
        if yaml_exists:
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
            # Migrate old format: authorized_users was a string "*" but is now a list
            if isinstance(raw.get("bot", {}).get("authorized_users"), str):
                raw.setdefault("bot", {})["authorized_users"] = []
                migrated = True
            # Strip removed fields that may exist in old config files
            raw.get("bot", {}).pop("auth_required_callback", None)
            raw.pop("_dashboard_managed", None)  # legacy field, no longer used

            tx = raw.get("transcription") or {}
            if isinstance(tx, dict):
                # One-shot: disable previous-text conditioning that caused
                # silence cutoffs. Later loads respect the user's checkbox.
                if int(tx.get("asr_defaults_version") or 0) < 2:
                    tx["condition_on_previous_text"] = False
                    tx["asr_defaults_version"] = 2
                    migrated = True
                if tx.get("engine") == "parakeet" and str(tx.get("model") or "") in _LEGACY_PARAKEET_MODELS:
                    tx["model"] = PARAKEET_TDT_V2
                    migrated = True
                tx.setdefault("vad_min_silence_ms", 700)
                tx.setdefault("vad_speech_pad_ms", 400)
                tx.setdefault("vad_threshold", 0.45)
                tx.setdefault("hallucination_silence_threshold", 2.0)
                raw["transcription"] = tx

            bot = raw.get("bot") or {}
            if isinstance(bot, dict) and "auth_required_transcribe" not in bot:
                bot["auth_required_transcribe"] = bot.get("auth_required_summarize", True)
                raw["bot"] = bot
                migrated = True

            base = cls.model_validate(raw)
            if migrated:
                base.save(path)
        else:
            # Create default config if it doesn't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            base = cls()
            base.save(path)

        # Secrets and deploy paths always come from the environment.
        token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        if token:
            base.telegram_bot_token = token

        llm_api_key = os.getenv("LLM_API_KEY", "")
        if llm_api_key:
            base.llm.api_key = llm_api_key

        data_dir = (
            os.getenv("DATA_DIR")
            or os.getenv("TELESCRIBE_DATA_DIR")
            or os.getenv("TALKSCRIBE_DATA_DIR")
            or ""
        )
        if data_dir:
            base.data_dir = data_dir

        # Dashboard-managed settings: env is only the first-run seed. After
        # config.yaml exists, Save & Reload must win (including LLM URL/model).
        if not yaml_exists:
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

            llm_url = os.getenv("LLM_BASE_URL", "")
            if llm_url:
                base.llm.base_url = llm_url

            llm_model = os.getenv("LLM_MODEL", "")
            if llm_model:
                base.llm.model = llm_model

            privacy = os.getenv("PRIVACY_MODE", "")
            if privacy:
                base.bot.privacy_mode = privacy.lower() in ("true", "1", "yes")

        return base

    def save(self, path: str | Path | None = None) -> None:
        """Serialize back to YAML, marking it as dashboard-managed."""
        path = Path(path) if path else DEFAULT_CONFIG_PATH

        data = {
            "bot": {
                "privacy_mode": self.bot.privacy_mode,
                "admin_user_ids": self.bot.admin_user_ids,
                "authorized_users": self.bot.authorized_users,
                "show_transcribing_feedback": self.bot.show_transcribing_feedback,
                "show_transcription_header": self.bot.show_transcription_header,
                "auth_required_summarize": self.bot.auth_required_summarize,
                "auth_required_reply": self.bot.auth_required_reply,
                "auth_required_transcribe": self.bot.auth_required_transcribe,
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
                "temperature": self.transcription.temperature,
                "best_of": self.transcription.best_of,
                "patience": self.transcription.patience,
                "length_penalty": self.transcription.length_penalty,
                "repetition_penalty": self.transcription.repetition_penalty,
                "no_repeat_ngram_size": self.transcription.no_repeat_ngram_size,
                "suppress_blank": self.transcription.suppress_blank,
                "condition_on_previous_text": self.transcription.condition_on_previous_text,
                "vad_filter": self.transcription.vad_filter,
                "vad_min_silence_ms": self.transcription.vad_min_silence_ms,
                "vad_speech_pad_ms": self.transcription.vad_speech_pad_ms,
                "vad_threshold": self.transcription.vad_threshold,
                "hallucination_silence_threshold": self.transcription.hallucination_silence_threshold,
                "no_speech_threshold": self.transcription.no_speech_threshold,
                "log_prob_threshold": self.transcription.log_prob_threshold,
                "compression_ratio_threshold": self.transcription.compression_ratio_threshold,
                "initial_prompt": self.transcription.initial_prompt,
                "language": self.transcription.language,
                "asr_defaults_version": self.transcription.asr_defaults_version,
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
