"""CLI entry point for the telescribe bot — runs both the bot and web dashboard."""

from __future__ import annotations

import sys
import threading

from telescribe import __version__
from telescribe.config import AppConfig
from telescribe.history import MessageStore
from telescribe.logger import get_logger, setup_logging
from telescribe.transcriber import create_transcriber

logger = get_logger("main")


def cli() -> None:
    """Main entry point — starts bot polling and web dashboard concurrently."""
    setup_logging()
    logger.info("=== TeleScribe v%s starting ===", __version__)

    config = AppConfig.load()

    # Add file logging for dashboard log viewer (truncate to clear previous session logs)
    with open(f"{config.data_dir}/telescribe.log", "w") as f:
        f.write("")
    setup_logging(log_path=f"{config.data_dir}/telescribe.log")

    if not config.telegram_bot_token:
        logger.error("TELEGRAM_BOT_TOKEN is required — set it in .env or environment")
        sys.exit(1)

    # Log the active configuration
    logger.info("Config:")
    logger.info("  ASR engine:  %s", config.transcription.engine)
    logger.info("  ASR model:   %s", config.transcription.model)
    if config.transcription.engine == "local":
        logger.info("  ASR device:  %s", config.transcription.device)
        logger.info("  ASR compute: %s", config.transcription.compute_type)
    logger.info("  LLM endpoint: %s", config.llm.base_url)
    logger.info("  LLM model:   %s", config.llm.model)
    logger.info("  Privacy mode: %s", config.bot.privacy_mode)
    logger.info("  Admin users:  %s", config.bot.admin_user_ids or "(none — all users authorized)")
    logger.info("  Data dir:     %s", config.data_dir)
    logger.info("  Web dashboard: %s:%s", config.web.host, config.web.port)

    # Initialize components
    logger.info("Initializing transcriber (%s)...", config.transcription.engine)
    transcriber = create_transcriber(config)

    logger.info("Initializing message store...")
    store = MessageStore(f"{config.data_dir}/telescribe.db")

    # Start web dashboard in a background thread
    logger.info("Starting web dashboard thread...")
    _start_web_dashboard(config)

    # Import and run the bot (blocking — run_polling never returns)
    from telescribe.bot import TalkscribeBot

    bot = TalkscribeBot(config, transcriber, store)
    logger.info("Bot initialized, starting polling...")
    bot.run()


def _start_web_dashboard(config: AppConfig) -> None:
    """Start the FastAPI web dashboard in a daemon thread."""
    import uvicorn

    def _run():
        logger.info("Web dashboard starting on %s:%s", config.web.host, config.web.port)
        uvicorn.run(
            "telescribe.web.app:app",
            host=config.web.host,
            port=config.web.port,
            log_level="info",
        )

    thread = threading.Thread(target=_run, daemon=True, name="web-dashboard")
    thread.start()
    logger.info("Web dashboard thread started")