"""Logging setup — output goes to stdout + optional file for dashboard log viewer."""

import logging
import os
import sys
from pathlib import Path


_LOG_FILE_PATH: str | None = None
_FILE_HANDLER: logging.Handler | None = None


def setup_logging(log_path: str | None = None) -> logging.Logger:
    """Configure and return root logger.

    Log level is read from LOG_LEVEL env var (default: INFO).
    All output goes to stdout for Docker log capture.
    If log_path is provided, also writes to that file for the dashboard log viewer.
    """
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger("telescribe")
    logger.setLevel(level)

    # Add stdout handler once
    if not any(isinstance(h, logging.StreamHandler) and h.stream == sys.stdout for h in logger.handlers):
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        stdout_handler.setLevel(level)
        logger.addHandler(stdout_handler)

    # Add file handler if path provided
    global _LOG_FILE_PATH, _FILE_HANDLER
    if log_path:
        _LOG_FILE_PATH = log_path
        log_dir = Path(log_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        if _FILE_HANDLER is None:
            _FILE_HANDLER = logging.FileHandler(log_path, encoding="utf-8")
            _FILE_HANDLER.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            _FILE_HANDLER.setLevel(level)
            logger.addHandler(_FILE_HANDLER)
            logger.info("Log file: %s", log_path)

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("aiosqlite").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logger.info("Log level set to %s", level_name)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a named child logger."""
    return logging.getLogger(f"telescribe.{name}")


def get_log_file_path() -> str | None:
    """Return the current log file path, or None if not set."""
    global _LOG_FILE_PATH
    return _LOG_FILE_PATH