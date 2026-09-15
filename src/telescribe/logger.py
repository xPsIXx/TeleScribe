"""Logging setup — stdout + rotating file for the dashboard log viewer."""

from __future__ import annotations

import logging
import os
import sys
import threading
import warnings
from logging.handlers import RotatingFileHandler
from pathlib import Path


_LOG_FILE_PATH: str | None = None
_FILE_HANDLER: logging.Handler | None = None
_STDOUT_HANDLER: logging.Handler | None = None

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 5


def _formatter() -> logging.Formatter:
    return logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)


def _excepthook(exc_type, exc_value, exc_tb) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    logging.getLogger("telescribe").error(
        "FAILURE uncaught_exception type=%s",
        getattr(exc_type, "__name__", exc_type),
        exc_info=(exc_type, exc_value, exc_tb),
    )


def _thread_excepthook(args: threading.ExceptHookArgs) -> None:
    if args.exc_type and issubclass(args.exc_type, KeyboardInterrupt):
        return
    logging.getLogger("telescribe").error(
        "FAILURE uncaught_thread_exception thread=%s type=%s",
        getattr(args.thread, "name", "?"),
        getattr(args.exc_type, "__name__", args.exc_type),
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )


def setup_logging(log_path: str | None = None) -> logging.Logger:
    """Configure root logging so bot, dashboard, and third-party errors all land in one place.

    Log level is read from LOG_LEVEL env var (default: INFO).
    Stdout is always attached (Docker). A rotating file is attached when log_path is set.
    Previous sessions are kept — the file is not truncated on startup.
    """
    global _LOG_FILE_PATH, _FILE_HANDLER, _STDOUT_HANDLER

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    if _STDOUT_HANDLER is None:
        _STDOUT_HANDLER = logging.StreamHandler(sys.stdout)
        _STDOUT_HANDLER.setFormatter(_formatter())
        _STDOUT_HANDLER.setLevel(level)
        root.addHandler(_STDOUT_HANDLER)

    if log_path:
        _LOG_FILE_PATH = log_path
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        if _FILE_HANDLER is None:
            _FILE_HANDLER = RotatingFileHandler(
                log_path,
                maxBytes=MAX_BYTES,
                backupCount=BACKUP_COUNT,
                encoding="utf-8",
            )
            _FILE_HANDLER.setFormatter(_formatter())
            _FILE_HANDLER.setLevel(level)
            root.addHandler(_FILE_HANDLER)
            logging.getLogger("telescribe").info("Log file: %s (rotate at %dMB x%d)", log_path, MAX_BYTES // (1024 * 1024), BACKUP_COUNT)

    logging.captureWarnings(True)
    warnings.simplefilter("default")
    sys.excepthook = _excepthook
    threading.excepthook = _thread_excepthook

    # Third-party noise down, their errors still reach us.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("aiosqlite").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("telegram.ext.ExtBot").setLevel(logging.WARNING)

    logging.getLogger("telescribe").info("Log level set to %s", level_name)
    return logging.getLogger("telescribe")


def get_logger(name: str) -> logging.Logger:
    """Get a named child logger."""
    return logging.getLogger(f"telescribe.{name}")


def get_log_file_path() -> str | None:
    """Return the current log file path, or None if not set."""
    return _LOG_FILE_PATH


def log_failure(log: logging.Logger, event: str, err: BaseException | None = None, **fields) -> None:
    """Log a searchable failure line: `FAILURE <event> k=v ...`.

    Always includes a traceback when `err` is set. Dashboard ERROR / Failures
    filters match these lines.
    """
    bits = [f"FAILURE {event}"]
    for key, value in fields.items():
        if value is None:
            continue
        bits.append(f"{key}={value}")
    line = " ".join(bits)
    if err is not None:
        log.error("%s err=%s: %s", line, type(err).__name__, err, exc_info=True)
    else:
        log.error("%s", line)


def line_matches_level(line: str, level: str) -> bool:
    """True if a formatted log line belongs to the dashboard level filter."""
    if not level:
        return True
    lv = level.upper()
    aliases = {
        "DEBUG": ("DEBUG",),
        "INFO": ("INFO",),
        "WARN": ("WARN", "WARNING"),
        "WARNING": ("WARN", "WARNING"),
        "ERROR": ("ERROR", "CRITICAL", "FATAL"),
        "FAILURE": ("ERROR", "CRITICAL", "FATAL", "FAILURE"),
        "FAIL": ("ERROR", "CRITICAL", "FATAL", "FAILURE"),
    }
    names = aliases.get(lv, (lv,))
    if lv in ("FAILURE", "FAIL"):
        if "FAILURE" in line:
            return True
    return any(f" - {name} - " in line for name in names)
