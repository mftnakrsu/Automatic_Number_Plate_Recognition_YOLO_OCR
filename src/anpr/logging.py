"""structlog configuration.

Emits human-readable colored output by default; switch to JSON via
ANPR_LOG_JSON=true for production / log shipping. structlog's contextvars
processor lets the FastAPI middleware bind a request_id that follows the
log line through async context.

Logs go to standard error, so `anpr infer --json` and `anpr watch --json`
can be piped as they are.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configure_logging(*, level: str = "INFO", json: bool = False) -> None:
    """Idempotent: re-applies on every call."""
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        timestamper,
        structlog.processors.StackInfoRenderer(),
    ]

    renderer: Any
    if json:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    level_int = logging.getLevelName(level.upper())
    if not isinstance(level_int, int):
        level_int = logging.INFO

    structlog.configure(
        processors=[
            *shared,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level_int),
        context_class=dict,
        logger_factory=_StderrLogger,
        cache_logger_on_first_use=True,
    )


class _StderrLogger:
    """Prints to whatever `sys.stderr` is at the time, not when the logger was made.

    Loggers are cached on first use, and test runners swap `sys.stderr` per
    command, so binding the stream up front would write to a stale one.
    """

    def __init__(self, *args: Any) -> None:
        pass

    def msg(self, message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    log = debug = info = warn = warning = err = error = critical = exception = fatal = msg


def get_logger(name: str | None = None) -> Any:
    return structlog.get_logger(name)
