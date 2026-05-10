"""structlog configuration.

Emits human-readable colored output by default; switch to JSON via
ANPR_LOG_JSON=true for production / log shipping. structlog's contextvars
processor lets the FastAPI middleware bind a request_id that follows the
log line through async context.
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
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> Any:
    return structlog.get_logger(name)
