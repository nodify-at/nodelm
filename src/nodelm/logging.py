from __future__ import annotations

import logging
import os
from typing import Any, cast

import structlog


def configure_structured_logging(level: str | None = None) -> None:
    """Configure newline-delimited JSON logs on stderr for future long-running workflows."""

    requested_level = (level or os.environ.get("NODELM_LOG_LEVEL", "INFO")).upper()
    numeric_level = getattr(logging, requested_level, None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"unsupported log level: {requested_level}")
    logging.basicConfig(level=numeric_level, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(sort_keys=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(**context: Any) -> structlog.stdlib.BoundLogger:
    return cast(
        structlog.stdlib.BoundLogger,
        structlog.get_logger("nodelm").bind(**context),
    )
