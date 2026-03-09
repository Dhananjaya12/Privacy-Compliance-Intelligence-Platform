from __future__ import annotations

import logging
import sys
from typing import Any


class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        import json
        import traceback

        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        if record.exc_info:
            payload["exception"] = traceback.format_exception(*record.exc_info)
        return json.dumps(payload)


def setup_logging(log_level: str = "INFO") -> None:
    """Configure root logger with JSON output to stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)

    # Quieten noisy third-party loggers
    for noisy in ("transformers", "faiss", "urllib3", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
