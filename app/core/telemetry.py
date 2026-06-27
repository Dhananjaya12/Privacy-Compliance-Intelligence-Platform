from __future__ import annotations

import logging
import os
import sys
import time
from typing import Optional

try:
    from opencensus.ext.azure.log_exporter import AzureLogHandler
except Exception:  # pragma: no cover - dependency may be optional in local dev
    AzureLogHandler = None  # type: ignore[assignment]


APP_INSIGHTS_HANDLER_MARKER = "_privacyguard_appinsights_handler"


def _visible_print(message: str) -> None:
    """Print immediately so Colab/VS Code terminals show progress."""
    print(message, flush=True)


def get_logger() -> logging.Logger:
    """
    Return the telemetry logger.

    Always keeps console logging visible. If APPLICATIONINSIGHTS_CONNECTION_STRING
    is set and opencensus-ext-azure is installed, also attaches AzureLogHandler.
    """
    logger = logging.getLogger("pdf_rag_agent")
    logger.setLevel(logging.INFO)
    logger.propagate = True

    # Add a simple console handler if this named logger has no direct console handler.
    has_stream_handler = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    if not has_stream_handler:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(console)

    conn_str = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    has_ai_handler = any(getattr(h, APP_INSIGHTS_HANDLER_MARKER, False) for h in logger.handlers)

    if conn_str and not has_ai_handler:
        if AzureLogHandler is None:
            _visible_print(
                "[telemetry] APPLICATIONINSIGHTS_CONNECTION_STRING is set, "
                "but opencensus-ext-azure is not installed. Run: pip install -r requirements.txt"
            )
        else:
            handler = AzureLogHandler(connection_string=conn_str)
            setattr(handler, APP_INSIGHTS_HANDLER_MARKER, True)
            handler.setLevel(logging.INFO)
            logger.addHandler(handler)
            _visible_print("[telemetry] Application Insights enabled and AzureLogHandler attached.")
    elif not conn_str:
        _visible_print("[telemetry] APPLICATIONINSIGHTS_CONNECTION_STRING not set; telemetry is console-only.")

    return logger


class QueryTelemetry:
    """Tracks one query through the compliance pipeline."""

    def __init__(self) -> None:
        self.logger = get_logger()
        self._start_time: Optional[float] = None

    def start(self, query: str) -> None:
        self._start_time = time.time()
        _visible_print(f"[telemetry] query_received | query={query[:160]}")
        self.logger.info(
            "query_received",
            extra={"custom_dimensions": {"query": query}},
        )

    def node(self, name: str, **dimensions: object) -> None:
        """Log a visible node checkpoint to console and Application Insights."""
        clean_dimensions = {k: str(v) for k, v in dimensions.items()}
        _visible_print(f"[telemetry] {name} | {clean_dimensions}")
        self.logger.info(
            name,
            extra={"custom_dimensions": clean_dimensions},
        )

    def finish(self, state: dict) -> None:
        latency_ms = round((time.time() - self._start_time) * 1000) if self._start_time else -1

        jurisdictions = state.get("jurisdictions") or []
        target_documents = state.get("target_documents") or []
        compliance_score = state.get("compliance_score")
        clarification = bool(state.get("clarification_needed"))
        gaps_count = len(state.get("gaps") or [])
        conflicts_count = len(state.get("conflicts") or [])

        dimensions = {
            "latency_ms": str(latency_ms),
            "jurisdictions": ",".join(jurisdictions),
            "target_documents": ",".join(target_documents),
            "compliance_score": str(compliance_score),
            "gaps_count": str(gaps_count),
            "conflicts_count": str(conflicts_count),
            "clarification_needed": str(clarification),
        }

        _visible_print(
            "[telemetry] query_completed | "
            f"latency_ms={latency_ms} | jurisdictions={jurisdictions} | "
            f"targets={target_documents} | compliance_score={compliance_score} | "
            f"gaps={gaps_count} | conflicts={conflicts_count} | clarification={clarification}"
        )
        self.logger.info(
            "query_completed",
            extra={"custom_dimensions": dimensions},
        )

        # Force Application Insights handler buffers to flush quickly for demos.
        for handler in self.logger.handlers:
            try:
                handler.flush()
            except Exception:
                pass

    def log_error(self, query: str, error: Exception) -> None:
        _visible_print(f"[telemetry] query_error | {type(error).__name__}: {error}")
        self.logger.error(
            "query_error",
            extra={
                "custom_dimensions": {
                    "query": query,
                    "error": str(error),
                    "error_type": type(error).__name__,
                }
            },
        )