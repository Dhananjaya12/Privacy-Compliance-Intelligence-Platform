import os
import time
import logging
from typing import Optional
from opencensus.ext.azure.log_exporter import AzureLogHandler
from opencensus.ext.azure import metrics_exporter
from opencensus.stats import aggregation, measure, stats, view


# ── Logging setup ──────────────────────────────────────────────────────────────

def get_logger() -> logging.Logger:
    """
    Returns a logger that ships to Application Insights via AzureLogHandler.
    Falls back to a plain console logger if the connection string is missing.
    """
    logger = logging.getLogger("pdf_rag_agent")

    if logger.handlers:
        return logger  # already configured

    conn_str = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")

    if conn_str:
        handler = AzureLogHandler(connection_string=conn_str)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        print("[telemetry] Application Insights enabled.")
    else:
        logging.basicConfig(level=logging.INFO)
        logger.warning("APPLICATIONINSIGHTS_CONNECTION_STRING not set — telemetry is console-only.")

    return logger


# ── Query telemetry ────────────────────────────────────────────────────────────

class QueryTelemetry:
    """
    Tracks one query's journey through the RAG pipeline.
    Call start() when a query arrives, finish() when it resolves.
    """

    def __init__(self):
        self.logger = get_logger()
        self._start_time: Optional[float] = None

    def start(self, query: str) -> None:
        self._start_time = time.time()
        self.logger.info(
            "query_received",
            extra={"custom_dimensions": {"query": query}},
        )

    def finish(self, state: dict) -> None:
        """
        Call after the LangGraph run completes.
        state is the final AgentState dict.
        """
        latency_ms = round((time.time() - self._start_time) * 1000) if self._start_time else -1

        jurisdictions   = state.get("jurisdictions") or []
        target_documents = state.get("target_documents") or []
        compliance_score = state.get("compliance_score")
        clarification     = bool(state.get("clarification_needed"))

        dimensions = {
            "latency_ms": latency_ms,
            "jurisdictions": ",".join(jurisdictions),
            "target_documents": ",".join(target_documents),
            "compliance_score": str(compliance_score),
            "gaps_count": str(len(state.get("gaps") or [])),
            "conflicts_count": str(len(state.get("conflicts") or [])),
            "clarification_needed": str(clarification),
        }

        self.logger.info(
            "query_completed",
            extra={"custom_dimensions": dimensions},
        )

        # Print locally so you can see it during development
        print(
            f"[telemetry] latency={latency_ms}ms | jurisdictions={jurisdictions} | "
            f"targets={target_documents} | compliance_score={compliance_score} | "
            f"clarification={clarification}"
        )

    def log_error(self, query: str, error: Exception) -> None:
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