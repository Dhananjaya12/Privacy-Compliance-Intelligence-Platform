"""
mlops/compliance_tracker.py

MLflow tracking for compliance audit runs.

Tracks per-audit:
  - Parameters  : query, jurisdictions, policy document name, chunking strategy
  - Metrics     : overall_score, per-regulation risk scores, gap counts,
                  latency, chunk/triple counts
  - Artifacts   : full compliance report (markdown), gaps JSON, risk scores JSON

Local usage (no Azure ML):
  mlflow server --host 0.0.0.0 --port 5000   # start UI
  Then open http://localhost:5000

Azure ML usage (swap tracking URI):
  Set MLFLOW_TRACKING_URI in .env to your Azure ML workspace URI:
  MLFLOW_TRACKING_URI = "azureml://eastus.api.azureml.ms/mlflow/v2.0/subscriptions/.../..."

  Get the URI from:
  Azure Portal → Azure ML workspace → Overview → MLflow tracking URI
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("pdf_rag_agent.mlops")

# ── Severity weights (same as compliance_nodes.py) ────────────────────────────
SEVERITY_WEIGHTS = {"critical": 10.0, "high": 7.0, "medium": 4.0, "low": 1.0}


class ComplianceTracker:
    """
    Wraps MLflow to log one compliance audit run.

    Usage
    -----
    tracker = ComplianceTracker()

    with tracker.start_run(query="Does Google comply with GDPR?", policy="google"):
        # ... run the agent ...
        tracker.log_result(state)          # pass the final AgentState dict
    """

    EXPERIMENT_NAME = "privacy-compliance-audits"

    def __init__(self) -> None:
        self._mlflow   = None      # lazy import
        self._run      = None
        self._run_start: Optional[float] = None

    # ── Lazy MLflow import ────────────────────────────────────────────────────

    def _get_mlflow(self):
        if self._mlflow is None:
            try:
                import mlflow
                self._mlflow = mlflow
                self._setup_tracking()
            except ImportError:
                raise ImportError(
                    "mlflow not installed. Run: pip install mlflow"
                )
        return self._mlflow

    def _setup_tracking(self) -> None:
        mlflow = self._mlflow

        # Tracking URI: env var → local ./mlruns fallback
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info("MLflow tracking → %s", tracking_uri)
        else:
            # Local: stores runs in ./mlruns/
            local_uri = Path("mlruns").resolve().as_uri()
            mlflow.set_tracking_uri(local_uri)
            logger.info("MLflow tracking → local (%s)", local_uri)

        # Create experiment if it doesn't exist
        if not mlflow.get_experiment_by_name(self.EXPERIMENT_NAME):
            mlflow.create_experiment(
                self.EXPERIMENT_NAME,
                tags={"project": "pdf-rag-agent", "phase": "compliance-audit"},
            )
        mlflow.set_experiment(self.EXPERIMENT_NAME)

    # ── Context manager ───────────────────────────────────────────────────────

    def start_run(
        self,
        query:        str,
        policy_name:  str = "unknown",
        run_name:     Optional[str] = None,
    ):
        """
        Context manager — wraps an MLflow run.

        with tracker.start_run(query=q, policy_name="google") as tracker:
            result = agent.invoke({"query": q})
            tracker.log_result(result)
        """
        self._query       = query
        self._policy_name = policy_name
        self._run_name    = run_name or f"{policy_name}_{int(time.time())}"
        return self

    def __enter__(self):
        mlflow = self._get_mlflow()
        self._run       = mlflow.start_run(run_name=self._run_name)
        self._run_start = time.time()

        # Log run parameters immediately
        mlflow.log_params({
            "query":         self._query[:250],   # MLflow param limit
            "policy_name":   self._policy_name,
            "run_name":      self._run_name,
        })
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        mlflow = self._get_mlflow()

        if exc_type is not None:
            mlflow.set_tag("run_status", "failed")
            mlflow.set_tag("error", str(exc_val)[:250])
            logger.error("Compliance run failed: %s", exc_val)
        else:
            mlflow.set_tag("run_status", "completed")

        mlflow.end_run()
        self._run = None
        return False   # don't suppress exceptions

    # ── Logging helpers ───────────────────────────────────────────────────────

    def log_result(self, state: Dict[str, Any]) -> None:
        """
        Log all compliance metrics and artifacts from a completed AgentState.
        Call this inside the `with tracker.start_run(...)` block after
        agent.invoke() returns.
        """
        mlflow = self._get_mlflow()
        if self._run is None:
            raise RuntimeError("log_result() called outside of start_run() context.")

        latency_ms = round((time.time() - self._run_start) * 1000)

        jurisdictions   = state.get("jurisdictions", [])
        overall_score   = state.get("overall_score")
        compliance_score = state.get("compliance_score")
        per_reg_comp    = state.get("per_reg_compliance", {})
        risk_scores     = state.get("risk_scores", {})
        gaps            = state.get("gaps", [])
        conflicts       = state.get("conflicts", [])
        financial_exp   = state.get("financial_exposure", "")
        report          = state.get("compliance_report", "")
        kg_chunks       = state.get("kg_chunks", [])
        kg_triples      = state.get("kg_triples", [])

        # ── Parameters ───────────────────────────────────────────────────────
        mlflow.log_params({
            "jurisdictions":      ",".join(jurisdictions) if jurisdictions else "none",
            "num_jurisdictions":  len(jurisdictions),
        })

        # ── Core metrics ─────────────────────────────────────────────────────
        metrics: Dict[str, float] = {
            "latency_ms":       float(latency_ms),
            "compliance_score": float(compliance_score) if compliance_score is not None else 0.0,
            "overall_risk_score": float(overall_score) if overall_score is not None else 0.0,
            "total_gaps":       float(len(gaps)),
            "total_conflicts":  float(len(conflicts)),
            "kg_chunks_used":   float(len(kg_chunks)),
            "kg_triples_used":  float(len(kg_triples)),
        }

        # Per-regulation risk + compliance scores
        for reg, score in risk_scores.items():
            metrics[f"risk_{reg.lower()}"] = float(score)
        for reg, score in per_reg_comp.items():
            metrics[f"compliance_{reg.lower()}"] = float(score)

        # Gap severity breakdown
        for severity in ("critical", "high", "medium", "low"):
            count = sum(1 for g in gaps if g.get("severity") == severity)
            metrics[f"gaps_{severity}"] = float(count)

        # Per-regulation gap counts
        reg_gap_counts: Dict[str, int] = {}
        for gap in gaps:
            reg = gap.get("regulation", "unknown")
            reg_gap_counts[reg] = reg_gap_counts.get(reg, 0) + 1
        for reg, count in reg_gap_counts.items():
            metrics[f"gaps_{reg.lower()}"] = float(count)

        mlflow.log_metrics(metrics)

        # ── Tags ─────────────────────────────────────────────────────────────
        mlflow.set_tags({
            "policy_name":        self._policy_name,
            "jurisdictions":      ",".join(jurisdictions),
            "has_critical_gaps":  str(any(g.get("severity") == "critical" for g in gaps)),
            "financial_exposure": financial_exp[:250] if financial_exp else "none",
        })

        # ── Artifacts ────────────────────────────────────────────────────────
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # 1. Full compliance report markdown
            if report:
                report_path = tmp / "compliance_report.md"
                report_path.write_text(report, encoding="utf-8")
                mlflow.log_artifact(str(report_path), artifact_path="report")

            # 2. Gaps JSON
            if gaps:
                gaps_path = tmp / "gaps.json"
                gaps_path.write_text(
                    json.dumps(gaps, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                mlflow.log_artifact(str(gaps_path), artifact_path="data")

            # 3. Risk scores JSON
            risk_path = tmp / "risk_scores.json"
            risk_path.write_text(
                json.dumps({
                    "overall_score":     overall_score,
                    "per_regulation":    risk_scores,
                    "financial_exposure": financial_exp,
                    "jurisdictions":     jurisdictions,
                }, indent=2),
                encoding="utf-8",
            )
            mlflow.log_artifact(str(risk_path), artifact_path="data")

            # 4. Summary text (human-readable one-liner per gap)
            if gaps:
                summary_lines = [
                    f"[{g.get('severity','?').upper():8s}] "
                    f"{g.get('regulation','?'):6s} | "
                    f"{g.get('ob_type','?'):20s} | "
                    f"{g.get('description','')[:100]}"
                    for g in sorted(gaps, key=lambda x: SEVERITY_WEIGHTS.get(x.get("severity","low"), 0), reverse=True)
                ]
                summary_path = tmp / "gap_summary.txt"
                summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
                mlflow.log_artifact(str(summary_path), artifact_path="report")

        logger.info(
            "[mlflow] run logged | policy=%s | score=%.2f | gaps=%d | latency=%dms",
            self._policy_name, overall_score or 0.0, len(gaps), latency_ms,
        )

    def log_ingestion_run(
        self,
        input_dir:     str,
        strategy:      str,
        total_files:   int,
        total_chunks:  int,
        uploaded:      int,
        errors:        int,
        latency_ms:    int,
    ) -> None:
        """
        Log a Spark ingestion run as a separate MLflow experiment.
        Call this from spark_ingestion.py after run_ingestion() completes.
        """
        mlflow = self._get_mlflow()
        mlflow.set_experiment("pdf-rag-ingestion")

        with mlflow.start_run(run_name=f"ingest_{int(time.time())}"):
            mlflow.log_params({
                "input_dir":       input_dir,
                "chunk_strategy":  strategy,
            })
            mlflow.log_metrics({
                "total_files":    float(total_files),
                "total_chunks":   float(total_chunks),
                "uploaded":       float(uploaded),
                "errors":         float(errors),
                "latency_ms":     float(latency_ms),
                "success_rate":   float(uploaded / max(total_chunks, 1)),
            })
            mlflow.set_tag("status", "ok" if errors == 0 else "partial")


# ── Convenience wrapper for rag_service.py ────────────────────────────────────

_tracker: Optional[ComplianceTracker] = None


def get_tracker() -> ComplianceTracker:
    """Singleton — one tracker instance per process."""
    global _tracker
    if _tracker is None:
        _tracker = ComplianceTracker()
    return _tracker