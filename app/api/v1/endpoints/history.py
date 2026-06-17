"""
app/api/v1/endpoints/history.py
--------------------------------
GET /api/v1/history  — recent compliance audit runs (from MLflow).
GET /api/v1/trends   — compliance score / gap counts over time.

Reads the MLflow runs logged by mlops/compliance_tracker.py. If MLflow isn't
available the endpoints return empty lists rather than erroring.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Query

logger = logging.getLogger(__name__)
router = APIRouter(tags=["history"])

EXPERIMENT_NAME = "privacy-compliance-audits"


def _search_runs(max_results: int):
    """Return MLflow runs for the audits experiment, newest first, or []."""
    try:
        from mlops.compliance_tracker import ComplianceTracker

        # Configure tracking URI / experiment exactly as the tracker does.
        mlflow = ComplianceTracker()._get_mlflow()
        exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if exp is None:
            return []
        df = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=max_results,
        )
        return df.to_dict(orient="records")
    except Exception as exc:
        logger.warning("MLflow history unavailable: %s", exc)
        return []


def _g(rec: dict, *keys):
    for k in keys:
        if k in rec and rec[k] is not None:
            return rec[k]
    return None


@router.get("/history", summary="Recent compliance audit runs")
async def history(limit: int = Query(default=50, ge=1, le=500)):
    runs = _search_runs(limit)
    out: List[dict] = []
    for r in runs:
        out.append({
            "run_id":           _g(r, "run_id"),
            "policy_name":      _g(r, "tags.policy_name", "params.policy_name"),
            "query":            _g(r, "params.query"),
            "compliance_score": _g(r, "metrics.compliance_score"),
            "overall_risk":     _g(r, "metrics.overall_risk_score"),
            "total_gaps":       _g(r, "metrics.total_gaps"),
            "total_conflicts":  _g(r, "metrics.total_conflicts"),
            "jurisdictions":    _g(r, "tags.jurisdictions", "params.jurisdictions"),
            "start_time":       str(_g(r, "start_time")),
        })
    return {"runs": out, "count": len(out)}


@router.get("/trends", summary="Compliance trends over time")
async def trends(
    limit: int = Query(default=200, ge=1, le=1000),
    policy_name: Optional[str] = Query(default=None),
):
    runs = _search_runs(limit)
    points: List[dict] = []
    for r in runs:
        name = _g(r, "tags.policy_name", "params.policy_name")
        if policy_name and name != policy_name:
            continue
        points.append({
            "start_time":       str(_g(r, "start_time")),
            "policy_name":      name,
            "compliance_score": _g(r, "metrics.compliance_score"),
            "total_gaps":       _g(r, "metrics.total_gaps"),
        })
    # Oldest → newest for charting.
    points.reverse()
    return {"points": points, "count": len(points)}
