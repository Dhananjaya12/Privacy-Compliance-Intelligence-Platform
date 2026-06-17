"""
tests/test_2_api.py

Layer 2 — FastAPI wiring tests using TestClient.
No credentials, no pipeline startup — mocks app.state directly.

Run:
    pip install httpx
    python -m pytest tests/test_2_api.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def app():
    """
    Import the FastAPI app without triggering the real lifespan
    (which would try to connect to Azure/Neo4j).
    """
    from contextlib import asynccontextmanager
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    # Override lifespan with a no-op so the real pipeline doesn't start
    @asynccontextmanager
    async def mock_lifespan(app):
        yield

    from app.core.config import get_settings
    from app.api.v1.router import api_router

    settings = get_settings()
    _app = FastAPI(title="test", lifespan=mock_lifespan)
    _app.include_router(api_router, prefix="/api/v1")
    return _app


@pytest.fixture(scope="module")
def client(app):
    return TestClient(app, raise_server_exceptions=False)


def _set_ready(app, agent=None, pipeline=None, config=None, compliance_retriever=None):
    """Inject mock pipeline state onto app.state."""
    app.state.ready    = True
    app.state.agent    = agent or MagicMock()
    app.state.pipeline = pipeline or MagicMock()
    app.state.config   = config or {}
    app.state.compliance_retriever = compliance_retriever


# ── Health endpoints ───────────────────────────────────────────────────────────

def test_liveness_always_200(client):
    """GET /api/v1/health must return 200 regardless of pipeline state."""
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_readiness_when_not_ready(client, app):
    """GET /api/v1/health/ready returns loading when pipeline not initialised."""
    app.state.ready = False
    r = client.get("/api/v1/health/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "loading"
    assert body["pipeline_ready"] is False


def test_readiness_when_ready(client, app):
    """GET /api/v1/health/ready returns ready once pipeline is up."""
    _set_ready(app)
    r = client.get("/api/v1/health/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["pipeline_ready"] is True


# ── Query endpoint ─────────────────────────────────────────────────────────────

def test_query_returns_503_when_pipeline_not_ready(client, app):
    """POST /api/v1/query must return 503 before pipeline is ready."""
    app.state.ready = False
    r = client.post("/api/v1/query", json={"query": "test"})
    assert r.status_code == 503


def test_query_validates_empty_string(client, app):
    """POST /api/v1/query must reject empty query (min_length=1)."""
    _set_ready(app)
    r = client.post("/api/v1/query", json={"query": ""})
    assert r.status_code == 422  # Pydantic validation error


def test_query_returns_answer_shape(client, app):
    """POST /api/v1/query must return QueryResponse schema."""
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {
        "query":                "test question",
        "final_answer":         "This is the answer.",
        "compliance_report":    "This is the answer.",
        "target_documents":     [],
        "clarification_needed": None,
        "jurisdictions":        None,
        "kg_chunks":            [],
    }
    _set_ready(app, agent=mock_agent)

    r = client.post("/api/v1/query", json={"query": "test question"})
    assert r.status_code == 200
    body = r.json()

    # Schema fields from app/models/schemas.py QueryResponse
    assert "query" in body
    assert "answer" in body
    assert "source_chunks" in body
    assert "clarification" in body
    assert "compliance" in body

    assert body["query"]  == "test question"
    assert body["answer"] == "This is the answer."
    assert body["clarification"] is None


def test_query_falls_back_to_compliance_report(client, app):
    """answer must fall back to compliance_report when final_answer is empty."""
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {
        "query":                "q",
        "final_answer":         "",
        "compliance_report":    "# Privacy Compliance Audit Report\n\n...",
        "target_documents":     [],
        "clarification_needed": None,
        "jurisdictions":        None,
        "kg_chunks":            [],
    }
    _set_ready(app, agent=mock_agent)

    r = client.post("/api/v1/query", json={"query": "q"})
    assert r.json()["answer"] == "# Privacy Compliance Audit Report\n\n..."


def test_query_returns_clarification_when_ambiguous(client, app):
    """clarification must be populated when doc_resolver can't pick a target document."""
    mock_agent = MagicMock()
    clarification_msg = "Multiple policies are available and the query doesn't name one."
    mock_agent.invoke.return_value = {
        "query":                "Does the policy cover encryption?",
        "final_answer":         f"# Clarification Needed\n\n{clarification_msg}\n",
        "compliance_report":    f"# Clarification Needed\n\n{clarification_msg}\n",
        "target_documents":     [],
        "clarification_needed": clarification_msg,
        "jurisdictions":        [],
        "kg_chunks":            [],
    }
    _set_ready(app, agent=mock_agent)

    r = client.post("/api/v1/query", json={"query": "Does the policy cover encryption?"})
    assert r.status_code == 200
    body = r.json()
    assert body["clarification"] == clarification_msg
    assert body["compliance"] is None


def test_query_500_on_agent_exception(client, app):
    """POST /api/v1/query must return 500 if agent.invoke raises."""
    mock_agent = MagicMock()
    mock_agent.invoke.side_effect = RuntimeError("LLM timeout")
    _set_ready(app, agent=mock_agent)

    r = client.post("/api/v1/query", json={"query": "q"})
    assert r.status_code == 500


# ── Ingest endpoint ────────────────────────────────────────────────────────────

def test_ingest_returns_503_when_not_ready(client, app):
    app.state.ready = False
    r = client.post("/api/v1/ingest", files=[])
    assert r.status_code in (400, 503)  # 503 if state check fires, 400 if files check fires first


def test_ingest_rejects_non_pdf(client, app):
    """POST /api/v1/ingest must reject files that are not application/pdf."""
    _set_ready(app)
    r = client.post(
        "/api/v1/ingest",
        files={"files": ("evil.txt", b"not a pdf", "text/plain")},
    )
    assert r.status_code == 415


def test_ingest_accepts_pdf_and_returns_schema(client, app):
    """POST /api/v1/ingest with a real-ish PDF must return IngestResponse schema."""
    from app.models.schemas import IngestResponse

    mock_svc = MagicMock()
    mock_svc.ingest_pdfs.return_value = IngestResponse(
        message="Successfully ingested 1 PDF(s).",
        files_processed=1,
        chunks_created=42,
    )

    # Patch RAGService construction inside the endpoint
    with patch("app.api.v1.endpoints.ingest.RAGService", return_value=mock_svc):
        _set_ready(app)
        # Minimal valid PDF magic bytes
        fake_pdf = b"%PDF-1.4 fake content"
        r = client.post(
            "/api/v1/ingest",
            files={"files": ("policy.pdf", fake_pdf, "application/pdf")},
        )

    assert r.status_code == 200
    body = r.json()
    assert "files_processed" in body
    assert "chunks_created" in body
    assert "message" in body


# ── Policy registry endpoint ────────────────────────────────────────────────────

def test_policies_endpoint_returns_registry(client, app):
    """GET /api/v1/policies must return the policy-document registry."""
    mock_retriever = MagicMock()
    mock_retriever.list_policy_documents.return_value = ["google_privacy_policy.pdf"]
    _set_ready(app, compliance_retriever=mock_retriever)

    r = client.get("/api/v1/policies")
    assert r.status_code == 200
    assert r.json() == {"policies": ["google_privacy_policy.pdf"]}


def test_policies_endpoint_503_when_retriever_missing(client, app):
    """GET /api/v1/policies must 503 if compliance_retriever isn't on app.state."""
    _set_ready(app, compliance_retriever=None)
    r = client.get("/api/v1/policies")
    assert r.status_code == 503


# ── Graph explorer endpoint ──────────────────────────────────────────────────────

def test_graph_endpoint_returns_nodes_and_links(client, app):
    """GET /api/v1/graph must return the Neo4j-backed {nodes, links} shape."""
    mock_retriever = MagicMock()
    mock_retriever.get_graph.return_value = {
        "nodes": [{"id": "GDPR Article 17", "label": "GDPR Article 17"}],
        "links": [],
    }
    _set_ready(app, compliance_retriever=mock_retriever)

    r = client.get("/api/v1/graph")
    assert r.status_code == 200
    body = r.json()
    assert "nodes" in body and "links" in body


# ── Regulations endpoints ────────────────────────────────────────────────────────

def test_regulations_endpoint_lists_regulations(client, app):
    """GET /api/v1/regulations must return indexed regulation names."""
    mock_retriever = MagicMock()
    mock_retriever.list_regulation_documents.return_value = ["GDPR", "CCPA"]
    _set_ready(app, compliance_retriever=mock_retriever)

    r = client.get("/api/v1/regulations")
    assert r.status_code == 200
    assert r.json() == {"regulations": ["GDPR", "CCPA"]}


def test_regulation_chunks_endpoint_browses_text(client, app):
    """GET /api/v1/regulations/chunks must return browsable chunk text."""
    mock_retriever = MagicMock()
    mock_retriever.get_regulation_chunks.return_value = [
        {"regulation": "GDPR", "paper_id": "gdpr.pdf", "chunk_id": "gdpr.pdf:0", "content": "..."}
    ]
    _set_ready(app, compliance_retriever=mock_retriever)

    r = client.get("/api/v1/regulations/chunks?regulation=GDPR")
    assert r.status_code == 200
    assert r.json()["chunks"][0]["regulation"] == "GDPR"


# ── History / trends endpoints (MLflow-backed) ────────────────────────────────────

def test_history_endpoint_returns_run_list_shape(client, app):
    """GET /api/v1/history must always return {runs, count}, even without MLflow."""
    r = client.get("/api/v1/history")
    assert r.status_code == 200
    body = r.json()
    assert "runs" in body and "count" in body


def test_trends_endpoint_returns_points_shape(client, app):
    """GET /api/v1/trends must always return {points, count}, even without MLflow."""
    r = client.get("/api/v1/trends")
    assert r.status_code == 200
    body = r.json()
    assert "points" in body and "count" in body


# ── Report PDF endpoint ──────────────────────────────────────────────────────────

def test_report_pdf_rejects_empty_markdown(client, app):
    """POST /api/v1/report/pdf must reject empty/whitespace-only markdown."""
    r = client.post("/api/v1/report/pdf", json={"markdown": "   "})
    assert r.status_code == 400


def test_report_pdf_returns_pdf_for_valid_markdown(client, app):
    """POST /api/v1/report/pdf must render Markdown to a downloadable PDF."""
    r = client.post("/api/v1/report/pdf", json={"markdown": "# Report\n\nHello"})
    if r.status_code == 501:
        pytest.skip("xhtml2pdf/markdown not installed")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"


# ── Route registration sanity ──────────────────────────────────────────────────

def test_all_expected_routes_registered(app):
    """Verify the exact set of routes the refactored main.py registers."""
    paths = {r.path for r in app.routes}
    expected = {
        "/api/v1/health",
        "/api/v1/health/ready",
        "/api/v1/query",
        "/api/v1/ingest",
        "/api/v1/policies",
        "/api/v1/report/pdf",
        "/api/v1/history",
        "/api/v1/trends",
        "/api/v1/graph",
        "/api/v1/regulations",
        "/api/v1/regulations/chunks",
    }
    for path in expected:
        assert path in paths, f"Route {path} not registered in app"


def test_no_legacy_routes_registered(app):
    """Old unprefixed routes from app/routers/ must not appear."""
    paths = {r.path for r in app.routes}
    legacy = {"/health", "/api/query", "/api/ingest"}
    for path in legacy:
        assert path not in paths, f"Legacy route {path} still registered"
