from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

@pytest.fixture()
def mock_agent():
    agent = MagicMock()
    agent.invoke.return_value = {
        "final_answer": "The transformer uses self-attention.",
        "rag_answer": "",
        "retrieved_docs": [],
        "retrieval_score": 0.85,
        "web_results": "",
    }
    return agent


@pytest.fixture()
def client(mock_agent):
    """Create a TestClient with the pipeline pre-loaded in app.state."""
    # Patch lifespan so we don't actually load models
    with patch("app.core.lifespan.lifespan"):
        from app.main import create_app
        app = create_app()

    # Manually inject state
    app.state.ready = True
    app.state.agent = mock_agent
    app.state.pipeline = MagicMock()
    app.state.config = {
        "llm": {
            "retrieval_prompt": "{context}{query}",
            "answer_grader_prompt": "{query}{answer}",
            "final_synthesizer_prompt": "{query}{retrieved_docs}{rag_answer}{web_results}",
            "web_search_results_prompt": "{query}{web_results}",
        }
    }

    return TestClient(app)

def test_liveness(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

def test_readiness_when_ready(client):
    resp = client.get("/api/v1/health/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["pipeline_ready"] is True

def test_query_success(client, mock_agent):
    resp = client.post("/api/v1/query", json={"query": "What is self-attention?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "The transformer uses self-attention."
    assert data["query"] == "What is self-attention?"
    mock_agent.invoke.assert_called_once()


def test_query_empty_string(client):
    resp = client.post("/api/v1/query", json={"query": ""})
    assert resp.status_code == 422  # validation error


def test_query_service_unavailable():
    """If pipeline not ready, should return 503."""
    with patch("app.core.lifespan.lifespan"):
        from app.main import create_app
        app = create_app()
    app.state.ready = False

    c = TestClient(app, raise_server_exceptions=False)
    resp = c.post("/api/v1/query", json={"query": "hello"})
    assert resp.status_code == 503
