"""
tests/test_e2e.py

End-to-end functional tests for the Privacy Compliance Intelligence Platform.

Tests four real scenarios in order:
  1. Azure AI Search  — can retrieve policy chunks for a compliance query
  2. Neo4j            — triples exist and return correctly for GDPR/HIPAA/CCPA/NIST
  3. Compliance chain — the full 8-node compliance graph produces a real audit report
  4. FastAPI server   — health, query, and ingest endpoints respond correctly

All tests require a valid .env. Run:
    python -m pytest tests/test_e2e.py -v -s

Individual scenario:
    python -m pytest tests/test_e2e.py::test_azure_search_retrieves_chunks -v -s
"""

import os
import pytest
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# 1. AZURE AI SEARCH
# Does it return relevant chunks for a real compliance question?
# ─────────────────────────────────────────────────────────────────────────────

def test_azure_search_retrieves_chunks():
    """
    Azure AI Search must return at least 1 chunk for a compliance query.
    If this fails → AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY, or index name is wrong,
    or the index is empty (run scripts/ingest_policy.py first).
    """
    from pipeline.compliance_retriever import ComplianceRetriever

    retriever = ComplianceRetriever()
    chunks = retriever._azure_search("breach notification GDPR", k=5)
    retriever.close()

    print(f"\n  Chunks returned: {len(chunks)}")
    if chunks:
        print(f"  First chunk preview: {chunks[0].get('page_content', '')[:120]}...")

    assert len(chunks) > 0, (
        "Azure AI Search returned 0 chunks. "
        "Either the index is empty or credentials are wrong. "
        "Run: python scripts/ingest_policy.py"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. NEO4J — triples return correctly
# This was the known bug: NEO4J_DATABASE=neo4j instead of 6a29e6b9
# ─────────────────────────────────────────────────────────────────────────────

def test_neo4j_returns_triples():
    """
    Neo4j must return KG triples. If this returns 0, either NEO4J_DATABASE is
    wrong (should be 'neo4j' for AuraDB) or the KG hasn't been built yet —
    run ComplianceKGBuilder().build_from_pdfs().
    """
    from pipeline.compliance_retriever import ComplianceRetriever

    retriever = ComplianceRetriever()

    # Use known entities that should exist in the KG
    triples = retriever._neo4j_triples(["GDPR", "breach notification", "data subject"])
    retriever.close()

    print(f"\n  Triples returned: {len(triples)}")
    if triples:
        t = triples[0]
        print(f"  Example: {t.get('source')} → [{t.get('relation')}] → {t.get('target')}")

    assert len(triples) > 0, (
        "Neo4j returned 0 triples. "
        f"NEO4J_DATABASE is currently '{os.getenv('NEO4J_DATABASE')}' — "
        "confirm it points at the populated AuraDB database and that the "
        "knowledge graph has been built (ComplianceKGBuilder().build_from_pdfs())."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. COMPLIANCE CHAIN — full 8-node graph produces a real audit report
# doc_resolver → jurisdiction_detector → kg_retriever → gap_analyzer
#   → conflict_detector → risk_scorer → remediation → report_generator
# ─────────────────────────────────────────────────────────────────────────────

def test_phase2_compliance_chain_produces_report():
    """
    The full 8-node compliance chain must produce a report with:
    - A resolved target document (or a clarification, in which case we skip)
    - At least 1 jurisdiction detected
    - A compliance score between 0 and 100
    - A non-empty markdown report in final_answer
    - The report saved to data/test_report.md for manual inspection
    """
    from pipeline.compliance_retriever import ComplianceRetriever
    from agent.compliance_nodes import ComplianceNodes

    retriever = ComplianceRetriever()
    nodes = ComplianceNodes(retriever=retriever)

    state = {
        "query":                "Does the policy comply with GDPR Article 17 right to erasure and HIPAA breach notification requirements?",
        "final_answer":         "",
        "policy_document":      None,
        "target_documents":     [],
        "clarification_needed": None,
        "jurisdictions":        [],
        "kg_chunks":            [],
        "kg_triples":           [],
        "obligations":          [],
        "per_doc_results":      {},
        "gaps":                 [],
        "conflicts":            [],
        "risk_scores":          {},
        "per_reg_compliance":   {},
        "overall_score":        0.0,
        "compliance_score":     0.0,
        "financial_exposure":   "",
        "remediations":         [],
        "compliance_report":    "",
    }

    # Node 0 — resolve which policy document(s) to audit
    state = nodes.doc_resolver_node(state)
    print(f"\n  [0] Targets: {state['target_documents']} | clarification: {state['clarification_needed']}")
    if state["clarification_needed"]:
        pytest.skip(f"No unambiguous policy document to audit: {state['clarification_needed']}")

    # Node 1 — jurisdiction detection
    state = nodes.jurisdiction_detector_node(state)
    print(f"  [1] Jurisdictions detected: {state['jurisdictions']}")
    assert len(state["jurisdictions"]) >= 1, "jurisdiction_detector returned nothing"
    assert any(j in state["jurisdictions"] for j in ["GDPR", "HIPAA"]), \
        f"Expected GDPR or HIPAA, got: {state['jurisdictions']}"

    # Node 2 — KG retrieval (per document)
    state = nodes.kg_retriever_node(state)
    print(f"  [2] Chunks: {len(state['kg_chunks'])} | Triples: {len(state['kg_triples'])} | Obligations: {len(state['obligations'])}")
    assert isinstance(state["kg_chunks"], list)
    assert isinstance(state["obligations"], list)

    # Node 3 — gap analysis (full, untruncated policy text)
    state = nodes.gap_analyzer_node(state)
    print(f"  [3] Gaps found: {len(state['gaps'])}")
    if state["gaps"]:
        g = state["gaps"][0]
        print(f"      Example gap: [{g.get('severity','?').upper()}] {g.get('description','')[:100]}")
    assert isinstance(state["gaps"], list)

    # Node 4 — cross-regulation conflict detection (KG, value-level)
    state = nodes.conflict_detector_node(state)
    print(f"  [4] Conflicts found: {len(state['conflicts'])}")
    assert isinstance(state["conflicts"], list)

    # Node 5 — risk scoring -> 0-100 compliance
    state = nodes.risk_scorer_node(state)
    print(f"  [5] Compliance score: {state['compliance_score']:.1f}/100")
    print(f"      Per-regulation compliance: {state['per_reg_compliance']}")
    assert 0.0 <= state["compliance_score"] <= 100.0

    # Node 6 — remediation recommendations
    state = nodes.remediation_node(state)
    print(f"  [6] Remediations: {len(state['remediations'])}")
    assert isinstance(state["remediations"], list)

    # Node 7 — render the final markdown report
    state = nodes.report_generator_node(state)
    assert len(state["compliance_report"]) > 200, "Report is suspiciously short"
    assert state["final_answer"] == state["compliance_report"], \
        "final_answer and compliance_report are out of sync"
    assert "# Privacy Compliance Audit Report" in state["compliance_report"]

    # Save for manual inspection
    out = Path("data/test_report.md")
    out.parent.mkdir(exist_ok=True)
    out.write_text(state["compliance_report"], encoding="utf-8")
    print(f"\n  Full report saved to: {out}")
    print(f"\n--- REPORT PREVIEW ---\n{state['compliance_report'][:600]}\n...")

    retriever.close()


# ─────────────────────────────────────────────────────────────────────────────
# 4. FASTAPI SERVER — health, query, ingest
# Start the server first: uvicorn app.main:app --port 8000
# ─────────────────────────────────────────────────────────────────────────────

SERVER_URL = os.getenv("TEST_SERVER_URL", "http://localhost:8000")


def _server_is_up() -> bool:
    try:
        import requests
        r = requests.get(f"{SERVER_URL}/api/v1/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="module")
def server_ready():
    if not _server_is_up():
        pytest.skip(
            f"Server not running at {SERVER_URL}. "
            "Start it with: uvicorn app.main:app --port 8000"
        )


def test_health_endpoint_returns_ok(server_ready):
    """GET /api/v1/health must return {status: ok}."""
    import requests
    r = requests.get(f"{SERVER_URL}/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    print(f"\n  Health: {r.json()}")


def test_readiness_endpoint(server_ready):
    """GET /api/v1/health/ready must return pipeline_ready=true once loaded."""
    import requests
    r = requests.get(f"{SERVER_URL}/api/v1/health/ready")
    assert r.status_code == 200
    body = r.json()
    print(f"\n  Readiness: {body}")
    assert "pipeline_ready" in body
    # If pipeline is still loading this will be False — that's OK,
    # the test just tells you the current state
    if not body["pipeline_ready"]:
        pytest.skip("Pipeline still loading — re-run after startup completes")


def test_compliance_query_via_api(server_ready):
    """
    POST /api/v1/query with a compliance question must return a non-empty answer.
    Tests the full path: HTTP → FastAPI → agent.invoke() → final_answer.
    """
    import requests
    payload = {"query": "Does the policy address GDPR breach notification requirements?"}
    r = requests.post(f"{SERVER_URL}/api/v1/query", json=payload, timeout=120)

    print(f"\n  Status: {r.status_code}")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text[:200]}"

    body = r.json()
    print(f"  Answer preview: {body.get('answer','')[:300]}")
    print(f"  Clarification: {body.get('clarification')}")
    print(f"  Compliance: {body.get('compliance')}")

    assert len(body.get("answer", "")) > 50, "Answer is empty or too short"
    assert "query" in body
    assert "source_chunks" in body


def test_ingest_endpoint_accepts_pdf(server_ready):
    """
    POST /api/v1/ingest with a real PDF must return files_processed=1.
    Uses a minimal valid PDF so we don't need an actual document.
    """
    import requests

    # Minimal real PDF (from RFC — just enough to not be rejected by pymupdf)
    minimal_pdf = (
        b"%PDF-1.0\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f \n"
        b"trailer<</Size 4/Root 1 0 R>>\n"
        b"startxref\n0\n%%EOF"
    )

    r = requests.post(
        f"{SERVER_URL}/api/v1/ingest",
        files={"files": ("test_policy.pdf", minimal_pdf, "application/pdf")},
        timeout=120,
    )
    print(f"\n  Ingest status: {r.status_code}")
    print(f"  Response: {r.json()}")

    assert r.status_code == 200
    body = r.json()
    assert body["files_processed"] == 1
    assert "chunks_created" in body
    assert "message" in body
