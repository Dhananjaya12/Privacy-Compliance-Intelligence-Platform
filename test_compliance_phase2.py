"""
test_compliance_phase2.py

Smoke test for:
  1. ComplianceRetriever.query()   — Azure Search + Neo4j hybrid
  2. Phase 2 nodes run directly    — jurisdiction_detector → kg_retriever
                                     → gap_analyzer → risk_scorer
  3. Full LangGraph graph run      — build_agent() + agent.invoke()
  4. Import / wiring check

Run:
  python test_compliance_phase2.py [import|retriever|nodes|graph|all]
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

TEST_QUESTIONS = [
    "Does the policy address breach notification timelines?",
    "What rights does the policy grant to data subjects for deletion of their data?",
    "Is there evidence of documented lawful basis for processing personal data?",
    "How does the policy handle data retention and deletion schedules?",
    "Are encryption requirements for data at rest and in transit addressed?",
]

FULL_GRAPH_QUESTION = (
    "Does the policy comply with GDPR Article 17 right to erasure "
    "and HIPAA breach notification?"
)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — ComplianceRetriever standalone
# ─────────────────────────────────────────────────────────────────────────────

def run_retriever_test():
    from pipeline.compliance_retriever import ComplianceRetriever

    print("\n" + "="*60)
    print("  STAGE 1 — ComplianceRetriever hybrid query test")
    print("="*60)

    retriever = ComplianceRetriever()

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[Q{i}] {question}")
        print("-"*55)
        try:
            result = retriever.query(question)
            print(f"  Jurisdictions: {result['jurisdictions']}")
            print(f"  Chunks used:   {result['raw_chunks_used']}")
            print(f"  Triples used:  {result['raw_triples_used']}")
            print(f"  Latency:       {result['latency_ms']}ms")
            print(f"  Gaps found:    {len(result['gaps'])}")
            print(f"\n  Answer:\n{result['answer'][:400]}...")
            if result["gaps"]:
                print(f"\n  Top gap: {result['gaps'][0]}")
        except Exception as exc:
            print(f"  ❌ FAILED: {exc}")

    retriever.close()
    print("\n✅ Stage 1 complete.")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — Phase 2 nodes run directly (no full graph init)
# ─────────────────────────────────────────────────────────────────────────────

def _blank_state(query: str) -> dict:
    """Minimal AgentState-compatible dict for direct node testing."""
    return {
        "query":              query,
        "retrieved_docs":     [],
        "retrieval_score":    0.0,
        "rag_answer":         "",
        "answer_score":       0,
        "web_results":        "",
        "final_answer":       "",
        "decision":           "internal",
        "jurisdictions":      [],
        "kg_chunks":          [],
        "kg_triples":         [],
        "obligations":        [],
        "gaps":               [],
        "conflicts":          [],
        "risk_scores":        {},
        "overall_score":      0.0,
        "financial_exposure": "",
        "compliance_report":  "",
    }


def run_nodes_test():
    from pipeline.compliance_retriever import ComplianceRetriever
    from agent.compliance_nodes import ComplianceNodes

    print("\n" + "="*60)
    print("  STAGE 2 — Phase 2 nodes (direct, no graph)")
    print("="*60)

    retriever = ComplianceRetriever()
    nodes     = ComplianceNodes(retriever=retriever)
    state     = _blank_state(FULL_GRAPH_QUESTION)

    print(f"\nQuery: {FULL_GRAPH_QUESTION}\n")

    print("→ jurisdiction_detector...")
    state = nodes.jurisdiction_detector_node(state)
    print(f"  Jurisdictions: {state['jurisdictions']}")

    print("→ kg_retriever...")
    state = nodes.kg_retriever_node(state)
    print(f"  Chunks: {len(state['kg_chunks'])} | "
          f"Triples: {len(state['kg_triples'])} | "
          f"Obligations: {len(state['obligations'])}")

    # Validate obligations have expected keys
    for ob in state["obligations"][:3]:
        assert all(k in ob for k in ("id", "regulation", "text", "type")), \
            f"Obligation missing keys: {ob}"

    print("→ gap_analyzer...")
    state = nodes.gap_analyzer_node(state)
    print(f"  Gaps: {len(state['gaps'])} | Conflicts: {len(state['conflicts'])}")

    # Validate gap structure
    for g in state["gaps"][:3]:
        assert all(k in g for k in ("obligation_id", "regulation", "severity")), \
            f"Gap missing keys: {g}"

    print("→ risk_scorer...")
    state = nodes.risk_scorer_node(state)

    # Validate scoring math: no regulation should be scored with 0 obligations
    for reg, score in state["risk_scores"].items():
        ob_count = sum(1 for ob in state["obligations"] if ob["regulation"] == reg)
        assert ob_count > 0, \
            f"[FAIL] {reg} has score {score} but 0 obligations — denominator bug"

    print(f"  Risk scores:  {state['risk_scores']}")
    print(f"  Overall:      {state['overall_score']:.2f}/10")
    print(f"  Exposure:\n{state['financial_exposure']}")

    # Validate no None score leaks into overall_score
    assert isinstance(state["overall_score"], float), "overall_score must be float"
    assert 0.0 <= state["overall_score"] <= 10.0, \
        f"overall_score out of range: {state['overall_score']}"

    print("\n── COMPLIANCE REPORT ─────────────────────────────────────\n")
    print(state["compliance_report"])

    out = Path("data/compliance_report_test.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(state["compliance_report"], encoding="utf-8")
    print(f"\n✅ Stage 2 complete. Report saved to {out}")

    retriever.close()


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — Full LangGraph graph invocation via build_agent()
# ─────────────────────────────────────────────────────────────────────────────

def run_graph_test():
    """
    Exercises the actual compiled graph including phase_router logic.
    Uses a stub Phase 1 retriever/generator so we don't need the
    full vectorstore loaded — compliance queries bypass Phase 1 anyway.
    """
    from unittest.mock import MagicMock
    from pipeline.compliance_retriever import ComplianceRetriever
    from pipeline.generator import LLMGenerator
    from agent.graph import build_agent

    print("\n" + "="*60)
    print("  STAGE 3 — Full build_agent() + graph.invoke()")
    print("="*60)

    # Stub Phase 1 components — compliance queries never touch them
    mock_retriever = MagicMock()
    mock_retriever.vectorstore.similarity_search_with_score.return_value = []
    mock_retriever.search_kwargs = {"k": 5}

    mock_generator = MagicMock(spec=LLMGenerator)
    mock_generator.generate_answer.return_value = {"generated_answer": "stub"}

    config = {
        "llm": {
            "retrieval_prompt":           "{query}{context}{retrieved_docs}{rag_answer}{web_results}",
            "web_search_results_prompt":  "{query}{context}{retrieved_docs}{rag_answer}{web_results}",
            "final_synthesizer_prompt":   "{query}{context}{retrieved_docs}{rag_answer}{web_results}",
            "answer_grader_prompt":       "{query}{answer}",
        }
    }

    compliance_retriever = ComplianceRetriever()
    agent = build_agent(
        retriever=mock_retriever,
        generator=mock_generator,
        config=config,
        compliance_retriever=compliance_retriever,
    )

    print(f"\nInvoking graph with: {FULL_GRAPH_QUESTION}\n")
    result = agent.invoke({"query": FULL_GRAPH_QUESTION})

    # Must have gone through Phase 2 (phase_router → jurisdiction_detector)
    assert result.get("jurisdictions"), \
        "phase_router failed to route to compliance chain — jurisdictions empty"
    assert result.get("compliance_report"), \
        "risk_scorer did not produce a compliance_report"
    assert "final_answer" in result and result["final_answer"], \
        "final_answer not set by risk_scorer"

    print(f"  Jurisdictions:  {result['jurisdictions']}")
    print(f"  Overall score:  {result.get('overall_score', 'N/A')}")
    print(f"  Gaps:           {len(result.get('gaps', []))}")
    print(f"  Report length:  {len(result.get('compliance_report', ''))} chars")

    compliance_retriever.close()
    print("\n✅ Stage 3 complete.")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — Import / wiring check (no env needed)
# ─────────────────────────────────────────────────────────────────────────────

def run_import_check():
    print("\n" + "="*60)
    print("  STAGE 4 — Import / wiring check")
    print("="*60)

    checks = [
        ("agent.state",              "AgentState"),
        ("agent.compliance_nodes",   "ComplianceNodes, compliance_router"),
        ("agent.graph",              "build_agent, phase_router"),
        ("pipeline.compliance_retriever", "ComplianceRetriever"),
        ("app.models.schemas",       "QueryResponse, ComplianceDetail"),
        ("app.services.rag_service", "RAGService"),
    ]

    all_ok = True
    for module, names in checks:
        try:
            exec(f"from {module} import {names}")
            print(f"  ✅ {module} ({names})")
        except Exception as e:
            print(f"  ❌ {module}: {e}")
            all_ok = False

    # Verify ComplianceDetail is in QueryResponse
    try:
        from app.models.schemas import QueryResponse, ComplianceDetail
        import inspect
        fields = QueryResponse.model_fields
        assert "compliance" in fields, "QueryResponse missing 'compliance' field"
        print("  ✅ QueryResponse.compliance field present")
    except Exception as e:
        print(f"  ❌ QueryResponse.compliance check: {e}")
        all_ok = False

    if all_ok:
        print("\n✅ Stage 4 complete — all imports OK.")
    else:
        print("\n⚠️  Stage 4 complete — some imports failed.")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    stage = sys.argv[1] if len(sys.argv) > 1 else "all"

    if stage in ("import", "all"):
        run_import_check()

    if stage in ("retriever", "all"):
        run_retriever_test()

    if stage in ("nodes", "all"):
        run_nodes_test()

    if stage in ("graph", "all"):
        run_graph_test()