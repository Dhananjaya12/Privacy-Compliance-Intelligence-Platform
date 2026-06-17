"""
graph.py

Builds the compliance-intelligence LangGraph workflow.

The app is compliance-only. Every query runs the same linear chain:

  doc_resolver → jurisdiction_detector → kg_retriever → gap_analyzer
              → conflict_detector → risk_scorer → remediation → report_generator → END

`doc_resolver` may short-circuit straight to `report_generator` when the query
is ambiguous (no target document resolved) — it emits a clarification message.
"""

from langgraph.graph import END, StateGraph

from agent.compliance_nodes import (
    ComplianceNodes,
    compliance_router,
    doc_resolver_router,
)
from agent.state import AgentState
from pipeline.compliance_retriever import ComplianceRetriever


def build_agent(
    config: dict | None = None,
    compliance_retriever: ComplianceRetriever | None = None,
):
    """
    Build and compile the compliance agent.

    Parameters
    ----------
    config               : Optional config dict (reserved; currently unused by the
                           compliance flow — kept for call-site compatibility).
    compliance_retriever : Optional pre-built ComplianceRetriever. Pass None to
                           let ComplianceNodes create it lazily.
    """
    nodes = ComplianceNodes(retriever=compliance_retriever)

    workflow = StateGraph(AgentState)

    workflow.add_node("doc_resolver",          nodes.doc_resolver_node)
    workflow.add_node("jurisdiction_detector", nodes.jurisdiction_detector_node)
    workflow.add_node("kg_retriever",          nodes.kg_retriever_node)
    workflow.add_node("gap_analyzer",          nodes.gap_analyzer_node)
    workflow.add_node("conflict_detector",     nodes.conflict_detector_node)
    workflow.add_node("risk_scorer",           nodes.risk_scorer_node)
    workflow.add_node("remediation",           nodes.remediation_node)
    workflow.add_node("report_generator",      nodes.report_generator_node)

    workflow.set_entry_point("doc_resolver")

    # doc_resolver → audit chain, or short-circuit to report (clarification).
    workflow.add_conditional_edges(
        "doc_resolver",
        doc_resolver_router,
        {
            "audit":   "jurisdiction_detector",
            "clarify": "report_generator",
        },
    )

    workflow.add_edge("jurisdiction_detector", "kg_retriever")
    workflow.add_edge("kg_retriever",          "gap_analyzer")
    workflow.add_edge("gap_analyzer",          "conflict_detector")
    workflow.add_edge("conflict_detector",     "risk_scorer")
    workflow.add_edge("risk_scorer",           "remediation")
    workflow.add_edge("remediation",           "report_generator")

    workflow.add_conditional_edges(
        "report_generator",
        compliance_router,
        {"done": END},
    )

    return workflow.compile()
