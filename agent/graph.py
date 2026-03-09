from langgraph.graph import END, StateGraph

from agent.nodes import (
    AgentNodes,
    answer_router,
    classifier_router,
    retrieval_router,
)
from agent.state import AgentState
from agent.websearch import WebSearcher
from pipeline.generator import LLMGenerator


def build_agent(retriever, generator: LLMGenerator, config):
    nodes = AgentNodes(
        retriever=retriever,
        generator=generator,
        web_searcher=WebSearcher(),
        config = config
    )

    workflow = StateGraph(AgentState)

    workflow.add_node("classifier", nodes.classifier_node)
    workflow.add_node("retriever", nodes.retriever_node)
    workflow.add_node("retrieval_grader", nodes.retrieval_grader_node)
    workflow.add_node("answer_grader", nodes.answer_grader_node)
    workflow.add_node("web_search", nodes.web_search_node)

    workflow.add_node("rag_generator", nodes.generator_node)
    workflow.add_node("web_generator", nodes.generator_node)

    workflow.set_entry_point("classifier")

    workflow.add_conditional_edges(
        "classifier",
        classifier_router,
        {
            "fresh": "web_search",
            "internal": "retriever",
        },
    )

    workflow.add_edge("retriever", "retrieval_grader")

    workflow.add_conditional_edges(
        "retrieval_grader",
        retrieval_router,
        {
            "sufficient": "rag_generator",
            "insufficient": "web_search",
        },
    )

    workflow.add_edge("rag_generator", "answer_grader")

    workflow.add_conditional_edges(
        "answer_grader",
        answer_router,
        {
            "sufficient": END,
            "insufficient": "web_search",
        },
    )

    workflow.add_edge("web_search", "web_generator")
    workflow.add_edge("web_generator", END)

    return workflow.compile()