from __future__ import annotations

from typing import Any

from agent.state import AgentState
from agent.websearch import WebSearcher
from pipeline.generator import LLMGenerator

RETRIEVAL_SCORE_THRESHOLD: float = 0.3   # Below this → fall back to web
ANSWER_SCORE_THRESHOLD: int = 3          # Below this → fall back to web


class AgentNodes:

    def __init__(
        self,
        retriever,
        generator: LLMGenerator,
        config: Dict,
        web_searcher: WebSearcher | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.config = config
        self.web_searcher = web_searcher or WebSearcher()

    def classifier_node(self, state: AgentState) -> AgentState:
        query = state["query"]

        # Simple heuristic: keywords that signal recency requirements
        fresh_signals = {"latest", "recent", "today", "current", "2024", "2025", "news"}
        if any(word in query for word in fresh_signals):
            state["decision"] = "fresh"
        else:
            state["decision"] = "internal"

        return state

    def retriever_node(self, state: AgentState) -> AgentState:
        docs_with_scores = self.retriever.vectorstore.similarity_search_with_score(
        state["query"],
        k=self.retriever.search_kwargs.get("k", 5),
        )

        docs = []
        scores = []

        for doc, score in docs_with_scores:
            docs.append(doc.page_content)
            scores.append(score)

        state["retrieved_docs"] = docs

        # FAISS returns L2 distance (lower = better)
        # Convert to similarity proxy
        if scores:
            best_score = min(scores)
            state["retrieval_score"] = 1 / (1 + best_score)
        else:
            state["retrieval_score"] = 0.0

        return state

    def retrieval_grader_node(self, state: AgentState) -> AgentState:
        score = state.get("retrieval_score", 0.0)
        docs = state.get("retrieved_docs", [])

        if not docs or score < RETRIEVAL_SCORE_THRESHOLD:
            state["decision"] = "insufficient"
        else:
            state["decision"] = "sufficient"

        return state

    def answer_grader_node(self, state: AgentState) -> AgentState:
        prompt_template = self.config["llm"]["answer_grader_prompt"]

        prompt = prompt_template.format(
            query=state["query"],
            answer=state.get("rag_answer", "")
        )

        raw_score = self.generator.generate_answer(
            docs=[],
            prompt=prompt,
            question=state["query"],
        )['generated_answer']

        try:
            state["answer_score"] = int(raw_score)
        except:
            state["answer_score"] = 0

        state["decision"] = (
            "sufficient"
            if state["answer_score"] >= ANSWER_SCORE_THRESHOLD
            else "insufficient"
        )

        return state

    def web_search_node(self, state: AgentState) -> AgentState:
        state["web_results"] = self.web_searcher.search(state["query"])
        return state

    def generator_node(self, state: AgentState) -> AgentState:
        query = state["query"]
        retrieved_docs = "\n\n---\n\n".join(state.get("retrieved_docs", []))
        rag_answer = state.get("rag_answer", "")
        web_results = state.get("web_results", "")

        state_keys = state.keys()

        # Determine mode automatically
        if "web_results" in state_keys and "rag_answer" in state_keys:
            mode = "final_synthesizer_prompt"
            state["skip_grading"] = True
        elif "web_results" in state_keys:
            mode = "web_search_results_prompt"
            state["skip_grading"] = True
        else:
            mode = "retrieval_prompt"
            state["skip_grading"] = False

        prompt_template = self.config["llm"][mode]

        prompt = prompt_template.format(
            query=query,
            context=retrieved_docs,
            retrieved_docs=retrieved_docs,
            rag_answer=rag_answer,
            web_results=web_results,
        )

        result = self.generator.generate_answer(
            docs=[],
            prompt=prompt,
            question=query,
        )

        if mode == "retrieval_prompt":
            state["rag_answer"] = result["generated_answer"]
        else:
            state["final_answer"] = result["generated_answer"]

        return state

def classifier_router(state: AgentState) -> str:
    return state.get("decision", "internal")

def retrieval_router(state: AgentState) -> str:
   return state.get("decision", "insufficient")

def answer_router(state: AgentState) -> str:
    return state.get("decision", "insufficient")