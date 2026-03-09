from typing import TypedDict, List

class AgentState(TypedDict):
    query: str
    retrieved_docs: List[str]
    retrieval_score: float
    rag_answer: str
    answer_score: int
    web_results: str
    final_answer: str
    decision: str
    