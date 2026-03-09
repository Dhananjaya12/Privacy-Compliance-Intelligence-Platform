from fastapi import APIRouter, HTTPException, Request
from app.schemas import QueryRequest, QueryResponse

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(body: QueryRequest, request: Request):
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(status_code=503, detail="Pipeline still loading.")

    agent = request.app.state.agent

    try:
        result = agent.invoke({"query": body.query})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    answer = result.get("final_answer") or result.get("rag_answer", "")

    return QueryResponse(
        query=body.query,
        answer=answer,
        retrieval_score=result.get("retrieval_score"),
        used_web_search=bool(result.get("web_results")),
    )
