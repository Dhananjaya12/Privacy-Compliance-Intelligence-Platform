from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])

@router.get("/health")
def health(request: Request):
    ready = getattr(request.app.state, "ready", False)
    return {"status": "ready" if ready else "loading"}
