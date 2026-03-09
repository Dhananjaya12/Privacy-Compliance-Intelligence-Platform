from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import query, ingest, health

from dotenv import load_dotenv
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    config = settings.to_pipeline_config()

    from pipeline.rag_pipeline import RAGPipeline
    from agent.graph import build_agent

    pipeline = RAGPipeline(config)
    retriever, generator = pipeline.build()

    app.state.agent = build_agent(retriever, generator, config)
    app.state.pipeline = pipeline
    app.state.config = config
    app.state.ready = True

    yield

    app.state.ready = False


app = FastAPI(
    title="PDF RAG Agent",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(query.router, prefix="/api")
app.include_router(ingest.router, prefix="/api")
