from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "PDF RAG Agent API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    API_V1_PREFIX: str = "/api/v1"
    ALLOWED_ORIGINS: list[str] = ["*"]

    DATA_DIR: Path = Path("data")
    CONFIG_PATH: Path = Path("config/default.json")

    # ── Chunking ───────────────────────────────────────────────────────────────
    CHUNK_STRATEGY: Literal["token", "semantic", "agentic"] = "token"
    TOKEN_CHUNK_SIZE: int = 512
    TOKEN_CHUNK_OVERLAP: int = 100
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNKING_LLM_MODEL: str = "meta-llama/Llama-3.1-8B-Instruct"

    # ── Azure AI Search ────────────────────────────────────────────────────────
    AZURE_SEARCH_ENDPOINT: Optional[str] = Field(default=None)
    AZURE_SEARCH_KEY: Optional[str] = Field(default=None)
    # Legacy single index (kept for back-compat; superseded by the two below).
    AZURE_SEARCH_INDEX_NAME: str = "pdf-rag-index"
    # Two separate indexes: regulation text vs uploaded company policies.
    AZURE_SEARCH_REGULATIONS_INDEX: str = "compliance-regulations"
    AZURE_SEARCH_POLICIES_INDEX: str = "compliance-policies"
    VECTORSTORE_TOP_K: int = 5

    # ── Azure Document Intelligence ────────────────────────────────────────────
    AZURE_DOC_INTEL_ENDPOINT: Optional[str] = Field(default=None)
    AZURE_DOC_INTEL_KEY: Optional[str] = Field(default=None)

    # ── Azure Application Insights ─────────────────────────────────────────────
    APPLICATIONINSIGHTS_CONNECTION_STRING: Optional[str] = Field(default=None)

    # ── Azure OpenAI ───────────────────────────────────────────────────────────
    AZURE_OPENAI_ENDPOINT: Optional[str] = Field(default=None)
    AZURE_OPENAI_KEY: Optional[str] = Field(default=None)
    AZURE_OPENAI_DEPLOYMENT: str = "gpt-4o-mini"

    # ── Neo4j AuraDB ───────────────────────────────────────────────────────────
    NEO4J_URI: Optional[str] = Field(default=None)
    NEO4J_USERNAME: str = Field(default="neo4j")
    NEO4J_PASSWORD: Optional[str] = Field(default=None)
    NEO4J_DATABASE: str = Field(default="neo4j")

    # ── LLM (HuggingFace — agentic/semantic chunking only) ────────────────────
    LLM_MODEL_NAME: str = "meta-llama/Llama-3.1-8B-Instruct"
    LLM_MAX_NEW_TOKENS: int = 300
    LLM_TEMPERATURE: float = 1e-9
    DEVICE: str = "cpu"

    # ── External APIs ──────────────────────────────────────────────────────────
    HUGGING_FACE_API: Optional[str] = Field(default=None)
    TAVILY_API_KEY: Optional[str] = Field(default=None)
    GROQ_API_KEY: Optional[str] = Field(default=None)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @field_validator("DATA_DIR", "CONFIG_PATH", mode="before")
    @classmethod
    def _to_path(cls, v):
        return Path(v)

    def as_pipeline_config(self) -> dict:
        return {
            "paths": {"save_dir": str(self.DATA_DIR)},
            "chunking": {
                "strategy": self.CHUNK_STRATEGY,
                "token_chunk_size": self.TOKEN_CHUNK_SIZE,
                "token_chunk_overlap": self.TOKEN_CHUNK_OVERLAP,
                "embedding_model": self.EMBEDDING_MODEL,
                "llm_model": self.CHUNKING_LLM_MODEL,
            },
            "vectorstore": {
                "endpoint": self.AZURE_SEARCH_ENDPOINT,
                "key": self.AZURE_SEARCH_KEY,
                "index_name": self.AZURE_SEARCH_INDEX_NAME,
                "regulations_index": self.AZURE_SEARCH_REGULATIONS_INDEX,
                "policies_index": self.AZURE_SEARCH_POLICIES_INDEX,
                "top_k": self.VECTORSTORE_TOP_K,
                "embedding_model": self.EMBEDDING_MODEL,
            },
            "azure_openai": {
                "endpoint":   self.AZURE_OPENAI_ENDPOINT,
                "key":        self.AZURE_OPENAI_KEY,
                "deployment": self.AZURE_OPENAI_DEPLOYMENT,
            },
            "neo4j": {
                "uri":      self.NEO4J_URI,
                "username": self.NEO4J_USERNAME,
                "password": self.NEO4J_PASSWORD,
                "database": self.NEO4J_DATABASE,
            },
            "llm": self._load_prompts(),
        }

    def _load_prompts(self) -> dict:
        if self.CONFIG_PATH.exists():
            with open(self.CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            llm_section = cfg.get("llm", {})
        else:
            llm_section = {}

        llm_section.update(
            {
                "llm_model_name": self.LLM_MODEL_NAME,
                "max_new_tokens": self.LLM_MAX_NEW_TOKENS,
                "temperature": self.LLM_TEMPERATURE,
            }
        )
        return llm_section


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
