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

    CHUNK_STRATEGY: Literal["token", "semantic", "agentic"] = "token"
    TOKEN_CHUNK_SIZE: int = 512
    TOKEN_CHUNK_OVERLAP: int = 100
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNKING_LLM_MODEL: str = "meta-llama/Llama-3.1-8B-Instruct"

    VECTORSTORE_TYPE: Literal["faiss", "chroma", "pgvector"] = "faiss"
    VECTORSTORE_TOP_K: int = 5
    PGVECTOR_CONNECTION: Optional[str] = None

    LLM_MODEL_NAME: str = "meta-llama/Llama-3.1-8B-Instruct"
    LLM_MAX_NEW_TOKENS: int = 300
    LLM_TEMPERATURE: float = 1e-9
    DEVICE: str = "cpu"

    HUGGING_FACE_API: Optional[str] = Field(default=None, alias="HUGGING_FACE_API")
    TAVILY_API_KEY: Optional[str] = Field(default=None, alias="TAVILY_API_KEY")

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
        """Return a plain dict shaped like the legacy config.json schema."""
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
                "type": self.VECTORSTORE_TYPE,
                "top_k": self.VECTORSTORE_TOP_K,
                "collection_name": f"{self.CHUNK_STRATEGY}_chunks",
                "pgvector_connection": self.PGVECTOR_CONNECTION or "",
                "embedding_model": self.EMBEDDING_MODEL,
            },
            "llm": self._load_prompts(),
        }

    def _load_prompts(self) -> dict:
        """Load prompt templates from config/default.json, merging LLM params."""
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
    """Return a cached singleton Settings instance."""
    return Settings()
