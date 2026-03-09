from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Paths
    DATA_DIR: Path = Path("data")
    CONFIG_PATH: Path = Path("config/config.json")

    # Chunking
    CHUNK_STRATEGY: Literal["token", "semantic", "agentic"] = "token"
    TOKEN_CHUNK_SIZE: int = 512
    TOKEN_CHUNK_OVERLAP: int = 100
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNKING_LLM_MODEL: str = "meta-llama/Llama-3.1-8B-Instruct"

    # Vector store
    VECTORSTORE_TYPE: Literal["faiss", "chroma", "pgvector"] = "faiss"
    VECTORSTORE_TOP_K: int = 5
    PGVECTOR_CONNECTION: Optional[str] = None

    # LLM
    LLM_MODEL_NAME: str = "meta-llama/Llama-3.1-8B-Instruct"
    LLM_MAX_NEW_TOKENS: int = 300
    LLM_TEMPERATURE: float = 1e-9
    DEVICE: str = "cpu"

    # API keys
    HUGGING_FACE_API: Optional[str] = None
    TAVILY_API_KEY: Optional[str] = None

    class Config:
        env_file = ".env"

    def to_pipeline_config(self) -> dict:
        import json
        with open(self.CONFIG_PATH) as f:
            cfg = json.load(f)

        cfg["paths"] = {"save_dir": str(self.DATA_DIR)}
        cfg["chunking"].update({
            "strategy": self.CHUNK_STRATEGY,
            "token_chunk_size": self.TOKEN_CHUNK_SIZE,
            "token_chunk_overlap": self.TOKEN_CHUNK_OVERLAP,
            "embedding_model": self.EMBEDDING_MODEL,
            "llm_model": self.CHUNKING_LLM_MODEL,
        })
        cfg["vectorstore"].update({
            "type": self.VECTORSTORE_TYPE,
            "top_k": self.VECTORSTORE_TOP_K,
            "embedding_model": self.EMBEDDING_MODEL,
            "pgvector_connection": self.PGVECTOR_CONNECTION or "",
        })
        cfg["llm"].update({
            "llm_model_name": self.LLM_MODEL_NAME,
            "max_new_tokens": self.LLM_MAX_NEW_TOKENS,
            "temperature": self.LLM_TEMPERATURE,
        })
        return cfg


@lru_cache
def get_settings() -> Settings:
    return Settings()
