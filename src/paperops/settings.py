"""Configuration boundary for the PaperOps application."""

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Load product configuration from environment variables and ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PAPEROPS_",
        extra="ignore",
    )

    artifacts_dir: Path = Path("artifacts")
    knowledge_dir: Path = Path("knowledge")
    checkpoint_db: Path = Path("paperops.db")
    client_mode: Literal["fake", "real"] = "fake"
    retrieval_backend: Literal[
        "native",
        "dense",
        "hybrid",
        "hybrid_reranked",
        "ragflow",
    ] = "native"

    mineru_base_url: str = "http://localhost:8000"
    mineru_backend: str = "pipeline"
    mineru_parse_method: Literal["auto", "txt", "ocr"] = "auto"
    mineru_poll_interval_seconds: float = Field(default=1.0, gt=0.0, le=60.0)
    mineru_task_timeout_seconds: float = Field(default=1800.0, gt=0.0)
    mineru_max_result_bytes: int = Field(
        default=200 * 1024 * 1024,
        ge=1024,
    )
    mineru_max_extracted_bytes: int = Field(
        default=500 * 1024 * 1024,
        ge=1024,
    )

    ragflow_base_url: str = "http://localhost:9380"
    ragflow_api_key: SecretStr = SecretStr("")
    ragflow_poll_interval_seconds: float = Field(default=1.0, gt=0.0, le=60.0)
    ragflow_index_timeout_seconds: float = Field(default=900.0, gt=0.0)
    ragflow_similarity_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    ragflow_page_size: int = Field(default=10, ge=1, le=100)

    native_index_db: Path = Path("paperops-index.db")
    native_chunk_size_chars: int = Field(default=1200, ge=200, le=10000)
    native_chunk_overlap_chars: int = Field(default=160, ge=0, le=2000)
    native_search_top_k: int = Field(default=10, ge=1, le=100)
    retrieval_embedding_model: str = "BAAI/bge-small-en-v1.5"
    retrieval_reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    retrieval_model_cache_dir: Path = Path(".paperops-models")
    retrieval_candidate_k: int = Field(default=20, ge=1, le=100)
    retrieval_rrf_k: int = Field(default=60, ge=1)

    external_connect_timeout_seconds: float = Field(default=10.0, gt=0.0)
    external_read_timeout_seconds: float = Field(default=60.0, gt=0.0)
    external_write_timeout_seconds: float = Field(default=300.0, gt=0.0)
    external_trust_env: bool = False
    max_upload_bytes: int = Field(default=50 * 1024 * 1024, ge=1024)

    max_parse_attempts: int = Field(default=2, ge=1, le=5)
    min_markdown_characters: int = Field(default=120, ge=1)
    min_section_count: int = Field(default=1, ge=0)
    max_replacement_character_ratio: float = Field(default=0.01, ge=0.0, le=1.0)
    min_retrieval_hits: int = Field(default=1, ge=1, le=10)
    retrieval_probe_top_k: int = Field(default=10, ge=1, le=100)
