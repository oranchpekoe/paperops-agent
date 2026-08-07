"""Configuration boundary for the PaperOps application."""

from pathlib import Path

from pydantic import Field
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
    max_parse_attempts: int = Field(default=2, ge=1, le=5)
    min_markdown_characters: int = Field(default=120, ge=1)
    min_section_count: int = Field(default=1, ge=0)
    max_replacement_character_ratio: float = Field(default=0.01, ge=0.0, le=1.0)
    min_retrieval_hits: int = Field(default=1, ge=1, le=10)
