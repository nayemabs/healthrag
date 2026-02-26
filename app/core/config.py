from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Healthcare Knowledge Assistant"
    environment: str = "development"
    log_level: str = "INFO"

    # pydantic-settings v2 parses list[str] as JSON; keep as str and split manually
    api_keys: str = "dev-key-healthrag"

    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    faiss_index_path: str = "data/faiss_index"

    # translation: "google" | "googletrans" | "argostranslate" | "mock"
    translation_backend: Literal["googletrans", "argostranslate", "google", "mock"] = "google"

    # llm: "template" (no key) | "openai" (OPENAI_API_KEY) | "claude" (ANTHROPIC_API_KEY)
    llm_backend: Literal["template", "openai", "claude"] = "template"
    llm_model: str = ""  # defaults: gpt-4o-mini / claude-sonnet-4-6

    chunk_size: int = 512
    chunk_overlap: int = 50
    max_context_chars: int = 2000

    log_file: str = ""
    log_max_bytes: int = 10 * 1024 * 1024
    log_backup_count: int = 5

    def get_api_keys(self) -> list[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    model_config = {"env_file": ".env", "case_sensitive": False}


@lru_cache
def get_settings() -> Settings:
    return Settings()
