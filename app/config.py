"""Application settings, loaded once from environment / .env."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # OpenAI. Empty key => deterministic fallback mode (no external calls).
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"

    # Model routing: cheap fast model on the hot path (chat, evals),
    # stronger model for the one-time per-document insights pass.
    chat_model: str = "gpt-4o-mini"
    insights_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"

    # Infrastructure (defaults match docker-compose.yml)
    database_url: str = "postgresql+psycopg://vault:vault@localhost:5433/vault"
    redis_url: str = "redis://localhost:6380/0"
    storage_dir: str = "./storage"

    # Generation caps — keep cost and latency bounded.
    max_tokens_chat: int = 400
    max_tokens_summary: int = 500


@lru_cache
def get_settings() -> Settings:
    return Settings()
