"""
modules/pluto/config.py
Centralized configuration using Pydantic Settings.

"""
from pathlib import Path
from typing import Optional
from pydantic import PostgresDsn, RedisDsn, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PlutoConfig(BaseSettings):
    # PostgreSQL / TimescaleDB
    pg_dsn: PostgresDsn = "postgresql://postgres:password@localhost:5432/pluto"
    pg_pool_min: int = 1
    pg_pool_max: int = 10

    # Redis
    redis_dsn: RedisDsn = "redis://localhost:6379/0"
    redis_cache_ttl: int = 300  # seconds

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_timeout: int = 10

    # Ollama
    ollama_host: str = "127.0.0.1"
    ollama_port: int = 11434
    ollama_model: str = "mistral"

    # Kimi (optional)
    kimi_api_key: Optional[str] = None
    kimi_endpoint: Optional[str] = None

    # Application
    currency: str = "₹"
    db_path: Path = Path("data/pluto/pluto.db")
    max_expense_report_rows: int = 200

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Pluto is only one of many modules sharing the repo-root .env file,
        # so it must ignore keys it doesn't declare (TELEGRAM_BOT_TOKEN, etc.)
        # instead of raising on them.
        extra="ignore",
    )