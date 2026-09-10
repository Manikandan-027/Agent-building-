"""Central configuration. Everything is env-driven (12-factor) via pydantic-settings.

Storage backends are pluggable: SQLite + in-memory implementations run anywhere
(dev/CI); Postgres/Redis/Qdrant activate purely through configuration.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from pydantic import AliasChoices, Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ARA_", env_file=".env", extra="ignore")

    env: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"

    # LLM (OpenAI-compatible). Empty -> deterministic scripted provider.
    # NOTE: these are intentionally read WITHOUT the ARA_ prefix (standard names,
    # matching docker-compose and provider conventions) via AliasChoices.
    openai_base_url: str = Field(default="", validation_alias=AliasChoices("OPENAI_BASE_URL", "ARA_OPENAI_BASE_URL"))
    openai_api_key: str = Field(default="", validation_alias=AliasChoices("OPENAI_API_KEY", "ARA_OPENAI_API_KEY"))
    openai_model: str = Field(default="gpt-4.1-mini", validation_alias=AliasChoices("OPENAI_MODEL", "ARA_OPENAI_MODEL"))
    # Task-based routing (optional): strongest model for planning, cheap model for
    # extractive roles (REASONER/ANSWERER/CLAIMER). Empty = use openai_model.
    openai_model_planner: str = Field(default="", validation_alias=AliasChoices("OPENAI_MODEL_PLANNER", "ARA_OPENAI_MODEL_PLANNER"))
    openai_model_fast: str = Field(default="", validation_alias=AliasChoices("OPENAI_MODEL_FAST", "ARA_OPENAI_MODEL_FAST"))
    openai_fallback_models: str = Field(default="", validation_alias=AliasChoices("OPENAI_FALLBACK_MODELS", "ARA_OPENAI_FALLBACK_MODELS"))  # comma separated
    llm_timeout_s: float = 60.0

    # ColPali-family visual retrieval service
    colpali_service_url: str = Field(default="http://localhost:8100", validation_alias=AliasChoices("COLPALI_SERVICE_URL", "ARA_COLPALI_SERVICE_URL"))
    colpali_mode: Literal["mock", "real"] = Field(default="mock", validation_alias=AliasChoices("COLPALI_MODE", "ARA_COLPALI_MODE"))
    retrieval_top_k: int = 5
    retrieval_min_score: float = 0.20

    # Storage
    database_url: str = Field(default="sqlite:///./data/ara.db", validation_alias=AliasChoices("DATABASE_URL", "ARA_DATABASE_URL"))
    redis_url: str = Field(default="", validation_alias=AliasChoices("REDIS_URL", "ARA_REDIS_URL"))
    qdrant_url: str = Field(default="", validation_alias=AliasChoices("QDRANT_URL", "ARA_QDRANT_URL"))

    # Security
    api_key_sha256: str = ""  # comma-separated accepted key hashes (production)
    dev_api_key: str = "ara-dev-key-change-me"
    secret_key: str = "change-me-32-bytes-min"
    rate_limit_per_minute: int = 60

    # Default budgets (per task)
    default_time_budget_s: float = 120.0
    default_max_tool_calls: int = 24
    default_max_llm_calls: int = 16
    default_max_tokens: int = 120_000
    default_max_cost_usd: float = 2.0
    max_iterations: int = 40
    max_replans: int = 3

    # Cost accounting (USD / 1K tokens)
    price_input_per_1k: float = 0.0005
    price_output_per_1k: float = 0.0015

    # Injection detection sensitivity 0..1
    injection_threshold: float = 0.55

    # Approvals auto-expiry
    approval_ttl_s: int = 3600

    @property
    def db_backend(self) -> Literal["sqlite", "postgres"]:
        return "postgres" if self.database_url.startswith(("postgres", "postgresql")) else "sqlite"

    @property
    def fallback_models(self) -> list[str]:
        return [m.strip() for m in self.openai_fallback_models.split(",") if m.strip()]

    def accepted_api_key_hashes(self) -> set[str]:
        return {h.strip().lower() for h in self.api_key_sha256.split(",") if h.strip()}

    def hash_api_key(self, key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()


@lru_cache
def get_settings() -> Settings:
    return Settings()
