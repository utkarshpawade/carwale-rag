"""Runtime settings (env vars / .env). Model IDs are config because Groq models get deprecated."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(_ROOT / ".env", _ROOT / "backend" / ".env"), extra="ignore")

    groq_api_key: str = ""
    database_url: str = "postgresql://postgres:postgres@localhost:5433/carwale"

    # role -> ordered fallback chain (comma separated)
    planner_models: str = "openai/gpt-oss-20b,qwen/qwen3.8-27b,llama-3.1-8b-instant"
    answer_models: str = "openai/gpt-oss-120b,llama-3.3-70b-versatile,openai/gpt-oss-20b"
    summary_models: str = "llama-3.1-8b-instant,openai/gpt-oss-20b"
    force_fail_models: str = ""  # comma separated: simulate 429s to demo/test the fallback chain

    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_cache_dir: str | None = None
    embed_threads: int | None = None

    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    rate_limit_per_min: int = 12
    rate_limit_per_day: int = 300
    max_input_chars: int = 1000

    # token budgets
    context_token_budget: int = 2200
    answer_max_tokens: int = 1100
    history_turns: int = 3
    summarize_after_messages: int = 8

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def chain(self, role: str) -> list[str]:
        raw = {"planner": self.planner_models, "answer": self.answer_models, "summary": self.summary_models}[role]
        return [m.strip() for m in raw.split(",") if m.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
