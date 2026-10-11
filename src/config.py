"""Central configuration. All secrets come from environment variables / .env (backend only)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _p(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


@dataclass
class Settings:
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "").strip())
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", "").strip())
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "anthropic"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "claude-sonnet-5-5"))
    llm_timeout_s: float = field(default_factory=lambda: float(os.getenv("LLM_TIMEOUT_S", "60")))
    llm_max_retries: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_RETRIES", "2")))
    dataset_path: Path = field(default_factory=lambda: _p(os.getenv("DATASET_PATH", "data/raw/crop_yield.csv")))
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./terramind.db"))
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "127.0.0.1"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    target_rows: int = 55_000
    random_seed: int = 42
    cors_origins: list[str] = field(default_factory=lambda: os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","))

    # directories
    raw_dir: Path = ROOT / "data" / "raw"
    processed_dir: Path = ROOT / "data" / "processed"
    sample_dir: Path = ROOT / "data" / "sample_data"
    index_dir: Path = ROOT / "artifacts" / "indexes"
    model_dir: Path = ROOT / "artifacts" / "models"
    eval_dir: Path = ROOT / "artifacts" / "evaluation"

    @property
    def cases_path(self) -> Path:
        return self.processed_dir / "cases.csv"

    @property
    def quality_report_path(self) -> Path:
        return self.processed_dir / "data_quality_report.json"

    @property
    def llm_configured(self) -> bool:
        key = self.groq_api_key if self.llm_provider == "groq" else self.anthropic_api_key
        return bool(key) and not key.startswith("your_")

    @property
    def db_path(self) -> Path:
        url = self.database_url
        if not url.startswith("sqlite:///"):
            raise ValueError("Only sqlite:/// URLs are supported")
        return _p(url[len("sqlite:///"):])


settings = Settings()
