import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")


@pytest.fixture(scope="session")
def store():
    from src.rag.store import get_store
    return get_store()


@pytest.fixture(scope="session")
def retriever(store):
    from src.rag.retriever import get_retriever
    return get_retriever(store)


@pytest.fixture(scope="session")
def predictor():
    from src.models.yield_model import YieldPredictor
    return YieldPredictor()


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    """Isolated SQLite DB so tests never touch the real one."""
    from src.config import settings
    monkeypatch.setattr(settings, "database_url", f"sqlite:///{tmp_path / 'test.db'}")
    return tmp_path


@pytest.fixture()
def no_llm(monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setattr(settings, "groq_api_key", "")


@pytest.fixture()
def client(tmp_db):
    from fastapi.testclient import TestClient
    from src.main import app
    return TestClient(app)


@pytest.fixture(autouse=True)
def _pin_provider(monkeypatch):
    """Tests are provider-independent: pin the Anthropic code path regardless of the local .env."""
    from src.config import settings
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
