import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

# Set before the app is imported, so app.settings sees the test database
os.environ.setdefault(
    "DATABASE_URL", "postgresql+psycopg://parseiq:parseiq_pw@localhost:5433/parseiq_test"
)
# Deliberately use local storage for tests even if the .env file says otherwise
os.environ["STORAGE_BACKEND"] = "local"

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def engine():
    # Safety guard: Only test against the test DB, not real one
    from sqlalchemy.engine import make_url

    db_name = make_url(os.environ["DATABASE_URL"]).database or ""
    if not db_name.endswith("_test"):
        pytest.exit(f"Refusing to run against database {db_name!r}; it must end in _test")

    from sqlalchemy import text
    from sqlmodel import SQLModel

    import app.models  # noqa: F401
    from app.db import engine

    # create_all can't create a vector column until the extension exists
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    SQLModel.metadata.create_all(engine)
    return engine


@contextmanager
def isolated_client(engine):
    """
    A TestClient with a clean DB and uploads in a throwaway folder. Everything
    is undone on exit. Skips without an OpenAI key, since uploads are embedded.
    """
    from fastapi.testclient import TestClient
    from sqlalchemy import text

    import app.storage as storage
    from app.main import app
    from app.settings import settings

    if settings.openai_api_key is None:
        pytest.skip("needs OPENAI_API_KEY: uploads and questions are embedded")

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE documents, chunks RESTART IDENTITY CASCADE"))

    with tempfile.TemporaryDirectory() as upload_dir, pytest.MonkeyPatch.context() as mp:
        mp.setattr(storage, "UPLOADS_DIR", Path(upload_dir))
        storage.get_storage.cache_clear()
        try:
            yield TestClient(app)
        finally:
            storage.get_storage.cache_clear()


@pytest.fixture
def client(engine):
    """Fresh, isolated app for a single test."""
    with isolated_client(engine) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def eval_corpus(engine):
    """
    Uploads every document in rag_eval/corpus/ ONCE per test module, through the
    real upload endpoint (so it's chunked and embedded exactly like production).
    Yields {document_id: filename}.
    """
    from rag_eval.harness import CORPUS_DIR

    with isolated_client(engine) as test_client:
        filenames_by_id = {}
        for path in sorted(CORPUS_DIR.glob("*.txt")):
            res = test_client.post(
                "/api/documents/upload",
                files={"file": (path.name, path.read_bytes(), "text/plain")},
            )
            assert res.status_code == 200, res.text
            assert "processing_error" not in res.json()["processing"], res.json()
            filenames_by_id[res.json()["id"]] = path.name
        yield filenames_by_id


@pytest.fixture
def fake_llm(monkeypatch):
    """
    Replace the hosted LLM with a stub that records the prompt it was given.
    Set `fake_llm.answer` to control what it "replies".
    """

    class FakeLLM:
        answer = "I could not find that information in the provided documents."
        calls: list[tuple[str, str]] = []

        def __call__(self, system: str, user: str) -> str:
            self.calls.append((system, user))
            return self.answer

    fake = FakeLLM()
    fake.calls = []
    monkeypatch.setattr("app.retrieval.answer.generate_with_llm", fake)
    return fake
