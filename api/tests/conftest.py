import os
import shutil
from contextlib import contextmanager

# Set before the app is imported, so app.settings sees the test database
os.environ.setdefault(
    "DATABASE_URL", "postgresql+psycopg://parseiq:parseiq_pw@localhost:5433/parseiq_test"
)
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def engine():
    # Safety guard: Only test against the test DB, not real one
    from sqlalchemy.engine import make_url

    db_name = make_url(os.environ["DATABASE_URL"]).database or ""
    if not db_name.endswith("_test"):
        pytest.exit(f"Refusing to run against database {db_name!r}; it must end in _test")

    from sqlmodel import SQLModel

    import app.models  # noqa: F401
    from app.db import engine

    SQLModel.metadata.create_all(engine)
    return engine


@contextmanager
def isolated_client(engine, faiss_dir):
    """
    A TestClient with a clean DB, an empty FAISS index in `faiss_dir`, and
    uploads in a throwaway folder. Everything is undone on exit.
    """
    from fastapi.testclient import TestClient
    from sqlalchemy import text

    import app.routes.documents as documents
    import app.services.faiss_index as faiss_index
    from app.main import app

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE documents, chunks RESTART IDENTITY CASCADE"))

    upload_root = documents.STORAGE_ROOT / "_pytest"
    with pytest.MonkeyPatch.context() as mp:
        # FAISS index -> a temp dir, so tests never touch storage/faiss/text.faiss.
        mp.setattr(faiss_index, "FAISS_DIR", faiss_dir)
        mp.setattr(documents, "STORAGE_ROOT", upload_root)
        try:
            yield TestClient(app)
        finally:
            shutil.rmtree(upload_root, ignore_errors=True)


@pytest.fixture
def client(engine, tmp_path):
    """Fresh, isolated app for a single test."""
    with isolated_client(engine, tmp_path) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def eval_corpus(engine, tmp_path_factory):
    """
    Uploads every document in rag_eval/corpus/ ONCE per test module, through the
    real upload endpoint (so it's chunked and embedded exactly like production).
    Yields {document_id: filename}.
    """
    from rag_eval.harness import CORPUS_DIR

    with isolated_client(engine, tmp_path_factory.mktemp("eval_faiss")) as test_client:
        filenames_by_id = {}
        for path in sorted(CORPUS_DIR.glob("*.txt")):
            res = test_client.post(
                "/documents/upload", files={"file": (path.name, path.read_bytes(), "text/plain")}
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
    monkeypatch.setattr("app.services.rag.generate_with_llm", fake)
    return fake
