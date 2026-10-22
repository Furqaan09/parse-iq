import os
import shutil

os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_PORT", "5433")
os.environ.setdefault("DB_USER", "parseiq")
os.environ.setdefault("DB_PASSWORD", "parseiq_pw")
os.environ.setdefault("DB_NAME", "parseiq_test")
os.environ.setdefault("HF_TOKEN", "test-token-never-used")  # the LLM is always mocked
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def engine():
    # Safety guard: Only test against the test DB, not real one
    if not os.environ["DB_NAME"].endswith("_test"):
        pytest.exit(
            f"Refusing to run against DB_NAME={os.environ['DB_NAME']!r}; it must end in _test"
        )

    from sqlmodel import SQLModel

    import app.models  # noqa: F401
    from app.core.database import engine

    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def client(engine, tmp_path, monkeypatch):
    """A TestClient with a clean DB, an empty FAISS index, and isolated upload storage."""
    from fastapi.testclient import TestClient
    from sqlalchemy import text

    import app.routes.documents as documents
    import app.services.faiss_index as faiss_index
    from app.main import app

    with engine.begin() as conn:
        conn.execute(
            text("TRUNCATE documents, chunks, kv_extractions, tasks RESTART IDENTITY CASCADE")
        )

    # FAISS index -> pytest's temp dir, so tests never touch storage/faiss/text.faiss.
    monkeypatch.setattr(faiss_index, "FAISS_DIR", tmp_path)

    # Temp storage dir for uploads
    upload_root = documents.STORAGE_ROOT / "_pytest"
    monkeypatch.setattr(documents, "STORAGE_ROOT", upload_root)

    yield TestClient(app)
    shutil.rmtree(upload_root, ignore_errors=True)


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
