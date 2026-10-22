import pytest

pytestmark = pytest.mark.integration

SYLLABUS = """CSCI 4144 Data Mining Syllabus

Important Dates
Study Break: Feb. 16-20
Assignment Deadlines: 23:59 on Feb. 6, Feb. 27, Mar. 13, Mar. 27
Final Exam: TBA

Evaluation Criteria
Assignments are worth 30% of the final grade.
The midterm exam is worth 30% and the final exam is worth 40%.

Exam Requirements
Photo ID is required.
Closed book. No calculators, notes or electronic aids are allowed.
Note: Students must achieve at least 50% on the final exam to pass the course.
"""


def upload(client, name="syllabus.txt", content=SYLLABUS):
    return client.post("/documents/upload", files={"file": (name, content.encode(), "text/plain")})


def test_upload_chunks_and_embeds_document(client):
    res = upload(client)

    assert res.status_code == 200
    processing = res.json()["processing"]
    assert "processing_error" not in processing
    assert processing["chunks_created"] > 0
    assert processing["embedded"] == processing["chunks_created"]


def test_upload_cannot_write_outside_storage(client):
    res = upload(client, name="../../../evil.txt")

    assert res.status_code == 200
    path = res.json()["storage_path"]
    assert path.startswith("storage/uploads/_pytest/")
    assert ".." not in path
    assert path.endswith("/evil.txt")


def test_list_and_get_document(client):
    doc_id = upload(client).json()["id"]

    listed = client.get("/documents").json()
    assert listed["total"] == 1
    assert listed["items"][0]["id"] == doc_id

    doc = client.get(f"/documents/{doc_id}").json()
    assert doc["file_url"].startswith("/files/")


def test_get_missing_document_returns_404(client):
    assert client.get("/documents/999").status_code == 404


def test_chunks_fit_embedding_model_token_limit(client, engine):
    # Regression: chunks over MiniLM's 256-token limit were silently truncated
    # before embedding, so their endings could never be retrieved.
    from sqlmodel import Session, select
    from transformers import AutoTokenizer

    from app.models import Chunk

    long_section = "Course Policies\n" + "\n".join(
        f"Policy {i}: students must read the full policy document before week {i}."
        for i in range(80)
    )
    upload(client, name="policies.txt", content=long_section)

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    with Session(engine) as session:
        chunks = session.exec(select(Chunk)).all()

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(tokenizer(chunk.content_text)["input_ids"]) <= 256


def test_chat_retrieves_relevant_chunk_and_validates_citations(client, fake_llm):
    upload(client)
    fake_llm.answer = "The study break is Feb. 16-20 [1]. See also the handbook [9]."

    res = client.post("/chat/ask", json={"message": "When is the study break?"})

    assert res.status_code == 200
    body = res.json()

    # Retrieval: the context handed to the LLM, item [1], is the Important Dates chunk.
    _, prompt = fake_llm.calls[0]
    first_context_item = prompt.split("[1]", 1)[1].split("[2]", 1)[0]
    assert "Study Break: Feb. 16-20" in first_context_item

    # Citation validation: [9] doesn't exist, so it's stripped; only [1] is returned.
    assert "[9]" not in body["answer"]
    assert [c["n"] for c in body["citations"]] == [1]
    assert "Study Break" in body["citations"][0]["snippet"]


def test_chat_with_no_documents_does_not_call_llm(client, fake_llm):
    res = client.post("/chat/ask", json={"message": "When is the study break?"})

    assert res.status_code == 200
    assert res.json()["citations"] == []
    assert fake_llm.calls == []
