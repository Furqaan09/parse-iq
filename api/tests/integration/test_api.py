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
    return client.post(
        "/api/documents/upload", files={"file": (name, content.encode(), "text/plain")}
    )


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

    listed = client.get("/api/documents").json()
    assert listed["total"] == 1
    assert listed["items"][0]["id"] == doc_id

    doc = client.get(f"/api/documents/{doc_id}").json()
    assert doc["file_url"].startswith("/files/")


def test_get_missing_document_returns_404(client):
    assert client.get("/api/documents/999").status_code == 404


def test_chat_retrieves_relevant_chunk_and_validates_citations(client, fake_llm):
    upload(client)
    fake_llm.answer = "The study break is Feb. 16-20 [1]. See also the handbook [9]."

    res = client.post("/api/ask", json={"message": "When is the study break?"})

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
    res = client.post("/api/ask", json={"message": "When is the study break?"})

    assert res.status_code == 200
    assert res.json()["citations"] == []
    assert fake_llm.calls == []


def test_reindex_reembeds_every_chunk(client, engine):
    from sqlalchemy import func
    from sqlmodel import Session, select

    from app.ingestion.indexing import reindex_all
    from app.models import Chunk

    upload(client)

    with Session(engine) as session:
        summary = reindex_all(session)
        unembedded = session.exec(
            select(func.count()).select_from(Chunk).where(Chunk.embedding.is_(None))
        ).one()

    assert summary["documents"] == 1
    assert summary["failed"] == []
    assert summary["embedded"] == summary["chunks"] > 0
    assert unembedded == 0


def test_vector_search_only_returns_requested_documents(client, engine):
    from sqlmodel import Session

    from app.retrieval.search import vector_search

    upload(client)
    other = upload(client, name="parking.txt", content="Parking permits are sold online.")
    other_id = other.json()["id"]

    with Session(engine) as session:
        hits = vector_search(session, "When is the study break?", document_ids=[other_id])

    assert hits
    assert {chunk.document_id for chunk in hits} == {other_id}


def test_keyword_search_matches_any_word_of_a_long_question(client, engine):
    from sqlmodel import Session

    from app.retrieval.search import keyword_search

    upload(client)

    with Session(engine) as session:
        hits = keyword_search(session, "Do penguins or calculators appear anywhere?")

    assert any("calculators" in (chunk.content_text or "") for chunk in hits)


def test_keyword_search_finds_exact_identifiers(client, engine):
    from sqlmodel import Session

    from app.retrieval.search import keyword_search

    upload(client)
    upload(client, name="po.txt", content="Purchase order PO-7781 covers 100 laptops.")

    with Session(engine) as session:
        hits = keyword_search(session, "What does PO-7781 cover?")

    assert hits
    assert "PO-7781" in (hits[0].content_text or "")


def test_keyword_search_only_returns_requested_documents(client, engine):
    from sqlmodel import Session

    from app.retrieval.search import keyword_search

    upload(client)
    other = upload(client, name="exam.txt", content="The final exam is held in the gym.")
    other_id = other.json()["id"]

    with Session(engine) as session:
        hits = keyword_search(session, "final exam", document_ids=[other_id])

    assert hits
    assert {chunk.document_id for chunk in hits} == {other_id}


def test_keyword_search_with_only_stop_words_returns_nothing(client, engine):
    from sqlmodel import Session

    from app.retrieval.search import keyword_search

    upload(client)

    with Session(engine) as session:
        assert keyword_search(session, "what is it?") == []


def test_context_keeps_all_six_chunks_of_a_long_document(client, fake_llm):
    long_section = "Course Policies\n" + "\n".join(
        f"Policy {i}: students must read the full policy document before week {i}."
        for i in range(120)
    )
    upload(client, name="policies.txt", content=long_section)

    client.post("/api/ask", json={"message": "What must students read?", "top_k": 6})

    _, prompt = fake_llm.calls[0]
    assert "[6]" in prompt


def test_ask_rejects_an_oversized_top_k(client):
    res = client.post("/api/ask", json={"message": "anything", "top_k": 1000})
    assert res.status_code == 422
