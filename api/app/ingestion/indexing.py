import logging

from sqlmodel import Session, select

from app.ingestion.chunking import chunk_document
from app.models import Chunk, Document
from app.retrieval.embeddings import embed_texts

logger = logging.getLogger(__name__)


# -----------------------------------------------
# Embed chunks and store each vector on its row
# -----------------------------------------------
def _embed_chunks(session: Session, chunks: list[Chunk]) -> int:
    """Embed the chunks that have text and save each vector on its row. Returns how many."""
    chunks = [c for c in chunks if (c.content_text or "").strip()]
    vectors = embed_texts([c.content_text for c in chunks])

    # Save the embedding for each chunk in the database
    for chunk, vector in zip(chunks, vectors, strict=True):
        chunk.embedding = vector
    session.add_all(chunks)
    session.commit()
    return len(chunks)


# -------------------------------------------------
# Index one document: create chunks -> embed them
# -------------------------------------------------
def index_document(session: Session, doc: Document) -> dict:
    """Chunk one document and store an embedding on each chunk."""
    chunks = chunk_document(session, doc)
    n = _embed_chunks(session, chunks)
    return {"chunks_created": len(chunks), "embedded": n}


# --------------------------------------
# Re-chunk and re-embed every document
# --------------------------------------
def reindex_all(session: Session) -> dict:
    """Re-chunk and re-embed every document. One failure doesn't stop the rest."""
    docs = session.exec(select(Document).order_by(Document.id)).all()
    summary = {"documents": len(docs), "chunks": 0, "embedded": 0, "failed": []}

    for doc in docs:
        try:
            result = index_document(session, doc)
        except Exception:
            logger.exception("Indexing failed for document %s", doc.id)
            session.rollback()
            summary["failed"].append(doc.id)
            continue

        summary["chunks"] += result["chunks_created"]
        summary["embedded"] += result["embedded"]
    return summary
