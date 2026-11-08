import logging

import numpy as np
from sqlmodel import Session, select

from app.models import Chunk, Document
from app.services import faiss_index
from app.services.chunking import run_chunking
from app.services.embeddings import embed_texts

logger = logging.getLogger(__name__)


# ----------------------------------
# Utility function to embed chunks
# ----------------------------------
def _vectors_for(chunks: list[Chunk]) -> tuple[np.ndarray, np.ndarray]:
    """Embed the chunks that have the text. Returns (vectors, chunk ids), aligned row by row."""
    chunks = [c for c in chunks if (c.content_text or "").strip()]
    if not chunks:
        return (
            np.zeros((0, faiss_index.DIMS["text"]), dtype=np.float32),
            np.zeros(0, dtype=np.int64),
        )

    vectors = embed_texts([c.content_text for c in chunks])
    ids = np.array([c.id for c in chunks], dtype=np.int64)
    return vectors, ids


# ---------------------------------------------------------------------------------------
# Function to index a document by chunking it and adding its vectors to the FAISS index
# ---------------------------------------------------------------------------------------
def index_document(session: Session, doc: Document) -> dict:
    """Chunk one document and add its vectors to the existing FAISS index."""
    chunks = run_chunking(session, doc.id, rebuild=True)
    vectors, ids = _vectors_for(chunks)

    index = faiss_index.load_or_new("text")
    faiss_index.add(index, vectors, ids)
    faiss_index.save(index, "text")
    return {"chunks_created": len(chunks), "embedded": len(ids)}


# -----------------------------------------------------------------
# Function to reindex all documents by rebuilding the FAISS index
# -----------------------------------------------------------------
def reindex_all(session: Session) -> dict:
    """Re-chunk every document and rebuild the FAISS index from scratch."""
    docs = session.exec(select(Document).order_by(Document.id)).all()
    chunks: list[Chunk] = []
    failed: list[int] = []

    for doc in docs:
        try:
            chunks += run_chunking(session, doc.id, rebuild=True)
        except Exception:
            # Do not stop if one document fails to chunk
            logger.exception("Chunking failed for document %s", doc.id)
            failed.append(doc.id)

    vectors, ids = _vectors_for(chunks)
    index = faiss_index.rebuild("text", vectors, ids)
    faiss_index.save(index, "text")
    return {"documents": len(docs), "chunks": len(chunks), "embedded": len(ids), "failed": failed}
