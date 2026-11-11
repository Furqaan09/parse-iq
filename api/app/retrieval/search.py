from sqlmodel import Session, select

from app.models import Chunk
from app.retrieval.embeddings import embed_texts


# -----------------------------------
# Vector search: closest by meaning
# -----------------------------------
def vector_search(
    session: Session, query: str, *, document_ids: list[int] | None = None, limit: int = 20
) -> list[Chunk]:
    """Chunks closest in meaning to the query, optionally only from `document_ids`."""
    query_vector = embed_texts([query])[0]

    # Select chunks with embeddings
    stmt = select(Chunk).where(Chunk.embedding.is_not(None))

    # Optional: filter by document IDs if provided
    if document_ids is not None:
        stmt = stmt.where(Chunk.document_id.in_(document_ids))

    # Order by cosine distance (1 - similarity) and limit results
    stmt = stmt.order_by(Chunk.embedding.cosine_distance(query_vector)).limit(limit)
    return list(session.exec(stmt).all())
