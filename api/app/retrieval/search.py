from collections import defaultdict

from sqlalchemy import Text, cast, func
from sqlalchemy.dialects.postgresql import REGCONFIG, TSQUERY
from sqlmodel import Session, select

from app.models import Chunk
from app.retrieval.embeddings import embed_texts
from app.services.reranker import rerank

RRF_K = 60


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


# -------------------------------------
# Keyword search: matches exact words
# -------------------------------------
def keyword_search(
    session: Session, query: str, *, document_ids: list[int] | None = None, limit: int = 20
) -> list[Chunk]:
    """Chunks containing any word of the query, best matches first."""
    # Convert query to tsquery (tokenized, stemmed and stopword-filtered)
    all_words = func.plainto_tsquery(cast("english", REGCONFIG), query)

    # Replace AND with OR to match 'any' word in the query
    replaced = func.replace(cast(all_words, Text), "&", "|")
    any_word = cast(replaced, TSQUERY)

    # Select chunks whose search_vector (tsvector) matches the tsquery
    stmt = select(Chunk).where(Chunk.search_vector.bool_op("@@")(any_word))
    if document_ids is not None:
        stmt = stmt.where(Chunk.document_id.in_(document_ids))

    # Chunk.id breaks ties, so the same query always gives the same order
    stmt = stmt.order_by(func.ts_rank_cd(Chunk.search_vector, any_word).desc(), Chunk.id)
    return list(session.exec(stmt.limit(limit)).all())


# ----------------------------------------------------
# Reciprocal Rank Fusion (RRF): combine ranked lists
# ----------------------------------------------------
def reciprocal_rank_fusion(rankings: list[list[Chunk]]) -> list[Chunk]:
    """Merge ranked lists using each chunk's rank alone: score = sum of 1 / (60 + rank)."""
    scores: dict[int, float] = defaultdict(float)
    chunks_by_id: dict[int, Chunk] = {}

    # Fist fill in scores for vector seach then add scores for keyword search
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.id] += 1 / (RRF_K + rank)
            chunks_by_id[chunk.id] = chunk

    best_first = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    return [chunks_by_id[chunk_id] for chunk_id in best_first]


# -------------------------------------------------
# Hybrid search: vector + keyword, fused with RRF
# -------------------------------------------------
def hybrid_search(
    session: Session, query: str, *, document_ids: list[int] | None = None, limit: int = 20
) -> list[Chunk]:
    """Fuse both searches. `limit` applies to each one, so up to 2 * limit chunks return."""
    return reciprocal_rank_fusion(
        [
            vector_search(session, query, document_ids=document_ids, limit=limit),
            keyword_search(session, query, document_ids=document_ids, limit=limit),
        ]
    )


# -----------------------------------------------------------------
# Main search function that combines hybrid search with reranking
# -----------------------------------------------------------------
def search(
    session: Session, query: str, *, document_ids: list[int] | None = None, top_k: int = 6
) -> list[Chunk]:
    """Hybrid search, then the cross-encoder picks the best top_k."""
    pool = hybrid_search(session, query, document_ids=document_ids, limit=max(top_k * 4, 20))
    return rerank(query, pool, top_k=top_k)
