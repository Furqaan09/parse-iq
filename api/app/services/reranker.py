import logging

from sentence_transformers import CrossEncoder

from app.models import Chunk

logger = logging.getLogger(__name__)


# ----------------------------------------------------------
# Singleton instance for the cross-encoder re-ranker model
# ----------------------------------------------------------
_RERANKER_MODEL: CrossEncoder | None = None


# ---------------------------------------------------
# Function to get the cross-encoder re-ranker model
# ---------------------------------------------------
def get_reranker_model() -> CrossEncoder:
    global _RERANKER_MODEL
    if _RERANKER_MODEL is None:
        logger.info("Loading cross-encoder re-ranker model...")
        _RERANKER_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _RERANKER_MODEL


# -----------------
# Rerank function
# -----------------
def rerank(query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
    """
    Re-rank `chunks` against `query` with the cross-encoder model and return the top_k results
    ordered by the predicted relevance (highest first).
    """

    if not chunks:
        return []

    model = get_reranker_model()
    pairs = [(query, c.content_text or "") for c in chunks]
    scores = model.predict(pairs)

    ranked = sorted(zip(chunks, scores, strict=True), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in ranked[:top_k]]
