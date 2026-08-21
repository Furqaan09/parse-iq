import logging
from sentence_transformers import CrossEncoder

from app.core.device import get_torch_device
from app.models import Chunk

logger = logging.getLogger(__name__)

_REANKER_MODEL: CrossEncoder | None = None

device = get_torch_device()


def get_reranker_model() -> CrossEncoder:
    global _REANKER_MODEL
    if _REANKER_MODEL is None:
        logger.info("Loading cross-encoder re-ranker model...")
        _REANKER_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=str(device))
    return _REANKER_MODEL


def rerank(query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
    """
    Re-rank `chunks` against `query` with the crosss-encoder model and return the top_k results
    ordered by the predicted relevance (highest first).
    """

    if not chunks:
        return []

    model = get_reranker_model()
    pairs = [(query, c.content_text or "") for c in chunks]
    scores = model.predict(pairs)

    ranked = sorted(zip(chunks, scores, strict=True), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in ranked[:top_k]]
