from functools import cache

from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.models import Chunk

RERANK_MODEL = "jinaai/jina-reranker-v1-turbo-en"


# -----------------------------------
# Load ONNX model once and reuse it
# -----------------------------------
@cache
def _model() -> TextCrossEncoder:
    return TextCrossEncoder(RERANK_MODEL)


# -----------------
# Rerank function
# -----------------
def rerank(query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
    if not chunks:
        return []

    chunk_texts = [c.content_text or "" for c in chunks]
    scores = list(_model().rerank(query, chunk_texts))
    ranked = sorted(zip(chunks, scores, strict=True), key=lambda pair: pair[1], reverse=True)
    return [chunk for chunk, _ in ranked[:top_k]]
