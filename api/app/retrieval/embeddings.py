from functools import cache

from openai import OpenAI

from app.settings import settings

BATCH_SIZE = 100


# --------------------------------------------------
# Connect to OpenAI and cache the client for reuse
# --------------------------------------------------
@cache
def _client() -> OpenAI:
    if settings.openai_api_key is None:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=settings.openai_api_key.get_secret_value(), timeout=30.0)


# --------------------
# Embedding function
# --------------------
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed non-empty texts in batches. Returns one vector per text, in input order."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        response = _client().embeddings.create(model=settings.embedding_model, input=batch)
        vectors += [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
    return vectors
