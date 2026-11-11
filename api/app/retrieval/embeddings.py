from app.openai_client import openai_client
from app.settings import settings

BATCH_SIZE = 100


# --------------------
# Embedding function
# --------------------
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed non-empty texts in batches. Returns one vector per text, in input order."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        response = openai_client().embeddings.create(model=settings.embedding_model, input=batch)
        vectors += [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
    return vectors
