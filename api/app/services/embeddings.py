import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.device import get_torch_device

# ---------------------------------------------------
# Singleton instances for models to avoid reloading
# ---------------------------------------------------
_TEXT_MODEL: SentenceTransformer | None = None

device = get_torch_device()


# ------------------------------------------
# Function to get the text embedding model
# ------------------------------------------
def get_text_model() -> SentenceTransformer:
    global _TEXT_MODEL
    if _TEXT_MODEL is None:
        _TEXT_MODEL = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device=str(device)
        )
    return _TEXT_MODEL


# -----------------------------
# Embedding utility functions
# -----------------------------
def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """L2-normalize the rows of a matrix."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10
    return (mat / norms).astype(np.float32)


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Generate L2-normalized text embeddings."""
    model = get_text_model()
    model.to(device)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    return _l2_normalize(embeddings)
