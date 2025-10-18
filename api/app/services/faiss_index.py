import faiss
import numpy as np
from pathlib import Path
from typing import Literal, Tuple

IndexModality = Literal["text", "image"]

# --------------------------------------------------------
# Ensure the FAISS storage directory exists or create it
# --------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2] # /api/
FAISS_DIR = BASE_DIR / "storage" / "faiss"
FAISS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------
# Dimensions of index for different modalities
# ----------------------------------------------
DIMS = {"text": 384, "image": 512}

# ------------------------------------------
# Find the index path for a given modality
# ------------------------------------------
def _index_path(modality: IndexModality) -> Path:
    return FAISS_DIR / f"{modality}.faiss"

# ----------------------------------------------------
# Function for loading or creating a new FAISS index
# ----------------------------------------------------
def load_or_new(modality: IndexModality) -> faiss.IndexIDMap2:
    path = _index_path(modality)

    # Load existing index if it exists
    if path.exists():
        index = faiss.read_index(str(path))
        # Ensure the index is an IDMap2
        if not isinstance(index, faiss.IndexIDMap2):
            index = faiss.IndexIDMap2(index)
        return index

    # Create a new index otherwise
    dim = DIMS[modality]
    base = faiss.IndexFlatIP(dim)
    return faiss.IndexIDMap2(base)

# ------------------------------------------
# Function to save the FAISS index to disk
# ------------------------------------------
def save(index: faiss.IndexIDMap2, modality: IndexModality) -> None:
    path = _index_path(modality)
    faiss.write_index(index, str(path))

# ----------------------------------------------------
# Function to add vectors and their IDs to the index
# ----------------------------------------------------
def add(index: faiss.IndexIDMap2, vectors: np.ndarray, ids: np.ndarray) -> None:
    if vectors.shape[0] == 0:
        return
    index.add_with_ids(vectors, ids)

# --------------------------------------------
# Function to rebuild the index from scratch
# --------------------------------------------
def rebuild(modality: IndexModality, vectors: np.ndarray, ids: np.ndarray) -> faiss.IndexIDMap2:
    dim = DIMS[modality]
    base = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap2(base)
    add(index, vectors, ids)
    return index

# ----------------------------------------------------------------
# Function to perform a search on the index using a query vector
# ----------------------------------------------------------------
def search(index: faiss.IndexIDMap2, query_vec: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    if query_vec.ndim == 1:
        query_vec = query_vec[None, :]
    D, I = index.search(query_vec.astype(np.float32), top_k)
    return D[0], I[0]