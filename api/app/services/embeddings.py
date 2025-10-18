import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Optional

from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

# ---------------------------------------------------
# Singleton instances for models to avoid reloading
# ---------------------------------------------------
_TEXT_MODEL: Optional[SentenceTransformer] = None
_IMAGE_MODEL: Optional[CLIPModel] = None
_IMAGE_PROC: Optional[CLIPProcessor] = None

# ------------------------------------------
# Function to get the text embedding model
# ------------------------------------------
def get_text_model() -> SentenceTransformer:
    global _TEXT_MODEL
    if _TEXT_MODEL is None:
        _TEXT_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _TEXT_MODEL

# -------------------------------------------
# Function to get the image embedding model
# -------------------------------------------
def get_image_model() -> tuple[CLIPModel, CLIPProcessor]:
    global _IMAGE_MODEL, _IMAGE_PROC
    if _IMAGE_MODEL is None or _IMAGE_PROC is None:
        _IMAGE_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _IMAGE_PROC = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _IMAGE_MODEL.eval()
        _IMAGE_MODEL.to("cpu")
        try:
            # reduce thread pressure
            torch.set_num_threads(1)
        except Exception:
            pass
    return _IMAGE_MODEL, _IMAGE_PROC



# -----------------------------
# Embedding utility functions
# -----------------------------
def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """L2-normalize the rows of a matrix."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10
    return (mat / norms).astype(np.float32)

def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """Generate L2-normalized text embeddings."""
    model = get_text_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False
    )
    return _l2_normalize(embeddings)

def embed_images(paths: List[Path], batch_size: int = 16) -> np.ndarray:
    """Generate L2-normalized image embeddings."""
    model, proc = get_image_model()

    imgs = [Image.open(p).convert("RGB") for p in paths]
    embs_list: list[np.ndarray] = []

    for i in range(0, len(imgs), batch_size):
        batch = imgs[i : i+batch_size]
        inputs = proc(images=batch, return_tensors="pt").to("cpu")

        with torch.no_grad():
            feats = model.get_image_features(**inputs)
        feats = feats.cpu().numpy()
        embs_list.append(feats)

    embs = np.vstack(embs_list) if embs_list else np.zeros((0, model.visual.output_dim), dtype=np.float32)
    return _l2_normalize(embs)