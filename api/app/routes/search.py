from fastapi import APIRouter, Body, Depends
from sqlmodel import Session, select
from pathlib import Path

from app.core.database import get_session
from app.models import Chunk, Document
from app.services.embeddings import embed_texts
from app.services.faiss_index import load_or_new, search as faiss_search

# API router for search-related endpoints
router = APIRouter(prefix="/search", tags=["search"])

# -------------------------------
# POST endpoint for text search
# -------------------------------
@router.post("/text")
def search_text(
    query: str = Body(..., embed=True),
    top_k: int = Body(5, embed=True),
    session: Session = Depends(get_session),
):
    # Embed the query text
    # It returns (N, D=384), we only have one query so N = 1
    # Hence we take the first row
    qvec = embed_texts([query])[0]  # (384,)

    # Load the FAISS index for text
    index = load_or_new("text")
    if index.ntotal == 0:
        return {"items": [], "total": 0}

    # Perform the search
    scores, ids = faiss_search(index, qvec, top_k=top_k)

    # Fetch matching chunks and docs
    id_list = [int(i) for i in ids if i != -1]
    if not id_list:
        return {"items": [], "total": 0}

    # Retrieve chunks and their associated documents
    chunks = session.exec(select(Chunk).where(Chunk.id.in_(id_list))).all()
    docs = {d.id: d for d in session.exec(select(Document).where(Document.id.in_([c.document_id for c in chunks]))).all()}

    # Order results by FAISS order
    id_to_rank = {int(i): r for r, i in enumerate(ids) if i != -1}
    chunks.sort(key=lambda c: id_to_rank.get(int(c.id), 1_000_000))

    # Build response items
    items = []
    for c, score in zip(chunks, scores[:len(chunks)]):
        d = docs.get(c.document_id)

        # Build file_url from storage path
        file_url = None
        if d and d.storage_path:
            abs_path = Path(__file__).resolve().parents[2] / d.storage_path
            if abs_path.exists():
                rel = abs_path.relative_to(Path(__file__).resolve().parents[2] / "storage" / "uploads")
                file_url = f"/files/{rel.as_posix()}"

        items.append({
            "chunk_id": c.id,
            "document_id": c.document_id,
            "title": d.title if d else "",
            "media_type": d.media_type if d else "",
            "page": c.page,
            "score": float(score),
            "snippet": (c.content_text or "")[:200],
            "file_url": file_url,
        })

    return {"items": items, "total": len(items)}