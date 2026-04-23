from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from sqlmodel import Session, select

from app.models import Chunk, Document
from app.services.embeddings import embed_texts
from app.services.faiss_index import load_or_new, search as faiss_search
from app.services.llm_provider import generate_with_local_llm

MIN_SCORE = 0.20

# ------------------------------------------------
# Function to convert storage path to files path
# ------------------------------------------------
def _file_url_from_storage_path(storage_path: str) -> Optional[str]:
    """
    Convert 'storage/uploads/YYYY/MM/file.pdf' -> '/files/YYYY/MM/file.pdf'
    Only if path exists under uploads.
    """
    try:
        base = Path(__file__).resolve().parents[2]  # api/
        abs_path = (base / storage_path).resolve()
        uploads_root = (base / "storage" / "uploads").resolve()
        rel = abs_path.relative_to(uploads_root)
        return f"/files/{rel.as_posix()}"
    except Exception:
        return None


# -------------------------------
# Function to get ranked chunks
# -------------------------------
def _ranked_chunks_by_ids(session: Session, ids: np.ndarray, scores: np.ndarray) -> List[Chunk]:
    id_list = [int(i) for i in ids if int(i) != -1]
    if not id_list:
        return []
    chunks = session.exec(select(Chunk).where(Chunk.id.in_(id_list))).all()
    order = {int(i): r for r, i in enumerate(ids) if int(i) != -1}
    chunks.sort(key=lambda c: order.get(int(c.id), 10**9))
    return chunks


# --------------------------------------------------------------
# Function to build the context and generate citations for llm
# --------------------------------------------------------------
def _build_context_and_citations(session: Session, chunks: List[Chunk], max_chars: int = 4500):
    # fetch docs
    doc_ids = list({c.document_id for c in chunks})
    docs = {
        d.id: d
        for d in session.exec(select(Document).where(Document.id.in_(doc_ids))).all()
    }

    context_parts = []
    citations = []
    total = 0
    idx = 1

    for c in chunks:
        d = docs.get(c.document_id)
        if not d:
            continue
        snippet = (c.content_text or "").strip()
        if not snippet:
            continue
        # budget context size
        if total + len(snippet) > max_chars and len(context_parts) > 0:
            break
        file_url = (
            _file_url_from_storage_path(d.storage_path) if d.storage_path else None
        )

        context_parts.append(f"[{idx}] {d.title} (p.{c.page or 1})\n{snippet}\n")
        citations.append(
            {
                "n": idx,
                "document_id": d.id,
                "title": d.title,
                "page": c.page,
                "file_url": file_url,
                "snippet": snippet[:220],
            }
        )
        total += len(snippet)
        idx += 1

    context = "\n".join(context_parts)
    return context, citations


# ---------------------------------
# Function to generate RAG answer
# ---------------------------------
def rag_answer(
    session: Session,
    query: str,
    top_k: int = 6,
    restrict_doc_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    # 1) embed query & retrieve text chunks
    qvec = embed_texts([query])[0]
    index = load_or_new("text")

    if index.ntotal == 0:
        return {
            "answer": "I could not find any relevant content yet. Try processing documents first.",
            "citations": [],
        }

    # Search a bit deeper then filter by doc IDs if provided
    scores, ids = faiss_search(index, qvec, top_k=max(top_k * 4, 20))

    scored_ids = [
        (float(score), int(chunk_id))
        for score, chunk_id in zip(scores, ids)
        if int(chunk_id) != -1
    ]

    if not scored_ids:
        return {
            "answer": "I could not find any relevant content in the indexed documents.",
            "citations": [],
        }

    scored_ids = [(s, i) for s, i in scored_ids if s >= MIN_SCORE]
    if not scored_ids:
        return {
            "answer": "I could not find relevant information for that question in the indexed documents.",
            "citations": [],
        }

    filtered_ids = np.array([i for _, i in scored_ids], dtype=np.int64)
    filtered_scores = np.array([s for s, _ in scored_ids], dtype=np.float32)

    chunks = _ranked_chunks_by_ids(session, filtered_ids, filtered_scores)

    if restrict_doc_ids:
        chunks = [c for c in chunks if c.document_id in set(restrict_doc_ids)]

    # keep top_k with text
    texty = [c for c in chunks if (c.content_text or "").strip()]
    texty = texty[:top_k]

    if not texty:
        return {
            "answer": "No text snippets matched your question in the current documents.",
            "citations": [],
        }

    # 2) build context and citations
    context, citations = _build_context_and_citations(session, texty, max_chars=4500)

    print("RAG: retrieval complete, building prompt...")
    print(f"RAG: number of citations = {len(citations)}")
    print("RAG: calling generator...")

    # 3) craft grounded prompt
    system = (
        "You are ParseIQ's document assistant.\n"
        "Answer ONLY what is asked in the QUESTION. Do NOT add more details.\n"
        "Answer ONLY from the provided CONTEXT.\n"
        "If the CONTEXT does not contain the answer, reply exactly:\n"
        "\"I could not find that information in the provided documents.\"\n"
        "Do not use outside knowledge.\n"
        "Do not guess.\n"
        "Cite sources in brackets like [1], [2] only when the answer is supported by context."
    )
    user = (
        f"QUESTION:\n{query}\n\nCONTEXT:\n{context}\n\nINSTRUCTIONS:\n"
        f"- Use only the CONTEXT. Do not invent facts.\n"
        f"- Include bracketed citations [n] aligned to the context items.\n"
    )

    # 4) generate with the local HF model
    answer = generate_with_local_llm(system, user)

    return {"answer": answer, "citations": citations}
