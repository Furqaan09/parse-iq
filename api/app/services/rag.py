import logging
import re
from typing import Any

from sqlmodel import Session, select

from app.models import Chunk, Document
from app.retrieval.search import vector_search
from app.services.llm_provider import generate_with_llm
from app.services.reranker import rerank
from app.settings import API_DIR, STORAGE_DIR

logger = logging.getLogger(__name__)

_CITATION_PATTERN = re.compile(r"\s?\[(\d+)\]")
REFUSAL_MESSAGE = "I could not find that information in the provided documents."


# ------------------------------------------------
# Function to convert storage path to files path
# ------------------------------------------------
def _file_url_from_storage_path(storage_path: str) -> str | None:
    """
    Convert 'storage/uploads/YYYY/MM/file.pdf' -> '/files/YYYY/MM/file.pdf'
    Only if path exists under uploads.
    """
    try:
        base = API_DIR
        abs_path = (base / storage_path).resolve()
        uploads_root = STORAGE_DIR / "uploads"
        rel = abs_path.relative_to(uploads_root)
        return f"/files/{rel.as_posix()}"
    except Exception:
        return None


# --------------------------------------------------------------
# Function to build the context and generate citations for llm
# --------------------------------------------------------------
def _build_context_and_citations(session: Session, chunks: list[Chunk], max_chars: int = 4500):
    # fetch docs
    doc_ids = list({c.document_id for c in chunks})
    docs = {d.id: d for d in session.exec(select(Document).where(Document.id.in_(doc_ids))).all()}

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
        file_url = _file_url_from_storage_path(d.storage_path) if d.storage_path else None

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


# ------------------------------------------------------
# Function to check if the answer is a refusal message
# ------------------------------------------------------
def _is_refusal(answer: str) -> bool:
    """True if the answer is exactly the refusal message, ignoring [n] markers and quotes"""
    without_markers = _CITATION_PATTERN.sub("", answer).strip().strip('"').strip()
    return without_markers == REFUSAL_MESSAGE


# ------------------------------------------------------------------------
# Function to validate citations in the answer against retrieved context
# ------------------------------------------------------------------------
def _validate_citations(answer: str, citations: list[dict]) -> tuple[str, list[dict]]:
    """
    Cross-check model's [n] markers against citations actually built from retrived context:
        - citations never referenced in the answer are dropped from the list
        - [n] markers that don't match any real citation are stripped from the answer text
    """
    if _is_refusal(answer):
        return REFUSAL_MESSAGE, []

    valid_ns = {c["n"] for c in citations}
    cited_ns = {int(n) for n in _CITATION_PATTERN.findall(answer)}

    used_ns = cited_ns & valid_ns
    filtered_citations = [c for c in citations if c["n"] in used_ns]

    def _strip_invalid(match: re.Match) -> str:
        n = int(match.group(1))
        return match.group(0) if n in valid_ns else ""

    cleaned_answer = _CITATION_PATTERN.sub(_strip_invalid, answer)
    return cleaned_answer, filtered_citations


# -----------------------------
# Function to retrieve chunks
# -----------------------------
def retrieve_chunks(
    session: Session,
    query: str,
    top_k: int = 6,
    restrict_doc_ids: list[int] | None = None,
    rerank_results: bool = True,
) -> list[Chunk]:
    """
    Two-stage retrieval.
        - pgvector pulls a candidate pool of max(top_k * 4, 20) chunks, filtered to restrict_doc_ids
        inside the SQL query.
        - A cross-encoder reranks that pool and the best top_k are returned.
        - rerank_results=False skips the cross-encoder and returns the top_k in vector order.
        Only used by the eval; production always reranks.
    """
    pool = vector_search(session, query, document_ids=restrict_doc_ids, limit=max(top_k * 4, 20))
    if rerank_results:
        return rerank(query, pool, top_k=top_k)
    return pool[:top_k]


# ---------------------------------
# Function to generate RAG answer
# ---------------------------------
def rag_answer(
    session: Session, query: str, top_k: int = 6, restrict_doc_ids: list[int] | None = None
) -> dict[str, Any]:
    # 1) retrieve the most relevant chunks
    chunks = retrieve_chunks(session, query, top_k=top_k, restrict_doc_ids=restrict_doc_ids)
    if not chunks:
        return {
            "answer": "I could not find relevant information for that question in the indexed documents.",
            "citations": [],
        }

    # 2) build context and citations
    context, citations = _build_context_and_citations(session, chunks, max_chars=4500)

    logger.debug("RAG: retrieval complete, building prompt (%d citations)", len(citations))
    logger.debug("RAG: calling generator...")

    # 3) craft grounded prompt
    system = (
        "You are ParseIQ's document assistant.\n"
        "Answer ONLY from the provided CONTEXT.\n"
        "If the CONTEXT does not contain the answer, reply exactly:\n"
        f'"{REFUSAL_MESSAGE}"\n'
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
    answer = generate_with_llm(system, user)

    # 5) verify [n] citations against what was actually retrieved
    answer, citations = _validate_citations(answer, citations)

    return {"answer": answer, "citations": citations}
