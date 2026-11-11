import logging
import re
import textwrap
from typing import Any

from sqlmodel import Session, select

from app.models import Chunk, Document
from app.openai_client import openai_client
from app.retrieval.search import search
from app.settings import API_DIR, STORAGE_DIR, settings

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
def _build_context_and_citations(session: Session, chunks: list[Chunk]):
    # fetch docs
    doc_ids = list({c.document_id for c in chunks})
    docs = {d.id: d for d in session.exec(select(Document).where(Document.id.in_(doc_ids))).all()}

    context_parts = []
    citations = []
    idx = 1

    for c in chunks:
        d = docs.get(c.document_id)
        if not d:
            continue

        snippet = (c.content_text or "").strip()
        if not snippet:
            continue

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


# --------------------------------------
# Function to generate answer with LLM
# --------------------------------------
def generate_with_llm(system_prompt: str, user_prompt: str) -> str:
    completion = openai_client().chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=300,
        reasoning_effort="none",
    )
    return (completion.choices[0].message.content or "").strip()


# ---------------------------------
# Function to generate RAG answer
# ---------------------------------
def rag_answer(
    session: Session, query: str, top_k: int = 6, restrict_doc_ids: list[int] | None = None
) -> dict[str, Any]:
    # 1) retrieve the most relevant chunks
    chunks = search(session, query, document_ids=restrict_doc_ids, top_k=top_k)
    if not chunks:
        return {"answer": REFUSAL_MESSAGE, "citations": []}

    # 2) build context and citations
    context, citations = _build_context_and_citations(session, chunks)

    logger.debug("RAG: retrieval complete, building prompt (%d citations)", len(citations))
    logger.debug("RAG: calling generator...")

    # 3) craft grounded prompt
    system = textwrap.dedent(f"""\
        You are ParseIQ's document assistant.

        - Answer ONLY from the provided CONTEXT.
        - If the CONTEXT does not contain the answer, reply exactly:
        {REFUSAL_MESSAGE}
        - Do NOT use outside knowledge.
        - Do NOT guess.
        - Cite sources in brackets like [1], [2] only when the answer is supported by context.
        """)

    user = textwrap.dedent(f"""\
        QUESTION:
        {query}

        CONTEXT:
        {context}

        INSTRUCTIONS:
        - Use only the CONTEXT. Do NOT invent facts.
        - Include bracketed citations [n] aligned to the context items.
        """)

    # 4) generate with the LLM
    answer = generate_with_llm(system, user)

    # 5) verify [n] citations against what was actually retrieved
    answer, citations = _validate_citations(answer, citations)

    return {"answer": answer, "citations": citations}
