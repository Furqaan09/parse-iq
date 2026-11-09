from __future__ import annotations

from fastapi import APIRouter, Body, Depends
from sqlmodel import Session

from app.db import get_session
from app.services.rag import rag_answer

# API router for question answering
router = APIRouter(tags=["ask"])


# ----------------------------------------
# POST endpoint to ask a question to LLM
# ----------------------------------------
@router.post("/ask")
def ask(
    message: str = Body(..., embed=True),
    top_k: int = Body(6, embed=True),
    document_ids: list[int] | None = Body(None, embed=True),
    session: Session = Depends(get_session),
):
    """
    RAG answer: retrieve the top-k chunks and answer with the hosted LLM, citing them.
    Optional 'document_ids' restricts retrieval to those documents.
    """
    return rag_answer(session=session, query=message, top_k=top_k, restrict_doc_ids=document_ids)
