from __future__ import annotations
from typing import Optional, List
from fastapi import APIRouter, Body, Depends
from sqlmodel import Session
from app.core.database import get_session
from app.services.rag import rag_answer

# API router for chat-related endpoints
router = APIRouter(prefix="/chat", tags=["chat"])

# ----------------------------------------
# POST endpoint to ask a question to LLM
# ----------------------------------------
@router.post("/ask")
def chat_ask(
    message: str = Body(..., embed=True),
    top_k: int = Body(6, embed=True),
    document_ids: Optional[List[int]] = Body(None, embed=True),
    session: Session = Depends(get_session),
):
    """
    RAG chat: retrieve top-k textual chunks and answer using a local HF LLM.
    Optional 'document_ids' restricts retrieval to a subset (selected docs).
    """
    result = rag_answer(session=session, query=message, top_k=top_k, restrict_doc_ids=document_ids)
    return result