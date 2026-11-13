import datetime
import re
import uuid
from io import BytesIO
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi import Path as FastAPIPath
from fastapi.responses import JSONResponse
from PIL import Image
from pypdf import PdfReader
from sqlalchemy import func
from sqlmodel import Session, select

from app.db import get_session
from app.ingestion.indexing import index_document
from app.models import Document
from app.storage import get_storage

# API router for document-related endpoints
router = APIRouter(prefix="/documents", tags=["documents"])

# Types for media
MediaType = Literal["pdf", "image", "text"]

_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


# -------------------------------------
# Utility functions for file handling
# -------------------------------------
def _infer_media_type(filename: str, content_type: str) -> MediaType:
    # Check content type first if provided by browser
    if content_type:
        if "pdf" in content_type:
            return "pdf"
        if content_type.startswith("image/"):
            return "image"
        if content_type.startswith("text/"):
            return "text"

    # Fallback to filename extension
    lower_name = filename.lower()
    if lower_name.endswith(".pdf"):
        return "pdf"
    if any(lower_name.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"]):
        return "image"
    if any(lower_name.endswith(ext) for ext in [".txt", ".md", ".csv"]):
        return "text"

    # Default to text if unsure
    return "text"


def _title_from_name(name: str) -> str:
    stem = Path(name).stem
    return " ".join(stem.replace("_", " ").split()).strip() or "Untitled"


def _sanitize_filename(filename: str) -> str:
    """
    Strip directory components and unsafe characters from a user-supplied
    filename before it is used to build a filesystem path (prevents path
    traversal via names like '../../etc/passwd' or embedded separators).
    """
    # Path(...).name drops any leading directories and '..' segments,
    # whether they came from POSIX ('/') or Windows ('\\') style input.
    base = Path(filename.replace("\\", "/")).name
    stem, suffix = Path(base).stem, Path(base).suffix
    safe_stem = _UNSAFE_FILENAME_CHARS.sub("_", stem).strip("._") or "file"
    safe_suffix = _UNSAFE_FILENAME_CHARS.sub("", suffix)
    return f"{safe_stem}{safe_suffix}"


def _new_storage_key(filename: str) -> str:
    """Returns a key e.g. '2026/11/3f9a0c1b2d4e_invoice.pdf'"""
    now = datetime.datetime.now(datetime.UTC)
    uid = uuid.uuid4().hex[:12]
    return f"{now:%Y/%m}/{uid}_{_sanitize_filename(filename)}"


def _inspect_file(content: bytes, media_type: MediaType) -> tuple[int | None, str]:
    """Check the file can be read before storing it. Returns (page count, content type to store)."""
    if media_type == "pdf":
        return len(PdfReader(BytesIO(content)).pages), "application/pdf"
    if media_type == "image":
        with Image.open(BytesIO(content)) as image:
            image.verify()
            return None, Image.MIME.get(image.format, "application/octet-stream")
    return None, "text/plain; charset=utf-8"


# ------------------------------------
# POST endpoint to upload a document
# ------------------------------------
@router.post("/upload")
def upload_document(file: UploadFile = File(...), session: Session = Depends(get_session)):
    # Validation for large files
    if file.size and file.size > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (>50MB).")

    media_type = _infer_media_type(file.filename, file.content_type)
    content = file.file.read()

    # Check the file before storing it, so a broken upload never reaches storage
    try:
        pages, content_type = _inspect_file(content, media_type)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File parsing failed: {e}") from e

    # Store the file before the row in db
    key = _new_storage_key(file.filename)
    get_storage().put(key, content, content_type)

    doc = Document(
        title=_title_from_name(file.filename),
        media_type=media_type,
        storage_key=key,
        pages=pages,
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)

    try:
        processing = index_document(session, doc)
    except Exception as e:
        # The upload itself still succeeds; the error is reported, not raised
        processing = {"chunks_created": 0, "embedded": 0, "processing_error": str(e)}

    return JSONResponse(
        {
            "id": doc.id,
            "title": doc.title,
            "media_type": doc.media_type,
            "pages": doc.pages,
            "created_at": doc.created_at.isoformat(),
            "processing": processing,
        }
    )


# --------------------------------------------------------------------
# GET endpoint to list all documents based on filters and pagination
# --------------------------------------------------------------------
@router.get("")
def list_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    media_type: MediaType | None = Query(None),
    q: str | None = Query(None),
    session: Session = Depends(get_session),
):
    # SQL quesries to fetch documents and count
    stmt = select(Document)
    count_stmt = select(func.count(Document.id))

    # Apply media type filter if provided
    if media_type:
        stmt = stmt.where(Document.media_type == media_type)
        count_stmt = count_stmt.where(Document.media_type == media_type)

    # Apply search filter if provided (Matches the doc title)
    if q:
        like = f"%{q.strip()}%"
        stmt = stmt.where(Document.title.ilike(like))
        count_stmt = count_stmt.where(Document.title.ilike(like))

    # Execute count query to get total count for pagination
    total = session.exec(count_stmt).one()

    # Apply pagination (To keep track of how many items to skip on next page)
    offset = (page - 1) * page_size
    docs = session.exec(stmt.order_by(Document.id.desc()).offset(offset).limit(page_size)).all()

    # Prepare response items
    items = [
        {
            "id": d.id,
            "title": d.title,
            "media_type": d.media_type,
            "pages": d.pages,
            "created_at": d.created_at.isoformat(),
        }
        for d in docs
    ]

    # Return paginated response
    return {
        "items": items,
        "page": page,
        "page_size": page_size,
        "total": total,
        "has_next": (offset + len(items)) < total,
    }


# -----------------------------------------------
# GET endpoint to fetch a single document by ID
# -----------------------------------------------
@router.get("/{doc_id}")
def get_document(doc_id: int = FastAPIPath(..., ge=1), session: Session = Depends(get_session)):
    # Fetch document from DB
    doc = session.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "id": doc.id,
        "title": doc.title,
        "media_type": doc.media_type,
        "pages": doc.pages,
        "file_url": get_storage().url(doc.storage_key),
    }
