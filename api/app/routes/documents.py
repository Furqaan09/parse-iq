import datetime
import re
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
from app.settings import API_DIR, STORAGE_DIR

# API router for document-related endpoints
router = APIRouter(prefix="/documents", tags=["documents"])

# Directory to store uploaded files
STORAGE_ROOT = STORAGE_DIR / "uploads"

# Types for media
MediaType = Literal["pdf", "image", "text"]


# -------------------------------------------------------------
# Function to detect media type from filename or content type
# -------------------------------------------------------------
def infer_media_type(filename: str, content_type: str) -> MediaType:
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


# ------------------------------------------
# Function to create a title from filename
# ------------------------------------------
def title_from_name(name: str) -> str:
    stem = Path(name).stem
    return " ".join(stem.replace("_", " ").split()).strip() or "Untitled"


_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(filename: str) -> str:
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


# ------------------------------------
# POST endpoint to upload a document
# ------------------------------------
@router.post("/upload")
async def upload_document(file: UploadFile = File(...), session: Session = Depends(get_session)):
    # Validation for large files
    if file.size and file.size > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (>50MB).")

    # Infer media type
    media_type = infer_media_type(file.filename, file.content_type)
    title = title_from_name(file.filename)
    safe_filename = sanitize_filename(file.filename)

    # Build a storage path
    today = datetime.datetime.now(datetime.UTC)
    dir_path = STORAGE_ROOT / f"{today:%Y}" / f"{today:%m}"
    dir_path.mkdir(parents=True, exist_ok=True)

    dest_path = dir_path / safe_filename

    # Handle name collisions by adding a counter suffix
    counter = 1
    while dest_path.exists():
        dest_path = dir_path / f"{Path(safe_filename).stem}_{counter}{Path(safe_filename).suffix}"
        counter += 1

    # Save the file on disk
    content = await file.read()
    with open(dest_path, "wb") as f:
        f.write(content)

    # Derieve metadata
    pages = None
    try:
        if media_type == "pdf":
            reader = PdfReader(str(dest_path))
            pages = len(reader.pages)
        elif media_type == "image":
            Image.open(dest_path).verify()
    except Exception as e:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"File parsing failed: {e}") from e

    # Insert document record in DB
    doc = Document(
        title=title,
        source_type="upload",
        media_type=media_type,
        storage_path=str(dest_path.relative_to(API_DIR)),
        pages=pages,
        meta_json=None,
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)

    try:
        processing = index_document(session, doc)
    except Exception as e:
        # The upload itself still succeeds; the error is reported, not raised
        processing = {"chunks_created": 0, "embedded": 0, "processing_error": str(e)}

    # Return a JSON response
    return JSONResponse(
        {
            "id": doc.id,
            "title": doc.title,
            "media_type": doc.media_type,
            "storage_path": doc.storage_path,
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
            "storage_path": d.storage_path,
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


# ----------------------------------------------
# Function to convert storage path to file URL
# ----------------------------------------------
def storage_path_to_file_url(storage_path: str) -> str | None:
    try:
        # Get relative path from storage root (e.g. 2025/09/file.pdf)
        rel = Path(storage_path).resolve().relative_to(STORAGE_DIR / "uploads")
        return f"/files/{rel.as_posix()}"
    except Exception:
        return None


# -----------------------------------------------
# GET endpoint to fetch a single document by ID
# -----------------------------------------------
@router.get("/{doc_id}")
def get_document(doc_id: int = FastAPIPath(..., ge=1), session: Session = Depends(get_session)):
    # Fetch document from DB
    doc = session.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Convert storage path to file URL
    file_url = None
    if doc.storage_path:
        abs_path = API_DIR / doc.storage_path
        if abs_path.exists():
            file_url = storage_path_to_file_url(str(abs_path))

    return {
        "id": doc.id,
        "title": doc.title,
        "media_type": doc.media_type,
        "pages": doc.pages,
        "storage_path": doc.storage_path,
        "file_url": file_url,
    }
