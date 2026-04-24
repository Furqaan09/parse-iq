import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional
from PyPDF2 import PdfReader
from fastapi import APIRouter, Body, Depends, File, HTTPException, Query, UploadFile
from fastapi import Path as FastAPIPath
from fastapi.responses import JSONResponse
from sqlalchemy import func
from sqlmodel import Session, select
from PIL import Image

from app.services.faiss_index import (
    load_or_new, add as faiss_add, save as faiss_save, rebuild as faiss_rebuild, DIMS
)
from app.services.embeddings import embed_images, embed_texts
from app.services.chunking import run_chunking
from app.core.database import get_session
from app.models import Chunk, Document

# API router for document-related endpoints
router = APIRouter(prefix="/documents", tags=["documents"])

# Directory to store uploaded files
STORAGE_ROOT = Path(__file__).resolve().parents[2] / "storage" / "uploads"

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


def process_document_after_upload(doc: Document, session: Session) -> dict:
    """
    Run chunking and embeddings for a document immediately after upload.
    Returns a summary of processing.
    """
    # Chunk the document
    chunks = run_chunking(session, doc.id, rebuild=True)

    # Fetch all chunks for embedding
    all_chunks: list[Chunk] = session.exec(select(Chunk).where(Chunk.document_id == doc.id)).all()

    text_chunks = [
        c for c in all_chunks
        if c.modality == "text" and (c.content_text or "").strip() and not c.embedding_key
    ]
    image_chunks = [
        c for c in all_chunks
        if c.modality == "image" and not c.embedding_key
    ]

    details = []

    # Embed text chunks
    if text_chunks:
        texts = [c.content_text or "" for c in text_chunks]
        vecs = embed_texts(texts)
        ids = np.array([c.id for c in text_chunks], dtype=np.int64)

        index = load_or_new("text")
        faiss_add(index, vecs, ids)
        faiss_save(index, "text")

        for c in text_chunks:
            c.embedding_key = "text"
        session.add_all(text_chunks)
        session.commit()

        details.append({"modality": "text", "count": len(text_chunks)})

    # Embed image chunks
    if image_chunks:
        abs_path = Path(__file__).resolve().parents[2] / doc.storage_path
        paths = [abs_path for _ in image_chunks]

        vecs = embed_images(paths)
        ids = np.array([c.id for c in image_chunks], dtype=np.int64)

        index = load_or_new("image")
        faiss_add(index, vecs, ids)
        faiss_save(index, "image")

        for c in image_chunks:
            c.embedding_key = "image"
        session.add_all(image_chunks)
        session.commit()

        details.append({"modality": "image", "count": len(image_chunks)})

    return {
        "chunks_created": len(chunks),
        "embedded": sum(d["count"] for d in details) if details else 0,
        "embedding_details": details,
    }


# ------------------------------------
# POST endpoint to upload a document
# ------------------------------------
@router.post("/upload")
async def upload_document(file: UploadFile = File(...), session: Session = Depends(get_session)):
    # Validation for large files
    if file.size and file.size > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (>50MB).")

    #Infer media type
    media_type = infer_media_type(file.filename, file.content_type)
    title = title_from_name(file.filename)

    # Build a storage path
    today = datetime.utcnow()
    dir_path = STORAGE_ROOT / f"{today:%Y}" / f"{today:%m}"
    dir_path.mkdir(parents=True, exist_ok=True)

    dest_path = dir_path / file.filename

    # Handle name collisions by adding a counter suffix
    counter = 1
    while dest_path.exists():
        dest_path = dir_path / f"{Path(file.filename).stem}_{counter}{Path(file.filename).suffix}"
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
        raise HTTPException(status_code=400, detail=f"File parsing failed: {e}")
    
    # Insert document record in DB
    doc = Document(
        user_id=None,
        title=title,
        source_type="upload",
        media_type=media_type,
        storage_path=str(dest_path.relative_to(Path(__file__).resolve().parents[2])),
        pages=pages,
        meta_json=None
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)

    processing = {
        "chunks_created": 0,
        "embedded": 0,
        "embedding_details": [],
    }

    try:
        processing = process_document_after_upload(doc, session)
    except Exception as e:
        # Do not fail the upload itself if processing fails
        processing = {
            "chunks_created": 0,
            "embedded": 0,
            "embedding_details": [],
            "processing_error": str(e),
        }

    # Return a JSON response
    return JSONResponse(
        {
            "id": doc.id,
            "title": doc.title,
            "media_type": doc.media_type,
            "storage_path": doc.storage_path,
            "pages": doc.pages,
            "created_at": doc.created_at.isoformat() if hasattr(doc, "created_at") else None,
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
    media_type: Optional[MediaType] = Query(None),
    q: Optional[str] = Query(None),
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
    docs = (
        session.exec(
            stmt.order_by(Document.id.desc()).offset(offset).limit(page_size)
        ).all()
    )

    # Prepare response items
    items = [
        {
            "id": d.id,
            "title": d.title,
            "media_type": d.media_type,
            "storage_path": d.storage_path,
            "pages": d.pages,
            "created_at": d.created_at.isoformat() if hasattr(d, "created_at") else None
        }
        for d in docs
    ]
    
    # Return paginated response
    return {
        "items": items,
        "page": page,
        "page_size": page_size,
        "total": total,
        "has_next": (offset + len(items)) < total
    }


# ----------------------------------------------
# Function to convert storage path to file URL
# ----------------------------------------------
def storage_path_to_file_url(storage_path: str) -> str | None:
    try:
        # Get relative path from storage root (e.g. 2025/09/file.pdf)
        rel = Path(storage_path).resolve().relative_to(
            Path(__file__).resolve().parents[2] / "storage" / "uploads"
        )
        return f"/files/{rel.as_posix()}"
    except Exception:
        return None


# -----------------------------------------------
# GET endpoint to fetch a single document by ID
# -----------------------------------------------
@router.get("/{doc_id}")
def get_document(
    doc_id: int = FastAPIPath(..., ge=1),
    session: Session = Depends(get_session)
):
    # Fetch document from DB
    doc = session.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Convert storage path to file URL
    file_url = None
    if doc.storage_path:
        abs_path = Path(__file__).resolve().parents[2] / doc.storage_path
        if abs_path.exists():
            file_url = storage_path_to_file_url(str(abs_path))

    return {
        "id": doc.id,
        "title": doc.title,
        "media_type": doc.media_type,
        "pages": doc.pages,
        "storage_path": doc.storage_path,
        "file_url": file_url
    }


# ---------------------------------------------
# POST endpoint to run chunking on a document
# ---------------------------------------------
@router.post("/{doc_id}/chunk")
def chunk_document(
    doc_id: int,
    rebuild: bool = Body(False, embed=True),
    session: Session = Depends(get_session)
):
    # Fetch document from DB
    doc = session.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Run chunking
    try:
        chunks = run_chunking(session, doc_id, rebuild=rebuild)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chunking failed: {e}")

    return {
        "document_id": doc.id,
        "created": len(chunks),
        "media_type": doc.media_type,
        "chunks": [
            {
                "id": ch.id,
                "modality": ch.modality,
                "page": ch.page,
                "has_text": bool(ch.content_text and ch.content_text.strip()),
            }
        for ch in chunks
        ]
    }


# ----------------------------------------
# POST endpoint to embed document chunks
# ----------------------------------------
@router.post("/{doc_id}/embed")
def embed_document_chunks(
    doc_id: int,
    session: Session = Depends(get_session)
):
    # Fetch document from DB
    doc = session.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Fetch all chunks for the document
    chunks: list[Chunk] = session.exec(
        select(Chunk).where(Chunk.document_id == doc_id)
    ).all()

    if not chunks:
        return {"document_id": doc.id, "embedded": 0, "details": []}

    # Filter chunks that need embedding
    text_chunks = [c for c in chunks if c.modality == "text" and (c.content_text or "").strip() and not c.embedding_key]
    image_chunks = [c for c in chunks if c.modality == "image" and not c.embedding_key]

    details = []

    # Embed TEXT chunks
    if text_chunks:
        texts = [c.content_text or "" for c in text_chunks]
        vecs = embed_texts(texts)
        ids = np.array([c.id for c in text_chunks], dtype=np.int64)
        index = load_or_new("text")
        faiss_add(index, vecs, ids)
        faiss_save(index, "text")

        for c in text_chunks:
            c.embedding_key = "text"
        session.add_all(text_chunks)
        session.commit()
        details.append({"modality": "text", "count": len(text_chunks)})

    # Embed IMAGE chunks
    if image_chunks:
        paths = []
        for c in image_chunks:
            abs_path = Path(__file__).resolve().parents[2] / doc.storage_path
            paths.append(abs_path)

        vecs = embed_images(paths)
        ids = np.array([c.id for c in image_chunks], dtype=np.int64)
        index = load_or_new("image")
        faiss_add(index, vecs, ids)
        faiss_save(index, "image")

        for c in image_chunks:
            c.embedding_key = "image"
        session.add_all(image_chunks)
        session.commit()
        details.append({"modality": "image", "count": len(image_chunks)})

    return {"document_id": doc.id, "embedded": sum(d["count"] for d in details) if details else 0, "details": details}


# --------------------------------
# POST endpoint to rebuild index
# --------------------------------
@router.post("/embed/rebuild")
def rebuild_index(
    modality: Literal["text", "image"] = Body(..., embed=True),
    session: Session = Depends(get_session)
):
    # TEXT
    if modality == "text":
        all_chunks = session.exec(select(Chunk).where(Chunk.modality == "text")).all()
        texts = [c.content_text or "" for c in all_chunks if (c.content_text or "").strip()]
        ids = np.array([c.id for c in all_chunks if (c.content_text or "").strip()], dtype=np.int64)
        if ids.size == 0:
            # write empty index
            index = faiss_rebuild("text", np.zeros((0, DIMS["text"]), dtype="float32"), np.zeros((0,), dtype=np.int64))
            faiss_save(index, "text")
            return {"modality": "text", "rebuilt": 0}
        vecs = embed_texts(texts)
        index = faiss_rebuild("text", vecs, np.array(ids, dtype=np.int64))
        faiss_save(index, "text")
        # mark embedding_key
        for c in all_chunks:
            if c.id in ids:
                c.embedding_key = "text"
        session.add_all(all_chunks)
        session.commit()
        return {"modality": "text", "rebuilt": len(ids)}

    # IMAGE
    all_chunks = session.exec(select(Chunk).where(Chunk.modality == "image")).all()
    # map chunk -> its document path
    docs = {d.id: d for d in session.exec(select(Document)).all()}  # cache
    paths = []
    ids = []
    for c in all_chunks:
        d = docs.get(c.document_id)
        if not d:
            continue
        abs_path = Path(__file__).resolve().parents[2] / d.storage_path
        if abs_path.exists():
            paths.append(abs_path)
            ids.append(c.id)
    if not ids:
        index = faiss_rebuild("image", np.zeros((0, DIMS["image"]), dtype="float32"), np.zeros((0,), dtype=np.int64))
        faiss_save(index, "image")
        return {"modality": "image", "rebuilt": 0}

    vecs = embed_images(paths)
    index = faiss_rebuild("image", vecs, np.array(ids, dtype=np.int64))
    faiss_save(index, "image")
    for c in all_chunks:
        if c.id in ids:
            c.embedding_key = "image"
    session.add_all(all_chunks)
    session.commit()
    return {"modality": "image", "rebuilt": len(ids)}