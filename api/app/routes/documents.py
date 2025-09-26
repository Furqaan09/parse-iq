from datetime import datetime
from pathlib import Path
from typing import Literal
from PyPDF2 import PdfReader
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from sqlmodel import Session
from PIL import Image

from app.core.database import get_session
from app.models import Document

# API router for document-related endpoints
router = APIRouter(prefix="/documents", tags=["documents"])

# Directory to store uploaded files
STORAGE_ROOT = Path(__file__).resolve().parents[2] / "storage" / "uploads"

# Types for media
MediaType = Literal["pdf", "image", "audio", "text"]

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
        if content_type.startswith("audio/"):
            return "audio"
        if content_type.startswith("text/"):
            return "text"
    
    # Fallback to filename extension
    lower_name = filename.lower()
    if lower_name.endswith(".pdf"):
        return "pdf"
    if any(lower_name.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"]):
        return "image"
    if any(lower_name.endswith(ext) for ext in [".mp3", ".wav", ".m4a", ".ogg", ".aac"]):
        return "audio"
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


# ------------------------------------
# Post endpoint to upload a document
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

    # Return a JSON response
    return JSONResponse(
        {
            "id": doc.id,
            "title": doc.title,
            "media_type": doc.media_type,
            "storage_path": doc.storage_path,
            "pages": doc.pages,
            "created_at": doc.created_at.isoformat() if hasattr(doc, "created_at") else None
        }
    )