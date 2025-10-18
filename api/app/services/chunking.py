import pytesseract
from pathlib import Path
from typing import Optional
from PIL import Image
from PyPDF2 import PdfReader
from sqlmodel import Session, select

from app.models import Chunk, Document

# ------------------------------------------------
# Chunking strategy for PDFs: one chunk per page
# ------------------------------------------------
def _abs_storage_path(doc: Document) -> Path:
    """
    Get the absolute storage path for a document.
    Document path: storage/upload/...
    Returned path: api/storage/upload/...
    """
    here = Path(__file__).resolve().parents[2]  # points to api/
    return (here / doc.storage_path).resolve()

def _safe_trim(text: Optional[str]) -> Optional[str]:
    """Trim text and return None if empty or None."""
    if text is None:
        return None
    t = text.strip()
    return t if t else None

def chunk_pdf(session: Session, doc: Document) -> list[Chunk]:
    """Create one text chunk per PDF page."""
    # Load the PDF file
    path = _abs_storage_path(doc)
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    reader = PdfReader(str(path))

    # Create chunks for each page
    created: list[Chunk] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        text = _safe_trim(txt)
        
        # Build and save chunk
        ch = Chunk(
            document_id=doc.id,
            modality="text",
            page=i,
            content_text=text,
            bbox=None,
            embedding_key=None
        )
        session.add(ch)
        created.append(ch)

    # Commit all chunks at once to the database
    session.commit()
    for ch in created:
        session.refresh(ch)

    return created


# ---------------------------------------------------
# Chunking strategy for Images: one chunk per image
# ---------------------------------------------------
def _try_ocr_image(path: Path) -> Optional[str]:
    """Try to OCR the image and return extracted text or None."""
    try:
        img = Image.open(path)
        text = pytesseract.image_to_string(img)
        return _safe_trim(text)
    except Exception:
        return None
    
def chunk_image(session: Session, doc: Document) -> list[Chunk]:
    """Create one chunk for the image, with optional OCR text."""
    # Load the image file
    path = _abs_storage_path(doc)
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    
    try:
        Image.open(path).verify()
    except Exception as e:
        raise RuntimeError(f"Invalid image for chunking: {e}")
    
    # Perform OCR to extract text
    text = _try_ocr_image(path)
    
    # Build and save chunk, commit to the database
    ch = Chunk(
        document_id=doc.id,
        modality="image",
        page=1,
        content_text=text,
        bbox=None,
        embedding_key=None
    )
    session.add(ch)
    session.commit()
    session.refresh(ch)
    
    return [ch]


# ----------------------------------
# Chunking strategy for text files
# ----------------------------------
def chunk_text(session: Session, doc: Document) -> list[Chunk]:
    """Placeholder function for text file chunking."""
    ch = Chunk(
        document_id=doc.id,
        modality="text",
        page=1,
        content_text=None,
        bbox=None,
        embedding_key=None
    )
    session.add(ch)
    session.commit()
    session.refresh(ch)

    return [ch]


# ---------------------------------------------------
# Function to delete existing chunks for a document
# ---------------------------------------------------
def delete_existing_chunks(session: Session, doc: Document) -> int:
    """Delete all existing chunks for a document. Returns number deleted."""
    chunks = session.exec(
        select(Chunk).where(Chunk.document_id == doc.id)
    ).all()

    # Delete all chunks
    for ch in chunks:
        session.delete(ch)
    session.commit()

    return len(chunks)


# -------------------------------------------------------
# Function to run chunking based on document media type
# -------------------------------------------------------
def run_chunking(session: Session, doc_id: int, rebuild: bool = False) -> list[Chunk]:
    """Run chunking for a document by ID. Optionally rebuild existing chunks."""
    # Fetch document from DB
    doc = session.get(Document, doc_id)
    if not doc:
        raise ValueError(f"Document not found with id: {doc_id}")

    # Delete existing chunks if rebuild is requested
    if rebuild:
        deleted_count = delete_existing_chunks(session, doc)
        print(f"Deleted {deleted_count} existing chunks for document id {doc.id}")

    # Run chunking based on media type
    if doc.media_type == "pdf":
        return chunk_pdf(session, doc)
    elif doc.media_type == "image":
        return chunk_image(session, doc)
    elif doc.media_type == "text":
        return chunk_text(session, doc)
    else:
        raise ValueError(f"Unsupported media type for chunking: {doc.media_type}")