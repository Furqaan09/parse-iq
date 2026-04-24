import pytesseract
import re
from pathlib import Path
from typing import Optional
from PIL import Image
from PyPDF2 import PdfReader
from sqlmodel import Session, select

from app.models import Chunk, Document


# ------------------------------------------------------
# Chunking strategy for PDFs: multiple chunks per page
# ------------------------------------------------------
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


def _clean_text(text: str) -> str:
    """Light cleanup for extracted/OCR text."""
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text_into_chunks(
    text: str, target_size: int = 900, overlap: int = 150
) -> list[str]:
    """
    Split text into paragraph-aware overlapping chunks.
    """
    text = _clean_text(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        # If adding this paragraph stays within target, append it
        if len(current) + len(para) + 2 <= target_size:
            current = f"{current}\n\n{para}".strip()
        else:
            # Save current chunk if it exists
            if current:
                chunks.append(current)

            # If paragraph itself is too large, split it further
            if len(para) > target_size:
                start = 0
                while start < len(para):
                    end = start + target_size
                    piece = para[start:end].strip()
                    if piece:
                        chunks.append(piece)
                    start += target_size - overlap
                current = ""
            else:
                # Start new chunk with this paragraph
                current = para

    if current:
        chunks.append(current)

    # Add overlap between adjacent chunks
    overlapped = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            overlapped.append(chunk)
        else:
            prev = chunks[i - 1]
            prefix = prev[-overlap:].strip()
            merged = f"{prefix}\n\n{chunk}".strip()
            overlapped.append(merged)

    return overlapped


def chunk_pdf(session: Session, doc: Document) -> list[Chunk]:
    """Create multiple text chunks per PDF page."""
    # Load the PDF file
    path = _abs_storage_path(doc)
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    reader = PdfReader(str(path))
    created: list[Chunk] = []
    running_chunk_index = 0

    # Create multiple chunks for each page
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
        except Exception:
            raw_text = ""

        page_text = _safe_trim(raw_text)
        if not page_text:
            continue
        subchunks = split_text_into_chunks(page_text, target_size=900, overlap=150)

        for subchunk in subchunks:
            enriched_text = (
                f"Document: {doc.title}\n" f"Page: {page_num}\n\n" f"{subchunk}"
            )

            ch = Chunk(
                document_id=doc.id,
                chunk_index=running_chunk_index,
                modality="text",
                page=page_num,
                content_text=enriched_text,
                bbox=None,
                embedding_key=None,
            )

            session.add(ch)
            created.append(ch)
            running_chunk_index += 1

    # Commit all chunks at once to the database
    session.commit()
    for ch in created:
        session.refresh(ch)

    return created


# ------------------------------------------------------------
# Chunking strategy for Images: one chunk per image for CLIP
# + multiple OCR text chunks for text retrieval
# ------------------------------------------------------------
def _try_ocr_image(path: Path) -> Optional[str]:
    """Try to OCR the image and return extracted text or None."""
    try:
        img = Image.open(path)
        text = pytesseract.image_to_string(img)
        return _safe_trim(text)
    except Exception:
        return None


def chunk_image(session: Session, doc: Document) -> list[Chunk]:
    """
    Create one chunk for the image for CLIP
    and multiple OCR text chunks for text retrieval.
    """
    # Load the image file
    path = _abs_storage_path(doc)
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    try:
        Image.open(path).verify()
    except Exception as e:
        raise RuntimeError(f"Invalid image for chunking: {e}")

    created: list[Chunk] = []
    running_chunk_index = 0

    # Image chunk for visual embeddings
    image_chunk = Chunk(
        document_id=doc.id,
        modality="image",
        page=1,
        chunk_index=running_chunk_index,
        content_text=None,
        bbox=None,
        embedding_key=None,
    )
    session.add(image_chunk)
    created.append(image_chunk)
    running_chunk_index += 1

    # OCR text -> multiple text chunks
    ocr_text = _try_ocr_image(path)
    if ocr_text:
        subchunks = split_text_into_chunks(
            ocr_text,
            target_size=900,
            overlap=150,
        )

        for subchunk in subchunks:
            enriched_text = f"Document: {doc.title}\n" f"Page: 1\n\n" f"{subchunk}"

            text_chunk = Chunk(
                document_id=doc.id,
                modality="text",
                page=1,
                chunk_index=running_chunk_index,
                content_text=enriched_text,
                bbox=None,
                embedding_key=None,
            )
            session.add(text_chunk)
            created.append(text_chunk)
            running_chunk_index += 1

    session.commit()
    for ch in created:
        session.refresh(ch)

    return created


# ----------------------------------
# Chunking strategy for text files
# ----------------------------------
def chunk_text(session: Session, doc: Document) -> list[Chunk]:
    """Placeholder function for text file chunking."""
    path = _abs_storage_path(doc)
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    raw_text = path.read_text(encoding="utf-8", errors="ignore")
    raw_text = _safe_trim(raw_text)
    if not raw_text:
        return []

    # Create chunks from the text content
    created: list[Chunk] = []
    subchunks = split_text_into_chunks(raw_text, target_size=900, overlap=150)

    for idx, subchunk in enumerate(subchunks):
        enriched_text = f"Document: {doc.title}\n" f"Page: 1\n\n" f"{subchunk}"

        ch = Chunk(
            document_id=doc.id,
            modality="text",
            page=1,
            chunk_index=idx,
            content_text=enriched_text,
            bbox=None,
            embedding_key=None,
        )
        session.add(ch)
        created.append(ch)

    session.commit()
    for ch in created:
        session.refresh(ch)

    return created


# ---------------------------------------------------
# Function to delete existing chunks for a document
# ---------------------------------------------------
def delete_existing_chunks(session: Session, doc: Document) -> int:
    """Delete all existing chunks for a document. Returns number deleted."""
    chunks = session.exec(select(Chunk).where(Chunk.document_id == doc.id)).all()

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
