import logging
import re
import textwrap
from pathlib import Path

import pytesseract
from PIL import Image
from pypdf import PdfReader
from sqlmodel import Session, select

from app.models import Chunk, Document

logger = logging.getLogger(__name__)
logging.getLogger("pypdf").setLevel(logging.ERROR)

CHUNK_TARGET_CHARS = 800  # header + body must stay under MiniLM's 256-token limit

_WHITESPACE = re.compile(r"[ \t]+")
_PAGE_NUMBER = re.compile(r"^\d{1,3}$")
_BULLET = re.compile(r"^(?:[•\-*·▪◦]|o\s)")
_LINE_WRAP_HYPHEN = re.compile(r"([a-z])-[ \t]*\n[ \t]*([a-z])")


# (page, text, is_heading)
Line = tuple[int, str, bool]


# ------------------
# Helper functions
# ------------------
def _is_heading(line: str, prev_blank: bool) -> bool:
    """Short, capitalized line right after a blank line, not a label/bullet/page number."""
    if not prev_blank:
        return False
    if len(line) > 60 or len(line.split()) > 7:
        return False
    if _PAGE_NUMBER.match(line) or _BULLET.match(line):
        return False
    if line.endswith((":", ".", ",", ";")):
        return False
    return line[0].isupper()


def _lines_from_text(text: str, page: int) -> list[Line]:
    """Normalize whitespace per line, drop page-number lines, flag headings."""
    # Rejoin words hyphenated across a line wrap ("mys-\nterious" -> "mysterious").
    text = _LINE_WRAP_HYPHEN.sub(r"\1\2", text)
    lines: list[Line] = []
    prev_blank = True

    for raw in text.split("\n"):
        line = _WHITESPACE.sub(" ", raw).strip()
        if not line:
            prev_blank = True
            continue
        if _PAGE_NUMBER.match(line):
            continue
        lines.append((page, line, _is_heading(line, prev_blank)))
        prev_blank = False
    return lines


def _split_into_sections(lines: list[Line]) -> list[tuple[str | None, list[tuple[int, str]]]]:
    """Group lines under the most recent heading. Text before any heading gets None."""
    sections: list[tuple[str | None, list[tuple[int, str]]]] = []
    heading: str | None = None
    body: list[tuple[int, str]] = []

    for page, text, is_heading in lines:
        if is_heading:
            if body:
                sections.append((heading, body))
                heading = text
            else:
                # Back-to-back headings: nest them instead of dropping the first.
                heading = f"{heading} > {text}" if heading else text
            body = []
        else:
            body.append((page, text))

    if body:
        sections.append((heading, body))
    elif heading:
        # Document ended on a heading: keep its text instead of dropping it.
        sections.append((heading, [(lines[-1][0], heading)]))
    return sections


def _pack_section(
    body: list[tuple[int, str]], target: int = CHUNK_TARGET_CHARS
) -> list[list[tuple[int, str]]]:
    """
    Greedily pack whole lines into chunks of up to `target` chars. Lines longer
    than `target` are wrapped at word boundaries. The previous chunk's last line
    is carried over as overlap, but only if it's short; otherwise it would
    create a near-duplicate chunk.
    """
    pieces = [(page, part) for page, line in body for part in textwrap.wrap(line, target)]
    chunks: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    size = 0
    carried = 0  # how many items in `current` are overlap, not new content

    for page, part in pieces:
        if size + len(part) + 1 > target and len(current) > carried:
            chunks.append(current)
            tail = current[-1]
            if len(tail[1]) <= target // 4:
                current, size, carried = [tail], len(tail[1]) + 1, 1
            else:
                current, size, carried = [], 0, 0
        current.append((page, part))
        size += len(part) + 1
    if len(current) > carried:
        chunks.append(current)
    return chunks


def _save_chunks(session: Session, doc: Document, lines: list[Line]) -> list[Chunk]:
    """Shared by PDF, image (OCR) and text: lines -> sections -> chunks -> DB rows."""
    created: list[Chunk] = []

    for heading, body in _split_into_sections(lines):
        for piece in _pack_section(body):
            page = piece[0][0]  # cite the page the chunk starts on
            header = f"Document: {doc.title}\n"
            if heading:
                header += f"Section: {heading}\n"
            header += f"Page: {page}\n\n"
            ch = Chunk(
                document_id=doc.id,
                chunk_index=len(created),
                modality="text",
                page=page,
                content_text=header + "\n".join(text for _, text in piece),
                bbox=None,
                embedding_key=None,
            )
            session.add(ch)
            created.append(ch)
    session.commit()

    for ch in created:
        session.refresh(ch)
    return created


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


def _safe_trim(text: str | None) -> str | None:
    """Trim text and return None if empty or None."""
    if text is None:
        return None
    t = text.strip()
    return t if t else None


def chunk_pdf(session: Session, doc: Document) -> list[Chunk]:
    """Create multiple text chunks per PDF page."""
    # Load the PDF file
    path = _abs_storage_path(doc)
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    lines: list[Line] = []
    for page_num, page in enumerate(PdfReader(str(path)).pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            logger.warning("Text extraction failed on page %d of document %s", page_num, doc.id)
            continue
        lines.extend(_lines_from_text(text, page_num))
    return _save_chunks(session, doc, lines)


# ---------------------------------------------------------------------------
# Chunking strategy for Images: multiple OCR text chunks for text retrieval
# ---------------------------------------------------------------------------
def _try_ocr_image(path: Path) -> str | None:
    """Try to OCR the image and return extracted text or None."""
    try:
        img = Image.open(path)
        text = pytesseract.image_to_string(img)
        return _safe_trim(text)
    except Exception:
        return None


def chunk_image(session: Session, doc: Document) -> list[Chunk]:
    """Create multiple OCR text chunks for text retrieval."""
    # Load the image file
    path = _abs_storage_path(doc)
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    try:
        Image.open(path).verify()
    except Exception as e:
        raise RuntimeError(f"Invalid image for chunking: {e}") from e

    # OCR text -> multiple text chunks
    ocr_text = _try_ocr_image(path)
    if not ocr_text:
        return []
    return _save_chunks(session, doc, _lines_from_text(ocr_text, page=1))


# ----------------------------------
# Chunking strategy for text files
# ----------------------------------
def chunk_text(session: Session, doc: Document) -> list[Chunk]:
    """Function for text file chunking."""
    path = _abs_storage_path(doc)
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    raw_text = path.read_text(encoding="utf-8", errors="ignore")
    return _save_chunks(session, doc, _lines_from_text(raw_text, page=1))


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
        logger.info("Deleted %d existing chunks for document id %s", deleted_count, doc.id)

    # Run chunking based on media type
    if doc.media_type == "pdf":
        return chunk_pdf(session, doc)
    elif doc.media_type == "image":
        return chunk_image(session, doc)
    elif doc.media_type == "text":
        return chunk_text(session, doc)
    else:
        raise ValueError(f"Unsupported media type for chunking: {doc.media_type}")
