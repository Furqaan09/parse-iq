import logging
import re
import textwrap

from sqlmodel import Session, select

from app.ingestion.extract import extract_pages
from app.models import Chunk, Document
from app.settings import API_DIR

logger = logging.getLogger(__name__)

CHUNK_TARGET_CHARS = 800

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
                page=page,
                content_text=header + "\n".join(text for _, text in piece),
            )
            session.add(ch)
            created.append(ch)
    session.commit()

    for ch in created:
        session.refresh(ch)
    return created


def chunk_document(session: Session, doc: Document) -> list[Chunk]:
    """Replace a document's chunks: extract pages -> lines -> sections -> chunk rows."""
    # Read the file first: if it's missing or broken, the existing chunks are left alone
    pages = extract_pages(API_DIR / doc.storage_path, doc.media_type)
    lines = [line for page, text in pages for line in _lines_from_text(text, page)]

    deleted = _delete_chunks(session, doc)
    logger.info("Deleted %d existing chunks for document %s", deleted, doc.id)
    return _save_chunks(session, doc, lines)


def _delete_chunks(session: Session, doc: Document) -> int:
    """Delete all of a document's chunks. Returns how many were deleted."""
    chunks = session.exec(select(Chunk).where(Chunk.document_id == doc.id)).all()
    for ch in chunks:
        session.delete(ch)
    session.commit()
    return len(chunks)
