import logging
from pathlib import Path

import pytesseract
from PIL import Image
from pypdf import PdfReader

from app.models import MediaType

logger = logging.getLogger(__name__)
logging.getLogger("pypdf").setLevel(logging.ERROR)

# (1-based page number, raw text of that page)
Page = tuple[int, str]


# --------------------------------------------------
# Extract text from a file based on its media type
# --------------------------------------------------
def extract_pages(path: Path, media_type: MediaType) -> list[Page]:
    """Raw text per page. Raises if the file is missing or unreadable."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if media_type == MediaType.pdf:
        return _pdf_pages(path)
    if media_type == MediaType.image:
        return _ocr_pages(path)
    if media_type == MediaType.text:
        return [(1, path.read_text(encoding="utf-8", errors="ignore"))]

    raise ValueError(f"Unsupported media type: {media_type}")


# ------------------------------------------------------------------------------
# Function to extract text from PDF and return a list of pages with their text
# ------------------------------------------------------------------------------
def _pdf_pages(path: Path) -> list[Page]:
    pages: list[Page] = []
    for number, page in enumerate(PdfReader(str(path)).pages, start=1):
        try:
            pages.append((number, page.extract_text() or ""))
        except Exception:
            # One broken page shouldn't lose the rest of the document
            logger.warning("Text extraction failed on page %d of %s", number, path.name)
    return pages


# ---------------------------------------------------------------------
# Function to OCR an image and return a list of pages with their text
# ---------------------------------------------------------------------
def _ocr_pages(path: Path) -> list[Page]:
    try:
        Image.open(path).verify()  # verify valid image
    except Exception as e:
        raise RuntimeError(f"Invalid image: {e}") from e

    try:
        text = pytesseract.image_to_string(Image.open(path))
    except Exception:
        logger.warning("OCR failed for %s", path.name)
        return []
    return [(1, text)] if text.strip() else []
