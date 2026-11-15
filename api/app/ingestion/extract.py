import logging
from io import BytesIO

import pytesseract
from PIL import Image
from pypdf import PdfReader

from app.models import MediaType

logger = logging.getLogger(__name__)
logging.getLogger("pypdf").setLevel(logging.ERROR)

# (1-based page number, raw text of that page)
Page = tuple[int, str]


# ------------------------------------------------------------------------------
# Function to extract text from PDF and return a list of pages with their text
# ------------------------------------------------------------------------------
def _pdf_pages(data: bytes) -> list[Page]:
    pages: list[Page] = []
    for number, page in enumerate(PdfReader(BytesIO(data)).pages, start=1):
        try:
            pages.append((number, page.extract_text() or ""))
        except Exception:
            # One broken page shouldn't lose the rest of the document
            logger.warning("Text extraction failed on page %d", number)
    return pages


# ---------------------------------------------------------------------
# Function to OCR an image and return a list of pages with their text
# ---------------------------------------------------------------------
def _ocr_pages(data: bytes) -> list[Page]:
    try:
        Image.open(BytesIO(data)).verify()  # verify valid image
    except Exception as e:
        raise RuntimeError(f"Invalid image: {e}") from e

    try:
        text = pytesseract.image_to_string(Image.open(BytesIO(data)))
    except Exception:
        logger.warning("OCR failed")
        return []
    return [(1, text)] if text.strip() else []


# --------------------------------------------------
# Extract text from a file based on its media type
# --------------------------------------------------
def extract_pages(data: bytes, media_type: MediaType) -> list[Page]:
    """Raw text per page. Raises if the file is unreadable."""
    if media_type == MediaType.pdf:
        return _pdf_pages(data)
    if media_type == MediaType.image:
        return _ocr_pages(data)
    if media_type == MediaType.text:
        return [(1, data.decode("utf-8", errors="ignore"))]

    raise ValueError(f"Unsupported media type: {media_type}")
