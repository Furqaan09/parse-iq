import re

import pytest

from app.routes.documents import _infer_media_type, _new_storage_key, _sanitize_filename


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("../../etc/passwd", "passwd"),
        ("..\\..\\windows\\system32\\evil.txt", "evil.txt"),
        ("/absolute/path/file.txt", "file.txt"),
        ("My Syllabus (Fall 2026).PDF", "My_Syllabus_Fall_2026.PDF"),
        ("normal_file.pdf", "normal_file.pdf"),
        ("", "file"),
        ("...", "file"),
    ],
)
def test_sanitize_filename_blocks_path_traversal(raw, expected):
    assert _sanitize_filename(raw) == expected


@pytest.mark.parametrize(
    "filename, content_type, expected",
    [
        ("a.pdf", "application/pdf", "pdf"),
        ("a.png", "image/png", "image"),
        ("notes.md", "", "text"),  # no content type -> falls back to the extension
        ("scan.JPEG", "", "image"),
        ("mystery.bin", "", "text"),  # unknown -> text default
    ],
)
def test_infer_media_type(filename, content_type, expected):
    assert _infer_media_type(filename, content_type) == expected


def test_new_storage_key_is_dated_random_and_safe():
    first = _new_storage_key("../../My Invoice.pdf")
    second = _new_storage_key("../../My Invoice.pdf")

    assert first != second
    assert re.fullmatch(r"\d{4}/\d{2}/[0-9a-f]{12}_My_Invoice\.pdf", first)
