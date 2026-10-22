import pytest

from app.routes.documents import infer_media_type, sanitize_filename


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
    assert sanitize_filename(raw) == expected


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
    assert infer_media_type(filename, content_type) == expected
