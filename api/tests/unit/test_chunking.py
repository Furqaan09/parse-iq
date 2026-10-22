import pytest

from app.services.chunking import (
    CHUNK_TARGET_CHARS,
    _is_heading,
    _lines_from_text,
    _pack_section,
    _split_into_sections,
)


@pytest.mark.parametrize(
    "line, prev_blank, expected",
    [
        ("Important Dates", True, True),  # short capitalized text after blank line
        ("Course Description", True, True),  # text after blank line
        ("Important Dates", False, False),  # not preceded by a blank line
        ("Note that:", True, False),  # a `:` makes it a label, not a heading
        ("12", True, False),  # page number
        ("• Final Exam: TBA", True, False),  # bullet
        ("lowercase start", True, False),  # not capitalized
        ("This line is far too long to be a heading in any real document", True, False),
    ],
)
def test_is_heading(line, prev_blank, expected):
    assert _is_heading(line, prev_blank) is expected


def test_lines_from_text_rejoins_hyphenated_words_and_drops_page_numbers():
    text = "Intro\n\n3\nsome mys-\nterious agency\nFeb. 16-20"
    lines = [text for _, text, _ in _lines_from_text(text, page=1)]

    assert "some mysterious agency" in lines  # rejoined hyphenated word
    assert "3" not in lines  # dropped page number
    assert "Feb. 16-20" in lines  # not dehyphenated


def test_split_into_sections_groups_lines_under_headings():
    text = (
        "intro text\n\nImportant Dates\nStudy Break: Feb. 16-20\n\nExam Requirements\nClosed book"
    )
    sections = _split_into_sections(_lines_from_text(text, page=1))

    assert [heading for heading, _ in sections] == [None, "Important Dates", "Exam Requirements"]
    assert sections[1][1] == [(1, "Study Break: Feb. 16-20")]


@pytest.mark.parametrize(
    "text",
    [
        "Course Overview\n\nImportant Dates\nStudy Break: Feb. 16-20",  # back-to-back headings
        "intro text here\n\nAppendix",  # document ends on a heading
    ],
)
def test_split_into_sections_never_drops_text(text):
    # Regression: a heading with no body of its own used to vanish from every chunk.
    lines = _lines_from_text(text, page=1)
    sections = _split_into_sections(lines)
    kept = " ".join(
        (heading or "") + " " + " ".join(t for _, t in body) for heading, body in sections
    )

    for _, line, _ in lines:
        assert line in kept  # nothing is lost


def test_pack_section_keeps_every_chunk_within_target():
    body = [(1, f"line {i} " + "word " * 15) for i in range(60)]
    for chunk in _pack_section(body):
        assert sum(len(text) + 1 for _, text in chunk) <= CHUNK_TARGET_CHARS + 1


def test_pack_section_carries_short_last_line_as_overlap():
    body = [(1, f"line {i} " + "w" * 60) for i in range(30)]
    chunks = _pack_section(body)

    assert len(chunks) > 1
    assert chunks[1][0] == chunks[0][-1]  # chunk 2 first line == chunk 1 last line


def test_pack_section_does_not_emit_overlap_only_duplicates():
    # Regression: a chunk that is only the last line of the previous chunk used to be emitted as a duplicate.
    body = [(1, "x" * 790), (1, "y" * 790), (2, "z" * 790)]
    chunks = _pack_section(body)

    assert [[text[0] for _, text in c] for c in chunks] == [["x"], ["y"], ["z"]]


def test_pack_section_wraps_oversized_line_at_word_boundaries():
    body = [(1, " ".join(["word"] * 400))]
    for chunk in _pack_section(body):
        for _, text in chunk:
            assert all(token == "word" for token in text.split())
