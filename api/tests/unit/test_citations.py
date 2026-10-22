from app.services.rag import _validate_citations

CITATIONS = [
    {"n": 1, "title": "Syllabus A"},
    {"n": 2, "title": "Syllabus B"},
    {"n": 3, "title": "Syllabus C"},
]


def test_keeps_only_citations_the_answer_references():
    _, kept = _validate_citations("Midterm is March 5 [1] and the final is TBA [3].", CITATIONS)
    assert [c["n"] for c in kept] == [1, 3]


def test_strips_hallucinated_citation_markers_cleanly():
    answer, kept = _validate_citations("Midterm is March 5 [1], per the schedule [7].", CITATIONS)

    assert answer == "Midterm is March 5 [1], per the schedule."
    assert [c["n"] for c in kept] == [1]


def test_refusal_returns_no_citations():
    refusal = "I could not find that information in the provided documents."
    answer, kept = _validate_citations(refusal, CITATIONS)

    assert answer == refusal
    assert kept == []


def test_keeps_original_numbering_instead_of_renumbering():
    _, kept = _validate_citations("Closed book exam [3].", CITATIONS)
    assert kept == [{"n": 3, "title": "Syllabus C"}]


def test_no_citations_available_strips_every_marker():
    answer, kept = _validate_citations("Something [1] [2].", [])
    assert answer == "Something."
    assert kept == []
