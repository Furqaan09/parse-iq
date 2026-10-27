import pytest

from rag_eval.harness import (
    CATEGORIES,
    CORPUS_DIR,
    AnswerResult,
    GoldenQuestion,
    RetrievalResult,
    first_relevant_rank,
    hit_at_k,
    is_refusal,
    is_relevant,
    load_golden,
    mentions_any,
    normalize,
    reciprocal_rank,
    section_of,
    summarize_answers,
    summarize_retrieval,
)

GOLDEN = load_golden()


def question(**overrides) -> GoldenQuestion:
    fields = {
        "id": "q",
        "category": "lookup",
        "question": "When is the study break?",
        "expected_doc": "syllabus.txt",
        "evidence": "Study Break: February 16 to 20",
        "answer_any": ("February 16",),
    }
    return GoldenQuestion(**{**fields, **overrides})


UNANSWERABLE = question(category="unanswerable", expected_doc=None, evidence=None, answer_any=())


# ---------------------------------------------
# Check if golden set is valid and consistent
# ---------------------------------------------
class TestGoldenSet:
    def test_question_ids_are_unique(self):
        ids = [q.id for q in GOLDEN]
        assert len(ids) == len(set(ids))

    def test_every_category_is_known(self):
        assert {q.category for q in GOLDEN} <= CATEGORIES

    def test_every_evidence_phrase_exists_in_its_document(self):
        for q in (q for q in GOLDEN if q.answerable):
            document = (CORPUS_DIR / q.expected_doc).read_text(encoding="utf-8")
            assert normalize(q.evidence) in normalize(document), q.id

    def test_answerable_questions_say_what_a_correct_answer_mentions(self):
        assert all(q.answer_any for q in GOLDEN if q.answerable)

    def test_unanswerable_questions_have_no_expected_document_or_answer(self):
        for q in (q for q in GOLDEN if q.category == "unanswerable"):
            assert q.expected_doc is None and q.evidence is None and q.answer_any == ()

    def test_set_covers_hard_cases_not_just_easy_lookups(self):
        counts = {c: sum(q.category == c for q in GOLDEN) for c in CATEGORIES}
        assert all(count >= 1 for count in counts.values()), counts
        assert counts["unanswerable"] >= 3


# --------------------------------------------------
# Check whether a retrieved chunk is the right one
# --------------------------------------------------
class TestIsRelevant:
    def test_right_document_with_evidence_is_relevant(self):
        chunk = "Section: Important Dates\nStudy Break: February 16 to 20, no classes"
        assert is_relevant(chunk, "syllabus.txt", question())

    def test_same_text_from_the_wrong_document_is_not_relevant(self):
        # Both syllabi share this exact line; only the right course counts.
        chunk = "Study Break: February 16 to 20, no classes"
        assert not is_relevant(chunk, "other_syllabus.txt", question())

    def test_right_document_without_the_evidence_is_not_relevant(self):
        assert not is_relevant("Final exam: 45%", "syllabus.txt", question())

    def test_line_wraps_and_case_do_not_break_the_match(self):
        chunk = "study break:\nFebruary   16 to 20"
        assert is_relevant(chunk, "syllabus.txt", question())

    def test_nothing_is_relevant_to_an_unanswerable_question(self):
        assert not is_relevant("anything", "syllabus.txt", UNANSWERABLE)


# -----------------
# Ranking Metrics
# -----------------
class TestRankingMetrics:
    def test_first_relevant_rank_is_one_based(self):
        assert first_relevant_rank([False, True, True]) == 2

    def test_first_relevant_rank_is_none_when_nothing_matches(self):
        assert first_relevant_rank([False, False]) is None

    @pytest.mark.parametrize(
        "rank, k, expected",
        [(1, 1, True), (3, 3, True), (4, 3, False), (None, 6, False)],
    )
    def test_hit_at_k(self, rank, k, expected):
        assert hit_at_k(rank, k) is expected

    @pytest.mark.parametrize("rank, expected", [(1, 1.0), (2, 0.5), (4, 0.25), (None, 0.0)])
    def test_reciprocal_rank(self, rank, expected):
        assert reciprocal_rank(rank) == expected

    def test_summary_averages_over_questions(self):
        results = [
            RetrievalResult(question(), rank=1),
            RetrievalResult(question(), rank=2),
            RetrievalResult(question(), rank=None),
            RetrievalResult(question(), rank=5),
        ]
        summary = summarize_retrieval(results)

        assert summary["n"] == 4
        assert summary["hit@1"] == 0.25
        assert summary["hit@3"] == 0.5
        assert summary["hit@6"] == 0.75
        assert summary["mrr"] == pytest.approx((1 + 0.5 + 0 + 0.2) / 4)

    def test_section_of_reads_the_chunk_header(self):
        chunk = "Document: x\nSection: Important Dates\nPage: 1\n\nbody"
        assert section_of(chunk) == "Important Dates"
        assert section_of("no header here") is None


# ---------------------
# Judging LLM answers
# ---------------------
class TestAnswerMatching:
    def test_matches_whole_words_only(self):
        assert mentions_any("No, notes are not allowed.", ("no",))
        assert not mentions_any("I know the notes policy.", ("no",))

    def test_matching_ignores_case(self):
        assert mentions_any("The exam is CLOSED BOOK.", ("closed book",))

    def test_matches_phrases_with_symbols(self):
        assert mentions_any("You receive $60 per month.", ("$60",))

    def test_refusal_is_detected(self):
        assert is_refusal("I could not find that information in the provided documents.")
        assert not is_refusal("The midterm is on March 3 [1].")


class TestAnswerSummary:
    def test_correct_cited_answer_scores_fully(self):
        result = AnswerResult(question(), "It is February 16 to 20 [1].", ["syllabus.txt"])
        s = summarize_answers([result])

        assert s["answer_accuracy"] == 1.0
        assert s["citation_rate"] == 1.0
        assert s["citation_precision"] == 1.0
        assert s["false_refusal_rate"] == 0.0

    def test_citing_the_wrong_document_lowers_precision(self):
        result = AnswerResult(
            question(), "February 16 [1][2].", ["syllabus.txt", "other_syllabus.txt"]
        )
        assert summarize_answers([result])["citation_precision"] == 0.5

    def test_refusing_an_answerable_question_is_a_false_refusal(self):
        result = AnswerResult(question(), "I could not find that information.", [])
        s = summarize_answers([result])

        assert s["false_refusal_rate"] == 1.0
        assert s["answer_accuracy"] == 0.0

    def test_unanswerable_question_needs_a_refusal_with_no_citations(self):
        refused = AnswerResult(UNANSWERABLE, "I could not find that information.", [])
        cited_refusal = AnswerResult(UNANSWERABLE, "I could not find it [1].", ["syllabus.txt"])
        hallucinated = AnswerResult(UNANSWERABLE, "Parking costs $200.", [])

        assert summarize_answers([refused])["refusal_accuracy"] == 1.0
        assert summarize_answers([cited_refusal])["refusal_accuracy"] == 0.0
        assert summarize_answers([hallucinated])["refusal_accuracy"] == 0.0

    def test_provider_errors_are_excluded_not_counted_as_wrong(self):
        failed = AnswerResult(question(), error="capacity exhausted")
        correct = AnswerResult(question(), "February 16 [1].", ["syllabus.txt"])
        s = summarize_answers([failed, correct])

        assert s["errors"] == 1
        assert s["answerable"] == 1
        assert s["answer_accuracy"] == 1.0
