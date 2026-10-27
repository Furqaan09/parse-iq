"""
End-to-end answer quality, through the REAL hosted LLM. Not run in CI

Skipped by default (see addopts in pyproject.toml). Run on demand with:
    HF_TOKEN=<your token> pytest -m llm
"""

import os
from pathlib import Path

import pytest
from sqlmodel import Session

from rag_eval.harness import (
    EVAL_DIR,
    AnswerResult,
    load_golden,
    render_answer_report,
    summarize_answers,
)

pytestmark = [pytest.mark.integration, pytest.mark.eval, pytest.mark.llm]

GOLDEN = load_golden()
PLACEHOLDER_TOKEN = "test-token-never-used"  # set by tests/conftest.py

# Loose floors: a broken pipeline fails these; normal model variance doesn't.
MIN_ANSWER_ACCURACY = 0.75
MAX_FALSE_REFUSAL_RATE = 0.20
MIN_CITATION_PRECISION = 0.80
MIN_REFUSAL_ACCURACY = 0.80
MAX_ERROR_RATE = 0.30  # provider error


@pytest.fixture(scope="module")
def answers(engine, eval_corpus) -> list[AnswerResult]:
    """Every golden question (answerable and not) through the full RAG pipeline."""
    if os.environ.get("HF_TOKEN", PLACEHOLDER_TOKEN) == PLACEHOLDER_TOKEN:
        pytest.skip("needs a real HF_TOKEN in the environment")

    from app.services.rag import rag_answer

    # Run for all questions and collect all results
    results = []
    with Session(engine) as session:
        for question in GOLDEN:
            try:
                response = rag_answer(session, question.question)
            except Exception as exc:  # provider outage: reported, never scored as wrong
                results.append(AnswerResult(question, error=f"{type(exc).__name__}: {exc}"[:150]))
                continue

            cited_docs = [eval_corpus[c["document_id"]] for c in response["citations"]]
            results.append(AnswerResult(question, response["answer"], cited_docs))

    # Write a report to be analyzed if needed
    report_dir = Path(os.environ.get("RAG_EVAL_REPORT_DIR", EVAL_DIR / "reports"))
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "answers.md").write_text(render_answer_report(results), encoding="utf-8")

    # If the provider is down, skip instead of failing
    errors = sum(r.error is not None for r in results)
    if errors / len(results) > MAX_ERROR_RATE:
        pytest.skip(f"LLM provider unavailable: {errors}/{len(results)} calls failed")
    return results


def test_answers_contain_the_expected_fact(answers):
    assert summarize_answers(answers)["answer_accuracy"] >= MIN_ANSWER_ACCURACY


def test_answerable_questions_are_not_refused(answers):
    assert summarize_answers(answers)["false_refusal_rate"] <= MAX_FALSE_REFUSAL_RATE


def test_citations_point_at_the_right_document(answers):
    # checks that retrieved chunk the model cited is from the correct document
    assert summarize_answers(answers)["citation_precision"] >= MIN_CITATION_PRECISION


def test_unanswerable_questions_are_refused_without_citations(answers):
    # ckecks that model refuses to answer unanswerable questions and does not cite any document
    assert summarize_answers(answers)["refusal_accuracy"] >= MIN_REFUSAL_ACCURACY
