"""
Retrieval quality, measured on a labeled golden set (rag_eval/golden.jsonl).
"""

import os
from pathlib import Path

import pytest
from sqlmodel import Session

from rag_eval.harness import (
    EVAL_DIR,
    RetrievalResult,
    by_category,
    first_relevant_rank,
    is_relevant,
    load_golden,
    render_retrieval_report,
    section_of,
    summarize_retrieval,
)

pytestmark = [pytest.mark.integration, pytest.mark.eval]

ANSWERABLE = [q for q in load_golden() if q.answerable]
CONTEXT_SIZE = 6  # chunks rag_answer() hands to the LLM
CANDIDATE_POOL = max(CONTEXT_SIZE * 4, 20)  # chunks FAISS hands to the reranker


# They exist to catch changes that make retrieval worse. When a change makes it
# better, raise them so the improvement can't quietly slip back.
MIN_HIT_AT_6 = 0.95
MIN_HIT_AT_3 = 0.875
MIN_MRR = 0.70
MIN_POOL_RECALL = 0.95
MIN_CATEGORY_HIT_AT_6 = 0.75  # smallest real category has 4 questions: 3 of 4


@pytest.fixture(scope="module")
def runs(engine, eval_corpus) -> dict[str, list[RetrievalResult]]:
    """Every answerable question, through each retrieval configuration, once."""
    from app.services.rag import retrieve_chunks

    configs = {
        "dense-only": {"top_k": CONTEXT_SIZE, "rerank_results": False},
        "dense+rerank": {"top_k": CONTEXT_SIZE, "rerank_results": True},  # production
        "candidate-pool": {"top_k": CANDIDATE_POOL, "rerank_results": False},
    }
    results: dict[str, list[RetrievalResult]] = {name: [] for name in configs}

    with Session(engine) as session:
        for question in ANSWERABLE:
            for name, kwargs in configs.items():
                chunks = retrieve_chunks(session, question.question, **kwargs)
                relevance = [
                    is_relevant(c.content_text or "", eval_corpus[c.document_id], question)
                    for c in chunks
                ]
                top_chunk = (
                    f"{eval_corpus[chunks[0].document_id]} > "
                    f"{section_of(chunks[0].content_text or '')}"
                    if chunks
                    else None
                )
                results[name].append(
                    RetrievalResult(question, first_relevant_rank(relevance), top_chunk)
                )

    _write_report(results)
    return results


def _write_report(results: dict[str, list[RetrievalResult]]) -> None:
    report = render_retrieval_report(
        {name: results[name] for name in ("dense-only", "dense+rerank")}
    )
    pool = summarize_retrieval(results["candidate-pool"], ks=(CANDIDATE_POOL,))
    report += f"\nCandidate pool recall@{CANDIDATE_POOL}: {pool[f'hit@{CANDIDATE_POOL}']:.0%}\n"

    report_dir = Path(os.environ.get("RAG_EVAL_REPORT_DIR", EVAL_DIR / "reports"))
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "retrieval.md").write_text(report, encoding="utf-8")


def test_correct_chunk_reaches_the_llm_context(runs):
    assert summarize_retrieval(runs["dense+rerank"])["hit@6"] >= MIN_HIT_AT_6


def test_correct_chunk_is_usually_in_the_top_3(runs):
    assert summarize_retrieval(runs["dense+rerank"])["hit@3"] >= MIN_HIT_AT_3


def test_correct_chunk_ranks_high_on_average(runs):
    assert summarize_retrieval(runs["dense+rerank"])["mrr"] >= MIN_MRR


def test_candidate_pool_contains_the_answer(runs):
    # The reranker only reorders what FAISS hands it. Answer must be in the candidate pool.
    pool = summarize_retrieval(runs["candidate-pool"], ks=(CANDIDATE_POOL,))
    assert pool[f"hit@{CANDIDATE_POOL}"] >= MIN_POOL_RECALL


def test_reranking_is_at_least_as_good_as_dense_only(runs):
    dense = summarize_retrieval(runs["dense-only"])["mrr"]
    reranked = summarize_retrieval(runs["dense+rerank"])["mrr"]
    assert reranked >= dense


@pytest.mark.parametrize("category", sorted({q.category for q in ANSWERABLE}))
def test_no_question_category_is_left_behind(runs, category):
    results = by_category(runs["dense+rerank"])[category]
    assert summarize_retrieval(results)["hit@6"] >= MIN_CATEGORY_HIT_AT_6
