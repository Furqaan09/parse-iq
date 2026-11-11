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
CANDIDATE_POOL = max(CONTEXT_SIZE * 4, 20)  # chunks each search leg returns


# Baseline set from running jina-reranker-v1-turbo on 35 answerable questions
# About one question below each measured value to account for noise
MIN_HIT_AT_6 = 0.97  # measured 100% (35/35)
MIN_HIT_AT_3 = 0.94  # measured 97% (34/35)
MIN_MRR = 0.85  # measured 0.89
MIN_POOL_RECALL = 0.95
MIN_CATEGORY_HIT_AT_6 = 0.75  # 1-10 questions per category; 1-question category must be perfect


@pytest.fixture(scope="module")
def runs(engine, eval_corpus) -> dict[str, list[RetrievalResult]]:
    """Every answerable question, through each retrieval configuration, once."""
    from app.retrieval.search import hybrid_search, keyword_search, search, vector_search

    configs = {
        "vector": lambda s, q: vector_search(s, q, limit=CONTEXT_SIZE),
        "keyword": lambda s, q: keyword_search(s, q, limit=CONTEXT_SIZE),
        "hybrid": lambda s, q: hybrid_search(s, q, limit=CANDIDATE_POOL)[:CONTEXT_SIZE],
        "hybrid+rerank": lambda s, q: search(s, q, top_k=CONTEXT_SIZE),  # production
        "candidate-pool": lambda s, q: hybrid_search(s, q, limit=CANDIDATE_POOL),
    }
    results: dict[str, list[RetrievalResult]] = {name: [] for name in configs}

    with Session(engine) as session:
        for question in ANSWERABLE:
            for name, run_search in configs.items():
                chunks = run_search(session, question.question)
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
        {name: results[name] for name in ("vector", "keyword", "hybrid", "hybrid+rerank")}
    )
    pool = summarize_retrieval(results["candidate-pool"], ks=(CANDIDATE_POOL,))
    report += f"\nCandidate pool recall@{CANDIDATE_POOL}: {pool[f'hit@{CANDIDATE_POOL}']:.0%}\n"

    report_dir = Path(os.environ.get("RAG_EVAL_REPORT_DIR", EVAL_DIR / "reports"))
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "retrieval.md").write_text(report, encoding="utf-8")


def test_correct_chunk_reaches_the_llm_context(runs):
    assert summarize_retrieval(runs["hybrid+rerank"])["hit@6"] >= MIN_HIT_AT_6


def test_correct_chunk_is_usually_in_the_top_3(runs):
    assert summarize_retrieval(runs["hybrid+rerank"])["hit@3"] >= MIN_HIT_AT_3


def test_correct_chunk_ranks_high_on_average(runs):
    assert summarize_retrieval(runs["hybrid+rerank"])["mrr"] >= MIN_MRR


def test_candidate_pool_contains_the_answer(runs):
    # The reranker only reorders what hybrid search hands it. Answer must be in the candidate pool.
    pool = summarize_retrieval(runs["candidate-pool"], ks=(CANDIDATE_POOL,))
    assert pool[f"hit@{CANDIDATE_POOL}"] >= MIN_POOL_RECALL


def test_reranking_is_at_least_as_good_as_dense_only(runs):
    dense = summarize_retrieval(runs["vector"])["mrr"]
    reranked = summarize_retrieval(runs["hybrid+rerank"])["mrr"]
    assert reranked >= dense


@pytest.mark.parametrize("category", sorted({q.category for q in ANSWERABLE}))
def test_no_question_category_is_left_behind(runs, category):
    results = by_category(runs["hybrid+rerank"])[category]
    assert summarize_retrieval(results)["hit@6"] >= MIN_CATEGORY_HIT_AT_6
