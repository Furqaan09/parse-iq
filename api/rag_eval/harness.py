import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
CORPUS_DIR = EVAL_DIR / "corpus"
GOLDEN_PATH = EVAL_DIR / "golden.jsonl"

CATEGORIES = {"lookup", "paraphrase", "disambiguation", "negation", "update", "unanswerable"}
REFUSAL_PHRASE = "could not find"


# ----------------------------------
# The golden set (labeled answers)
# ----------------------------------
@dataclass(frozen=True)
class GoldenQuestion:
    id: str
    category: str
    question: str
    expected_doc: str | None  # filename of correct document
    evidence: str | None  # exact phrase the correct chunk must contain
    answer_any: tuple[str, ...]  # a correct answer mentions at least one of these

    @property
    def answerable(self) -> bool:
        return self.expected_doc is not None


def load_golden(path: Path = GOLDEN_PATH) -> list[GoldenQuestion]:
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            questions.append(GoldenQuestion(**{**row, "answer_any": tuple(row["answer_any"])}))
    return questions


# -------------------------------------------------
# Matching: is this chunk / answer the right one?
# -------------------------------------------------
def normalize(text: str) -> str:
    """Lowercase and collapse whitespace so line wraps don't break a match."""
    return " ".join(text.lower().split())


def is_relevant(chunk_text: str, chunk_doc: str, question: GoldenQuestion) -> bool:
    """A chunk is correct only if it comes from the right document AND contains the evidence."""
    if not question.answerable:
        return False
    return chunk_doc == question.expected_doc and normalize(question.evidence) in normalize(
        chunk_text
    )


def mentions_any(answer: str, phrases: tuple[str, ...]) -> bool:
    """True if the answer contains any phrase as a whole word or phrase"""
    return any(
        re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", answer, re.IGNORECASE) for phrase in phrases
    )


def is_refusal(answer: str) -> bool:
    return REFUSAL_PHRASE in answer.lower()


# -------------------
# Retrieval metrics
# -------------------
@dataclass
class RetrievalResult:
    question: GoldenQuestion
    rank: int | None  # position of the first correct chunk; None = not retrieved
    top_chunk: str | None = None  # "document > section" of the #1 chunk, to see what won instead


def first_relevant_rank(relevance: list[bool]) -> int | None:
    """[False, True, True] -> 2. None if nothing in the list is relevant."""
    for position, relevant in enumerate(relevance, start=1):
        if relevant:
            return position
    return None


def hit_at_k(rank: int | None, k: int) -> bool:
    """Did a correct chunk appear in the top k?"""
    return rank is not None and rank <= k


def reciprocal_rank(rank: int | None) -> float:
    """1.0 if correct chunk is #1, 0.5 if #2, 0.33 if #3 ... 0.0 if missing."""
    return 0.0 if rank is None else 1.0 / rank


def summarize_retrieval(results: list[RetrievalResult], ks: tuple[int, ...] = (1, 3, 6)) -> dict:
    """Hit@k for each k plus MRR, averaged over the questions given."""
    n = len(results)
    summary: dict = {"n": n}
    for k in ks:
        summary[f"hit@{k}"] = sum(hit_at_k(r.rank, k) for r in results) / n if n else 0.0
    summary["mrr"] = sum(reciprocal_rank(r.rank) for r in results) / n if n else 0.0
    return summary


def by_category(results: list) -> dict[str, list]:
    grouped: dict[str, list] = defaultdict(list)
    for result in results:
        grouped[result.question.category].append(result)
    return dict(sorted(grouped.items()))


def section_of(chunk_text: str) -> str | None:
    """Pull 'Important Dates' out of a chunk header line 'Section: Important Dates'."""
    match = re.search(r"^Section: (.+)$", chunk_text, re.MULTILINE)
    return match.group(1) if match else None


# ----------------------
# Answer (LLM) metrics
# ----------------------
@dataclass
class AnswerResult:
    question: GoldenQuestion
    answer: str = ""
    cited_docs: list[str] = field(default_factory=list)  # filename of each cited document
    error: str | None = None  # provider failure (e.g. capacity exhausted), not a wrong answer


def summarize_answers(results: list[AnswerResult]) -> dict:
    """
    answer_accuracy      answerable questions answered with a correct fact, not refused
    false_refusal_rate   answerable questions the model refused anyway
    citation_rate        answerable questions where the answer cited at least one source
    citation_precision   of all citations given, the share pointing at the right document
    refusal_accuracy     unanswerable questions refused with zero citations
    errors               calls that failed at the provider, excluded from every rate
    """
    ok = [r for r in results if r.error is None]
    answerable = [r for r in ok if r.question.answerable]
    unanswerable = [r for r in ok if not r.question.answerable]
    citations = [(r.question.expected_doc, doc) for r in answerable for doc in r.cited_docs]

    def rate(hits: int, total: int) -> float:
        return hits / total if total else 0.0

    return {
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "errors": len(results) - len(ok),
        "answer_accuracy": rate(
            sum(
                not is_refusal(r.answer) and mentions_any(r.answer, r.question.answer_any)
                for r in answerable
            ),
            len(answerable),
        ),
        "false_refusal_rate": rate(sum(is_refusal(r.answer) for r in answerable), len(answerable)),
        "citation_rate": rate(sum(bool(r.cited_docs) for r in answerable), len(answerable)),
        "citation_precision": rate(sum(exp == got for exp, got in citations), len(citations)),
        "refusal_accuracy": rate(
            sum(is_refusal(r.answer) and not r.cited_docs for r in unanswerable), len(unanswerable)
        ),
    }


# -----------------------------------------------
# Write and save reports in Markdown for review
# -----------------------------------------------
def _pct(value: float) -> str:
    return f"{value:.0%}"


def render_retrieval_report(configs: dict[str, list[RetrievalResult]]) -> str:
    """Markdown report comparing retrieval configurations side by side."""
    names = list(configs)

    lines = ["# Retrieval evaluation", ""]
    lines += ["| Metric | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    summaries = {name: summarize_retrieval(results) for name, results in configs.items()}
    for metric in ("hit@1", "hit@3", "hit@6", "mrr"):
        cells = [
            f"{summaries[n][metric]:.2f}" if metric == "mrr" else _pct(summaries[n][metric])
            for n in names
        ]
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    lines += ["", f"{summaries[names[0]]['n']} answerable questions.", ""]

    lines += ["## Hit@6 by category", ""]
    lines += ["| Category | n | " + " | ".join(names) + " |", "|---" * (len(names) + 2) + "|"]
    grouped = {name: by_category(results) for name, results in configs.items()}
    for category, results in grouped[names[0]].items():
        cells = [_pct(summarize_retrieval(grouped[n][category])["hit@6"]) for n in names]
        lines.append(f"| {category} | {len(results)} | " + " | ".join(cells) + " |")

    final = names[-1]
    misses = [r for r in configs[final] if not hit_at_k(r.rank, 1)]
    lines += ["", f"## Not ranked #1 by `{final}`", ""]
    if not misses:
        lines.append("None.")
    for r in misses:
        where = f"rank {r.rank}" if r.rank else "not retrieved"
        lines.append(f"- `{r.question.id}` ({where}): #1 was `{r.top_chunk}`")
    return "\n".join(lines) + "\n"


def render_answer_report(results: list[AnswerResult]) -> str:
    s = summarize_answers(results)
    lines = [
        "# Answer evaluation (live LLM)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Answer accuracy | {_pct(s['answer_accuracy'])} of {s['answerable']} answerable |",
        f"| False refusal rate | {_pct(s['false_refusal_rate'])} |",
        f"| Citation rate | {_pct(s['citation_rate'])} |",
        f"| Citation precision | {_pct(s['citation_precision'])} |",
        f"| Refusal accuracy | {_pct(s['refusal_accuracy'])} of {s['unanswerable']} unanswerable |",
        f"| Provider errors (excluded) | {s['errors']} |",
        "",
        "## Answers that need a look",
        "",
    ]
    flagged = 0
    for r in results:
        q = r.question
        if r.error:
            problem = f"provider error: {r.error}"
        elif q.answerable and is_refusal(r.answer):
            problem = "refused an answerable question"
        elif q.answerable and not mentions_any(r.answer, q.answer_any):
            problem = f"expected one of {list(q.answer_any)}"
        elif q.answerable and any(doc != q.expected_doc for doc in r.cited_docs):
            problem = f"cited {r.cited_docs}, expected {q.expected_doc}"
        elif not q.answerable and not (is_refusal(r.answer) and not r.cited_docs):
            problem = "answered an unanswerable question"
        else:
            continue
        flagged += 1
        lines.append(f"- `{q.id}`: {problem}  \n  > {r.answer[:200]}")
    if not flagged:
        lines.append("None.")
    return "\n".join(lines) + "\n"
