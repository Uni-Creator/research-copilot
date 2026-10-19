"""Evaluation metrics for a finished run.

Known-item / ranked-retrieval metrics (per-topic, over the candidate pool)
precision_at_k    : share of top-k retrieved papers that are must-cites
recall_at_k       : share of must-cites appearing in the top-k retrieved papers
ndcg_at_k         : normalised discounted cumulative gain  (graded relevance: 1 if must-cite)
mrr               : mean reciprocal rank of the first must-cite in the pool ranking
average_precision : area under the precision-recall curve

Pipeline-stage coverage metrics
coverage_pool      : recall of must-cite papers among ALL retrieved papers
coverage_shortlist : recall of must-cite papers among papers the system chose to read
coverage_cited     : recall of must-cite papers among papers the report actually cites

Topic-coverage metric
subquestion_coverage : dict[sq_id -> fraction of that sub-question's keywords found in
                       any shortlisted paper's title+abstract]

Citation-quality metric
support_rate : judge-verified share of cited claims backed by the cited text
               (supported=1, partial=0.5, unsupported=0)
stripped_citations : fabricated citations the Writer produced and the pipeline removed
"""
from __future__ import annotations

import math
from typing import Any, Callable

from copilot.agents.critic import split_claims, support_rate, verify_claims
from copilot.graph.state import CopilotState, Paper, ResearchPlan
from copilot.tools.chunking import paper_id_of
from copilot.utils import normalize_title, tokenize

TITLE_JACCARD = 0.8
DEFAULT_KS = (5, 10, 20)


# Paper identity matching
def match_paper(must: dict, paper: Paper) -> bool:
    """A must-cite entry ({title, arxiv_id?, doi?}) matches by id, else by title similarity."""
    if must.get("arxiv_id") and must["arxiv_id"] == paper.arxiv_id:
        return True
    if must.get("doi") and paper.doi and must["doi"].lower() == paper.doi.lower():
        return True
    a = set(normalize_title(must.get("title", "")).split())
    b = set(normalize_title(paper.title).split())
    return bool(a and b) and len(a & b) / len(a | b) >= TITLE_JACCARD


# Stage-level coverage (pool / shortlist / cited)
def coverage(must_cite: list[dict], papers: list[Paper]) -> tuple[float | None, list[str]]:
    """Recall of must-cite items over *papers*.  Returns (score, list_of_missed_titles)."""
    if not must_cite:
        return None, []
    missed = [
        m.get("title") or m.get("arxiv_id", "?")
        for m in must_cite
        if not any(match_paper(m, p) for p in papers)
    ]
    return round(1 - len(missed) / len(must_cite), 4), missed


# Ranked-retrieval metrics  (require a ranked pool list)
def _relevance_vector(must_cite: list[dict], ranked: list[Paper]) -> list[int]:
    """Binary relevance list aligned with *ranked* (1 = must-cite, 0 = not)."""
    return [int(any(match_paper(m, p) for m in must_cite)) for p in ranked]


def precision_at_k(must_cite: list[dict], ranked: list[Paper], k: int) -> float | None:
    """Share of top-k papers that are must-cites. None when k=0 or no must-cites."""
    if not must_cite or k <= 0:
        return None
    rel = _relevance_vector(must_cite, ranked[:k])
    return round(sum(rel) / k, 4)


def recall_at_k(must_cite: list[dict], ranked: list[Paper], k: int) -> float | None:
    """Share of must-cites that appear within the top-k papers. None when no must-cites."""
    if not must_cite or k <= 0:
        return None
    rel = _relevance_vector(must_cite, ranked[:k])
    return round(sum(rel) / len(must_cite), 4)


def ndcg_at_k(must_cite: list[dict], ranked: list[Paper], k: int) -> float | None:
    """Normalised DCG at k with binary relevance (must-cites score 1, others 0).
    Returns None when there are no must-cites or k=0."""
    if not must_cite or k <= 0:
        return None
    rel = _relevance_vector(must_cite, ranked[:k])
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    n_ideal = min(len(must_cite), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_ideal))
    if idcg == 0:
        return None
    return round(dcg / idcg, 4)


def mrr(must_cite: list[dict], ranked: list[Paper]) -> float | None:
    """Reciprocal rank of the first must-cite in the ranked pool.  None if not found."""
    if not must_cite:
        return None
    for rank, p in enumerate(ranked, 1):
        if any(match_paper(m, p) for m in must_cite):
            return round(1.0 / rank, 4)
    return 0.0


def average_precision(must_cite: list[dict], ranked: list[Paper]) -> float | None:
    """Area-under-precision-recall curve (AP). None when no must-cites in corpus."""
    if not must_cite:
        return None
    rel = _relevance_vector(must_cite, ranked)
    found = 0
    total_p = 0.0
    for i, r in enumerate(rel, 1):
        if r:
            found += 1
            total_p += found / i
    if not found:
        return 0.0
    return round(total_p / len(must_cite), 4)


# Sub-question topic coverage
def subquestion_coverage(
    plan: ResearchPlan | None,
    papers: list[Paper],
) -> dict[str, float]:
    """For each sub-question estimate how well the shortlisted papers cover its keywords.

    Returns {sq_id: score ∈ [0,1]} where score = fraction of the sub-question's
    significant tokens that appear in at least one paper's title or abstract.
    An absent plan returns {}.
    """
    if not plan or not plan.sub_questions:
        return {}
    paper_tokens = set()
    for p in papers:
        paper_tokens.update(tokenize(p.title))
        if p.abstract:
            paper_tokens.update(tokenize(p.abstract))

    result: dict[str, float] = {}
    for sq in plan.sub_questions:
        sq_tokens = set(tokenize(sq.question))
        for q in sq.queries:
            sq_tokens.update(tokenize(q))
        if not sq_tokens:
            result[sq.id] = 0.0
            continue
        covered = sq_tokens & paper_tokens
        result[sq.id] = round(len(covered) / len(sq_tokens), 4)
    return result


# Main evaluation entry point
def evaluate_run(
    topic: dict,
    state: CopilotState,
    chunk_text: Callable[[str], str | None],
    judge_llm: Any,
    batch_size: int = 8,
    ks: tuple[int, ...] = DEFAULT_KS,
) -> dict[str, Any]:
    """Compute all evaluation metrics for a finished pipeline run.

    Parameters
    ----------
    topic       Dataset topic dict (must have ``id`` and optionally ``must_cite``).
    state       Finished CopilotState.
    chunk_text  Callable mapping chunk_id -> text (from the pipeline's vector store).
    judge_llm   LLM used to verify claim support.
    batch_size  Claims per judge call.
    ks          Tuple of k values for Precision@k / Recall@k / nDCG@k metrics.
    """
    papers: dict[str, Paper] = state.get("papers", {})
    shortlist_ids: list[str] = state.get("shortlist", [])
    shortlist: list[Paper] = [papers[i] for i in shortlist_ids if i in papers]
    draft = state.get("draft")
    plan: ResearchPlan | None = state.get("plan")

    cited_ids = {paper_id_of(c) for c in (draft.cited_chunk_ids if draft else [])}
    cited: list[Paper] = [papers[i] for i in cited_ids if i in papers]
    must = topic.get("must_cite", [])

    #stage-level coverage 
    all_papers: list[Paper] = list(papers.values())
    cov_pool, _missed_pool = coverage(must, all_papers)
    cov_short, missed_short = coverage(must, shortlist)
    cov_cited, _ = coverage(must, cited)

    # ranked-retrieval metrics 
    # Sort the pool by score descending for ranking metrics.
    ranked_pool = sorted(all_papers, key=lambda p: (-p.score, -p.citation_count, p.title))

    ranked_metrics: dict[str, Any] = {}
    for k in ks:
        ranked_metrics[f"precision_at_{k}"] = precision_at_k(must, ranked_pool, k)
        ranked_metrics[f"recall_at_{k}"] = recall_at_k(must, ranked_pool, k)
        ranked_metrics[f"ndcg_at_{k}"] = ndcg_at_k(must, ranked_pool, k)
    ranked_metrics["mrr"] = mrr(must, ranked_pool)
    ranked_metrics["average_precision"] = average_precision(must, ranked_pool)

    # sub-question / topic coverage 
    sq_cov = subquestion_coverage(plan, shortlist)
    mean_sq_cov = (
        round(sum(sq_cov.values()) / len(sq_cov), 4) if sq_cov else None
    )

    # claim verification 
    def safe_chunk_text(cid: str) -> str | None:
        t = chunk_text(cid)
        if t:
            return t
        # fallback: paper abstract is better than nothing for claim verification
        pid = paper_id_of(cid)
        p = papers.get(pid)
        if p and p.abstract:
            return p.abstract
        return None

    claims = split_claims(draft.markdown) if draft else []
    checks, judge_usage = verify_claims(judge_llm, claims, safe_chunk_text, batch_size)

    usage = state.get("usage")
    return {
        "topic": topic["id"],
        # stage coverage
        "coverage_pool": cov_pool,
        "coverage_shortlist": cov_short,
        "coverage_cited": cov_cited,
        "missed_in_shortlist": missed_short,
        # ranked-retrieval metrics
        **ranked_metrics,
        # sub-question coverage
        "subquestion_coverage": sq_cov,
        "mean_subquestion_coverage": mean_sq_cov,
        # citation / quality
        "n_papers_cited": len(cited_ids),
        "n_claims": len(claims),
        "support_rate": support_rate(checks),
        "unsupported_claims": [c.claim for c in checks if c.verdict == "unsupported"],
        "stripped_citations": draft.stripped_citations if draft else 0,
        "words": len(draft.markdown.split()) if draft else 0,
        "iterations": state.get("iteration", 0),
        "tokens": usage.total_tokens if usage else 0,
        "judge_tokens": judge_usage.total_tokens,
        "errors": len(state.get("errors", [])),
    }
