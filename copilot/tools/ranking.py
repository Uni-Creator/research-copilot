"""Heuristic paper ranking: rank-fusion relevance + citations + recency + keyword overlap.

Ranking pipeline
----------------
1. Score each paper: weighted sum of (RRF, log-citations, recency, keyword-overlap).
2. MMR (Maximal Marginal Relevance) re-rank: trade relevance for diversity so the
   shortlist does not consist entirely of papers about the same narrow sub-topic.

Cheap and deterministic. Week 2+ can swap the overlap term for embedding similarity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from copilot.graph.state import Paper
from copilot.utils import tokenize


@dataclass(frozen=True)
class RankWeights:
    rrf: float = 0.5         # reciprocal-rank fusion across (query, source) result lists
    citations: float = 0.2   # log-scaled citation count
    recency: float = 0.1     # linear decay over 10 years
    overlap: float = 0.2     # share of question keywords present in title+abstract


@dataclass(frozen=True)
class MMRWeights:
    relevance: float = 0.7   # λ in MMR = λ·rel - (1-λ)·max_sim
    similarity: float = 0.3  # 1-λ: penalty for similarity to already-selected papers


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _paper_tokens(p: Paper) -> set[str]:
    tokens = set(tokenize(p.title))
    if p.abstract:
        tokens.update(tokenize(p.abstract))
    return tokens


def rank_papers(
    papers: list[Paper],
    rrf_scores: list[float],
    question: str,
    *,
    sub_questions: list[str] | None = None,
    weights: RankWeights = RankWeights(),
    mmr: MMRWeights = MMRWeights(),
    now_year: int | None = None,
    mmr_enabled: bool = True,
) -> list[Paper]:
    """Return copies of ``papers`` with ``score`` set, sorted best-first.

    First scores all papers with a weighted heuristic, then applies MMR
    re-ranking to improve diversity across the shortlist.
    """
    if not papers:
        return []
    now_year = now_year or datetime.now(timezone.utc).year
    max_rrf = max(rrf_scores) or 1.0
    max_cites = max(p.citation_count for p in papers)
    q_tokens = set(tokenize(question))

    sq_token_sets = [set(tokenize(sq)) for sq in (sub_questions or []) if sq]

    # Step 1: compute relevance scores
    scored: list[tuple[Paper, float]] = []
    for p, r in zip(papers, rrf_scores, strict=True):
        cite = math.log1p(p.citation_count) / math.log1p(max_cites) if max_cites > 0 else 0.0
        recency = 0.3 if p.year is None else max(0.0, 1 - max(0, now_year - p.year) / 10)

        title_tokens = set(tokenize(p.title))
        abstract_tokens = set(tokenize(p.abstract))
        text_tokens = title_tokens | abstract_tokens

        # Global overlap against the main question
        global_overlap = len(q_tokens & text_tokens) / len(q_tokens) if q_tokens else 0.0

        # Sub-question / query-specific relevance
        best_sq_score = 0.0
        for sq_tokens in sq_token_sets:
            if not sq_tokens:
                continue
            title_match = len(sq_tokens & title_tokens) / len(sq_tokens)
            text_match = len(sq_tokens & text_tokens) / len(sq_tokens)
            sq_score = 0.5 * title_match + 0.5 * text_match
            if sq_score > best_sq_score:
                best_sq_score = sq_score

        # Query that surfaced this paper (found_by)
        for fq in p.found_by:
            fq_tokens = set(tokenize(fq))
            if not fq_tokens:
                continue
            title_match = len(fq_tokens & title_tokens) / len(fq_tokens)
            text_match = len(fq_tokens & text_tokens) / len(fq_tokens)
            fq_score = 0.6 * title_match + 0.4 * text_match
            if fq_score > best_sq_score:
                best_sq_score = fq_score

        overlap = max(global_overlap, best_sq_score)

        score = (
            weights.rrf * (r / max_rrf)
            + weights.citations * cite
            + weights.recency * recency
            + weights.overlap * overlap
        )
        scored.append((p.model_copy(update={"score": round(score, 4)}), score))

    # Step 2: MMR re-ranking for diversity
    if not mmr_enabled or len(scored) <= 1:
        result = [p for p, _ in sorted(scored, key=lambda t: (-t[1], -t[0].citation_count, t[0].title))]
        return result

    candidate_tokens = [_paper_tokens(p) for p, _ in scored]
    relevance = [s for _, s in scored]
    papers_list = [p for p, _ in scored]

    selected: list[int] = []
    remaining = list(range(len(papers_list)))

    while remaining:
        best_idx = -1
        best_mmr = float("-inf")
        for i in remaining:
            # max similarity to already-selected papers
            if not selected:
                max_sim = 0.0
            else:
                max_sim = max(
                    _jaccard(candidate_tokens[i], candidate_tokens[j])
                    for j in selected
                )
            mmr_score = mmr.relevance * relevance[i] - mmr.similarity * max_sim
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = i
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [papers_list[i] for i in selected]
