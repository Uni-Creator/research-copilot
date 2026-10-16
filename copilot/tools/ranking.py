"""Heuristic paper ranking: rank-fusion relevance + citations + recency + keyword overlap.

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


def rank_papers(
    papers: list[Paper],
    rrf_scores: list[float],
    question: str,
    *,
    weights: RankWeights = RankWeights(),
    now_year: int | None = None,
) -> list[Paper]:
    """Return copies of ``papers`` with ``score`` set, sorted best-first."""
    if not papers:
        return []
    now_year = now_year or datetime.now(timezone.utc).year
    max_rrf = max(rrf_scores) or 1.0
    max_cites = max(p.citation_count for p in papers)
    q_tokens = set(tokenize(question))

    scored: list[Paper] = []
    for p, r in zip(papers, rrf_scores, strict=True):
        cite = math.log1p(p.citation_count) / math.log1p(max_cites) if max_cites > 0 else 0.0
        recency = 0.3 if p.year is None else max(0.0, 1 - max(0, now_year - p.year) / 10)
        text = set(tokenize(f"{p.title} {p.abstract}"))
        overlap = len(q_tokens & text) / len(q_tokens) if q_tokens else 0.0
        score = (
            weights.rrf * (r / max_rrf)
            + weights.citations * cite
            + weights.recency * recency
            + weights.overlap * overlap
        )
        scored.append(p.model_copy(update={"score": round(score, 4)}))
    scored.sort(key=lambda p: (-p.score, -p.citation_count, p.title))
    return scored
