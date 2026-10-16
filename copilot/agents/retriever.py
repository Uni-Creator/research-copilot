"""Retriever agent: run the plan's queries against all sources, dedupe, rank, shortlist.

Pipeline
--------
1. Collect queries from the plan that have not been searched yet.
2. Fan out (source x query) calls in a thread pool; a failing call is logged, not fatal.
3. Merge duplicates across sources (arXiv id / DOI / normalised title).
4. Score = reciprocal-rank fusion across result lists + citations + recency + keyword overlap.
5. First pass: shortlist the top ``max_papers``. Follow-up passes (Critic asked for more
   evidence): keep the old shortlist and append the best ``extra_per_pass`` new papers.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from copilot.config import Settings
from copilot.graph.state import CopilotState, Paper, SearchFilters, make_event
from copilot.tools.papers import PaperIndex
from copilot.tools.ranking import RankWeights, rank_papers
from copilot.tools.sources import SearchFn, build_sources

RRF_K = 60  # standard reciprocal-rank-fusion constant


def _in_range(p: Paper, f: SearchFilters) -> bool:
    if p.year is None:
        return True
    if f.year_from and p.year < f.year_from:
        return False
    if f.year_to and p.year > f.year_to:
        return False
    return True


def make_retriever_node(
    settings: Settings,
    sources: dict[str, SearchFn] | None = None,
    weights: RankWeights = RankWeights(),
) -> Callable[[CopilotState], dict]:
    sources = sources if sources is not None else build_sources(settings)

    def retriever_node(state: CopilotState) -> dict:
        question = state["question"]
        filters = state.get("filters") or SearchFilters()
        plan = state.get("plan")
        if plan is None:
            raise ValueError("retriever needs a plan; run the planner first")

        done = {q.lower() for q in state.get("searched_queries", [])}
        queries = [q for q in plan.all_queries() if q.lower() not in done]
        if not queries:
            return {"trace": [make_event("retriever", "no new queries to search")]}

        # Preload what we already know so duplicates map onto existing records.
        existing = state.get("papers", {})
        old_shortlist = list(state.get("shortlist", []))
        index = PaperIndex()
        preloaded = {index.add(p) for p in existing.values()}

        # Fan out. One failure must not sink the run.
        jobs = [(name, q) for q in queries for name in sources]
        with ThreadPoolExecutor(max_workers=settings.retriever_workers) as pool:
            futures = [
                pool.submit(sources[name], q, filters, settings.max_results_per_query)
                for name, q in jobs
            ]
            outcomes = []
            for (name, q), fut in zip(jobs, futures, strict=True):
                try:
                    outcomes.append((name, q, fut.result(), None))
                except Exception as exc:
                    outcomes.append((name, q, [], exc))

        rrf: defaultdict[int, float] = defaultdict(float)
        errors: list[str] = []
        per_source: Counter[str] = Counter()
        for name, q, results, exc in outcomes:
            if exc is not None:
                errors.append(f"retriever: {name} failed for {q!r}: {exc!r}")
                continue
            per_source[name] += len(results)
            for rank, paper in enumerate(results):
                rrf[index.add(paper)] += 1.0 / (RRF_K + rank + 1)

        candidates = [
            (i, p) for i, p in enumerate(index.papers) if i not in preloaded and _in_range(p, filters)
        ]
        ranked = rank_papers(
            [p for _, p in candidates], [rrf[i] for i, _ in candidates], question, weights=weights
        )

        if old_shortlist:  # follow-up pass: grow, don't reshuffle
            fresh = [p.paper_id for p in ranked[: settings.extra_per_pass]]
            shortlist = old_shortlist + [pid for pid in fresh if pid not in old_shortlist]
        else:
            shortlist = [p.paper_id for p in ranked[: filters.max_papers or settings.max_papers]]

        by_id = {p.paper_id: p for p in ranked}
        if not shortlist:
            errors.append("retriever: no papers found; check queries, filters or API errors")

        return {
            "papers": by_id,
            "shortlist": shortlist,
            "searched_queries": queries,
            "errors": errors,
            "trace": [
                make_event(
                    "retriever",
                    f"searched {len(queries)} queries x {len(sources)} sources; "
                    f"{len(ranked)} new unique papers; shortlist={len(shortlist)}",
                    raw_results=dict(per_source),
                    top=[
                        {"id": pid, "title": by_id[pid].title, "score": by_id[pid].score}
                        for pid in shortlist
                        if pid in by_id
                    ][:5],
                )
            ],
        }

    return retriever_node
