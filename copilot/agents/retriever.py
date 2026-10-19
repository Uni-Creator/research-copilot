"""Retriever agent: run the plan's queries against all sources, dedupe, rank, shortlist.

Pipeline
--------
1. Collect queries from the plan that have not been searched yet.
2. Fan out (source x query) calls in a thread pool, reporting each one live as it finishes;
   a failing call is logged, not fatal.
3. Merge duplicates across sources (arXiv id / DOI / normalised title).
4. Score = reciprocal-rank fusion across result lists + citations + recency + keyword overlap.
5. First pass: shortlist the top ``max_papers``. Follow-up passes (Critic asked for more
   evidence): keep the old shortlist and append the best ``extra_per_pass`` new papers.
"""
from __future__ import annotations

import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from copilot.config import Settings
from copilot.graph.state import CopilotState, Paper, SearchFilters
from copilot.progress import Tracer
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
        trace = Tracer("retriever")
        question = state["question"]
        filters = state.get("filters") or SearchFilters()
        plan = state.get("plan")
        if plan is None:
            raise ValueError("retriever needs a plan; run the planner first")

        done = {q.lower() for q in state.get("searched_queries", [])}
        queries = [q for q in plan.all_queries() if q.lower() not in done]
        if not queries:
            trace("no new queries to search", "info")
            return {"trace": trace.events}

        # Preload what we already know so duplicates map onto existing records.
        existing = state.get("papers", {})
        old_shortlist = list(state.get("shortlist", []))
        index = PaperIndex()
        preloaded = {index.add(p) for p in existing.values()}

        trace(f"searching {len(queries)} queries \u00d7 {len(sources)} sources ({', '.join(sources)})", "search")

        def timed(name: str, q: str):
            t0 = time.perf_counter()
            return sources[name](q, filters, settings.max_results_per_query), time.perf_counter() - t0

        # Fan out; report each call as it completes. One failure must not sink the run.
        jobs = [(name, q) for q in queries for name in sources]
        done_jobs: dict[tuple[str, str], tuple[list[Paper], Exception | None]] = {}
        with ThreadPoolExecutor(max_workers=settings.retriever_workers) as pool:
            futures = {pool.submit(timed, name, q): (name, q) for name, q in jobs}
            for fut in as_completed(futures):
                name, q = futures[fut]
                try:
                    results, dt = fut.result()
                    done_jobs[(name, q)] = (results, None)
                    trace(f"{name} \u2190 \u201c{q}\u201d \u00b7 {len(results)} results ({dt:.1f}s)", "search",
                          source=name, query=q, n=len(results))
                except Exception as exc:
                    done_jobs[(name, q)] = ([], exc)
                    trace(f"{name} failed for \u201c{q}\u201d: {exc!r}", "warn", source=name, query=q)

        # Process in job order so ranking is deterministic regardless of completion order.
        rrf: defaultdict[int, float] = defaultdict(float)
        errors: list[str] = []
        per_source: Counter[str] = Counter()
        raw_total = 0
        for name, q in jobs:
            results, exc = done_jobs[(name, q)]
            if exc is not None:
                errors.append(f"retriever: {name} failed for {q!r}: {exc!r}")
                continue
            per_source[name] += len(results)
            raw_total += len(results)
            for rank, paper in enumerate(results):
                rrf[index.add(paper)] += 1.0 / (RRF_K + rank + 1)

        fresh = [(i, p) for i, p in enumerate(index.papers) if i not in preloaded]
        candidates = [(i, p) for i, p in fresh if _in_range(p, filters)]
        dropped_year = len(fresh) - len(candidates)
        sq_contexts = (
            [f"{sq.question} {' '.join(sq.queries)}" for sq in plan.sub_questions]
            if plan and plan.sub_questions else []
        )
        ranked = rank_papers(
            [p for _, p in candidates],
            [rrf[i] for i, _ in candidates],
            question,
            sub_questions=sq_contexts,
            weights=weights,
        )

        if old_shortlist:  # follow-up pass: grow, don't reshuffle
            fresh_ids = [p.paper_id for p in ranked[: settings.extra_per_pass]]
            shortlist = old_shortlist + [pid for pid in fresh_ids if pid not in old_shortlist]
            new_ids = [pid for pid in fresh_ids if pid not in old_shortlist]
        else:
            shortlist = [p.paper_id for p in ranked[: filters.max_papers or settings.max_papers]]
            new_ids = shortlist

        by_id = {p.paper_id: p for p in ranked}
        if not shortlist:
            errors.append("retriever: no papers found; check queries, filters or API errors")
            trace("no papers found; check the queries, the year filter, or API errors above", "warn")
        else:
            cutoff = by_id[new_ids[-1]].score if new_ids else 0.0
            note = f"; {dropped_year} outside the year filter" if dropped_year else ""
            trace(
                f"{raw_total} raw results \u2192 {len(fresh)} unique papers "
                f"({raw_total - len(fresh)} duplicates merged{note}). "
                f"Selected {len(new_ids)} to read (score cut-off {cutoff:.2f})",
                "papers",
                papers=[
                    {"rank": r, "id": pid, "title": by_id[pid].title, "year": by_id[pid].year,
                     "citations": by_id[pid].citation_count, "score": by_id[pid].score,
                     "sources": by_id[pid].sources, "n_queries": len(by_id[pid].found_by)}
                    for r, pid in enumerate(new_ids, 1)
                ],
                raw_results=dict(per_source), unique=len(fresh), not_selected=len(ranked) - len(new_ids),
            )

        return {
            "papers": by_id,
            "shortlist": shortlist,
            "searched_queries": queries,
            "errors": errors,
            "trace": trace.events,
        }

    return retriever_node
