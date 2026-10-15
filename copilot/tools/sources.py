"""Wires the concrete search backends (arXiv, Semantic Scholar) behind one interface."""
from __future__ import annotations

from typing import Callable

import httpx

from copilot.config import Settings
from copilot.graph.state import Paper, SearchFilters
from copilot.tools.arxiv import search_arxiv
from copilot.tools.http import DiskCache, RateLimiter
from copilot.tools.semantic_scholar import search_semantic_scholar

# (query, filters, limit) -> papers
SearchFn = Callable[[str, SearchFilters, int], list[Paper]]


def build_sources(settings: Settings) -> dict[str, SearchFn]:
    client = httpx.Client(
        timeout=settings.http_timeout,
        headers={"User-Agent": settings.user_agent},
        follow_redirects=True,
    )
    cache = DiskCache(settings.cache_dir / "http", ttl_seconds=settings.cache_ttl_days * 86400)
    arxiv_limiter = RateLimiter(3.0)  # arXiv asks for <= 1 request / 3 s
    s2_limiter = RateLimiter(0.25 if settings.s2_api_key else 1.1)

    def arxiv_fn(query: str, filters: SearchFilters, limit: int) -> list[Paper]:
        return search_arxiv(query, filters, limit, client=client, cache=cache, limiter=arxiv_limiter)

    def s2_fn(query: str, filters: SearchFilters, limit: int) -> list[Paper]:
        return search_semantic_scholar(
            query, filters, limit,
            client=client, cache=cache, limiter=s2_limiter, api_key=settings.s2_api_key,
        )

    return {"arxiv": arxiv_fn, "semantic_scholar": s2_fn}
