"""Wire concrete search backends (arXiv and OpenAlex) behind one interface."""
from __future__ import annotations

import logging
from typing import Callable

import httpx

from copilot.config import Settings
from copilot.graph.state import Paper, SearchFilters
from copilot.tools.arxiv import search_arxiv
from copilot.tools.http import DiskCache, RateLimiter
from copilot.tools.openalex import search_openalex

# (query, filters, limit) -> papers
SearchFn = Callable[[str, SearchFilters, int], list[Paper]]
logger = logging.getLogger(__name__)


def build_sources(settings: Settings) -> dict[str, SearchFn]:
    if not settings.openalex_api_key:
        logger.warning(
            "OPENALEX_API_KEY is unset; unauthenticated OpenAlex access is only suitable for limited testing. "
            "Configure a key in .env for regular use."
        )

    client = httpx.Client(
        timeout=settings.http_timeout,
        headers={"User-Agent": settings.user_agent},
        follow_redirects=True,
    )
    cache = DiskCache(
        settings.cache_dir / "http",
        ttl_seconds=settings.cache_ttl_days * 86400,
    )
    arxiv_limiter = RateLimiter(3.0)  # arXiv asks for <= 1 request / 3 s
    # Conservative search pacing; all concurrent OpenAlex calls share this limiter.
    openalex_limiter = RateLimiter(1.0)

    def arxiv_fn(query: str, filters: SearchFilters, limit: int) -> list[Paper]:
        return search_arxiv(
            query, filters, limit,
            client=client, cache=cache, limiter=arxiv_limiter,
        )

    def openalex_fn(query: str, filters: SearchFilters, limit: int) -> list[Paper]:
        return search_openalex(
            query,
            filters,
            limit,
            client=client,
            cache=cache,
            limiter=openalex_limiter,
            api_key=settings.openalex_api_key,
            mailto=settings.openalex_mailto,
        )

    return {"arxiv": arxiv_fn, "openalex": openalex_fn}
