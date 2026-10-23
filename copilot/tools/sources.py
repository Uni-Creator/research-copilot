"""Wire concrete search backends (arXiv and OpenAlex) behind one interface."""

from __future__ import annotations

import logging
from typing import Any, Callable

import httpx

from copilot.config import Settings
from copilot.graph.state import Paper, SearchFilters
from copilot.tools.arxiv import lookup_arxiv, search_arxiv
from copilot.tools.http import DiskCache, RateLimiter
from copilot.tools.openalex import lookup_openalex, search_openalex

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
            query,
            filters,
            limit,
            client=client,
            cache=cache,
            limiter=arxiv_limiter,
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


def lookup_known_item(
    entry: dict[str, Any],
    settings: Settings,
    *,
    client: httpx.Client | None = None,
    cache: DiskCache | None = None,
    arxiv_limiter: RateLimiter | None = None,
    openalex_limiter: RateLimiter | None = None,
) -> list[Paper]:
    """Explicit known-item retrieval by arXiv ID, DOI, and exact title."""
    from eval.metrics import match_paper

    client = client or httpx.Client(
        timeout=settings.http_timeout,
        headers={"User-Agent": settings.user_agent},
        follow_redirects=True,
    )
    cache = cache or DiskCache(
        settings.cache_dir / "http",
        ttl_seconds=settings.cache_ttl_days * 86400,
    )
    arxiv_limiter = arxiv_limiter or RateLimiter(3.0)
    openalex_limiter = openalex_limiter or RateLimiter(1.0)

    arxiv_id = entry.get("arxiv_id") or entry.get("arxiv")
    doi = entry.get("doi")
    title = entry.get("title")

    results: list[Paper] = []
    # 1. Look up by arXiv ID
    if arxiv_id:
        try:
            ar_papers = lookup_arxiv(
                arxiv_id=arxiv_id,
                client=client,
                cache=cache,
                limiter=arxiv_limiter,
            )
            for p in ar_papers:
                if match_paper(entry, p):
                    results.append(p)
        except Exception:
            pass

        if not results:
            try:
                oa_papers = lookup_openalex(
                    arxiv_id=arxiv_id,
                    client=client,
                    cache=cache,
                    limiter=openalex_limiter,
                    api_key=settings.openalex_api_key,
                    mailto=settings.openalex_mailto,
                )
                for p in oa_papers:
                    if match_paper(entry, p):
                        results.append(p)
            except Exception:
                pass

    # 2. Look up by DOI
    if doi and not results:
        try:
            oa_papers = lookup_openalex(
                doi=doi,
                client=client,
                cache=cache,
                limiter=openalex_limiter,
                api_key=settings.openalex_api_key,
                mailto=settings.openalex_mailto,
            )
            for p in oa_papers:
                if match_paper(entry, p):
                    results.append(p)
        except Exception:
            pass

    # 3. Look up by exact title
    if title and not results:
        try:
            oa_papers = lookup_openalex(
                title=title,
                client=client,
                cache=cache,
                limiter=openalex_limiter,
                api_key=settings.openalex_api_key,
                mailto=settings.openalex_mailto,
            )
            for p in oa_papers:
                if match_paper(entry, p):
                    results.append(p)
        except Exception:
            pass

        if not results:
            try:
                ar_papers = lookup_arxiv(
                    title=title,
                    client=client,
                    cache=cache,
                    limiter=arxiv_limiter,
                )
                for p in ar_papers:
                    if match_paper(entry, p):
                        results.append(p)
            except Exception:
                pass

    tag = f"known_item:{title or arxiv_id or doi}"
    for p in results:
        if tag not in p.found_by:
            p.found_by.append(tag)
    return results

