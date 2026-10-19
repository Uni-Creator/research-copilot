"""Semantic Scholar Graph API search."""

from __future__ import annotations

from typing import Any

import httpx

from copilot.graph.state import Paper, SearchFilters
from copilot.tools.http import DiskCache, RateLimiter, get_with_retry

S2_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "title,abstract,year,authors,externalIds,citationCount,venue,openAccessPdf,url"


def year_param(filters: SearchFilters) -> str | None:
    """S2 year filter: '2020-2024', '2020-' or '-2024'."""
    if not (filters.year_from or filters.year_to):
        return None
    return f"{filters.year_from or ''}-{filters.year_to or ''}"


def parse_s2_item(item: dict[str, Any], query: str = "") -> Paper | None:
    title = (item.get("title") or "").strip()
    if not title:
        return None
    ext = item.get("externalIds") or {}
    arxiv_id = ext.get("ArXiv")
    pdf = (item.get("openAccessPdf") or {}).get("url") or (f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None)
    return Paper(
        title=title,
        authors=[a.get("name", "") for a in item.get("authors") or [] if a.get("name")],
        abstract=(item.get("abstract") or "").strip(),
        year=item.get("year"),
        venue=(item.get("venue") or None),
        url=item.get("url"),
        pdf_url=pdf,
        arxiv_id=arxiv_id,
        doi=ext.get("DOI"),
        s2_id=item.get("paperId"),
        citation_count=int(item.get("citationCount") or 0),
        sources=["semantic_scholar"],
        found_by=[query] if query else [],
    )


def search_semantic_scholar(
    query: str,
    filters: SearchFilters,
    limit: int,
    *,
    client: httpx.Client,
    cache: DiskCache,
    limiter: RateLimiter,
    api_key: str | None = None,
) -> list[Paper]:
    params: dict[str, Any] = {"query": query, "limit": limit, "fields": S2_FIELDS}
    if (yp := year_param(filters)) is not None:
        params["year"] = yp
    data = cache.get("s2", params)
    if data is None:
        headers = {"x-api-key": api_key} if api_key else None
        data = get_with_retry(client, S2_SEARCH, params=params, headers=headers, limiter=limiter).json()
        cache.set("s2", params, data)
    papers = (parse_s2_item(it, query) for it in data.get("data") or [])
    return [p for p in papers if p is not None]
