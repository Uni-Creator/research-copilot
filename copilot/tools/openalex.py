"""OpenAlex Works API search adapter."""

from __future__ import annotations

import re
from typing import Any

import httpx

from copilot.graph.state import Paper, SearchFilters
from copilot.tools.http import DiskCache, RateLimiter, get_with_retry

OPENALEX_WORKS = "https://api.openalex.org/works"
OPENALEX_FIELDS = ",".join(
    (
        "id",
        "doi",
        "title",
        "publication_year",
        "authorships",
        "abstract_inverted_index",
        "cited_by_count",
        "primary_location",
        "best_oa_location",
        "locations",
        "ids",
    )
)
_ARXIV_URL = re.compile(r"arxiv\.org/(?:abs|pdf)/([^/?#]+)", re.IGNORECASE)


def year_filter(filters: SearchFilters) -> str | None:
    """Build OpenAlex publication-date filters for the requested year range."""
    parts: list[str] = []
    if filters.year_from:
        parts.append(f"from_publication_date:{filters.year_from}-01-01")
    if filters.year_to:
        parts.append(f"to_publication_date:{filters.year_to}-12-31")
    return ",".join(parts) if parts else None


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """Reconstruct OpenAlex's abstract inverted index into reading order."""
    if not inverted_index:
        return ""

    indexed: list[tuple[int, str]] = []
    for token, positions in inverted_index.items():
        if not isinstance(token, str) or not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int) and position >= 0:
                indexed.append((position, token))

    if not indexed:
        return ""

    # Sorting pairs preserves repeated tokens at the same position rather than
    # silently overwriting them. OpenAlex indices normally have unique positions.
    indexed.sort(key=lambda pair: pair[0])
    return " ".join(token for _, token in indexed).strip()


def _location_urls(item: dict[str, Any]) -> tuple[str | None, str | None, list[str]]:
    """Return landing URL, PDF URL, and all candidate URLs from a work record."""
    locations = item.get("locations") or []
    best = item.get("best_oa_location") or {}
    primary = item.get("primary_location") or {}
    candidates = [best, primary, *locations]

    landing_url = None
    pdf_url = None
    all_urls: list[str] = []
    for location in candidates:
        if not isinstance(location, dict):
            continue
        for key in ("landing_page_url", "pdf_url"):
            value = location.get(key)
            if isinstance(value, str) and value:
                all_urls.append(value)
                if key == "landing_page_url" and landing_url is None:
                    landing_url = value
                if key == "pdf_url" and pdf_url is None:
                    pdf_url = value
    return landing_url, pdf_url, all_urls


def clean_search_query(query: str) -> str:
    """Prepare query for OpenAlex: strip conversational framing and overly long noise."""
    q = " ".join(query.split()).strip(" \"'.,:;?!")
    if len(q.split()) <= 8:
        return q
    cleaned = re.sub(
        r"^(?:compare|explain|review|assess|investigate|discuss|what are|how to|how should|please)\s+",
        "",
        q,
        flags=re.IGNORECASE,
    ).strip(" ?.")
    from copilot.utils import tokenize

    words = tokenize(cleaned)
    if len(words) > 8:
        return " ".join(words[:8])
    return cleaned or q


def _arxiv_id(item: dict[str, Any], urls: list[str]) -> str | None:
    ids = item.get("ids") or {}
    candidates = [
        *urls,
        str(ids.get("arxiv") or ""),
        str(ids.get("openalex") or ""),
        str(ids.get("doi") or ""),
        str(item.get("id") or ""),
        str(item.get("doi") or ""),
    ]
    for value in candidates:
        if not value:
            continue
        match = _ARXIV_URL.search(value)
        if match:
            return re.sub(r"v\d+$", "", match.group(1))
        direct = re.search(
            r"(?:arxiv(?:\.org)?(?:/abs/|:|\.))?(\d{4}\.\d{4,5}|[a-z\-]+/\d{7})",
            value,
            re.IGNORECASE,
        )
        if "arxiv" in value.lower() and direct:
            return re.sub(r"v\d+$", "", direct.group(1))
    return None


def parse_openalex_work(item: dict[str, Any], query: str = "") -> Paper | None:
    """Convert one OpenAlex work into the project's common Paper model."""
    title = (item.get("title") or item.get("display_name") or "").strip()
    if not title:
        return None

    ids = item.get("ids") or {}
    doi = item.get("doi") or ids.get("doi")
    if isinstance(doi, str):
        doi = re.sub(r"^https?://doi\.org/", "", doi.strip(), flags=re.IGNORECASE) or None

    landing_url, pdf_url, urls = _location_urls(item)
    arxiv_id = _arxiv_id(item, urls)
    if not pdf_url and arxiv_id:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

    authors: list[str] = []
    for authorship in item.get("authorships") or []:
        if not isinstance(authorship, dict):
            continue
        author = authorship.get("author") or {}
        name = author.get("display_name") if isinstance(author, dict) else None
        if name:
            authors.append(name)

    primary_location = item.get("primary_location") or {}
    source = primary_location.get("source") if isinstance(primary_location, dict) else None
    venue = source.get("display_name") if isinstance(source, dict) else None

    return Paper(
        title=title,
        authors=authors,
        abstract=reconstruct_abstract(item.get("abstract_inverted_index")),
        year=item.get("publication_year"),
        venue=venue or None,
        url=landing_url or (f"https://doi.org/{doi}" if doi else item.get("id")),
        pdf_url=pdf_url,
        arxiv_id=arxiv_id,
        doi=doi,
        citation_count=int(item.get("cited_by_count") or 0),
        sources=["openalex"],
        found_by=[query] if query else [],
    )


def search_openalex(
    query: str,
    filters: SearchFilters,
    limit: int,
    *,
    client: httpx.Client,
    cache: DiskCache,
    limiter: RateLimiter,
    api_key: str | None = None,
    mailto: str | None = None,
) -> list[Paper]:
    """Search OpenAlex works, using cache and the shared HTTP retry policy."""
    search_q = clean_search_query(query)
    params: dict[str, Any] = {
        "search": search_q,
        "per_page": max(1, min(int(limit), 100)),
        "select": OPENALEX_FIELDS,
    }
    if (year_range := year_filter(filters)) is not None:
        params["filter"] = year_range
    if mailto:
        params["mailto"] = mailto

    # Keep secrets out of cache keys and make cached results reusable across keys.
    data = cache.get("openalex", params)
    if data is None:
        request_params = dict(params)
        if api_key:
            request_params["api_key"] = api_key
        response = get_with_retry(
            client,
            OPENALEX_WORKS,
            params=request_params,
            limiter=limiter,
        )
        data = response.json()
        cache.set("openalex", params, data)

    records = data.get("results") or []
    papers = (parse_openalex_work(record, query) for record in records)
    return [paper for paper in papers if paper is not None]
