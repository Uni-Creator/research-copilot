"""arXiv search (Atom API): query building, fetching, parsing."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import httpx

from copilot.graph.state import Paper, SearchFilters
from copilot.tools.http import DiskCache, RateLimiter, get_with_retry
from copilot.utils import tokenize

ARXIV_API = "https://export.arxiv.org/api/query"
_NS = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
_ID = re.compile(r"abs/(.+?)(?:v\d+)?$")


def build_arxiv_query(query: str, filters: SearchFilters, mode: str = "and") -> str:
    """Natural-language query -> arXiv ``search_query`` string.

    Strategy
    --------
    * AND mode  (strict): top-8 tokens matched across ALL fields, plus title-field
      boost (ti:) for the top-4 tokens so that papers where the title contains the
      keywords are ranked higher.
    * OR mode  (broad fallback): top-6 tokens across all fields.
    * A year-range filter is appended when requested.

    This two-field approach dramatically improves recall for known-item retrieval
    where the paper's distinctive title words are part of the query (e.g. a model
    name like "Signs as Tokens" will match ``ti:signs AND ti:tokens`` as well as
    the broader ``all:sign AND all:language`` query).
    """
    tokens = tokenize(query)
    if not tokens:
        base = f"all:{query.strip()}"
    elif mode == "and":
        # Combine all: tokens (broad) with ti: tokens (precise title match)
        all_terms = " AND ".join(f"all:{t}" for t in tokens[:8])
        ti_terms = " AND ".join(f"ti:{t}" for t in tokens[:4])
        # Match broad fields AND (any of: title tokens match)
        base = f"({all_terms}) OR ({ti_terms})"
    else:
        # OR fallback: broader, uses top 6 tokens
        base = " OR ".join(f"all:{t}" for t in tokens[:6])

    if filters.year_from or filters.year_to:
        lo = f"{filters.year_from or 1991}01010000"
        hi = f"{filters.year_to or 2100}12312359"
        base = f"({base}) AND submittedDate:[{lo} TO {hi}]"
    return base


def _text(node: ET.Element | None) -> str:
    return " ".join((node.text or "").split()) if node is not None else ""


def parse_arxiv_feed(xml_text: str, query: str = "") -> list[Paper]:
    root = ET.fromstring(xml_text)
    papers: list[Paper] = []
    for entry in root.findall("a:entry", _NS):
        m = _ID.search(_text(entry.find("a:id", _NS)))
        title = _text(entry.find("a:title", _NS))
        if not m or not title:
            continue
        arxiv_id = m.group(1)
        pdf_url = next(
            (ln.get("href") for ln in entry.findall("a:link", _NS) if ln.get("title") == "pdf"),
            f"https://arxiv.org/pdf/{arxiv_id}",
        )
        published = _text(entry.find("a:published", _NS))
        papers.append(
            Paper(
                title=title,
                authors=[_text(a.find("a:name", _NS)) for a in entry.findall("a:author", _NS)],
                abstract=_text(entry.find("a:summary", _NS)),
                year=int(published[:4]) if published[:4].isdigit() else None,
                venue=_text(entry.find("x:journal_ref", _NS)) or None,
                url=f"https://arxiv.org/abs/{arxiv_id}",
                pdf_url=pdf_url,
                arxiv_id=arxiv_id,
                doi=_text(entry.find("x:doi", _NS)) or None,
                sources=["arxiv"],
                found_by=[query] if query else [],
            )
        )
    return papers


def search_arxiv(
    query: str,
    filters: SearchFilters,
    limit: int,
    *,
    client: httpx.Client,
    cache: DiskCache,
    limiter: RateLimiter,
) -> list[Paper]:
    """Search arXiv; falls back from AND to OR if the strict query finds nothing."""
    for mode in ("and", "or"):
        params = {
            "search_query": build_arxiv_query(query, filters, mode),
            "start": 0,
            "max_results": limit,
            "sortBy": "relevance",
        }
        xml_text = cache.get("arxiv", params)
        if xml_text is None:
            xml_text = get_with_retry(client, ARXIV_API, params=params, limiter=limiter).text
            cache.set("arxiv", params, xml_text)
        papers = parse_arxiv_feed(xml_text, query)
        if papers:
            return papers
    return []
