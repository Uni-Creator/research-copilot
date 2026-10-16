"""PDF download + text extraction (PyMuPDF), with caching and a polite rate limit."""
from __future__ import annotations

import re

import httpx

from copilot.graph.state import Paper
from copilot.tools.http import DiskCache, RateLimiter, get_with_retry

_REFS = re.compile(r"^\s*(references|bibliography)\s*$", re.IGNORECASE | re.MULTILINE)


def clean_text(text: str) -> str:
    """Undo PDF line-wrapping: de-hyphenate, join wrapped lines, keep paragraph breaks."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def parse_pdf_bytes(data: bytes, max_pages: int = 15) -> list[tuple[int, str]]:
    """Return [(page_number, text)] up to ``max_pages``; stops at the references section."""
    import pymupdf  # lazy: keeps imports light for tests that don't read PDFs

    pages: list[tuple[int, str]] = []
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            raw = page.get_text("text")
            m = _REFS.search(raw)
            if m and i >= 2:                       # ignore a "References" mention on page 1-2
                head = clean_text(raw[: m.start()])
                if head:
                    pages.append((i + 1, head))
                break
            text = clean_text(raw)
            if text:
                pages.append((i + 1, text))
    return pages


class PdfFetcher:
    """Callable ``fetch(paper) -> pages``; downloads, parses and caches paper text."""

    def __init__(self, client: httpx.Client, cache: DiskCache, limiter: RateLimiter, max_pages: int = 15):
        self.client, self.cache, self.limiter, self.max_pages = client, cache, limiter, max_pages

    def __call__(self, paper: Paper) -> list[tuple[int, str]]:
        if not paper.pdf_url:
            return []
        key = {"url": paper.pdf_url, "max_pages": self.max_pages}
        if (hit := self.cache.get("pdf_text", key)) is not None:
            return [(int(p), t) for p, t in hit]
        resp = get_with_retry(self.client, paper.pdf_url, params={}, limiter=self.limiter)
        pages = parse_pdf_bytes(resp.content, self.max_pages)
        self.cache.set("pdf_text", key, pages)
        return pages
