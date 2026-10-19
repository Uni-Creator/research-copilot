"""Split parsed PDF pages into citable chunks."""

from __future__ import annotations

import re

from copilot.graph.state import Chunk

MIN_CHUNK_CHARS = 80  # drops page numbers, running headers, stray captions


def abstract_chunk_id(paper_id: str) -> str:
    return f"{paper_id}#abs"


def paper_id_of(chunk_id: str) -> str:
    return chunk_id.rsplit("#", 1)[0]


def _windows(text: str, size: int, overlap: int) -> list[str]:
    """Split an over-long paragraph into overlapping windows, preferring sentence ends."""
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if cut > start + size // 2:
                end = cut + 1
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return out


def chunk_pages(
    paper_id: str,
    pages: list[tuple[int, str]],
    *,
    size: int = 1000,
    overlap: int = 150,
    max_chunks: int = 40,
) -> list[Chunk]:
    """Paragraph-aware chunking. Small paragraphs are packed together up to ``size``
    characters; long ones are windowed. Chunks never span pages (keeps page numbers exact)."""
    pieces: list[tuple[int, str]] = []
    for page_no, text in pages:
        buf = ""
        for para in (p.strip() for p in re.split(r"\n\s*\n", text)):
            if not para:
                continue
            if len(para) > size:
                if buf:
                    pieces.append((page_no, buf))
                    buf = ""
                pieces += [(page_no, w) for w in _windows(para, size, overlap)]
            elif len(buf) + len(para) + 1 <= size:
                buf = f"{buf}\n{para}".strip()
            else:
                pieces.append((page_no, buf))
                buf = para
        if buf:
            pieces.append((page_no, buf))

    pieces = [(pg, t) for pg, t in pieces if len(t) >= MIN_CHUNK_CHARS][:max_chunks]
    return [
        Chunk(chunk_id=f"{paper_id}#{i}", paper_id=paper_id, text=t, page=pg) for i, (pg, t) in enumerate(pieces, 1)
    ]
