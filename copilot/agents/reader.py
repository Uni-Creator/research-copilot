"""Reader agent: one paper -> full text chunks in the vector store + structured PaperNotes.

Runs once per shortlisted paper, in parallel (LangGraph ``Send`` fan-out). If the PDF cannot be
fetched or parsed it degrades to the abstract, so one bad PDF never breaks the run.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

from copilot.config import Settings
from copilot.graph.state import Chunk, Paper, PaperNotes, Usage, make_event
from copilot.llm import structured_call
from copilot.tools.chunking import abstract_chunk_id, chunk_pages
from copilot.tools.vectorstore import VectorStore

FACET_QUERIES = {
    "method": "proposed method model architecture approach",
    "data": "datasets benchmark corpus experiments setup",
    "metrics": "evaluation metrics results comparison BLEU accuracy",
    "limits": "limitations future work failure cases discussion",
}
MAX_CONTEXT_CHARS = 6000
FetchPages = Callable[[Paper], list[tuple[int, str]]]


class NotesOut(BaseModel):
    problem: str = ""
    method: str = ""
    datasets: list[str] = []
    metrics: list[str] = []
    key_results: str = ""
    limitations: str = ""


@lru_cache(maxsize=1)
def _system_prompt() -> str:
    return (Path(__file__).parent.parent / "prompts" / "reader.md").read_text(encoding="utf-8")


def select_context(store: VectorStore, paper: Paper) -> list[Chunk]:
    """Abstract first, then the best chunk(s) per facet (method / data / metrics / limits)."""
    picked: dict[str, Chunk] = {}
    if (a := store.get(abstract_chunk_id(paper.paper_id))) is not None:
        picked[a.chunk_id] = a
    for query in FACET_QUERIES.values():
        for chunk, _ in store.search(query, k=2, paper_ids={paper.paper_id}):
            picked.setdefault(chunk.chunk_id, chunk)
    out, total = [], 0
    for c in picked.values():
        if total + len(c.text) > MAX_CONTEXT_CHARS and out:
            break
        out.append(c)
        total += len(c.text)
    return out


def make_reader_node(
    llm: Any, store: VectorStore, fetch_pages: FetchPages | None, settings: Settings
) -> Callable[[dict], dict]:
    def reader_node(task: dict) -> dict:
        paper: Paper = task["paper"]
        errors: list[str] = []

        pages: list[tuple[int, str]] = []
        if fetch_pages is not None and paper.pdf_url:
            try:
                pages = fetch_pages(paper)
            except Exception as exc:
                errors.append(f"reader: PDF unavailable for {paper.paper_id} ({exc!r}); using abstract only")

        chunks: list[Chunk] = []
        if paper.abstract:
            chunks.append(
                Chunk(chunk_id=abstract_chunk_id(paper.paper_id), paper_id=paper.paper_id,
                      text=paper.abstract, kind="abstract")
            )
        chunks += chunk_pages(
            paper.paper_id, pages, size=settings.chunk_chars,
            overlap=settings.chunk_overlap, max_chunks=settings.max_chunks_per_paper,
        )
        store.add(chunks)

        context = select_context(store, paper)
        prompt = f"Title: {paper.title}\nYear: {paper.year or 'n.d.'}\n\n" + "\n\n".join(
            f"[excerpt {i}{' - abstract' if c.kind == 'abstract' else ''}]\n{c.text}"
            for i, c in enumerate(context, 1)
        )
        try:
            out, usage = structured_call(llm, NotesOut, [("system", _system_prompt()), ("human", prompt)])
        except Exception as exc:
            out, usage = NotesOut(problem=paper.abstract[:300]), Usage()
            errors.append(f"reader: extraction failed for {paper.paper_id} ({exc!r}); notes are abstract-only")

        notes = PaperNotes(paper_id=paper.paper_id, chunk_ids=[c.chunk_id for c in context], **out.model_dump())
        return {
            "notes": {paper.paper_id: notes},
            "usage": usage,
            "errors": errors,
            "trace": [
                make_event(
                    "reader",
                    f"read {paper.title[:60]!r}: {len(chunks)} chunks "
                    f"({'full text' if pages else 'abstract only'})",
                    paper_id=paper.paper_id, full_text=bool(pages), n_chunks=len(chunks),
                )
            ],
        }

    return reader_node
