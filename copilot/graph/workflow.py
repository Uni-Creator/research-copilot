"""Graph wiring.

    START -> planner -> retriever --(Send x N)--> reader -> writer -> critic_agent
                           ^                                   ^          |
                           |                                   |          | revise_draft
                           +----------- retrieve_more ---------+----------+  pass -> END

``until`` truncates the pipeline, which is what the ablations use:
    "retrieve"  planner -> retriever            (retrieval-only evals, cheap)
    "read"      ... -> reader
    "write"     ... -> writer                   ("multi-agent, no Critic" variant)
    "critic"    ... -> critic_agent loop        (full system)
Node names must differ from state keys (``critique`` is the key, ``critic_agent`` the node).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal

import httpx
from langgraph.graph import END, START, StateGraph

from copilot.agents.critic import make_critic_node
from copilot.agents.planner import make_planner_node
from copilot.agents.reader import FetchPages, make_reader_node
from copilot.agents.retriever import make_retriever_node
from copilot.agents.writer import make_writer_node
from copilot.config import Settings
from copilot.graph.routing import route_after_critic, route_after_retriever
from copilot.graph.state import CopilotState, SearchFilters, TraceEvent, make_initial_state
from copilot.llm import get_llm
from copilot.tools.embeddings import Embedder, get_embedder
from copilot.tools.http import DiskCache, RateLimiter
from copilot.tools.pdf import PdfFetcher
from copilot.tools.sources import SearchFn
from copilot.tools.vectorstore import VectorStore

Until = Literal["retrieve", "read", "write", "critic"]
STAGES: list[str] = ["retrieve", "read", "write", "critic"]
_UNSET: Any = object()


@dataclass
class Copilot:
    graph: Any
    store: VectorStore | None
    settings: Settings
    until: str

    def run(self, question: str, filters: SearchFilters | None = None) -> CopilotState:
        return self.graph.invoke(make_initial_state(question, filters), config={"recursion_limit": 60})

    def stream(
        self,
        question: str,
        filters: SearchFilters | None = None,
        on_event: Callable[[TraceEvent], None] | None = None,
    ):
        """Yield the full state after each graph step (as before) and, *as they happen*, call
        ``on_event`` for every progress event the agents emit (including from parallel Readers)."""
        for mode, chunk in self.graph.stream(
            make_initial_state(question, filters),
            stream_mode=["custom", "values"],
            config={"recursion_limit": 60},
        ):
            if mode == "custom":
                if on_event is not None and isinstance(chunk, dict) and "event" in chunk:
                    on_event(TraceEvent(**chunk["event"]))
            else:
                yield chunk

    def run_live(
        self,
        question: str,
        filters: SearchFilters | None = None,
        on_event: Callable[[TraceEvent], None] | None = None,
    ) -> CopilotState:
        """Run to completion, reporting progress through ``on_event``; returns the final state."""
        last = deque(self.stream(question, filters, on_event), maxlen=1)
        return last[0] if last else make_initial_state(question, filters)

    def chunk_text(self, chunk_id: str) -> str | None:
        c = self.store.get(chunk_id) if self.store else None
        return c.text if c else None


def default_fetcher(settings: Settings) -> PdfFetcher:
    client = httpx.Client(timeout=60.0, headers={"User-Agent": settings.user_agent}, follow_redirects=True)
    cache = DiskCache(Path(settings.cache_dir) / "http", ttl_seconds=settings.cache_ttl_days * 86400)
    return PdfFetcher(client, cache, RateLimiter(3.0), settings.pdf_max_pages)


def build_copilot(
    settings: Settings | None = None,
    *,
    until: Until = "critic",
    planner_llm: Any = None,
    reader_llm: Any = None,
    writer_llm: Any = None,
    critic_llm: Any = None,
    sources: dict[str, SearchFn] | None = None,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    fetch_pages: FetchPages | None = _UNSET,   # None => abstract-only reading
) -> Copilot:
    """Compile the pipeline. Every dependency can be injected (tests, ablations)."""
    settings = settings or Settings.from_env()
    stage = STAGES.index(until)

    def llm(given: Any, model: str, max_tokens: int):
        return given or get_llm(model, settings.model_provider, max_tokens=max_tokens)

    g = StateGraph(CopilotState)
    g.add_node("planner", make_planner_node(llm(planner_llm, settings.planner_model, 1500)))
    g.add_node("retriever", make_retriever_node(settings, sources))
    g.add_edge(START, "planner")
    g.add_edge("planner", "retriever")

    if stage == 0:
        g.add_edge("retriever", END)
        return Copilot(g.compile(), None, settings, until)

    if store is None:  # NB: not `store or ...`: an empty VectorStore is falsy (it defines __len__)
        store = VectorStore(embedder if embedder is not None else get_embedder(settings))
    fetch = default_fetcher(settings) if fetch_pages is _UNSET else fetch_pages
    after_read = "writer" if stage >= 2 else END
    g.add_node("reader", make_reader_node(llm(reader_llm, settings.reader_model, 1500), store, fetch, settings))
    g.add_conditional_edges(
        "retriever", partial(route_after_retriever, next_node=after_read),
        list(dict.fromkeys(["reader", after_read, END])),   # END = empty shortlist
    )
    g.add_edge("reader", after_read)

    if stage >= 2:
        g.add_node("writer", make_writer_node(llm(writer_llm, settings.writer_model, 3000), store, settings))
        if stage == 2:
            g.add_edge("writer", END)
        else:
            g.add_node("critic_agent", make_critic_node(llm(critic_llm, settings.critic_model, 2000), store, settings))
            g.add_edge("writer", "critic_agent")
            g.add_conditional_edges("critic_agent", route_after_critic, ["writer", "planner", END])
    return Copilot(g.compile(), store, settings, until)


def build_graph(settings: Settings | None = None, *, planner_llm: Any = None, sources: dict[str, SearchFn] | None = None):
    """Retrieval-only graph (Week 1 API, kept for backward compatibility)."""
    return build_copilot(settings, until="retrieve", planner_llm=planner_llm, sources=sources).graph
