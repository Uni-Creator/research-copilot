"""CopilotFactory: builds pipelines per request while sharing the expensive, stateful parts.

Shared across every run served by one process:
  * search clients + rate limiters (so concurrent runs cannot exceed arXiv's 1 request / 3 s)
  * the PDF fetcher (same reason, plus the on-disk PDF-text cache)
  * the embedding model (loaded once; ~130 MB for bge-small)
  * the vector store (a paper read in one run is not re-embedded in the next)
Anything can be overridden with keyword arguments (tests inject fakes this way).
"""

from __future__ import annotations

import threading
from typing import Any, Callable

from copilot.config import Settings
from copilot.graph.workflow import Copilot, build_copilot, default_fetcher
from copilot.tools.embeddings import get_embedder
from copilot.tools.sources import build_sources
from copilot.tools.vectorstore import VectorStore


class CopilotFactory:
    def __init__(self, settings: Settings | None = None, **overrides: Any) -> None:
        self.settings = settings or Settings.from_env()
        self._overrides = overrides
        self._shared: dict[str, Any] = {}
        self._lock = threading.Lock()

    def _shared_get(self, key: str, make: Callable[[], Any]) -> Any:
        with self._lock:
            if key not in self._shared:
                self._shared[key] = make()
            return self._shared[key]

    def __call__(self, until: str = "critic", fulltext: bool = True) -> Copilot:
        kw = dict(self._overrides)
        if "sources" not in kw:
            kw["sources"] = self._shared_get("sources", lambda: build_sources(self.settings))
        if until != "retrieve":
            if "embedder" not in kw:
                kw["embedder"] = self._shared_get("embedder", lambda: get_embedder(self.settings))
            if "store" not in kw:
                kw["store"] = self._shared_get("store", lambda: VectorStore(kw["embedder"]))
            if not fulltext:
                kw["fetch_pages"] = None
            elif "fetch_pages" not in kw:
                kw["fetch_pages"] = self._shared_get("fetcher", lambda: default_fetcher(self.settings))
        return build_copilot(self.settings, until=until, **kw)  # type: ignore[arg-type]
