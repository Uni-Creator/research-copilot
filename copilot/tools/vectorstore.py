"""Minimal thread-safe vector store (numpy cosine search, JSON+npy persistence).

At review scale (a few thousand chunks) brute force is instant and dependency-free.
Swap for Chroma/FAISS later behind the same four methods: add / search / get / has_paper.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from copilot.graph.state import Chunk
from copilot.tools.embeddings import Embedder


class VectorStore:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._chunks: dict[str, Chunk] = {}
        self._ids: list[str] = []
        self._mat: np.ndarray | None = None
        self._lock = threading.RLock()

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, chunks: list[Chunk]) -> int:
        """Embed and store new chunks (existing ids are skipped). Returns how many were added."""
        with self._lock:
            new = [c for c in chunks if c.chunk_id not in self._chunks]
            if not new:
                return 0
            vecs = self.embedder.embed_documents([c.text for c in new])
            self._mat = vecs if self._mat is None else np.vstack([self._mat, vecs])
            for c in new:
                self._chunks[c.chunk_id] = c
                self._ids.append(c.chunk_id)
            return len(new)

    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def has_paper(self, paper_id: str) -> bool:
        return any(c.paper_id == paper_id for c in self._chunks.values())

    def search(self, query: str, k: int = 5, paper_ids: set[str] | None = None) -> list[tuple[Chunk, float]]:
        with self._lock:
            if self._mat is None or not self._ids:
                return []
            scores = self._mat @ self.embedder.embed_query(query)
            if paper_ids is not None:
                mask = np.array([self._chunks[i].paper_id in paper_ids for i in self._ids])
                scores = np.where(mask, scores, -np.inf)
            order = np.argsort(-scores)[:k]
            return [(self._chunks[self._ids[i]], float(scores[i])) for i in order if np.isfinite(scores[i])]

    # ---- persistence ------------------------------------------------------ #
    def save(self, directory: Path) -> None:
        with self._lock:
            directory.mkdir(parents=True, exist_ok=True)
            if self._mat is not None:
                np.save(directory / "vectors.npy", self._mat)
            (directory / "chunks.json").write_text(
                json.dumps([self._chunks[i].model_dump() for i in self._ids]), encoding="utf-8"
            )

    @classmethod
    def load(cls, directory: Path, embedder: Embedder) -> "VectorStore":
        store = cls(embedder)
        chunks = [Chunk(**d) for d in json.loads((directory / "chunks.json").read_text(encoding="utf-8"))]
        store._chunks = {c.chunk_id: c for c in chunks}
        store._ids = [c.chunk_id for c in chunks]
        vec_path = directory / "vectors.npy"
        store._mat = np.load(vec_path) if vec_path.exists() else None
        return store
