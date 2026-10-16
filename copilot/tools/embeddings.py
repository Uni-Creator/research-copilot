"""Embedders. ``bge`` for real runs (small enough for a 6 GB GPU or CPU); ``hash`` for tests/dev."""
from __future__ import annotations

import zlib
from typing import Protocol

import numpy as np

from copilot.utils import tokenize

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / np.where(norms == 0, 1.0, norms)).astype(np.float32)


class HashingEmbedder:
    """Deterministic signed bag-of-(uni+bi)grams. Not semantic, but offline and stable."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = tokenize(text)
        for t in toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]:
            h = zlib.crc32(t.encode())
            v[h % self.dim] += 1.0 if (zlib.crc32(t.encode() + b"#") & 1) else -1.0
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.stack([self._vec(t) for t in texts])) if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is not installed. `pip install sentence-transformers`, "
                "or set COPILOT_EMBEDDER=hash for a quick offline run."
            ) from exc
        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self._bge = "bge" in model_name.lower()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, batch_size=32), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        q = (BGE_QUERY_PREFIX + text) if self._bge else text
        return np.asarray(self.model.encode([q], normalize_embeddings=True)[0], dtype=np.float32)


def get_embedder(settings) -> Embedder:
    if settings.embedder == "hash":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(settings.embedder_model)
