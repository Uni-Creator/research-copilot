"""Runtime settings, read from environment variables (.env supported)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    # models: cheap model for structured extraction, stronger one for writing/judging
    planner_model: str = "gemini-3.5-flash-lite"
    reader_model: str = "gemini-3.5-flash-lite"
    writer_model: str = "gemini-3.5-flash-lite"
    critic_model: str = "gemini-3.5-flash-lite"
    model_provider: str = "google_genai"
    # retrieval
    openalex_api_key: str | None = None
    openalex_mailto: str | None = None
    max_papers: int = 20            # shortlist size on the first pass
    extra_per_pass: int = 5         # papers added per "retrieve more" loop
    max_results_per_query: int = 15
    retriever_workers: int = 4
    # reading
    embedder: str = "bge"           # "bge" (sentence-transformers) | "hash" (offline, tests/dev)
    embedder_model: str = "BAAI/bge-small-en-v1.5"
    pdf_max_pages: int = 15
    chunk_chars: int = 1000
    chunk_overlap: int = 150
    max_chunks_per_paper: int = 40
    # writing / critic
    evidence_per_subquestion: int = 5
    max_evidence_chunks: int = 24
    support_threshold: float = 0.85
    verify_batch_size: int = 8
    # budget / loop guards
    max_iterations: int = 2
    max_total_tokens: int = 400_000
    # http / cache
    http_timeout: float = 20.0
    user_agent: str = "research-copilot/0.1"
    cache_dir: Path = Path(".cache")
    cache_ttl_days: int = 7
    traces_dir: Path = Path("traces")

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        d = cls()
        e = os.environ.get
        return cls(
            planner_model=e("COPILOT_PLANNER_MODEL", d.planner_model),
            reader_model=e("COPILOT_READER_MODEL", d.reader_model),
            writer_model=e("COPILOT_WRITER_MODEL", d.writer_model),
            critic_model=e("COPILOT_CRITIC_MODEL", d.critic_model),
            model_provider=e("COPILOT_MODEL_PROVIDER", d.model_provider),
            openalex_api_key=e("OPENALEX_API_KEY") or None,
            openalex_mailto=e("OPENALEX_MAILTO") or None,
            max_papers=_int("COPILOT_MAX_PAPERS", d.max_papers),
            extra_per_pass=_int("COPILOT_EXTRA_PER_PASS", d.extra_per_pass),
            max_results_per_query=_int("COPILOT_MAX_RESULTS_PER_QUERY", d.max_results_per_query),
            embedder=e("COPILOT_EMBEDDER", d.embedder),
            embedder_model=e("COPILOT_EMBEDDER_MODEL", d.embedder_model),
            pdf_max_pages=_int("COPILOT_PDF_MAX_PAGES", d.pdf_max_pages),
            support_threshold=_float("COPILOT_SUPPORT_THRESHOLD", d.support_threshold),
            max_iterations=_int("COPILOT_MAX_ITERATIONS", d.max_iterations),
            max_total_tokens=_int("COPILOT_MAX_TOTAL_TOKENS", d.max_total_tokens),
            user_agent=e("COPILOT_USER_AGENT", d.user_agent),
            cache_dir=Path(e("COPILOT_CACHE_DIR", d.cache_dir)),
            traces_dir=Path(e("COPILOT_TRACES_DIR", d.traces_dir)),
        )
