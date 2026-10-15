"""Runtime settings, read from environment variables (.env supported)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    # models
    planner_model: str = "claude-haiku-5-5"
    model_provider: str = "anthropic"
    # retrieval
    s2_api_key: str | None = None
    max_papers: int = 20            # shortlist size on the first pass
    extra_per_pass: int = 5         # papers added per "retrieve more" loop
    max_results_per_query: int = 10
    retriever_workers: int = 4
    # budget / loop guards
    max_iterations: int = 2
    max_total_tokens: int = 400_000
    # http / cache
    http_timeout: float = 20.0
    user_agent: str = "research-copilot/0.1 (mailto:you@example.com)"
    cache_dir: Path = Path(".cache")
    cache_ttl_days: int = 7
    traces_dir: Path = Path("traces")

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        d = cls()
        return cls(
            planner_model=os.environ.get("COPILOT_PLANNER_MODEL", d.planner_model),
            model_provider=os.environ.get("COPILOT_MODEL_PROVIDER", d.model_provider),
            s2_api_key=os.environ.get("S2_API_KEY") or None,
            max_papers=_int("COPILOT_MAX_PAPERS", d.max_papers),
            extra_per_pass=_int("COPILOT_EXTRA_PER_PASS", d.extra_per_pass),
            max_results_per_query=_int("COPILOT_MAX_RESULTS_PER_QUERY", d.max_results_per_query),
            max_iterations=_int("COPILOT_MAX_ITERATIONS", d.max_iterations),
            max_total_tokens=_int("COPILOT_MAX_TOTAL_TOKENS", d.max_total_tokens),
            user_agent=os.environ.get("COPILOT_USER_AGENT", d.user_agent),
            cache_dir=Path(os.environ.get("COPILOT_CACHE_DIR", d.cache_dir)),
            traces_dir=Path(os.environ.get("COPILOT_TRACES_DIR", d.traces_dir)),
        )
