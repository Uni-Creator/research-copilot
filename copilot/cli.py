"""CLI: python -m copilot.cli "your research question" [--year-from 2021] [--max-papers 15]"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel

from copilot.config import Settings
from copilot.graph.state import SearchFilters, make_initial_state
from copilot.graph.workflow import build_graph


def _encode(o: Any) -> Any:
    if isinstance(o, BaseModel):
        return o.model_dump(mode="json")
    if isinstance(o, Enum):
        return o.value
    raise TypeError(f"not serialisable: {type(o)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Multi-agent research copilot (Week 1: plan + retrieve)")
    ap.add_argument("question")
    ap.add_argument("--year-from", type=int)
    ap.add_argument("--year-to", type=int)
    ap.add_argument("--max-papers", type=int)
    args = ap.parse_args(argv)

    settings = Settings.from_env()
    graph = build_graph(settings)
    state = make_initial_state(
        args.question, SearchFilters(year_from=args.year_from, year_to=args.year_to, max_papers=args.max_papers)
    )

    final, seen = state, 0
    for final in graph.stream(state, stream_mode="values"):
        trace = final.get("trace", [])
        for ev in trace[seen:]:
            print(f"[{ev.agent}] {ev.message}")
        seen = len(trace)

    papers = final.get("papers", {})
    print("\nShortlist")
    for i, pid in enumerate(final.get("shortlist", []), 1):
        p = papers[pid]
        print(f"{i:>2}. {p.score:.2f}  {p.year or '----'}  cites={p.citation_count:<5} {p.title[:90]}")
    for err in final.get("errors", []):
        print(f"! {err}")
    usage = final.get("usage")
    if usage:
        print(f"\nLLM calls={usage.llm_calls} tokens in/out={usage.input_tokens}/{usage.output_tokens}")

    settings.traces_dir.mkdir(parents=True, exist_ok=True)
    out = settings.traces_dir / f"run-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(final, default=_encode, indent=2), encoding="utf-8")
    print(f"trace saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
