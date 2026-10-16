"""CLI: python -m copilot.cli "your research question" [--out runs/slp] [--until critic]"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from copilot.config import Settings
from copilot.graph.state import SearchFilters
from copilot.graph.workflow import STAGES, build_copilot
from copilot.render import save_run


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Multi-agent research copilot")
    ap.add_argument("question")
    ap.add_argument("--year-from", type=int)
    ap.add_argument("--year-to", type=int)
    ap.add_argument("--max-papers", type=int)
    ap.add_argument("--until", choices=STAGES, default="critic", help="stop the pipeline after this stage")
    ap.add_argument("--out", type=Path, help="directory for report.md / report.html / run.json")
    ap.add_argument("--no-fulltext", action="store_true", help="read abstracts only (fast, cheap)")
    ap.add_argument("--embedder", choices=["bge", "hash"], help="override COPILOT_EMBEDDER")
    args = ap.parse_args(argv)

    settings = Settings.from_env()
    if args.embedder:
        settings = replace(settings, embedder=args.embedder)
    kw = {"fetch_pages": None} if args.no_fulltext else {}
    copilot = build_copilot(settings, until=args.until, **kw)
    filters = SearchFilters(year_from=args.year_from, year_to=args.year_to, max_papers=args.max_papers)

    final, seen = {}, 0
    for final in copilot.stream(args.question, filters):
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
    if usage := final.get("usage"):
        print(f"\nLLM calls={usage.llm_calls} tokens in/out={usage.input_tokens}/{usage.output_tokens}")

    out = args.out or settings.traces_dir / "last-run"
    final.setdefault("question", args.question)
    for name, path in save_run(out, final).items():
        print(f"saved {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
