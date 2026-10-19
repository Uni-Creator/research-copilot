"""CLI: python -m copilot.cli "your research question" [--out runs/slp] [--until critic]"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from copilot.config import Settings
from copilot.graph.state import SearchFilters
from copilot.graph.workflow import STAGES, build_copilot
from copilot.progress import format_event_text
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
    ap.add_argument("--brief", action="store_true", help="one line per step (hide plans, paper lists, verdict reasons)")
    args = ap.parse_args(argv)

    settings = Settings.from_env()
    if args.embedder:
        settings = replace(settings, embedder=args.embedder)
    kw = {"fetch_pages": None} if args.no_fulltext else {}
    copilot = build_copilot(settings, until=args.until, **kw)
    filters = SearchFilters(year_from=args.year_from, year_to=args.year_to, max_papers=args.max_papers)

    # every agent step is printed the moment it happens (plan, searches, papers chosen, reading, checks...)
    final = copilot.run_live(
        args.question, filters, on_event=lambda ev: print(format_event_text(ev, brief=args.brief), flush=True)
    )

    papers = final.get("papers", {})
    print(f"\nShortlist ({len(final.get('shortlist', []))} papers)")
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
