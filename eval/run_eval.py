"""Run ablations over an eval set and print/save a results table.

    python -m eval.run_eval --topics eval/datasets/slp.json \
        --variants baseline,multi,multi_critic --out eval/results [--no-fulltext] [--limit 3]

Variants
    baseline      one search, abstracts, one write call
    multi         planner + retriever + readers + writer           (no Critic)
    multi_critic  full system with the Critic loop
Use --no-fulltext to run the multi variants on abstracts only: that isolates the effect of the
agent architecture from the effect of reading full PDFs.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from copilot.baseline import run_baseline
from copilot.config import Settings
from copilot.graph.state import CopilotState, SearchFilters
from copilot.graph.workflow import build_copilot
from copilot.llm import get_llm
from copilot.tools.sources import SearchFn, build_sources
from eval.metrics import evaluate_run

Runner = Callable[[str, SearchFilters], tuple[CopilotState, Callable[[str], str | None]]]
COLUMNS = ["coverage_shortlist", "coverage_cited", "support_rate", "n_claims",
           "stripped_citations", "tokens", "seconds"]


def evaluate(
    topics: list[dict], variants: dict[str, Runner], judge_llm: Any, settings: Settings, log=print
) -> list[dict]:
    results = []
    for topic in topics:
        filters = SearchFilters(**topic.get("filters", {}))
        for name, runner in variants.items():
            t0 = time.perf_counter()
            try:
                state, chunk_text = runner(topic["question"], filters)
                row = evaluate_run(topic, state, chunk_text, judge_llm, settings.verify_batch_size)
                row["error"] = None
            except Exception as exc:  # one failed run must not abort the whole eval
                row = {"topic": topic["id"], "error": repr(exc)}
            row.update(variant=name, seconds=round(time.perf_counter() - t0, 1))
            results.append(row)
            log(f"[{topic['id']}] {name}: " + ", ".join(f"{c}={row.get(c)}" for c in COLUMNS))
    return results


def _mean(values: list) -> float | None:
    vals = [v for v in values if isinstance(v, (int, float))]
    return round(statistics.mean(vals), 3) if vals else None


def results_table(results: list[dict]) -> str:
    """Markdown ablation table: mean of each metric per variant."""
    variants = list(dict.fromkeys(r["variant"] for r in results))
    lines = ["| Variant | " + " | ".join(COLUMNS) + " | topics | failed |",
             "|---|" + "---|" * (len(COLUMNS) + 2)]
    for v in variants:
        rows = [r for r in results if r["variant"] == v]
        ok = [r for r in rows if not r.get("error")]
        cells = ["-" if (m := _mean([r.get(c) for r in ok])) is None else str(m) for c in COLUMNS]
        lines.append(f"| {v} | " + " | ".join(cells) + f" | {len(ok)} | {len(rows) - len(ok)} |")
    return "\n".join(lines)


def make_variants(
    names: list[str], settings: Settings, sources: dict[str, SearchFn], fulltext: bool
) -> dict[str, Runner]:
    variants: dict[str, Runner] = {}
    kw: dict[str, Any] = {} if fulltext else {"fetch_pages": None}
    for name in names:
        if name == "baseline":
            llm = get_llm(settings.writer_model, settings.model_provider, max_tokens=3000)
            variants[name] = lambda q, f, llm=llm: run_baseline(q, f, settings, llm, sources)
        elif name in ("multi", "multi_critic"):
            cp = build_copilot(settings, until="write" if name == "multi" else "critic", sources=sources, **kw)
            variants[name] = lambda q, f, cp=cp: (cp.run(q, f), cp.chunk_text)
        else:
            raise SystemExit(f"unknown variant {name!r}")
    return variants


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", default="eval/datasets/slp.json")
    ap.add_argument("--variants", default="baseline,multi,multi_critic")
    ap.add_argument("--out", default="eval/results")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-fulltext", action="store_true")
    ap.add_argument("--embedder", choices=["bge", "hash"])
    ap.add_argument("--judge-model", help="defaults to the critic model; use a different one for independence")
    args = ap.parse_args(argv)

    settings = Settings.from_env()
    if args.embedder:
        settings = replace(settings, embedder=args.embedder)
    topics = json.loads(Path(args.topics).read_text(encoding="utf-8"))["topics"][: args.limit]
    sources = build_sources(settings)
    judge = get_llm(args.judge_model or settings.critic_model, settings.model_provider, max_tokens=2000)
    variants = make_variants(args.variants.split(","), settings, sources, fulltext=not args.no_fulltext)

    results = evaluate(topics, variants, judge, settings)
    table = results_table(results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out / "results.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
