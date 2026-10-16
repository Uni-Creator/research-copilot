"""Turn a finished run into a readable report: numbered citations, comparison table, references.

Citations are stored as [cite:<chunk_id>]; here they become [1], [2] ... numbered by paper in
order of first appearance, and the same numbers key the reference list and the table.
"""
from __future__ import annotations

import html
import json
import re
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from copilot.graph.state import CopilotState, Paper, PaperNotes
from copilot.tools.chunking import paper_id_of

_RUN = re.compile(r"(?:\[cite:[^\]]+\]\s*)+")
_TAG = re.compile(r"\[cite:([^\]]+)\]")


def _cell(value: Any, limit: int = 160) -> str:
    text = ", ".join(value) if isinstance(value, list) else str(value or "")
    text = " ".join(text.replace("|", "/").split())
    return (text[: limit - 1] + "…") if len(text) > limit else (text or "-")


def _authors(p: Paper) -> str:
    names = p.authors[:3]
    return (", ".join(names) + (" et al." if len(p.authors) > 3 else "")) if names else "Unknown"


def number_citations(markdown: str) -> tuple[str, list[str]]:
    """Replace cite tags with [n]; returns (text, paper_ids in numbering order)."""
    order: list[str] = []

    def num(pid: str) -> int:
        if pid not in order:
            order.append(pid)
        return order.index(pid) + 1

    def sub(m: re.Match) -> str:
        nums = sorted({num(paper_id_of(i.strip())) for i in _TAG.findall(m.group(0))})
        tail = " " if m.group(0)[-1:].isspace() else ""
        return "[" + ", ".join(map(str, nums)) + "]" + tail

    return _RUN.sub(sub, markdown), order


def render_markdown(state: CopilotState) -> str:
    draft = state.get("draft")
    papers: dict[str, Paper] = state.get("papers", {})
    notes: dict[str, PaperNotes] = state.get("notes", {})
    body, order = number_citations(draft.markdown if draft else "_No draft was produced._")

    out = [f"# {state['question']}", "", body.strip(), ""]

    rows = [pid for pid in order if pid in notes]
    if rows:
        out += ["## Comparison of cited papers", "",
                "| # | Paper | Method | Datasets | Metrics | Limitations |",
                "|---|---|---|---|---|---|"]
        for pid in rows:
            n, p = notes[pid], papers.get(pid)
            label = f"{_authors(p).split(',')[0]} ({p.year or 'n.d.'})" if p else pid
            out.append(f"| {order.index(pid) + 1} | {_cell(label, 60)} | {_cell(n.method)} | "
                       f"{_cell(n.datasets)} | {_cell(n.metrics)} | {_cell(n.limitations)} |")
        out.append("")

    out += ["## References", ""]
    for i, pid in enumerate(order, 1):
        p = papers.get(pid)
        if p is None:
            out.append(f"{i}. {pid}")
            continue
        venue = f" {p.venue}." if p.venue else ""
        link = f" <{p.url}>" if p.url else ""
        out.append(f"{i}. {_authors(p)} ({p.year or 'n.d.'}). *{p.title}*.{venue}{link}")

    crit = state.get("critique")
    if crit and crit.support_rate is not None:
        out += ["", "---", f"_Automated check: {len(crit.checks)} cited claims verified against source text, "
                f"support rate {crit.support_rate:.0%}; draft v{draft.version if draft else 1}._"]
    return "\n".join(out) + "\n"


_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>
:root{{--bg:#fff;--fg:#1a1a1a;--muted:#666;--line:#ddd;--head:#f5f5f5}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16181c;--fg:#e8e8e8;--muted:#9aa;--line:#333;--head:#22252b}}}}
body{{background:var(--bg);color:var(--fg);font:16px/1.65 system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem}}
h1{{font-size:1.6rem}}h2{{margin-top:2rem;border-bottom:1px solid var(--line);padding-bottom:.2rem}}
table{{border-collapse:collapse;display:block;overflow-x:auto;font-size:.85rem}}
th,td{{border:1px solid var(--line);padding:.4rem .6rem;vertical-align:top;text-align:left}}
th{{background:var(--head)}}a{{color:#4a8cf0}}em{{color:var(--muted)}}
</style></head><body>{body}</body></html>"""


def render_html(markdown_text: str, title: str) -> str:
    import markdown

    body = markdown.markdown(markdown_text, extensions=["tables", "sane_lists"])
    return _HTML.format(title=html.escape(title), body=body)


def _encode(o: Any) -> Any:
    if isinstance(o, BaseModel):
        return o.model_dump(mode="json")
    if isinstance(o, Enum):
        return o.value
    raise TypeError(f"not serialisable: {type(o)}")


def save_run(out_dir: Path, state: CopilotState) -> dict[str, Path]:
    """Write report.md, report.html and run.json (full state: plan, papers, notes, trace)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    md = render_markdown(state)
    paths = {
        "report.md": out_dir / "report.md",
        "report.html": out_dir / "report.html",
        "run.json": out_dir / "run.json",
    }
    paths["report.md"].write_text(md, encoding="utf-8")
    paths["report.html"].write_text(render_html(md, state["question"]), encoding="utf-8")
    paths["run.json"].write_text(json.dumps(state, default=_encode, indent=2), encoding="utf-8")
    return paths
