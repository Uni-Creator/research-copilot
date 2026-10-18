"""Live progress: rich trace events while agents work, plus formatters for the CLI and the UI.

Every node makes a ``Tracer`` and calls it for each meaningful step. A call (a) records the event so the
node can return it in ``state["trace"]`` (saved in run.json) and (b) pushes it *live* through LangGraph's
custom stream, so consumers see steps as they happen, even inside parallel Reader branches.
Outside a graph run (unit tests calling a node directly) step (b) is silently skipped.
"""
from __future__ import annotations

import textwrap
from typing import Any

from copilot.graph.state import TraceEvent, make_event


def _push_live(event: TraceEvent) -> None:
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
    except Exception:  # not inside a graph run, or an old langgraph without custom streaming
        return
    writer({"event": event.model_dump(mode="json")})


class Tracer:
    def __init__(self, agent: str) -> None:
        self.agent = agent
        self.events: list[TraceEvent] = []

    def __call__(self, message: str, kind: str = "info", **data: Any) -> TraceEvent:
        ev = make_event(self.agent, message, kind=kind, **data)
        self.events.append(ev)
        _push_live(ev)
        return ev


def short(text: str, n: int = 80) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


# Formatting
_VERDICT_TXT = {"supported": "[ok]  ", "partial": "[~]   ", "unsupported": "[FAIL]"}
_VERDICT_MD = {"supported": "✅", "partial": "🟡", "unsupported": "❌"}
_KIND_MD = {"thinking": "💭", "plan": "🧭", "search": "🔎", "papers": "📄", "read": "📖", "notes": "🧾",
            "evidence": "🧩", "draft": "✍️", "decision": "⚖️", "warn": "⚠️", "info": "•"}


def _paper_line(p: dict) -> str:
    srcs = ", ".join(p.get("sources", [])) or "?"
    return (f"{p['rank']:>2}. score {p['score']:.2f} · {p.get('year') or 'n.d.'} · "
            f"{p['citations']} cites · {short(p['title'], 90)}  [{srcs}; {p['n_queries']} queries]")


def _details(ev: TraceEvent) -> list[str]:
    d = ev.data
    if ev.kind == "plan":
        return [f"search: {q}" for q in d.get("queries", [])]
    if ev.kind == "papers":
        return [_paper_line(p) for p in d.get("papers", [])]
    if ev.kind == "notes":
        lines = []
        for k in ("method", "datasets", "metrics", "limitations"):
            v = d.get(k)
            if v:
                lines.append(f"{k}: {short(', '.join(v) if isinstance(v, list) else str(v), 200)}")
        return lines
    if ev.kind == "evidence":
        return [f"{short(g['title'], 70)}  [{', '.join(g['passages'])}]" for g in d.get("papers", [])]
    if ev.kind == "check" and d.get("reason"):
        return [f"why: {d['reason']}"]
    return []


def format_event_text(ev: TraceEvent, *, brief: bool = False) -> str:
    """Plain-text rendering for terminals."""
    head = f"[{ev.agent}] "
    pad = " " * (len(head) + 2)
    if ev.kind == "check" and "verdict" in ev.data:
        first = f"{_VERDICT_TXT.get(ev.data['verdict'], '')} {short(ev.message, 160)}"
    elif ev.kind == "thinking":
        first = "thinking: " + ev.message
    elif ev.kind == "warn":
        first = "WARNING: " + ev.message
    else:
        first = ev.message
    if ev.kind == "thinking" and not brief:
        wrapped = textwrap.wrap(first, 100)
        return "\n".join([head + wrapped[0]] + [pad + w for w in wrapped[1:]])
    lines = [head + first]
    if not brief:
        lines += [pad + line for line in _details(ev)]
    return "\n".join(lines)


def format_event_md(ev: TraceEvent) -> str:
    """Markdown rendering for the Streamlit status box."""
    icon = _KIND_MD.get(ev.kind, "•")
    if ev.kind == "check" and "verdict" in ev.data:
        icon = _VERDICT_MD.get(ev.data["verdict"], "•")
    if ev.kind == "thinking":
        body = f"*{ev.message}*"
    elif ev.kind == "plan":
        body = f"**{ev.message}**"
    else:
        body = ev.message
    out = f"{icon} **{ev.agent}** · {body}"
    details = _details(ev)
    if details:
        out += "\n" + "\n".join(f"  - {line}" for line in details)
    return out
