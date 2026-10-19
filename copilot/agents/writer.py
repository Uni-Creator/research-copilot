"""Writer agent: evidence-grounded literature review with verifiable citations.

Grounding mechanics
-------------------
* Evidence = top chunks per sub-question (from the vector store) + the previous draft's chunks.
* The LLM sees short aliases ([C1], [C2] ...) instead of long chunk ids, which it copies reliably.
* Aliases are mapped back to real chunk ids; any alias that does not exist is *stripped* and
  counted (``Draft.stripped_citations``), so a fabricated citation can never reach the report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from copilot.config import Settings
from copilot.graph.state import (
    ClaimCheck,
    CopilotState,
    Draft,
    Usage,
)
from copilot.llm import text_call
from copilot.progress import Tracer
from copilot.tools.chunking import abstract_chunk_id
from copilot.tools.vectorstore import VectorStore

EVIDENCE_CHARS = 700
_STRICT = re.compile(r"\[cite:\s*([^\]]+?)\s*\]")
_BARE = re.compile(r"\[(C\d+(?:\s*,\s*C\d+)*)\]")
_REAL = re.compile(r"\[cite:([^\]]+)\]")


@dataclass(frozen=True)
class EvidenceItem:
    chunk_id: str
    paper_id: str
    title: str
    year: int | None
    text: str
    page: int | None = None  # None for abstract chunks


@lru_cache(maxsize=1)
def _system_prompt() -> str:
    return (Path(__file__).parent.parent / "prompts" / "writer.md").read_text(encoding="utf-8")


def gather_evidence(
    state: CopilotState, store: VectorStore, settings: Settings, keep_chunk_ids: list[str] | None = None
) -> list[EvidenceItem]:
    papers = state.get("papers", {})
    shortlist = state.get("shortlist", [])
    allowed = set(shortlist)
    plan = state.get("plan")
    queries = (
        [f"{sq.question} {' '.join(sq.queries)}" for sq in plan.sub_questions]
        if plan and plan.sub_questions
        else [state["question"]]
    )

    picked: dict[str, Any] = {}
    for cid in keep_chunk_ids or []:  # revision: keep what was already cited
        c = store.get(cid)
        if c is not None and c.paper_id in allowed:
            picked.setdefault(cid, c)
    for q in queries:
        for chunk, _ in store.search(q, k=settings.evidence_per_subquestion, paper_ids=allowed):
            picked.setdefault(chunk.chunk_id, chunk)
    represented = {c.paper_id for c in picked.values()}
    for pid in shortlist:  # make sure every paper has a voice
        if len(picked) >= settings.max_evidence_chunks:
            break
        if pid not in represented and (a := store.get(abstract_chunk_id(pid))) is not None:
            picked[a.chunk_id] = a

    items = []
    for c in list(picked.values())[: settings.max_evidence_chunks]:
        p = papers.get(c.paper_id)
        items.append(
            EvidenceItem(c.chunk_id, c.paper_id, p.title if p else c.paper_id, p.year if p else None, c.text, c.page)
        )
    return items


def to_aliases(markdown: str, real_to_alias: dict[str, str]) -> str:
    """Rewrite real [cite:<chunk_id>] tags as [cite:C#] for the prompt; drop unknown ones."""

    def sub(m: re.Match) -> str:
        alias = real_to_alias.get(m.group(1).strip())
        return f"[cite:{alias}]" if alias else ""

    return _REAL.sub(sub, markdown)


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[\"'(])")


def _resolve_sentence(sent: str, alias_to_real: dict[str, str], cited: list[str]) -> tuple[str, int, int]:
    """Replace alias tags in one sentence. Returns (text, n_valid_ids, n_invalid_ids)."""
    valid = invalid = 0

    def sub(m: re.Match) -> str:
        nonlocal valid, invalid
        tags = []
        for raw in m.group(1).split(","):
            alias = raw.strip().removeprefix("cite:").strip()
            real = alias_to_real.get(alias)
            if real is None:
                invalid += 1
                continue
            valid += 1
            if real not in cited:
                cited.append(real)
            tags.append(f"[cite:{real}]")
        return "".join(tags)

    sent = _BARE.sub(sub, _STRICT.sub(sub, sent))
    return re.sub(r"[ \t]+([.,;])", r"\1", sent), valid, invalid


def resolve_citations(text: str, alias_to_real: dict[str, str]) -> tuple[str, list[str], int]:
    """Map LLM aliases to real chunk ids. Returns (markdown, cited_ids, n_stripped).

    Invented aliases are removed. A sentence whose citations were *all* invented is dropped
    entirely: it would otherwise survive as an unsupported, uncited assertion.
    """
    cited: list[str] = []
    stripped = 0
    lines = []
    for line in text.splitlines():
        kept = []
        for sent in _SENT.split(line):
            new, valid, invalid = _resolve_sentence(sent, alias_to_real, cited)
            stripped += invalid
            if invalid and not valid:
                continue
            kept.append(new)
        lines.append(" ".join(kept).rstrip())
    return "\n".join(lines).strip(), cited, stripped


def generate_draft(
    llm: Any,
    question: str,
    sub_questions: list[str],
    evidence: list[EvidenceItem],
    *,
    previous: Draft | None = None,
    flagged: list[ClaimCheck] | None = None,
    missing_topics: list[str] | None = None,
) -> tuple[Draft, Usage]:
    alias_to_real = {f"C{i}": e.chunk_id for i, e in enumerate(evidence, 1)}
    real_to_alias = {v: k for k, v in alias_to_real.items()}

    parts = [f"Research question: {question}"]
    if sub_questions:
        parts.append("Topics to cover (one section each):\n" + "\n".join(f"- {q}" for q in sub_questions))
    parts.append(
        "EVIDENCE:\n"
        + "\n\n".join(
            f"[{real_to_alias[e.chunk_id]}] {e.title} ({e.year or 'n.d.'})\n{e.text[:EVIDENCE_CHARS]}" for e in evidence
        )
    )
    if previous:
        parts.append("PREVIOUS DRAFT:\n" + to_aliases(previous.markdown, real_to_alias))
    if flagged:
        parts.append(
            "Reviewer problems (fix or remove these claims):\n"
            + "\n".join(f"- {c.claim} -> {c.verdict}: {c.reason}" for c in flagged)
        )
    if missing_topics:
        parts.append("New evidence was added for these gaps: " + "; ".join(missing_topics))

    text, usage = text_call(llm, [("system", _system_prompt()), ("human", "\n\n".join(parts))])
    markdown, cited, stripped = resolve_citations(text, alias_to_real)
    draft = Draft(
        markdown=markdown,
        cited_chunk_ids=cited,
        stripped_citations=stripped,
        version=(previous.version + 1) if previous else 1,
    )
    return draft, usage


def group_evidence(evidence: list[EvidenceItem]) -> list[dict]:
    """[{title, passages: ['p3', 'abstract', ...]}] per paper, for display."""
    groups: dict[str, dict] = {}
    for e in evidence:
        g = groups.setdefault(e.paper_id, {"title": e.title, "passages": []})
        g["passages"].append(f"p{e.page}" if e.page else "abstract")
    return list(groups.values())


def make_writer_node(llm: Any, store: VectorStore, settings: Settings) -> Callable[[CopilotState], dict]:
    def writer_node(state: CopilotState) -> dict:
        trace = Tracer("writer")
        previous = state.get("draft")
        critique = state.get("critique")
        flagged = (
            [c for c in critique.checks if c.verdict != "supported"]
            if critique and critique.action == "revise_draft"
            else []
        )
        missing = critique.missing_topics if critique and critique.action == "retrieve_more" else []
        version = (previous.version + 1) if previous else 1

        evidence = gather_evidence(state, store, settings, previous.cited_chunk_ids if previous else None)
        if not evidence:
            trace("no evidence available; cannot write", "warn")
            return {
                "draft": Draft(markdown="_No evidence could be retrieved for this question._", version=version),
                "errors": ["writer: no evidence available"],
                "trace": trace.events,
            }
        groups = group_evidence(evidence)
        plan = state.get("plan")
        trace(
            f"selected {len(evidence)} evidence passages from {len(groups)} papers (best matches per sub-question)",
            "evidence",
            papers=groups,
        )

        if flagged:
            trace(f"revising draft v{version}: fixing {len(flagged)} flagged claim(s)", "draft")
        elif missing:
            trace(f"rewriting as draft v{version} with new evidence on: {'; '.join(missing)}", "draft")
        else:
            trace(f"writing draft v{version} (every factual sentence must cite a passage)", "draft")

        draft, usage = generate_draft(
            llm,
            state["question"],
            [sq.question for sq in plan.sub_questions] if plan else [],
            evidence,
            previous=previous,
            flagged=flagged,
            missing_topics=missing,
        )
        extra = f"; {draft.stripped_citations} invented citation(s) removed" if draft.stripped_citations else ""
        trace(
            f"draft v{draft.version} ready: {len(draft.markdown.split())} words, "
            f"{len(draft.cited_chunk_ids)} passages cited{extra}",
            "draft",
            words=len(draft.markdown.split()),
            cited=len(draft.cited_chunk_ids),
            stripped=draft.stripped_citations,
            flagged=len(flagged),
        )
        return {"draft": draft, "usage": usage, "trace": trace.events}

    return writer_node
