"""Single-agent baseline: ONE search with the raw question, abstracts as evidence, ONE writing call.

No planner, no full text, no critic. It is the control group for every ablation.
"""

from __future__ import annotations

from typing import Any, Callable

from copilot.agents.retriever import make_retriever_node
from copilot.agents.writer import EvidenceItem, generate_draft
from copilot.config import Settings
from copilot.graph.state import (
    CopilotState,
    ResearchPlan,
    SearchFilters,
    SubQuestion,
    Usage,
    make_event,
    make_initial_state,
)
from copilot.tools.chunking import abstract_chunk_id
from copilot.tools.sources import SearchFn


def run_baseline(
    question: str,
    filters: SearchFilters | None,
    settings: Settings,
    llm: Any,
    sources: dict[str, SearchFn],
) -> tuple[CopilotState, Callable[[str], str | None]]:
    state = make_initial_state(question, filters)
    state["plan"] = ResearchPlan(sub_questions=[SubQuestion(id="sq1", question=question, queries=[question])])
    update = make_retriever_node(settings, sources)(state)
    papers = update.get("papers", {})
    shortlist = update.get("shortlist", [])

    evidence, texts = [], {}
    for pid in shortlist:
        p = papers[pid]
        if not p.abstract:
            continue
        cid = abstract_chunk_id(pid)
        texts[cid] = p.abstract
        evidence.append(EvidenceItem(cid, pid, p.title, p.year, p.abstract))

    draft, usage = generate_draft(llm, question, [], evidence) if evidence else (None, Usage())
    state.update(
        papers=papers,
        shortlist=shortlist,
        draft=draft,
        usage=usage,
        errors=update.get("errors", []),
        trace=update.get("trace", []) + [make_event("baseline", f"single-pass draft from {len(evidence)} abstracts")],
    )
    return state, texts.get
