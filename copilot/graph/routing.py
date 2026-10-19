"""Routing helpers and loop/budget guards."""

from __future__ import annotations

from langgraph.graph import END
from langgraph.types import Send

from copilot.config import Settings
from copilot.graph.state import CopilotState


def budget_exceeded(state: CopilotState, settings: Settings) -> bool:
    usage = state.get("usage")
    return bool(usage and usage.total_tokens >= settings.max_total_tokens)


def iterations_left(state: CopilotState, settings: Settings) -> bool:
    return state.get("iteration", 0) < settings.max_iterations


def route_after_retriever(state: CopilotState, *, next_node: str):
    """Fan out one Reader per shortlisted paper that has no notes yet (parallel ``Send``).

    Nothing to read (all papers already have notes, e.g. after a retrieve-more pass that found
    nothing new) -> go straight to ``next_node``. Empty shortlist -> nothing to write -> END.
    """
    shortlist = state.get("shortlist", [])
    if not shortlist:
        return END
    notes = state.get("notes", {})
    papers = state["papers"]
    todo = [pid for pid in shortlist if pid not in notes]
    if not todo:
        return next_node
    return [Send("reader", {"paper": papers[pid], "question": state["question"]}) for pid in todo]


def route_after_critic(state: CopilotState) -> str:
    return {"pass": END, "revise_draft": "writer", "retrieve_more": "planner"}[state["critique"].action]
