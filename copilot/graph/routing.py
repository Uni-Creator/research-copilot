"""Routing helpers and loop/budget guards (the Critic edge is wired in Week 4)."""
from __future__ import annotations

from copilot.config import Settings
from copilot.graph.state import CopilotState


def budget_exceeded(state: CopilotState, settings: Settings) -> bool:
    usage = state.get("usage")
    return bool(usage and usage.total_tokens >= settings.max_total_tokens)


def iterations_left(state: CopilotState, settings: Settings) -> bool:
    return state.get("iteration", 0) < settings.max_iterations
