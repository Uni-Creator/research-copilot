"""Graph wiring. Week 1: START -> planner -> retriever -> END.

Planned (see README roadmap):
  retriever -> reader (parallel via Send) -> writer -> critic_agent
  critic_agent --pass--> END | --revise_draft--> writer | --retrieve_more--> planner
Node names must differ from state keys (``critique`` is the key, ``critic_agent`` the node).
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from copilot.agents.planner import make_planner_node
from copilot.agents.retriever import make_retriever_node
from copilot.config import Settings
from copilot.graph.state import CopilotState
from copilot.llm import get_llm
from copilot.tools.sources import SearchFn


def build_graph(
    settings: Settings | None = None,
    *,
    planner_llm: Any | None = None,
    sources: dict[str, SearchFn] | None = None,
):
    """Compile the graph. ``planner_llm`` / ``sources`` can be injected for tests."""
    settings = settings or Settings.from_env()
    llm = planner_llm or get_llm(settings.planner_model, settings.model_provider, max_tokens=1500)

    g = StateGraph(CopilotState)
    g.add_node("planner", make_planner_node(llm))
    g.add_node("retriever", make_retriever_node(settings, sources))
    g.add_edge(START, "planner")
    g.add_edge("planner", "retriever")
    g.add_edge("retriever", END)
    return g.compile()
