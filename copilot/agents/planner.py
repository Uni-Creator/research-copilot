"""Planner agent: research question -> sub-questions + search queries."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

from copilot.graph.state import (
    CopilotState,
    ResearchPlan,
    SearchFilters,
    SubQuestion,
    Usage,
)
from copilot.llm import structured_call
from copilot.progress import Tracer, short

MAX_SUBQUESTIONS = 6
MAX_QUERIES_PER_SUBQUESTION = 3


# ---- LLM output schema (kept loose; we sanitise afterwards) ---------------- #
class PlannedSubQuestion(BaseModel):
    question: str
    queries: list[str] = []


class PlannerOutput(BaseModel):
    reasoning: str = ""          # listed first so the model "thinks" before it plans; shown to the user
    sub_questions: list[PlannedSubQuestion] = []
    scope_notes: str = ""
    year_from: int | None = None


@lru_cache(maxsize=1)
def _system_prompt() -> str:
    return (Path(__file__).parent.parent / "prompts" / "planner.md").read_text(encoding="utf-8")


def build_user_prompt(
    question: str,
    filters: SearchFilters,
    prior: ResearchPlan | None,
    missing_topics: list[str],
) -> str:
    parts = [f"Research question: {question}"]
    if filters.year_from or filters.year_to:
        parts.append(f"User year filter: {filters.year_from or 'any'} to {filters.year_to or 'now'}")
    if prior and prior.sub_questions:
        parts.append("Queries already searched (do not repeat):")
        parts += [f"- {q}" for q in prior.all_queries()]
    if missing_topics:
        parts.append("Missing topics to cover (return ONLY sub-questions for these):")
        parts += [f"- {t}" for t in missing_topics]
    return "\n".join(parts)


# ---- sanitising ------------------------------------------------------------ #
def fallback_plan(question: str) -> ResearchPlan:
    """Deterministic plan used when the LLM call fails: just search the question itself."""
    return ResearchPlan(
        sub_questions=[SubQuestion(id="sq1", question=question, queries=[question])],
        scope_notes="Fallback plan: planner LLM unavailable, searching the raw question.",
    )


def to_plan(out: PlannerOutput, question: str, prior: ResearchPlan | None = None) -> ResearchPlan:
    """Clean the LLM output: trim, dedupe, cap sizes, continue ids after ``prior``."""
    existing = list(prior.sub_questions) if prior else []
    seen_questions = {sq.question.lower() for sq in existing}
    seen_queries = {q.lower() for sq in existing for q in sq.queries}

    added: list[SubQuestion] = []
    for item in out.sub_questions[:MAX_SUBQUESTIONS]:
        text = item.question.strip()
        if not text or text.lower() in seen_questions:
            continue
        queries: list[str] = []
        for q in item.queries:
            q = " ".join(q.split()).strip(" \"'")
            if q and q.lower() not in seen_queries and len(queries) < MAX_QUERIES_PER_SUBQUESTION:
                queries.append(q)
                seen_queries.add(q.lower())
        if not queries:
            continue
        seen_questions.add(text.lower())
        added.append(SubQuestion(id=f"sq{len(existing) + len(added) + 1}", question=text, queries=queries))

    if not existing and not added:
        return fallback_plan(question)
    notes = out.scope_notes.strip() or (prior.scope_notes if prior else "")
    return ResearchPlan(sub_questions=existing + added, scope_notes=notes)


# ---- node ------------------------------------------------------------------ #
def make_planner_node(llm: Any) -> Callable[[CopilotState], dict]:
    def planner_node(state: CopilotState) -> dict:
        trace = Tracer("planner")
        question = state["question"]
        filters = state.get("filters") or SearchFilters()
        prior = state.get("plan")
        critique = state.get("critique")
        missing = critique.missing_topics if critique and critique.action == "retrieve_more" else []

        if missing:
            trace("follow-up: the reviewer found gaps; planning searches for: " + "; ".join(missing), "thinking")
        else:
            trace(f"breaking the question into sub-questions: \u201c{short(question, 120)}\u201d", "info")

        messages = [
            ("system", _system_prompt()),
            ("human", build_user_prompt(question, filters, prior, missing)),
        ]
        update: dict = {}
        n_before = len(prior.sub_questions) if prior else 0
        try:
            out, usage = structured_call(llm, PlannerOutput, messages)
            plan = to_plan(out, question, prior)
            if out.year_from and filters.year_from is None:
                update["filters"] = filters.model_copy(update={"year_from": out.year_from})
                trace(f"applying year filter from {out.year_from} (the question asks for recent work)", "info")
            errors: list[str] = []
            if out.reasoning.strip():
                trace(out.reasoning.strip(), "thinking")
            if plan.scope_notes and not missing:
                trace("scope: " + plan.scope_notes, "thinking")
        except Exception as exc:  # network / parsing / provider errors: degrade, don't crash
            usage = Usage()
            plan = prior or fallback_plan(question)
            msg = f"planner: LLM call failed ({exc!r}); using {'previous' if prior else 'fallback'} plan"
            errors = [msg]
            trace(msg, "warn")

        for sq in plan.sub_questions[n_before:]:
            trace(f"{sq.id} \u00b7 {sq.question}", "plan", queries=sq.queries)
        n_sq, n_q = len(plan.sub_questions), len(plan.all_queries())
        trace(f"plan ready: {n_sq} sub-question{'s' * (n_sq != 1)}, {n_q} search quer{'ies' if n_q != 1 else 'y'}", "info",
              sub_questions=[sq.question for sq in plan.sub_questions], queries=plan.all_queries())

        update.update(plan=plan, usage=usage, errors=errors, trace=trace.events)
        return update

    return planner_node
