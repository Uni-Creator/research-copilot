"""Shared state and typed data contracts for the research-copilot graph.

Design rules
------------
* Agents talk to each other **only** through ``CopilotState`` (no free-text hand-offs).
* Fields written by several nodes, or by parallel branches, carry a *reducer*
  (``Annotated[..., reducer]``) so concurrent updates merge instead of overwrite.
* Models for later stages (Reader / Writer / Critic) are defined now so the contract
  is frozen early; nodes are added week by week.
* State keys must never share a name with a graph node (LangGraph restriction), so
  nodes are named ``planner``, ``retriever``, ``reader``, ``writer``, ``critic_agent``.
"""
from __future__ import annotations

import operator
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field, model_validator

from copilot.utils import normalize_title, short_hash


# --------------------------------------------------------------------------- #
# Papers
# --------------------------------------------------------------------------- #
def compute_paper_id(
    arxiv_id: str | None, doi: str | None, s2_id: str | None, title: str
) -> str:
    """Canonical id: arXiv > DOI > Semantic Scholar > hash of the normalised title."""
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    if doi:
        return f"doi:{doi.lower()}"
    if s2_id:
        return f"s2:{s2_id}"
    return f"title:{short_hash(normalize_title(title))}"


def _union(a: list[str], b: list[str]) -> list[str]:
    return list(dict.fromkeys([*a, *b]))


class Paper(BaseModel):
    paper_id: str = ""
    title: str
    authors: list[str] = []
    abstract: str = ""
    year: int | None = None
    venue: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    arxiv_id: str | None = None
    doi: str | None = None
    s2_id: str | None = None
    citation_count: int = 0
    sources: list[str] = []      # which APIs returned it: "arxiv", "semantic_scholar"
    found_by: list[str] = []     # which search queries surfaced it
    score: float = 0.0           # retrieval rank score in [0, 1]

    @model_validator(mode="after")
    def _fill_id(self) -> "Paper":
        if not self.paper_id:
            self.paper_id = compute_paper_id(self.arxiv_id, self.doi, self.s2_id, self.title)
        return self

    def keys(self) -> list[str]:
        """Identity keys used to detect duplicates across sources."""
        ks: list[str] = []
        if self.arxiv_id:
            ks.append(f"arxiv:{self.arxiv_id}")
        if self.doi:
            ks.append(f"doi:{self.doi.lower()}")
        nt = normalize_title(self.title)
        if nt:
            ks.append(f"title:{nt}")
        return ks

    def merged_with(self, other: "Paper") -> "Paper":
        """Combine two records of the same paper, keeping the richest data."""

        def pick(a: Any, b: Any) -> Any:
            return a if a not in (None, "", []) else b

        m = self.model_copy(deep=True)
        m.abstract = max((self.abstract, other.abstract), key=len)
        for f in ("year", "venue", "url", "pdf_url", "arxiv_id", "doi", "s2_id"):
            setattr(m, f, pick(getattr(self, f), getattr(other, f)))
        m.authors = self.authors or other.authors
        m.citation_count = max(self.citation_count, other.citation_count)
        m.sources = _union(self.sources, other.sources)
        m.found_by = _union(self.found_by, other.found_by)
        m.score = max(self.score, other.score)
        m.paper_id = compute_paper_id(m.arxiv_id, m.doi, m.s2_id, m.title)
        return m


# --------------------------------------------------------------------------- #
# Planner output
# --------------------------------------------------------------------------- #
class SearchFilters(BaseModel):
    year_from: int | None = None
    year_to: int | None = None
    max_papers: int | None = None


class SubQuestion(BaseModel):
    id: str
    question: str
    queries: list[str]


class ResearchPlan(BaseModel):
    sub_questions: list[SubQuestion] = []
    scope_notes: str = ""

    def all_queries(self) -> list[str]:
        """Unique search queries across sub-questions, in plan order."""
        seen: set[str] = set()
        out: list[str] = []
        for sq in self.sub_questions:
            for q in sq.queries:
                if q.lower() not in seen:
                    seen.add(q.lower())
                    out.append(q)
        return out


# --------------------------------------------------------------------------- #
# Later-stage contracts (Reader / Writer / Critic) -- nodes arrive in Weeks 2-4
# --------------------------------------------------------------------------- #
class PaperNotes(BaseModel):
    paper_id: str
    problem: str = ""
    method: str = ""
    datasets: list[str] = []
    metrics: list[str] = []
    key_results: str = ""
    limitations: str = ""
    chunk_ids: list[str] = []    # evidence chunks stored in the vector DB


class Draft(BaseModel):
    markdown: str
    cited_chunk_ids: list[str] = []
    version: int = 1


class ClaimCheck(BaseModel):
    claim: str
    chunk_ids: list[str] = []
    verdict: Literal["supported", "partial", "unsupported"]
    reason: str = ""


class CriticReport(BaseModel):
    passed: bool
    action: Literal["pass", "revise_draft", "retrieve_more"] = "pass"
    checks: list[ClaimCheck] = []
    missing_topics: list[str] = []
    support_rate: float | None = None


# --------------------------------------------------------------------------- #
# Bookkeeping
# --------------------------------------------------------------------------- #
class Usage(BaseModel):
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            llm_calls=self.llm_calls + other.llm_calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class TraceEvent(BaseModel):
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    agent: str
    message: str
    data: dict[str, Any] = {}


def make_event(agent: str, message: str, **data: Any) -> TraceEvent:
    return TraceEvent(agent=agent, message=message, data=data)


# --------------------------------------------------------------------------- #
# Reducers
# --------------------------------------------------------------------------- #
def merge_papers(left: dict[str, Paper] | None, right: dict[str, Paper] | None) -> dict[str, Paper]:
    """Merge paper dicts by id; duplicate ids are fused with ``Paper.merged_with``."""
    out = dict(left or {})
    for pid, paper in (right or {}).items():
        out[pid] = out[pid].merged_with(paper) if pid in out else paper
    return out


def merge_dicts(left: dict | None, right: dict | None) -> dict:
    """Right-biased dict merge (used for per-paper Reader notes from parallel branches)."""
    return {**(left or {}), **(right or {})}


def merge_usage(left: Usage | None, right: Usage | None) -> Usage:
    return (left or Usage()) + (right or Usage())


# --------------------------------------------------------------------------- #
# Graph state
# --------------------------------------------------------------------------- #
class CopilotState(TypedDict, total=False):
    # inputs
    question: str
    filters: SearchFilters

    # planner
    plan: ResearchPlan

    # retriever
    papers: Annotated[dict[str, Paper], merge_papers]        # candidate pool
    shortlist: list[str]                                     # paper_ids to read, ranked
    searched_queries: Annotated[list[str], operator.add]     # avoids repeat searches

    # reader (Week 2) -- written in parallel, one entry per paper
    notes: Annotated[dict[str, PaperNotes], merge_dicts]

    # writer / critic (Weeks 3-4)
    draft: Draft
    critique: CriticReport
    iteration: int

    # bookkeeping (every node may append)
    usage: Annotated[Usage, merge_usage]
    trace: Annotated[list[TraceEvent], operator.add]
    errors: Annotated[list[str], operator.add]


def make_initial_state(question: str, filters: SearchFilters | None = None) -> CopilotState:
    return CopilotState(
        question=question.strip(),
        filters=filters or SearchFilters(),
        iteration=0,
    )
