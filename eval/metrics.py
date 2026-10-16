"""Eval metrics for a finished run.

coverage_shortlist : recall of must-cite papers among papers the system chose to read
coverage_cited     : recall of must-cite papers among papers the report actually cites
support_rate       : judge-verified share of cited claims actually backed by the cited text
                     (supported=1, partial=0.5, unsupported=0)
stripped_citations : fabricated citations the Writer produced and the pipeline removed
"""
from __future__ import annotations

from typing import Any, Callable

from copilot.agents.critic import split_claims, support_rate, verify_claims
from copilot.graph.state import CopilotState, Paper
from copilot.tools.chunking import paper_id_of
from copilot.utils import normalize_title

TITLE_JACCARD = 0.8


def match_paper(must: dict, paper: Paper) -> bool:
    """A must-cite entry ({title, arxiv_id?, doi?}) matches by id, else by title similarity."""
    if must.get("arxiv_id") and must["arxiv_id"] == paper.arxiv_id:
        return True
    if must.get("doi") and paper.doi and must["doi"].lower() == paper.doi.lower():
        return True
    a, b = set(normalize_title(must.get("title", "")).split()), set(normalize_title(paper.title).split())
    return bool(a and b) and len(a & b) / len(a | b) >= TITLE_JACCARD


def coverage(must_cite: list[dict], papers: list[Paper]) -> tuple[float | None, list[str]]:
    if not must_cite:
        return None, []
    missed = [m.get("title") or m.get("arxiv_id", "?") for m in must_cite
              if not any(match_paper(m, p) for p in papers)]
    return round(1 - len(missed) / len(must_cite), 4), missed


def evaluate_run(
    topic: dict,
    state: CopilotState,
    chunk_text: Callable[[str], str | None],
    judge_llm: Any,
    batch_size: int = 8,
) -> dict[str, Any]:
    papers = state.get("papers", {})
    shortlist = [papers[i] for i in state.get("shortlist", []) if i in papers]
    draft = state.get("draft")
    cited_ids = {paper_id_of(c) for c in (draft.cited_chunk_ids if draft else [])}
    cited = [papers[i] for i in cited_ids if i in papers]
    must = topic.get("must_cite", [])

    cov_short, missed = coverage(must, shortlist)
    cov_cited, _ = coverage(must, cited)

    claims = split_claims(draft.markdown) if draft else []
    checks, judge_usage = verify_claims(judge_llm, claims, chunk_text, batch_size)
    usage = state.get("usage")
    return {
        "topic": topic["id"],
        "coverage_pool": coverage(must, list(papers.values()))[0],
        "coverage_shortlist": cov_short,
        "coverage_cited": cov_cited,
        "missed_in_shortlist": missed,
        "n_papers_cited": len(cited_ids),
        "n_claims": len(claims),
        "support_rate": support_rate(checks),
        "unsupported_claims": [c.claim for c in checks if c.verdict == "unsupported"],
        "stripped_citations": draft.stripped_citations if draft else 0,
        "words": len(draft.markdown.split()) if draft else 0,
        "iterations": state.get("iteration", 0),
        "tokens": usage.total_tokens if usage else 0,
        "judge_tokens": judge_usage.total_tokens,
        "errors": len(state.get("errors", [])),
    }
