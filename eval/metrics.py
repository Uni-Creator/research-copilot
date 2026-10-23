"""Evaluation metrics for Research Copilot.

Known-item metrics answer whether a curated must-cite list was retrieved.
Graded relevance metrics answer a different question and are available only when
explicit relevance_judgments are supplied in the dataset. Subquestion coverage
is a lexical overlap proxy, not a semantic relevance judgment. Claim support is
LLM-judged against the cited passage and should be spot-checked by humans.
"""
from __future__ import annotations

import math
import re
import unicodedata
from typing import Any, Callable

from copilot.agents.critic import split_claims, support_rate, verify_claims
from copilot.graph.state import CopilotState, Paper, ResearchPlan
from copilot.tools.chunking import paper_id_of
from copilot.utils import tokenize
from eval.retrieval_metrics import (
    graded_relevance_metrics,
    judged_aspect_coverage,
    normalize_arxiv,
    normalize_doi,
)

TITLE_JACCARD = 0.8
DEFAULT_KS = (5, 10, 20)


def normalize_title(title: str | None) -> str:
    """Normalize a paper title for robust comparison."""
    if not title:
        return ""

    title = unicodedata.normalize("NFKC", title).casefold()
    title = re.sub(r"[^\w\s]", " ", title)
    return " ".join(title.split())


def match_paper(must: dict, paper: Paper) -> bool:
    """Match a known-item entry by normalized arXiv/DOI first, then title."""
    must_arxiv = normalize_arxiv(must.get("arxiv_id") or must.get("arxiv"))
    paper_arxiv = normalize_arxiv(getattr(paper, "arxiv_id", None))
    if must_arxiv and paper_arxiv and must_arxiv == paper_arxiv:
        return True
    must_doi = normalize_doi(must.get("doi"))
    paper_doi = normalize_doi(getattr(paper, "doi", None))
    if must_doi and paper_doi and must_doi == paper_doi:
        return True
    a = set(normalize_title(must.get("title", "")).split())
    b = set(normalize_title(getattr(paper, "title", "")).split())
    return bool(a and b) and len(a & b) / len(a | b) >= TITLE_JACCARD


def coverage(must_cite: list[dict], papers: list[Paper]) -> tuple[float | None, list[str]]:
    """Recall of must-cite anchors among papers at one pipeline stage."""
    if not must_cite:
        return None, []
    missed = [m.get("title") or m.get("arxiv_id", "?") for m in must_cite
              if not any(match_paper(m, p) for p in papers)]
    return round(1 - len(missed) / len(must_cite), 4), missed


def _legacy_known_item_metrics(must: list[dict], ranked: list[Paper], ks: tuple[int, ...]) -> dict[str, Any]:
    """Backward-compatible ranked metrics against the must-cite list only."""
    rel = [int(any(match_paper(m, p) for m in must)) for p in ranked]
    out: dict[str, Any] = {"known_item_metric_basis": "must_cite_list"}
    for k in ks:
        if not must or k <= 0:
            precision = recall = ndcg = None
        else:
            top = rel[:k]
            precision = round(sum(top) / k, 4)
            recall = round(sum(top) / len(must), 4)
            dcg = sum(value / math.log2(i + 2) for i, value in enumerate(top))
            ideal_n = min(len(must), k)
            idcg = sum(1 / math.log2(i + 2) for i in range(ideal_n))
            ndcg = round(dcg / idcg, 4) if idcg else None
        # Existing field names retained for result compatibility. They are NOT
        # open-ended relevance metrics; they measure must-cite retrieval only.
        out[f"precision_at_{k}"] = precision
        out[f"recall_at_{k}"] = recall
        out[f"ndcg_at_{k}"] = ndcg
    if not must:
        out["mrr"] = out["average_precision"] = None
    else:
        first = next((i for i, value in enumerate(rel, 1) if value), None)
        out["mrr"] = round(1 / first, 4) if first else 0.0
        found = 0
        total_precision = 0.0
        for rank, value in enumerate(rel, 1):
            if value:
                found += 1
                total_precision += found / rank
        out["average_precision"] = round(total_precision / len(must), 4)
    return out


def subquestion_coverage(plan: ResearchPlan | None, papers: list[Paper]) -> dict[str, float]:
    """Lexical token overlap between each planned subquestion and shortlisted titles/abstracts.

    This is a cheap diagnostic, not a semantic relevance metric or evidence check.
    """
    if not plan or not plan.sub_questions:
        return {}
    paper_tokens: set[str] = set()
    for paper in papers:
        paper_tokens.update(tokenize(paper.title))
        if paper.abstract:
            paper_tokens.update(tokenize(paper.abstract))
    result: dict[str, float] = {}
    for sq in plan.sub_questions:
        tokens = set(tokenize(sq.question))
        for query in sq.queries:
            tokens.update(tokenize(query))
        result[sq.id] = round(len(tokens & paper_tokens) / len(tokens), 4) if tokens else 0.0
    return result


def _claim_check_dict(check: Any) -> dict[str, Any]:
    return {
        "claim": getattr(check, "claim", ""),
        "chunk_ids": list(getattr(check, "chunk_ids", []) or []),
        "verdict": getattr(check, "verdict", "unsupported"),
        "reason": getattr(check, "reason", ""),
    }


def _sentence_count(markdown: str) -> int:
    """Rough count of prose sentences, excluding headings and table rows."""
    count = 0
    for line in markdown.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        if not line or line.startswith(("#", "|", "```")):
            continue
        count += len([s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\[\"'(])", line) if s.strip()])
    return count

def precision_at_k(must_cite, ranked, k):
    return _legacy_known_item_metrics(must_cite, ranked, (k,))[
        f"precision_at_{k}"
    ]


def recall_at_k(must_cite, ranked, k):
    return _legacy_known_item_metrics(must_cite, ranked, (k,))[
        f"recall_at_{k}"
    ]


def ndcg_at_k(must_cite, ranked, k):
    return _legacy_known_item_metrics(must_cite, ranked, (k,))[
        f"ndcg_at_{k}"
    ]


def mrr(must_cite, ranked):
    return _legacy_known_item_metrics(must_cite, ranked, ())["mrr"]


def average_precision(must_cite, ranked):
    return _legacy_known_item_metrics(must_cite, ranked, ())[
        "average_precision"
    ]

def evaluate_run(
    topic: dict,
    state: CopilotState,
    chunk_text: Callable[[str], str | None],
    judge_llm: Any,
    batch_size: int = 8,
    ks: tuple[int, ...] = DEFAULT_KS,
) -> dict[str, Any]:
    """Compute retrieval, topic, citation-support, and efficiency metrics for one run."""
    papers: dict[str, Paper] = state.get("papers", {})
    all_papers = list(papers.values())
    shortlist_ids: list[str] = state.get("shortlist", [])
    shortlist = [papers[i] for i in shortlist_ids if i in papers]
    draft = state.get("draft")
    plan: ResearchPlan | None = state.get("plan")
    cited_chunk_ids = list(draft.cited_chunk_ids if draft else [])
    cited_ids = {paper_id_of(cid) for cid in cited_chunk_ids}
    cited = [papers[i] for i in cited_ids if i in papers]
    must = topic.get("must_cite", [])

    cov_pool, missed_pool = coverage(must, all_papers)
    cov_short, missed_short = coverage(must, shortlist)
    cov_cited, missed_cited = coverage(must, cited)
    ranked_pool = sorted(all_papers, key=lambda p: (-getattr(p, "score", 0.0),
                                                     -getattr(p, "citation_count", 0), p.title))

    result: dict[str, Any] = {
        "topic": topic["id"],
        "coverage_pool": cov_pool,
        "coverage_shortlist": cov_short,
        "coverage_cited": cov_cited,
        "must_cite_count": len(must),
        "missed_in_pool": missed_pool,
        "missed_in_shortlist": missed_short,
        "missed_in_cited_report": missed_cited,
    }
    result.update(_legacy_known_item_metrics(must, ranked_pool, ks))

    # Independent relevance scores require an explicit, curated judgment set.
    # Empty/missing judgments deliberately produce None, not a fabricated zero.
    result.update(graded_relevance_metrics(topic.get("relevance_judgments", []), ranked_pool, ks))
    result.update(judged_aspect_coverage(
        topic.get("topic_aspects", []), topic.get("relevance_judgments", []), shortlist
    ))

    sq_cov = subquestion_coverage(plan, shortlist)
    mean_sq_cov = round(sum(sq_cov.values()) / len(sq_cov), 4) if sq_cov else None
    result.update({
        "lexical_subquestion_coverage": sq_cov,
        "mean_lexical_subquestion_coverage": mean_sq_cov,
        "subquestion_coverage": sq_cov,
        "mean_subquestion_coverage": mean_sq_cov,
        "subquestion_coverage_method": "lexical_token_overlap_title_abstract",
    })

    evidence_basis: dict[str, str] = {}

    def safe_chunk_text(cid: str) -> str | None:
        try:
            text = chunk_text(cid)
        except Exception:
            text = None
        if text:
            evidence_basis[cid] = "retrieved_text_chunk"
            return text
        paper_id = paper_id_of(cid)
        paper = papers.get(paper_id)
        if paper and paper.abstract:
            evidence_basis[cid] = "abstract_fallback"
            return paper.abstract
        evidence_basis[cid] = "missing"
        return None

    claims = split_claims(draft.markdown) if draft else []
    checks, judge_usage = verify_claims(judge_llm, claims, safe_chunk_text, batch_size)
    check_dicts = [_claim_check_dict(check) for check in checks]
    verdicts = [item["verdict"] for item in check_dicts]
    usage = state.get("usage")
    evidence_counts = {key: sum(value == key for value in evidence_basis.values())
                       for key in ("retrieved_text_chunk", "abstract_fallback", "missing")}
    n_judge_errors = sum("judge call failed" in str(item["reason"]).lower()
                         or "no verdict returned" in str(item["reason"]).lower()
                         for item in check_dicts)
    trace = state.get("trace", []) or []
    trace_actions = [getattr(event, "data", {}) or {} for event in trace]
    route_counts = {action: sum((data.get("action") == action) for data in trace_actions)
                    for action in ("pass", "revise_draft", "retrieve_more")}
    markdown = draft.markdown if draft else ""

    valid_refs = sum(paper_id_of(cid) in papers for cid in cited_chunk_ids)
    ref_validity = round(valid_refs / len(cited_chunk_ids), 4) if cited_chunk_ids else None
    sup_rate = support_rate(checks)
    n_unsupported = sum(v == "unsupported" for v in verdicts)
    n_missing_ev_chunks = evidence_counts.get("missing", 0)
    n_insufficient_claims = sum(
        1 for item in check_dicts
        if item["verdict"] != "supported" and any(
            w in str(item.get("reason", "")).lower()
            for w in ("not found", "missing", "insufficient", "no verdict", "not contain", "does not state")
        )
    )
    missing_or_insufficient_ev = n_missing_ev_chunks + n_insufficient_claims

    result.update({
        "n_papers_retrieved": len(all_papers),
        "n_papers_shortlisted": len(shortlist),
        "n_papers_cited": len(cited),
        "n_citation_refs": len(cited_chunk_ids),
        "n_valid_citation_refs": valid_refs,
        "reference_validity": ref_validity,
        "citation_reference_validity": ref_validity,
        "n_claims": len(claims),
        "n_supported_claims": sum(v == "supported" for v in verdicts),
        "n_partial_claims": sum(v == "partial" for v in verdicts),
        "n_unsupported_claims": n_unsupported,
        "unsupported_claim_count": n_unsupported,
        "n_claim_judge_errors": n_judge_errors,
        "claim_support_rate": sup_rate,
        "support_rate": sup_rate,
        "missing_or_insufficient_evidence_count": missing_or_insufficient_ev,
        "unsupported_claims": [c.claim for c in checks if c.verdict == "unsupported"],
        "claim_checks": check_dicts,
        "evidence_basis_counts": evidence_counts,
        "n_prose_sentences_estimate": _sentence_count(markdown),
        "stripped_citations": draft.stripped_citations if draft else 0,
        "words": len(markdown.split()),
        "iterations": state.get("iteration", 0),
        "trace_event_count": len(trace),
        "route_action_counts": route_counts,
        "tokens": usage.total_tokens if usage else 0,
        "judge_tokens": judge_usage.total_tokens,
        "errors": len(state.get("errors", [])),
    })
    return result
