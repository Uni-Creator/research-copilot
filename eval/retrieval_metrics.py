"""Pure-Python relevance metrics for Research Copilot evaluation.

This module deliberately has no dependency on the Copilot runtime so its
matching and metric semantics can be unit-tested offline.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlparse


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        for name in names:
            if obj.get(name) is not None:
                return obj[name]
    else:
        for name in names:
            value = getattr(obj, name, None)
            if value is not None:
                return value
    return default


def normalize_title(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def normalize_doi(value: Any) -> str:
    doi = str(value or "").strip().lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    return doi.removeprefix("doi:").strip().rstrip(".")


def normalize_arxiv(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    if raw.startswith("http"):
        parsed = urlparse(raw)
        raw = parsed.path.rsplit("/", 1)[-1]
    raw = re.sub(r"^(arxiv:|abs/|pdf/)", "", raw)
    raw = re.sub(r"\.pdf$", "", raw)
    raw = re.sub(r"v\d+$", "", raw)
    return raw.strip()


def paper_matches_entry(entry: Mapping[str, Any], paper: Any, title_jaccard: float = 0.8) -> bool:
    """Match a judgment/known-item entry to a retrieved paper using IDs first."""
    entry_arxiv = normalize_arxiv(entry.get("arxiv_id") or entry.get("arxiv"))
    paper_arxiv = normalize_arxiv(_get(paper, "arxiv_id", "arxivId", "arxiv"))
    if entry_arxiv and paper_arxiv and entry_arxiv == paper_arxiv:
        return True

    entry_doi = normalize_doi(entry.get("doi"))
    paper_doi = normalize_doi(_get(paper, "doi"))
    if entry_doi and paper_doi and entry_doi == paper_doi:
        return True

    a = set(normalize_title(entry.get("title", "")).split())
    b = set(normalize_title(_get(paper, "title", "paper_title", "name", default="")).split())
    return bool(a and b) and len(a & b) / len(a | b) >= title_jaccard


def _grade_for_paper(paper: Any, judgments: Sequence[Mapping[str, Any]]) -> int | None:
    matched: list[int] = []
    for judgment in judgments:
        if paper_matches_entry(judgment, paper):
            try:
                grade = int(judgment.get("grade", judgment.get("relevance", 0)))
            except (TypeError, ValueError):
                continue
            if grade in (0, 1, 2):
                matched.append(grade)
    return max(matched) if matched else None


def graded_relevance_metrics(
    judgments: Sequence[Mapping[str, Any]] | None,
    ranked_papers: Sequence[Any],
    ks: Sequence[int] = (5, 10, 20),
) -> dict[str, Any]:
    """Compute relevance metrics from explicit human/curated graded judgments.

    Grades: 0 = not relevant, 1 = partially/background relevant, 2 = directly relevant.
    Unjudged papers are excluded from precision/recall/AP/MRR calculations; the
    judgment-coverage metrics expose how much of each top-k was actually judged.
    nDCG is computed over judged papers appearing in the top-k window, with ideal
    DCG based on all positive judgments. No judgments => metrics are unavailable
    (None), not silently reported as zero.
    """
    judged = list(judgments or [])
    judged = [j for j in judged if isinstance(j, Mapping) and j.get("title")]
    valid = []
    seen = set()
    for j in judged:
        arxiv_key = normalize_arxiv(j.get("arxiv_id"))
        doi_key = normalize_doi(j.get("doi"))
        title_key = normalize_title(j.get("title"))
        key = ("arxiv", arxiv_key) if arxiv_key else ("doi", doi_key) if doi_key else ("title", title_key)
        if key in seen:
            continue
        try:
            grade = int(j.get("grade", j.get("relevance", -1)))
        except (TypeError, ValueError):
            continue
        if grade not in (0, 1, 2):
            continue
        seen.add(key)
        valid.append(dict(j, grade=grade))

    output: dict[str, Any] = {
        "relevance_judgment_count": len(valid),
        "relevance_judged_relevant_count": sum(j["grade"] > 0 for j in valid),
    }
    if not valid:
        for k in ks:
            output[f"judged_precision_at_{k}"] = None
            output[f"judged_recall_at_{k}"] = None
            output[f"judged_ndcg_at_{k}"] = None
            output[f"judgment_coverage_at_{k}"] = None
            output[f"judged_results_at_{k}"] = 0
        output.update({"judged_mrr": None, "judged_average_precision": None})
        return output

    ranked_grades: list[int | None] = [_grade_for_paper(p, valid) for p in ranked_papers]
    relevant_total = sum(j["grade"] > 0 for j in valid)

    for k in ks:
        if k <= 0:
            raise ValueError("k values must be positive integers")
        top = ranked_grades[:k]
        judged_top = [grade for grade in top if grade is not None]
        retrieved_relevant = sum(grade > 0 for grade in judged_top)
        denominator = len(judged_top)
        result_count = min(k, len(ranked_papers))
        output[f"judged_results_at_{k}"] = denominator
        output[f"judgment_coverage_at_{k}"] = round(denominator / result_count, 4) if result_count else None
        output[f"judged_precision_at_{k}"] = round(retrieved_relevant / denominator, 4) if denominator else None
        output[f"judged_recall_at_{k}"] = round(retrieved_relevant / relevant_total, 4) if relevant_total else None

        # Keep original rank positions for DCG; unjudged results contribute no gain.
        dcg = sum((2**grade - 1) / math.log2(rank + 2)
                  for rank, grade in enumerate(top) if grade is not None)
        ideal = sorted((j["grade"] for j in valid), reverse=True)[:k]
        idcg = sum((2**grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(ideal))
        output[f"judged_ndcg_at_{k}"] = round(dcg / idcg, 4) if idcg else None

    judged_ranked = [grade for grade in ranked_grades if grade is not None]
    first_relevant = next((i for i, grade in enumerate(judged_ranked, 1) if grade > 0), None)
    output["judged_mrr"] = round(1 / first_relevant, 4) if first_relevant else (0.0 if relevant_total else None)
    found = 0
    precision_sum = 0.0
    for rank, grade in enumerate(judged_ranked, 1):
        if grade > 0:
            found += 1
            precision_sum += found / rank
    output["judged_average_precision"] = round(precision_sum / relevant_total, 4) if relevant_total else None
    return output


def judged_aspect_coverage(
    aspects: Sequence[Mapping[str, Any]] | None,
    judgments: Sequence[Mapping[str, Any]] | None,
    shortlisted_papers: Sequence[Any],
) -> dict[str, Any]:
    """Coverage of curated topic aspects by positively judged shortlisted papers.

    A paper covers an aspect only when a human/curator explicitly tags that paper
    with the aspect in relevance_judgments[].aspects. Lexical overlap is not used.
    """
    aspect_rows = [a for a in (aspects or []) if isinstance(a, Mapping) and a.get("id")]
    judgment_rows = [j for j in (judgments or []) if isinstance(j, Mapping)]
    if not aspect_rows or not judgment_rows:
        return {"judged_aspect_coverage": None, "aspect_coverage": {}}

    per_aspect: dict[str, float | None] = {}
    for aspect in aspect_rows:
        aspect_id = str(aspect["id"])
        has_positive_judgment = any(
            int(j.get("grade", j.get("relevance", 0)) or 0) > 0
            and aspect_id in (j.get("aspects") or [])
            for j in judgment_rows
        )
        if not has_positive_judgment:
            per_aspect[aspect_id] = None
            continue
        covered = any(
            int(j.get("grade", j.get("relevance", 0)) or 0) > 0
            and aspect_id in (j.get("aspects") or [])
            and any(paper_matches_entry(j, paper) for paper in shortlisted_papers)
            for j in judgment_rows
        )
        per_aspect[aspect_id] = 1.0 if covered else 0.0

    scored = [value for value in per_aspect.values() if value is not None]
    return {
        "judged_aspect_coverage": round(sum(scored) / len(scored), 4) if scored else None,
        "aspect_coverage": per_aspect,
    }
