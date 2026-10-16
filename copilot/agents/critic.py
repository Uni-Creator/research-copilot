"""Critic agent: claim-to-evidence verification + coverage review, then a routing decision.

* Claims are extracted *mechanically*: every sentence carrying [cite:...] tags is a claim.
* A judge LLM checks each claim against only the chunks it cites (batched to save tokens).
* ``verify_claims`` is reused by the eval harness as the citation-support judge.
* Decision: missing topics -> retrieve_more; low support -> revise_draft; else pass.
  Loops are capped by ``settings.max_iterations`` and a token budget.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel

from copilot.config import Settings
from copilot.graph.routing import budget_exceeded, iterations_left
from copilot.graph.state import ClaimCheck, CopilotState, CriticReport, Usage, make_event
from copilot.llm import structured_call

_CITE = re.compile(r"\[cite:([^\]]+)\]")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[\"'(])")
VERDICT_SCORE = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}
EVIDENCE_CHARS = 900


class VerdictItem(BaseModel):
    index: int
    verdict: Literal["supported", "partial", "unsupported"]
    reason: str = ""


class VerifyOutput(BaseModel):
    verdicts: list[VerdictItem] = []


class CoverageOutput(BaseModel):
    missing_topics: list[str] = []


@lru_cache(maxsize=None)
def _prompt(name: str) -> str:
    return (Path(__file__).parent.parent / "prompts" / name).read_text(encoding="utf-8")


def split_claims(markdown: str) -> list[tuple[str, list[str]]]:
    """[(claim_text_without_tags, [chunk_ids])] for each cited sentence (headings/tables skipped)."""
    claims: list[tuple[str, list[str]]] = []
    for line in markdown.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        if not line or line.startswith(("#", "|")):
            continue
        for sent in _SENT.split(line):
            ids = [i.strip() for i in _CITE.findall(sent)]
            if ids:
                text = re.sub(r"\s+([.,;:!?])", r"\1", _CITE.sub("", sent)).strip()
                if text:
                    claims.append((text, list(dict.fromkeys(ids))))
    return claims


def support_rate(checks: list[ClaimCheck]) -> float | None:
    if not checks:
        return None
    return round(sum(VERDICT_SCORE[c.verdict] for c in checks) / len(checks), 4)


def verify_claims(
    llm: Any,
    claims: list[tuple[str, list[str]]],
    chunk_text: Callable[[str], str | None],
    batch_size: int = 8,
) -> tuple[list[ClaimCheck], Usage]:
    """Judge each claim against the chunks it cites. Order of results matches ``claims``."""
    checks: list[ClaimCheck | None] = [None] * len(claims)
    usage = Usage()

    todo: list[int] = []
    for i, (claim, ids) in enumerate(claims):
        if not any(chunk_text(cid) for cid in ids):
            checks[i] = ClaimCheck(claim=claim, chunk_ids=ids, verdict="unsupported",
                                   reason="cited chunk not found")
        else:
            todo.append(i)

    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        blocks = []
        for n, i in enumerate(batch):
            claim, ids = claims[i]
            ev = "\n".join(f"  [{cid}] {(chunk_text(cid) or '')[:EVIDENCE_CHARS]}" for cid in ids if chunk_text(cid))
            blocks.append(f"Claim {n}: {claim}\nCited evidence:\n{ev}")
        err = "no verdict returned"
        try:
            out, u = structured_call(llm, VerifyOutput,
                                     [("system", _prompt("critic_verify.md")), ("human", "\n\n".join(blocks))])
            usage = usage + u
            by_idx = {v.index: v for v in out.verdicts}
        except Exception as exc:
            by_idx = {}
            err = f"judge call failed: {exc!r}"
        for n, i in enumerate(batch):
            v = by_idx.get(n)
            claim, ids = claims[i]
            if v is None:
                checks[i] = ClaimCheck(claim=claim, chunk_ids=ids, verdict="unsupported",
                                       reason=err)
            else:
                checks[i] = ClaimCheck(claim=claim, chunk_ids=ids, verdict=v.verdict, reason=v.reason)
    return [c for c in checks if c is not None], usage


def make_critic_node(llm: Any, store, settings: Settings) -> Callable[[CopilotState], dict]:
    def critic_node(state: CopilotState) -> dict:
        draft = state["draft"]
        iteration = state.get("iteration", 0)
        can_loop = iterations_left(state, settings) and not budget_exceeded(state, settings)

        claims = split_claims(draft.markdown)
        checks, usage = verify_claims(
            llm, claims, lambda cid: (c.text if (c := store.get(cid)) else None), settings.verify_batch_size
        )
        rate = support_rate(checks)

        missing: list[str] = []
        if can_loop and claims:
            plan = state.get("plan")
            try:
                cov, u = structured_call(llm, CoverageOutput, [
                    ("system", _prompt("critic_coverage.md")),
                    ("human", f"Question: {state['question']}\nSub-questions:\n"
                              + "\n".join(f"- {sq.question}" for sq in (plan.sub_questions if plan else []))
                              + f"\n\nDraft:\n{_CITE.sub('', draft.markdown)}"),
                ])
                usage = usage + u
                missing = [t.strip() for t in cov.missing_topics if t.strip()][:4]
            except Exception:
                missing = []

        low_support = rate is not None and rate < settings.support_threshold
        action: Literal["pass", "revise_draft", "retrieve_more"]
        if can_loop and missing:
            action = "retrieve_more"
        elif can_loop and low_support:
            action = "revise_draft"
        else:
            action = "pass"
        report = CriticReport(
            passed=(rate is not None and not low_support and not missing),
            action=action, checks=checks, missing_topics=missing, support_rate=rate,
        )
        flagged = sum(c.verdict != "supported" for c in checks)
        return {
            "critique": report,
            "iteration": iteration + 1,
            "usage": usage,
            "trace": [make_event(
                "critic",
                f"{len(claims)} claims checked, support={rate}, flagged={flagged}, "
                f"missing={missing or 'none'} -> {action}",
                support_rate=rate, action=action,
            )],
        }

    return critic_node
