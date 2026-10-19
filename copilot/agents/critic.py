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
from copilot.graph.state import ClaimCheck, CopilotState, CriticReport, Usage
from copilot.llm import structured_call
from copilot.progress import Tracer

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
        trace = Tracer("critic")
        draft = state["draft"]
        iteration = state.get("iteration", 0)
        within_iterations = iterations_left(state, settings)
        over_budget = budget_exceeded(state, settings)
        can_loop = within_iterations and not over_budget

        claims = split_claims(draft.markdown)
        n_calls = -(-len(claims) // settings.verify_batch_size) if claims else 0
        trace(f"fact-checking {len(claims)} cited claims against the passages they cite "
              f"({n_calls} judge call{'s' if n_calls != 1 else ''})", "check")
        def get_chunk(cid: str) -> str | None:
            if (c := store.get(cid)) is not None:
                return c.text
            from copilot.tools.chunking import paper_id_of
            pid = paper_id_of(cid)
            p = state.get("papers", {}).get(pid)
            if p and p.abstract:
                return p.abstract
            return None

        checks, usage = verify_claims(
            llm, claims, get_chunk, settings.verify_batch_size
        )
        for c in checks:
            trace(c.claim, "check", verdict=c.verdict, claim=c.claim, reason=c.reason, chunks=c.chunk_ids)
        rate = support_rate(checks)
        if rate is not None:
            trace(f"support rate {rate:.0%} (threshold {settings.support_threshold:.0%}); "
                  f"{sum(c.verdict != 'supported' for c in checks)} of {len(checks)} claims flagged", "check")

        missing: list[str] = []
        if can_loop and claims:
            plan = state.get("plan")
            trace("reviewing coverage: does the draft answer every part of the question?", "check")
            try:
                cov, u = structured_call(llm, CoverageOutput, [
                    ("system", _prompt("critic_coverage.md")),
                    ("human", f"Question: {state['question']}\nSub-questions:\n"
                              + "\n".join(f"- {sq.question}" for sq in (plan.sub_questions if plan else []))
                              + f"\n\nDraft:\n{_CITE.sub('', draft.markdown)}"),
                ])
                usage = usage + u
                missing = [t.strip() for t in cov.missing_topics if t.strip()][:4]
            except Exception as exc:
                trace(f"coverage review failed ({exc!r}); skipping it", "warn")
            trace("coverage gaps: " + "; ".join(missing) if missing else "no coverage gaps found", "check",
                  missing=missing)

        low_support = rate is not None and rate < settings.support_threshold
        action: Literal["pass", "revise_draft", "retrieve_more"]
        if can_loop and missing:
            action, why = "retrieve_more", "gaps found; searching for papers that cover them"
        elif can_loop and low_support:
            action, why = "revise_draft", "too many unsupported claims; sending the flagged ones back to the writer"
        else:
            action = "pass"
            limit = "token budget" if over_budget else "iteration limit"
            if not claims:
                why = "no cited claims to verify"
            elif low_support:                      # can only happen when looping is no longer allowed
                why = f"{limit} reached; finishing with the quality bar NOT met"
            elif not can_loop:
                why = f"claims check out; {limit} reached, so coverage was not re-checked"
            else:
                why = "quality bar met"
        report = CriticReport(
            passed=(rate is not None and not low_support and not missing),
            action=action, checks=checks, missing_topics=missing, support_rate=rate,
        )
        trace(f"decision: {action} \u2014 {why}", "decision", action=action, reason=why, support_rate=rate)
        return {
            "critique": report,
            "iteration": iteration + 1,
            "usage": usage,
            "trace": trace.events,
        }

    return critic_node
