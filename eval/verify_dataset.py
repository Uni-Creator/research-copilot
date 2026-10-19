"""Check an eval dataset against arXiv + Semantic Scholar before you trust it.

    python -m eval.verify_dataset                          # dry run: report only
    python -m eval.verify_dataset --promote --write        # fill IDs, move verified candidates into must_cite

For every must_cite and candidate entry it searches by title and reports:
    ok           found; title/ID matches (IDs and DOIs are filled in when missing)
    id_mismatch  found by title, but the paper's arXiv id differs from the one in the file
    not_found    nothing matched; the closest titles are printed so a typo is easy to spot
    error        every source failed (network / rate limit): retry
Exit code 1 if any must_cite entry is not ok, so a broken gold set cannot silently skew scores.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

from copilot.config import Settings
from copilot.graph.state import Paper, SearchFilters
from copilot.tools.sources import SearchFn, build_sources
from eval.metrics import match_paper


@dataclass
class Resolution:
    status: str  # ok | id_mismatch | not_found | error
    paper: Paper | None = None
    hints: list[str] = field(default_factory=list)


def resolve(entry: dict, sources: dict[str, SearchFn], limit: int = 5) -> Resolution:
    """Find the paper an entry refers to. Semantic Scholar first: better at exact-title lookup."""
    hints: list[str] = []
    failures = 0
    for _, fn in sorted(sources.items(), key=lambda kv: kv[0] != "semantic_scholar"):
        try:
            results = fn(entry["title"], SearchFilters(), limit)
        except Exception:
            failures += 1
            continue
        for p in results:
            if match_paper(entry, p):
                bad_id = entry.get("arxiv_id") and p.arxiv_id and entry["arxiv_id"] != p.arxiv_id
                return Resolution("id_mismatch" if bad_id else "ok", p)
        hints += [p.title for p in results[:3]]
    if failures == len(sources):
        return Resolution("error", hints=["every source failed"])
    return Resolution("not_found", hints=list(dict.fromkeys(hints))[:5])


def verify_dataset(data: dict, sources: dict[str, SearchFn], *, promote: bool = False) -> tuple[list[dict], dict]:
    """Returns (report rows, updated copy of the dataset). The input is never modified."""
    out = copy.deepcopy(data)
    rows: list[dict] = []
    for topic in out["topics"]:
        promoted: list[dict] = []
        for kind in ("must_cite", "candidates"):
            for entry in topic.get(kind, []):
                res = resolve(entry, sources)
                rows.append(
                    {
                        "topic": topic["id"],
                        "list": kind,
                        "title": entry["title"],
                        "status": res.status,
                        "resolved_title": res.paper.title if res.paper else None,
                        "arxiv_id": res.paper.arxiv_id if res.paper else None,
                        "hints": res.hints,
                    }
                )
                if res.status == "ok" and res.paper:
                    if res.paper.arxiv_id:
                        entry.setdefault("arxiv_id", res.paper.arxiv_id)
                    if res.paper.doi:
                        entry.setdefault("doi", res.paper.doi)
                    entry["verified"] = "api"
                    if kind == "candidates" and promote:
                        promoted.append(entry)
        if promote and promoted:
            topic["candidates"] = [e for e in topic["candidates"] if e not in promoted]
            topic["must_cite"] += promoted
    return rows, out


def format_report(rows: list[dict]) -> str:
    icon = {"ok": "OK      ", "id_mismatch": "ID?     ", "not_found": "MISSING ", "error": "ERROR   "}
    lines = []
    for r in rows:
        lines.append(f"{icon[r['status']]} [{r['topic']}/{r['list']}] {r['title'][:95]}")
        if r["status"] == "id_mismatch":
            lines.append(f"           file id differs from the paper's arXiv id ({r['arxiv_id']}); fix the file")
        if r["status"] in ("not_found", "error") and r["hints"]:
            lines += [f"           closest: {h[:90]}" for h in r["hints"]]
    counts = {s: sum(r["status"] == s for r in rows) for s in icon}
    lines.append("\n" + ", ".join(f"{n} {s}" for s, n in counts.items() if n))
    return "\n".join(lines)


def main(argv: list[str] | None = None, sources: dict[str, SearchFn] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topics", default="eval/datasets/slp.json")
    ap.add_argument("--promote", action="store_true", help="move candidates that resolve into must_cite")
    ap.add_argument("--write", action="store_true", help="save filled-in IDs / promotions back to the file")
    args = ap.parse_args(argv)

    path = Path(args.topics)
    data = json.loads(path.read_text(encoding="utf-8"))
    sources = sources if sources is not None else build_sources(Settings.from_env())
    rows, updated = verify_dataset(data, sources, promote=args.promote)
    print(format_report(rows))
    if args.write:
        path.write_text(json.dumps(updated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nwrote {path}")
    return 1 if any(r["list"] == "must_cite" and r["status"] != "ok" for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
