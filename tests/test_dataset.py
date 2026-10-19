import json
import re
from pathlib import Path

from copilot.graph.state import Paper, SearchFilters
from eval.metrics import coverage, evaluate_run

DATA = json.loads((Path(__file__).parent.parent / "eval" / "datasets" / "slp.json").read_text(encoding="utf-8"))


def test_dataset_schema_is_valid():
    ids = [t["id"] for t in DATA["topics"]]
    assert len(ids) == len(set(ids)) and len(ids) >= 4
    for t in DATA["topics"]:
        assert len(t["question"]) > 30 and SearchFilters(**t["filters"])
        assert len(t["must_cite"]) >= 2
        titles = [e["title"].lower() for e in t["must_cite"] + t.get("candidates", [])]
        assert len(titles) == len(set(titles)), f"duplicate title in {t['id']}"
        for e in t["must_cite"]:
            assert "verified" not in e or e["verified"] in (
                "search",
                "listing",
                "api",
            )  # unverified entries belong in `candidates`
            assert "arxiv_id" not in e or re.fullmatch(r"\d{4}\.\d{4,5}", e["arxiv_id"])
            assert "verified" not in e or e["verified"] != "listing" or "arxiv_id" not in e
        for e in t.get("candidates", []):
            assert "verified" not in e


def test_every_entry_matches_a_paper_built_from_it():
    """The gold list is self-consistent with the matcher (title/ID/DOI) it is scored with."""
    for t in DATA["topics"]:
        papers = [Paper(title=e["title"], arxiv_id=e.get("arxiv_id"), doi=e.get("doi")) for e in t["must_cite"]]
        assert coverage(t["must_cite"], papers) == (1.0, [])
        assert coverage(t["must_cite"], papers[:-1])[0] == round(1 - 1 / len(papers), 4)


def test_arxiv_version_title_changes_still_match_by_id():
    entry = next(
        (e for t in DATA["topics"] for e in t["must_cite"] if e.get("arxiv_id") == "2308.16082"),
        next(e for t in DATA["topics"] for e in t["must_cite"] if e.get("arxiv_id")),
    )
    v2 = Paper(title=f"{entry['title']} (Extended)", arxiv_id=entry["arxiv_id"])
    assert coverage([entry], [v2]) == (1.0, [])


def test_dataset_topic_runs_through_the_evaluator():
    from tests.conftest import ScriptedLLM

    topic = DATA["topics"][0]
    state = {"papers": {}, "shortlist": [], "usage": None}
    row = evaluate_run(topic, state, lambda _: None, ScriptedLLM())
    assert row["coverage_shortlist"] == 0.0 and len(row["missed_in_shortlist"]) == len(topic["must_cite"])
