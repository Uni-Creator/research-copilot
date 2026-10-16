import re

from copilot.agents.critic import VerdictItem, VerifyOutput
from copilot.baseline import run_baseline
from copilot.config import Settings
from copilot.graph.state import SearchFilters
from eval.metrics import coverage, evaluate_run, match_paper
from eval.run_eval import evaluate, results_table
from tests.conftest import ScriptedLLM, fake_source, mk

S = Settings(max_papers=3, retriever_workers=1)
TOPIC = {"id": "t1", "question": "sign language production",
         "must_cite": [{"title": "Progressive Transformers for End-to-End Sign Language Production"},
                       {"arxiv_id": "2203.1"}, {"title": "A Paper Nobody Retrieved Ever"}]}


def _src():
    return {"arxiv": fake_source({"sign language production": [
        mk("Progressive transformers for end-to-end sign language production", arxiv="2004.1", cites=300,
           abstract="Progressive transformers generate sign pose from text."),
        mk("Some other paper", arxiv="2203.1", abstract="Another abstract about signing."),
        mk("Irrelevant", arxiv="1111.1", abstract="Unrelated.")]})}


def _writer(messages):
    labels = re.findall(r"^\[(C\d+)\]", messages[1][1], flags=re.MULTILINE)
    return f"Progressive transformers generate pose [cite:{labels[0]}]. Second claim [cite:{labels[1]}]."


def _judge(verdicts):
    return ScriptedLLM({VerifyOutput: VerifyOutput(verdicts=verdicts)})


def test_match_paper_and_coverage():
    p = mk("Progressive Transformers for End-to-End Sign Language Production!", arxiv="2004.1")
    assert match_paper({"title": "progressive transformers for end-to-end sign language production"}, p)
    assert match_paper({"arxiv_id": "2004.1", "title": "totally different"}, p)
    assert not match_paper({"title": "Sign Language Production: A Review"}, p)
    assert coverage([], [p]) == (None, [])
    assert coverage([{"title": "Nope nope nope"}, {"arxiv_id": "2004.1"}], [p]) == (0.5, ["Nope nope nope"])


def test_baseline_runs_one_search_and_one_write_call():
    writer = ScriptedLLM(text=_writer)
    state, text = run_baseline(TOPIC["question"], SearchFilters(), S, writer, _src())
    assert state["shortlist"][0] == "arxiv:2004.1" and writer.calls == ["text"]
    assert state["draft"].cited_chunk_ids == ["arxiv:2004.1#abs", "arxiv:2203.1#abs"]
    assert text("arxiv:2004.1#abs").startswith("Progressive transformers") and text("nope") is None


def test_evaluate_run_metrics():
    state, text = run_baseline(TOPIC["question"], SearchFilters(), S, ScriptedLLM(text=_writer), _src())
    row = evaluate_run(TOPIC, state, text, _judge([VerdictItem(index=0, verdict="supported"),
                                                   VerdictItem(index=1, verdict="unsupported", reason="x")]))
    assert row["coverage_shortlist"] == round(2 / 3, 4) and row["coverage_cited"] == round(2 / 3, 4)
    assert row["n_claims"] == 2 and row["support_rate"] == 0.5 and row["n_papers_cited"] == 2
    assert row["unsupported_claims"] == ["Second claim."] and row["missed_in_shortlist"] == ["A Paper Nobody Retrieved Ever"]
    assert row["tokens"] == 200 + 100 and row["stripped_citations"] == 0


def test_evaluate_aggregates_and_survives_a_failing_variant():
    def good(q, f):
        return run_baseline(q, f, S, ScriptedLLM(text=_writer), _src())

    def bad(q, f):
        raise RuntimeError("variant exploded")

    judge = _judge([VerdictItem(index=0, verdict="supported"), VerdictItem(index=1, verdict="supported")])
    rows = evaluate([TOPIC], {"baseline": good, "broken": bad}, judge, S, log=lambda *_: None)
    assert [r["variant"] for r in rows] == ["baseline", "broken"]
    assert rows[1]["error"] == "RuntimeError('variant exploded')" and rows[0]["support_rate"] == 1.0
    table = results_table(rows)
    assert "| baseline |" in table and "| broken | - |" in table and table.count("\n") == 3
