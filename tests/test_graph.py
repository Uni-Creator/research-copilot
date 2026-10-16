from copilot.agents.planner import PlannedSubQuestion, PlannerOutput
from copilot.config import Settings
from copilot.graph.state import CriticReport, SearchFilters, make_initial_state
from copilot.graph.workflow import build_graph
from tests.conftest import FakeLLM, fake_source, mk


def _settings(**kw):
    return Settings(max_papers=3, extra_per_pass=2, retriever_workers=2, **kw)


def _plan(*pairs):
    return PlannerOutput(sub_questions=[PlannedSubQuestion(question=q, queries=qs) for q, qs in pairs])


def test_end_to_end_plan_retrieve_dedupe_rank():
    def top(s):
        return mk("Gloss-free Sign Language Production", year=2024, cites=80, arxiv="2401.1", abstract="gloss-free sign language production", source=s)

    arxiv = fake_source({
        "sign language production": [top("arxiv"), mk("Other SLP paper", arxiv="2301.2"), mk("Old", year=2010, arxiv="1001.1")],
        "sign pose synthesis": [mk("Pose paper", arxiv="2302.3")],
    })
    s2 = fake_source({
        "sign language production": [top("semantic_scholar").model_copy(update={"arxiv_id": None, "s2_id": "S2A"}), mk("S2 only", cites=500, s2="S2B", source="semantic_scholar")],
    })
    graph = build_graph(
        _settings(), planner_llm=FakeLLM(_plan(("approaches?", ["sign language production", "sign pose synthesis"]))),
        sources={"arxiv": arxiv, "semantic_scholar": s2},
    )
    final = graph.invoke(make_initial_state("gloss-free sign language production", SearchFilters(year_from=2015)))

    assert len(final["shortlist"]) == 3                      # capped by max_papers
    assert final["shortlist"][0] == "arxiv:2401.1"           # found by both sources + relevant -> top
    assert final["papers"]["arxiv:2401.1"].sources == ["arxiv", "semantic_scholar"]
    assert "arxiv:1001.1" not in final["papers"]             # filtered by year_from
    assert sum(1 for p in final["papers"].values() if "Gloss-free" in p.title) == 1   # deduped
    assert final["searched_queries"] == ["sign language production", "sign pose synthesis"]
    assert final["usage"].total_tokens == 150 and final["errors"] == []
    assert [e.agent for e in final["trace"]] == ["planner", "retriever"]


def test_followup_pass_only_searches_new_queries_and_grows_shortlist():
    a = fake_source({
        "q1": [mk("P1", arxiv="1", cites=9), mk("P2", arxiv="2"), mk("P3", arxiv="3")],
        "q2 metrics": [mk("M1", arxiv="4", cites=50), mk("M2", arxiv="5"), mk("M3", arxiv="6")],
    })
    llm = FakeLLM(_plan(("core", ["q1"])))
    graph = build_graph(_settings(), planner_llm=llm, sources={"arxiv": a})
    first = graph.invoke(make_initial_state("question"))
    assert len(first["shortlist"]) == 3 and a.calls == ["q1"]

    llm.parsed = _plan(("metrics", ["q2 metrics"]))
    second_in = {**first, "critique": CriticReport(passed=False, action="retrieve_more", missing_topics=["metrics"])}
    second = graph.invoke(second_in)

    assert a.calls == ["q1", "q2 metrics"]                       # q1 not re-run
    assert second["shortlist"][:3] == first["shortlist"]         # old picks keep their place
    assert len(second["shortlist"]) == 5                         # + extra_per_pass new papers
    assert {"arxiv:4", "arxiv:5"} == set(second["shortlist"][3:])


def test_source_failure_is_reported_not_fatal():
    def broken(q, f, n):
        raise TimeoutError("api down")

    ok = fake_source({"q": [mk("Survivor", arxiv="9")]})
    graph = build_graph(_settings(), planner_llm=FakeLLM(_plan(("c", ["q"]))), sources={"s2": broken, "arxiv": ok})
    final = graph.invoke(make_initial_state("question"))
    assert final["shortlist"] == ["arxiv:9"]
    assert any("api down" in e for e in final["errors"])
