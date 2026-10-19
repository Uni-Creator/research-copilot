from copilot.agents.planner import PlannedSubQuestion, PlannerOutput, make_planner_node
from copilot.graph.state import CriticReport, SearchFilters, make_initial_state
from tests.conftest import FakeLLM


def _out():
    return PlannerOutput(
        sub_questions=[
            PlannedSubQuestion(
                question="What architectures exist?",
                queries=[
                    ' "sign language production" ',
                    "sign language generation",
                    "Sign Language Production",
                    "a",
                    "b",
                ],
            ),
            PlannedSubQuestion(question="   ", queries=["ignored"]),
            PlannedSubQuestion(question="Which datasets?", queries=["sign language datasets benchmark"]),
            PlannedSubQuestion(question="No queries", queries=[]),
        ],
        scope_notes="Focus on pose-based SLP.",
        year_from=2021,
    )


def test_planner_sanitises_output():
    node = make_planner_node(FakeLLM(_out()))
    r = node(make_initial_state("gloss-free SLP"))
    plan = r["plan"]
    assert [sq.id for sq in plan.sub_questions] == ["sq1", "sq2"]
    assert plan.sub_questions[0].queries == [
        "sign language production",
        "sign language generation",
        "a",
    ]  # deduped, capped at 3
    assert r["usage"].total_tokens == 150 and r["usage"].llm_calls == 1
    assert r["filters"].year_from == 2021 and r["errors"] == []


def test_planner_respects_user_year_filter():
    state = make_initial_state("q", SearchFilters(year_from=2018))
    assert "filters" not in make_planner_node(FakeLLM(_out()))(state)


def test_planner_falls_back_when_llm_fails():
    r = make_planner_node(FakeLLM(error=RuntimeError("boom")))(make_initial_state("my question"))
    assert r["plan"].all_queries() == ["my question"]
    assert r["usage"].llm_calls == 0 and "boom" in r["errors"][0]


def test_planner_followup_appends_only_new_subquestions():
    node = make_planner_node(FakeLLM(_out()))
    s = make_initial_state("q")
    s["plan"] = node(s)["plan"]
    follow = PlannerOutput(
        sub_questions=[
            PlannedSubQuestion(
                question="Which datasets?", queries=["dup question text"]
            ),  # duplicate question -> dropped
            PlannedSubQuestion(
                question="Evaluation metrics?", queries=["back translation BLEU sign", "sign language generation"]
            ),
        ]
    )
    s["critique"] = CriticReport(passed=False, action="retrieve_more", missing_topics=["metrics"])
    plan = make_planner_node(FakeLLM(follow))(s)["plan"]
    assert [sq.id for sq in plan.sub_questions] == ["sq1", "sq2", "sq3"]
    assert plan.sub_questions[2].queries == ["back translation BLEU sign"]  # already-seen query removed
