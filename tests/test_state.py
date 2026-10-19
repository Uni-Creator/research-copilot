from copilot.graph.state import (
    ResearchPlan,
    SubQuestion,
    Usage,
    make_initial_state,
    merge_papers,
    merge_usage,
)
from copilot.tools.papers import PaperIndex
from tests.conftest import mk


def test_paper_id_priority():
    assert mk("A", arxiv="2401.1").paper_id == "arxiv:2401.1"
    assert mk("A", doi="10.1/X").paper_id == "doi:10.1/x"
    assert mk("A", s2="abc").paper_id == "s2:abc"
    assert mk("Some Title").paper_id.startswith("title:")


def test_merge_keeps_richest_data():
    a = mk("Sign Language Production", cites=0, s2=None, abstract="short")
    b = mk(
        "Sign Language Production",
        cites=42,
        arxiv="2401.1",
        abstract="a much longer abstract",
        source="semantic_scholar",
    )
    m = a.merged_with(b)
    assert m.citation_count == 42 and m.arxiv_id == "2401.1"
    assert m.abstract == "a much longer abstract"
    assert m.paper_id == "arxiv:2401.1"
    assert m.sources == ["arxiv", "semantic_scholar"]


def test_index_dedupes_across_sources_by_title_then_ids():
    idx = PaperIndex()
    i = idx.add(mk("Progressive Transformers for Sign Language Production!"))
    j = idx.add(
        mk(
            "progressive transformers for sign language production",
            cites=300,
            arxiv="2004.1",
            source="semantic_scholar",
        )
    )
    k = idx.add(mk("Totally Different Paper"))
    assert i == j and k != i and len(idx) == 2
    # the arXiv id learned from the second record is now a lookup key too
    assert idx.add(mk("Renamed Title Entirely", arxiv="2004.1")) == i


def test_reducers():
    left = {"arxiv:1": mk("A", arxiv="1", cites=1)}
    right = {"arxiv:1": mk("A", arxiv="1", cites=9), "arxiv:2": mk("B", arxiv="2")}
    merged = merge_papers(left, right)
    assert set(merged) == {"arxiv:1", "arxiv:2"} and merged["arxiv:1"].citation_count == 9
    u = merge_usage(Usage(llm_calls=1, input_tokens=10), Usage(llm_calls=2, output_tokens=5))
    assert (u.llm_calls, u.total_tokens) == (3, 15)


def test_plan_queries_unique_and_initial_state():
    plan = ResearchPlan(
        sub_questions=[
            SubQuestion(id="sq1", question="q", queries=["a b", "C d"]),
            SubQuestion(id="sq2", question="r", queries=["A B", "e f"]),
        ]
    )
    assert plan.all_queries() == ["a b", "C d", "e f"]
    s = make_initial_state("  hello  ")
    assert s["question"] == "hello" and s["iteration"] == 0
