import threading

from copilot.graph.state import SearchFilters, make_event
from copilot.progress import Tracer, format_event_md, format_event_text
from tests.test_full_graph import _build


def test_tracer_works_outside_a_graph_run():
    t = Tracer("planner")
    ev = t("hello", "thinking", x=1)
    assert t.events == [ev] and ev.kind == "thinking" and ev.data == {"x": 1}


def test_events_stream_live_with_detail_and_in_a_sensible_order():
    cp, *_ = _build()
    events, states = [], []
    for st in cp.stream("How is sign language produced?", SearchFilters(), on_event=events.append):
        states.append(st)

    # the stream still yields graph states, ending in the finished one
    assert states[-1]["critique"].action == "pass" and states[-1]["iteration"] == 2

    kinds = [e.kind for e in events]
    assert kinds[0] == "info" and events[0].agent == "planner"                      # first sign of life is immediate
    assert "plan" in kinds and "search" in kinds and "papers" in kinds
    # the plan (with its queries) is visible, and comes before any search result
    plans = [e for e in events if e.kind == "plan"]
    assert plans[0].data["queries"] == ["q1"] and kinds.index("plan") < kinds.index("papers")
    # searches report per-source results; the shortlist names the papers chosen and why
    assert any(e.kind == "search" and e.data.get("n") == 3 for e in events)
    top = next(e for e in events if e.kind == "papers").data["papers"]
    assert [p["rank"] for p in top] == [1, 2, 3] and top[0]["score"] >= top[-1]["score"] and top[0]["sources"]
    # each paper is announced before its notes arrive (parallel readers, events pushed live)
    for pid in {e.data["paper_id"] for e in events if e.kind == "notes"}:
        order = [e.kind for e in events if e.data.get("paper_id") == pid]
        assert order.index("read") < order.index("notes")
    # writer shows its evidence; critic shows each verdict and a final decision with a reason
    ev_event = next(e for e in events if e.kind == "evidence")
    assert ev_event.data["papers"] and ev_event.data["papers"][0]["passages"]
    verdicts = [e for e in events if e.kind == "check" and "verdict" in e.data]
    assert verdicts and all(e.data["verdict"] == "supported" for e in verdicts)
    decisions = [e for e in events if e.kind == "decision"]
    assert [d.data["action"] for d in decisions] == ["retrieve_more", "pass"] and decisions[0].data["reason"]
    # live events (chronological, readers interleaved) and the saved trace (grouped per node) hold the same events
    assert sorted(e.message for e in events) == sorted(e.message for e in states[-1]["trace"])


def test_events_come_from_multiple_reader_threads_without_loss():
    cp, *_ = _build()
    threads, events = set(), []

    def collect(ev):
        threads.add(threading.current_thread().name)
        events.append(ev)

    for _ in cp.stream("How is sign language produced?", on_event=collect):
        pass
    assert sum(e.kind == "notes" for e in events) == 5
    assert threads == {threading.main_thread().name}        # delivered to the caller's thread, safe for Streamlit


def test_formatters_show_the_details():
    plan = make_event("planner", "sq1 · What architectures exist?", kind="plan", queries=["sign language production", "pose synthesis"])
    txt = format_event_text(plan)
    assert txt.splitlines()[0] == "[planner] sq1 · What architectures exist?"
    assert "search: sign language production" in txt and "search: pose synthesis" in txt
    assert format_event_text(plan, brief=True).count("\n") == 0

    papers = make_event("retriever", "picked", kind="papers", papers=[
        {"rank": 1, "id": "arxiv:1", "title": "Progressive Transformers", "year": 2020, "citations": 300,
         "score": 0.91, "sources": ["arxiv", "semantic_scholar"], "n_queries": 2}])
    assert "Progressive Transformers" in format_event_text(papers) and "0.91" in format_event_text(papers)
    assert "[arxiv, semantic_scholar; 2 queries]" in format_event_text(papers)

    bad = make_event("critic", "BLEU is 15", kind="check", verdict="unsupported", reason="number differs")
    assert "[FAIL]" in format_event_text(bad) and "why: number differs" in format_event_text(bad)
    assert "❌" in format_event_md(bad) and "number differs" in format_event_md(bad)

    think = make_event("planner", "I will split this by method family and evaluation. " * 6, kind="thinking")
    assert format_event_text(think).startswith("[planner] thinking: ") and format_event_md(think).count("*") >= 2
    assert format_event_md(plan).startswith("🧭 **planner** · **sq1")
