import re

from copilot.agents.critic import CoverageOutput, VerdictItem, VerifyOutput
from copilot.agents.planner import PlannedSubQuestion, PlannerOutput
from copilot.agents.reader import NotesOut
from copilot.config import Settings
from copilot.graph.state import SearchFilters
from copilot.graph.workflow import build_copilot
from copilot.render import render_html, render_markdown, save_run
from copilot.tools.embeddings import HashingEmbedder
from tests.conftest import ScriptedLLM, fake_source, mk

SETTINGS = Settings(embedder="hash", max_papers=3, extra_per_pass=2, max_iterations=2, retriever_workers=2)


def _paper(title, arxiv, cites=0):
    p = mk(
        title,
        year=2023,
        cites=cites,
        arxiv=arxiv,
        abstract=f"{title}: we study sign language pose generation with transformers.",
    )
    p.pdf_url = f"https://arxiv.org/pdf/{arxiv}"
    p.authors = ["Ada Lovelace", "Alan Turing"]
    return p


def _plan(question, *queries):
    return PlannerOutput(sub_questions=[PlannedSubQuestion(question=question, queries=list(queries))])


def _writer_text(messages):
    labels = re.findall(r"^\[(C\d+)\]", messages[1][1], flags=re.MULTILINE)
    assert labels, "writer prompt must contain labelled evidence"
    return (
        f"## Approaches\nTransformers generate pose sequences [cite:{labels[0]}]. "
        f"Another system differs in its encoder [cite:{labels[-1]}]. Invented claim [cite:C99]."
    )


def _verify_all_supported(messages):
    n = len(re.findall(r"^Claim \d+:", messages[1][1], flags=re.MULTILINE))
    return VerifyOutput(verdicts=[VerdictItem(index=i, verdict="supported") for i in range(n)])


def overrides():
    """Fresh fakes for one full run (the scripted LLM lists are consumed as they are used)."""
    src = fake_source(
        {
            "q1": [
                _paper("Alpha pose model", "1", 90),
                _paper("Beta gloss model", "2", 50),
                _paper("Gamma sign model", "3", 5),
            ],
            "q2 metrics": [_paper("Delta evaluation metrics", "4", 70), _paper("Epsilon BLEU study", "5", 10)],
        }
    )
    llms = dict(
        planner_llm=ScriptedLLM(
            {PlannerOutput: [_plan("core approaches", "q1"), _plan("evaluation metrics", "q2 metrics")]}
        ),
        reader_llm=ScriptedLLM({NotesOut: NotesOut(method="a transformer", datasets=["PHOENIX14T"], metrics=["BLEU"])}),
        writer_llm=ScriptedLLM(text=_writer_text),
        critic_llm=ScriptedLLM(
            {
                VerifyOutput: _verify_all_supported,
                CoverageOutput: [CoverageOutput(missing_topics=["evaluation metrics"]), CoverageOutput()],
            }
        ),
    )
    fetched = []

    def fetch(paper):
        fetched.append(paper.paper_id)
        return [(1, f"{paper.title} describes a transformer that maps gloss to pose sequences. " * 4)]

    kw = dict(sources={"arxiv": src}, embedder=HashingEmbedder(), fetch_pages=fetch, **llms)
    return kw, src, llms, fetched


def _build():
    kw, src, llms, fetched = overrides()
    return build_copilot(SETTINGS, until="critic", **kw), src, llms, fetched


def test_full_pipeline_with_critic_retrieve_more_loop():
    cp, src, llms, fetched = _build()
    final = cp.run("How is sign language produced?", SearchFilters())

    # loop ran exactly once: plan -> read -> write -> critic(retrieve_more) -> plan -> ... -> pass
    kinds = [e.kind for e in final["trace"]]
    assert kinds.count("papers") == 2 and kinds.count("decision") == 2  # two retrieval passes, two critic runs
    assert kinds.count("notes") == 5  # 3 + 2 new; never re-read
    assert [e.data["action"] for e in final["trace"] if e.kind == "decision"] == ["retrieve_more", "pass"]
    assert sum(e.message.startswith("draft v") and "ready" in e.message for e in final["trace"]) == 2
    assert sorted(fetched) == sorted(set(fetched)) and len(fetched) == 5
    assert src.calls == ["q1", "q2 metrics"]  # q1 not searched twice
    assert len(final["shortlist"]) == 5 and set(final["notes"]) == set(final["shortlist"])

    assert final["iteration"] == 2 and final["draft"].version == 2
    assert final["critique"].action == "pass" and final["critique"].passed
    assert llms["planner_llm"].calls.count("PlannerOutput") == 2
    assert llms["reader_llm"].calls.count("NotesOut") == 5

    # invented citation C99 never reaches the draft; every cited chunk exists in the store
    assert final["draft"].stripped_citations == 1 and "C99" not in final["draft"].markdown
    assert "Invented claim" not in final["draft"].markdown  # unsupported sentence removed, not left uncited
    assert final["draft"].cited_chunk_ids and all(cp.chunk_text(c) for c in final["draft"].cited_chunk_ids)
    assert final["usage"].llm_calls > 10 and final["errors"] == []


def test_until_write_skips_critic_and_until_read_skips_writer():
    cp, *_ = _build()
    no_critic = build_copilot(
        SETTINGS,
        until="write",
        sources={"arxiv": fake_source({"q1": [_paper("A", "1")]})},
        embedder=HashingEmbedder(),
        fetch_pages=None,
        planner_llm=ScriptedLLM({PlannerOutput: _plan("c", "q1")}),
        reader_llm=ScriptedLLM({NotesOut: NotesOut()}),
        writer_llm=ScriptedLLM(text=_writer_text),
    ).run("question")
    assert "critique" not in no_critic and no_critic["draft"].version == 1

    read_only = build_copilot(
        SETTINGS,
        until="read",
        sources={"arxiv": fake_source({"q1": [_paper("A", "1")]})},
        embedder=HashingEmbedder(),
        fetch_pages=None,
        planner_llm=ScriptedLLM({PlannerOutput: _plan("c", "q1")}),
        reader_llm=ScriptedLLM({NotesOut: NotesOut()}),
    ).run("question")
    assert "draft" not in read_only and len(read_only["notes"]) == 1


def test_empty_search_ends_gracefully():
    cp = build_copilot(
        SETTINGS,
        until="critic",
        sources={"arxiv": fake_source({})},
        embedder=HashingEmbedder(),
        fetch_pages=None,
        planner_llm=ScriptedLLM({PlannerOutput: _plan("c", "nothing")}),
        reader_llm=ScriptedLLM(),
        writer_llm=ScriptedLLM(),
        critic_llm=ScriptedLLM(),
    )
    final = cp.run("question")
    assert final["shortlist"] == [] and "draft" not in final
    assert any("no papers found" in e for e in final["errors"])


def test_render_report(tmp_path):
    cp, *_ = _build()
    final = cp.run("How is sign language produced?")
    md = render_markdown(final)
    assert md.startswith("# How is sign language produced?")
    assert "[cite:" not in md and re.search(r"\[1\]", md) and "## References" in md
    assert "| # | Paper | Method |" in md and "PHOENIX14T" in md
    assert "Lovelace, Alan Turing" in md and "support rate 100%" in md
    html = render_html(md, "T <x>")
    assert "<table>" in html and "&lt;x&gt;" in html and "prefers-color-scheme" in html
    paths = save_run(tmp_path / "out", final)
    assert all(p.exists() for p in paths.values()) and "arxiv:" in paths["run.json"].read_text()
