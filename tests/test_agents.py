from copilot.agents.critic import (
    CoverageOutput, VerdictItem, VerifyOutput, make_critic_node, split_claims, support_rate, verify_claims,
)
from copilot.agents.reader import NotesOut, make_reader_node
from copilot.agents.writer import resolve_citations, to_aliases
from copilot.config import Settings
from copilot.graph.state import ClaimCheck, Chunk, Draft, make_initial_state
from copilot.tools.embeddings import HashingEmbedder
from copilot.tools.vectorstore import VectorStore
from tests.conftest import ScriptedLLM, mk

S = Settings(embedder="hash", max_iterations=2)


def test_resolve_citations_maps_aliases_and_strips_invented_ones():
    amap = {"C1": "arxiv:1#abs", "C2": "arxiv:2#3"}
    text = "A works [cite:C1]. B too [cite: C2, C9]. Bare form [C1, C2]. Ghost [cite:C7]."
    md, cited, stripped = resolve_citations(text, amap)
    assert "[cite:arxiv:1#abs]" in md and "[cite:arxiv:2#3]" in md
    assert "C9" not in md and "C7" not in md and "[cite:]" not in md
    assert "Ghost" not in md and "B too" in md          # sentence resting only on an invented cite is dropped
    assert cited == ["arxiv:1#abs", "arxiv:2#3"] and stripped == 2
    # round trip for revision prompts; unknown ids are dropped
    back = to_aliases("X [cite:arxiv:2#3] Y [cite:arxiv:9#1]", {"arxiv:2#3": "C5"})
    assert back == "X [cite:C5] Y "


def test_split_claims_only_cited_sentences():
    md = ("# Title [cite:x#1]\nIntro without citation. Method A uses a transformer [cite:p1#1][cite:p2#1]. "
          "Another claim without cite.\n- Bullet claim [cite:p3#abs].\n| table | row [cite:p1#1] |")
    assert split_claims(md) == [
        ("Method A uses a transformer.", ["p1#1", "p2#1"]),
        ("Bullet claim.", ["p3#abs"]),
    ]


def test_verify_claims_batches_and_handles_missing_chunks_and_verdicts():
    texts = {"a#1": "model uses transformer", "b#1": "BLEU of 12"}
    llm = ScriptedLLM({VerifyOutput: lambda m: VerifyOutput(verdicts=[
        VerdictItem(index=0, verdict="supported"),
        VerdictItem(index=1, verdict="partial", reason="number differs"),
    ])})
    claims = [("uses transformer", ["a#1"]), ("BLEU is 15", ["b#1"]), ("ghost", ["zzz#1"]), ("no verdict", ["a#1"])]
    checks, usage = verify_claims(llm, claims, texts.get, batch_size=3)
    assert [c.verdict for c in checks] == ["supported", "partial", "unsupported", "unsupported"]
    assert checks[2].reason == "cited chunk not found" and checks[3].reason == "no verdict returned"
    # "ghost" is rejected without an LLM call; the other 3 fit in one batch
    assert llm.calls == ["VerifyOutput"] and usage.llm_calls == 1
    assert support_rate(checks) == round((1 + 0.5 + 0 + 0) / 4, 4)
    llm2 = ScriptedLLM({VerifyOutput: VerifyOutput(verdicts=[VerdictItem(index=0, verdict="supported")])})
    verify_claims(llm2, claims, texts.get, batch_size=1)
    assert llm2.calls == ["VerifyOutput"] * 3                      # batch_size=1 -> one call per judged claim


def test_reader_full_text_and_pdf_failure_fallback():
    store = VectorStore(HashingEmbedder())
    llm = ScriptedLLM({NotesOut: NotesOut(method="diffusion over pose tokens", datasets=["PHOENIX14T"], metrics=["BLEU"])})
    paper = mk("Gloss to Pose Diffusion", arxiv="2208.1", abstract="We generate sign poses with diffusion.")
    paper.pdf_url = "https://arxiv.org/pdf/2208.1"
    pages = [(1, "Method: a discrete diffusion model over pose tokens produces sign sequences from gloss input. " * 3),
             (2, "Experiments on PHOENIX14T report back translation BLEU scores against baselines in detail. " * 3)]

    ok = make_reader_node(llm, store, lambda p: pages, S)({"paper": paper, "question": "q"})
    n = ok["notes"]["arxiv:2208.1"]
    assert n.method == "diffusion over pose tokens" and n.datasets == ["PHOENIX14T"]
    assert "arxiv:2208.1#abs" in n.chunk_ids and any(c.endswith("#1") for c in n.chunk_ids)
    assert store.get("arxiv:2208.1#abs") is not None and len(store) >= 3
    assert "full text" in ok["trace"][0].message and ok["errors"] == []

    def boom(p):
        raise TimeoutError("pdf down")

    store2 = VectorStore(HashingEmbedder())
    bad = make_reader_node(llm, store2, boom, S)({"paper": paper, "question": "q"})
    assert "abstract only" in bad["trace"][0].message and "pdf down" in bad["errors"][0]
    assert len(store2) == 1                                   # only the abstract chunk exists


def _critic_state(draft_md, version=1, iteration=0):
    s = make_initial_state("What works?")
    s.update(draft=Draft(markdown=draft_md, version=version), iteration=iteration)
    return s


def _store():
    st = VectorStore(HashingEmbedder())
    st.add([Chunk(chunk_id="p1#1", paper_id="p1", text="the model uses a transformer encoder")])
    return st


def test_critic_routing_decisions():
    md = "It uses a transformer [cite:p1#1]."
    supported = VerifyOutput(verdicts=[VerdictItem(index=0, verdict="supported")])
    weak = VerifyOutput(verdicts=[VerdictItem(index=0, verdict="unsupported", reason="no such claim")])

    def run(verify, cover, iteration=0):
        llm = ScriptedLLM({VerifyOutput: verify, CoverageOutput: cover})
        return make_critic_node(llm, _store(), S)(_critic_state(md, iteration=iteration))

    r = run(supported, CoverageOutput())
    assert r["critique"].action == "pass" and r["critique"].passed and r["iteration"] == 1
    r = run(supported, CoverageOutput(missing_topics=["evaluation metrics"]))
    assert r["critique"].action == "retrieve_more" and r["critique"].missing_topics == ["evaluation metrics"]
    r = run(weak, CoverageOutput())
    assert r["critique"].action == "revise_draft" and not r["critique"].passed
    # out of iterations: never loops, but is honest that quality bar was not met
    r = run(weak, CoverageOutput(missing_topics=["x"]), iteration=2)
    assert r["critique"].action == "pass" and not r["critique"].passed and r["critique"].missing_topics == []
