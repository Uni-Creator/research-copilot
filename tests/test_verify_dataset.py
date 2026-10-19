import json

from eval.verify_dataset import format_report, main, resolve, verify_dataset
from tests.conftest import mk


def _src(papers_by_title, fail=False):
    def fn(query, filters, limit):
        if fail:
            raise TimeoutError("down")
        return [p.model_copy(deep=True) for p in papers_by_title.get(query, [])][:limit]

    return fn


DATA = {
    "topics": [
        {
            "id": "t",
            "question": "q",
            "must_cite": [
                {"title": "Progressive Transformers for End-to-End Sign Language Production", "verified": "listing"},
                {"title": "Wrong Id Paper Title Here", "arxiv_id": "1111.1111", "verified": "listing"},
                {"title": "A Paper That Does Not Exist Anywhere"},
            ],
            "candidates": [{"title": "Neural Discrete Representation Learning"}, {"title": "Some Other Candidate"}],
        }
    ]
}


def _sources():
    s2 = {
        "Progressive Transformers for End-to-End Sign Language Production": [
            mk("Progressive transformers for end-to-end sign language production", arxiv="2004.14874", doi="10.1/pt")
        ],
        "Wrong Id Paper Title Here": [mk("Wrong Id Paper Title Here", arxiv="2222.2222")],
        "Neural Discrete Representation Learning": [mk("Neural Discrete Representation Learning", arxiv="1711.00937")],
        "A Paper That Does Not Exist Anywhere": [mk("Something Quite Different About Images")],
    }
    return {"semantic_scholar": _src(s2), "arxiv": _src({})}


def test_resolve_statuses_and_hints():
    srcs = _sources()
    assert resolve(DATA["topics"][0]["must_cite"][0], srcs).status == "ok"
    assert resolve(DATA["topics"][0]["must_cite"][1], srcs).status == "id_mismatch"
    missing = resolve(DATA["topics"][0]["must_cite"][2], srcs)
    assert missing.status == "not_found" and missing.hints == ["Something Quite Different About Images"]
    assert resolve({"title": "x y z"}, {"s2": _src({}, fail=True)}).status == "error"
    # one broken source must not hide a hit from the other
    both = {"semantic_scholar": _src({}, fail=True), "arxiv": _src({"T": [mk("T", arxiv="9.9")]})}
    assert resolve({"title": "T"}, both).status == "ok"


def test_verify_fills_ids_promotes_and_never_mutates_input():
    snapshot = json.dumps(DATA, sort_keys=True)
    rows, out = verify_dataset(DATA, _sources(), promote=True)
    assert json.dumps(DATA, sort_keys=True) == snapshot
    topic = out["topics"][0]
    first = topic["must_cite"][0]
    assert first["arxiv_id"] == "2004.14874" and first["doi"] == "10.1/pt" and first["verified"] == "api"
    assert [e["title"] for e in topic["must_cite"]][-1] == "Neural Discrete Representation Learning"  # promoted
    assert topic["must_cite"][-1]["arxiv_id"] == "1711.00937"
    assert [e["title"] for e in topic["candidates"]] == ["Some Other Candidate"]  # unresolved stays
    assert [r["status"] for r in rows] == ["ok", "id_mismatch", "not_found", "ok", "not_found"]
    _, no_promote = verify_dataset(DATA, _sources(), promote=False)
    assert len(no_promote["topics"][0]["must_cite"]) == 3 and len(no_promote["topics"][0]["candidates"]) == 2


def test_cli_report_exit_code_and_write(tmp_path, capsys):
    path = tmp_path / "d.json"
    path.write_text(json.dumps(DATA), encoding="utf-8")
    assert main(["--topics", str(path)], sources=_sources()) == 1  # a must_cite entry is not ok
    out = capsys.readouterr().out
    assert "MISSING " in out and "closest: Something Quite Different" in out and "ID?" in out
    assert json.loads(path.read_text()) == DATA  # dry run writes nothing

    good = {
        "topics": [
            {
                "id": "t",
                "question": "q",
                "must_cite": [DATA["topics"][0]["must_cite"][0]],
                "candidates": [DATA["topics"][0]["candidates"][0]],
            }
        ]
    }
    path.write_text(json.dumps(good), encoding="utf-8")
    assert main(["--topics", str(path), "--promote", "--write"], sources=_sources()) == 0
    saved = json.loads(path.read_text())
    assert len(saved["topics"][0]["must_cite"]) == 2 and saved["topics"][0]["candidates"] == []
    assert "ERROR" in format_report(
        [
            {
                "topic": "t",
                "list": "must_cite",
                "title": "x",
                "status": "error",
                "resolved_title": None,
                "arxiv_id": None,
                "hints": ["every source failed"],
            }
        ]
    )
