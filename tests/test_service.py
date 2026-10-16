import json
import tempfile
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from copilot.factory import CopilotFactory  # noqa: E402
from copilot.service import create_app  # noqa: E402
from tests.test_full_graph import SETTINGS, overrides  # noqa: E402

BODY = {"question": "How is sign language produced?", "max_papers": 3}


def _client(factory=None, **kw):
    kw.setdefault("api_key", None)
    kw.setdefault("out_dir", Path(tempfile.mkdtemp()))      # never write into the repo
    factory = factory or CopilotFactory(SETTINGS, **overrides()[0])
    return TestClient(create_app(factory, settings=SETTINGS, **kw))


def _wait(client, run_id, headers=None, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        r = client.get(f"/runs/{run_id}", headers=headers).json()
        if r["status"] in ("done", "error"):
            return r
        time.sleep(0.02)
    raise AssertionError("run did not finish")


def test_full_run_summary_events_and_reports(tmp_path):
    with _client(out_dir=tmp_path) as client:
        r = client.post("/runs", json=BODY)
        assert r.status_code == 202
        run_id = r.json()["id"]

        # SSE: every agent step arrives as a `trace` event, then `done`; ids allow resuming
        with client.stream("GET", f"/runs/{run_id}/events") as stream:
            lines = [ln for ln in stream.iter_lines() if ln]
        events = [json.loads(ln[6:]) for ln in lines if ln.startswith("data: ")]
        agents = [e["agent"] for e in events if e["type"] == "trace"]
        assert agents[0] == "planner" and agents.count("reader") == 5 and agents[-1] == "critic"
        assert events[-1] == {"type": "done", "status": "done", "error": None}
        assert "id: 0" in lines

        with client.stream("GET", f"/runs/{run_id}/events", headers={"Last-Event-ID": str(len(events) - 2)}) as stream:
            resumed = [ln for ln in stream.iter_lines() if ln.startswith("event:")]
        assert resumed == ["event: done"]                                 # only what was missed

        s = _wait(client, run_id)
        assert s["status"] == "done" and len(s["shortlist"]) == 5 and s["support_rate"] == 1.0
        assert s["iterations"] == 2 and s["draft_version"] == 2 and s["tokens"] > 0 and s["errors"] == []
        assert s["report"] == f"/runs/{run_id}/report.md"

        md = client.get(f"/runs/{run_id}/report.md").text
        assert md.startswith("# How is sign language produced?") and "## References" in md and "[cite:" not in md
        assert "<table>" in client.get(f"/runs/{run_id}/report.html").text
        assert json.loads(client.get(f"/runs/{run_id}/run.json").text)["iteration"] == 2
        assert (tmp_path / run_id / "report.md").exists()                 # persisted to disk


def test_validation_and_unknown_run():
    with _client() as client:
        assert client.post("/runs", json={"question": "short"}).status_code == 422
        assert client.post("/runs", json={**BODY, "until": "banana"}).status_code == 422
        assert client.post("/runs", json={**BODY, "max_papers": 500}).status_code == 422
        for path in ("", "/events", "/report.md", "/run.json"):
            assert client.get(f"/runs/nope{path}").status_code == 404
        assert client.get("/health").json()["ok"] is True


def test_api_key_required_when_configured():
    with _client(api_key="s3cret") as client:
        assert client.get("/health").json()["auth"] is True              # health stays open
        assert client.post("/runs", json=BODY).status_code == 401
        assert client.post("/runs", json=BODY, headers={"x-api-key": "wrong"}).status_code == 401
        ok = client.post("/runs", json=BODY, headers={"x-api-key": "s3cret"})
        assert ok.status_code == 202
        run_id = ok.json()["id"]
        assert client.get(f"/runs/{run_id}").status_code == 401
        assert _wait(client, run_id, headers={"x-api-key": "s3cret"})["status"] == "done"
        assert client.get(f"/runs/{run_id}?api_key=s3cret").status_code == 200   # EventSource-style


class _Slow:
    """Stub copilot whose stream blocks until released (to observe in-flight behaviour)."""

    def __init__(self, gate):
        self.gate = gate

    def stream(self, question, filters):
        self.gate.wait(5)
        yield {"question": question}


def test_report_not_ready_then_available_semantics():
    gate = threading.Event()
    with _client(lambda until, fulltext: _Slow(gate)) as client:
        run_id = client.post("/runs", json=BODY).json()["id"]
        assert client.get(f"/runs/{run_id}/report.md").status_code == 409     # still running
        assert client.get(f"/runs/{run_id}/run.json").status_code == 409
        gate.set()
        assert _wait(client, run_id)["status"] == "done"
        assert client.get(f"/runs/{run_id}/report.md").status_code == 404     # finished, but no draft was produced


def test_failed_run_reports_error_not_hang():
    def broken(until, fulltext):
        raise RuntimeError("no API key")

    with _client(broken) as client:
        run_id = client.post("/runs", json=BODY).json()["id"]
        s = _wait(client, run_id)
        assert s["status"] == "error" and "RuntimeError: no API key" in s["error"]
        with client.stream("GET", f"/runs/{run_id}/events") as stream:
            last = [ln for ln in stream.iter_lines() if ln.startswith("data: ")][-1]
        assert json.loads(last[6:])["status"] == "error"


def test_queue_limit_and_eviction():
    gate = threading.Event()
    with _client(lambda u, f: _Slow(gate), max_concurrent=1, max_queue=2) as client:
        a = client.post("/runs", json=BODY)
        b = client.post("/runs", json=BODY)
        assert (a.status_code, b.status_code) == (202, 202)
        assert client.post("/runs", json=BODY).status_code == 429             # queue full
        gate.set()
        for r in (a, b):
            _wait(client, r.json()["id"])

    with _client(lambda u, f: _Slow(threading.Event()) if False else _Instant(), max_runs=2) as client:
        ids = []
        for _ in range(3):
            ids.append(client.post("/runs", json=BODY).json()["id"])
            _wait(client, ids[-1])
        assert client.get(f"/runs/{ids[0]}").status_code == 404                # oldest finished run evicted
        assert client.get(f"/runs/{ids[2]}").status_code == 200


class _Instant:
    def stream(self, question, filters):
        yield {"question": question}
