"""HTTP API: start a run, stream agent events live (SSE), fetch the finished report.

    uvicorn copilot.service:create_app --factory --port 8000

POST /runs                      start a run            -> 202 {id, links}
GET  /runs/{id}                 status + summary (shortlist, support rate, tokens, errors)
GET  /runs/{id}/events          Server-Sent Events: one `trace` event per agent step, then `done`
                                (supports Last-Event-ID, so a dropped connection can resume)
GET  /runs/{id}/report.md|html  finished report
GET  /runs/{id}/run.json        full state: plan, papers, notes, draft, critique, trace
GET  /health

Auth: set COPILOT_API_KEY and send it as `x-api-key` (or ?api_key= for EventSource, which cannot
set headers). Unset = open, which is fine locally but NOT for a public deployment: every run spends
LLM tokens. Concurrency is bounded; finished runs are evicted oldest-first.
"""

from __future__ import annotations

import hmac
import json
import os
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from copilot.config import Settings
from copilot.factory import CopilotFactory
from copilot.graph.state import SearchFilters
from copilot.render import dumps_state, render_html, render_markdown, save_run

_UNSET: Any = object()


class RunRequest(BaseModel):
    question: str = Field(min_length=10, max_length=500)
    year_from: int | None = Field(None, ge=1990, le=2100)
    year_to: int | None = Field(None, ge=1990, le=2100)
    max_papers: int | None = Field(None, ge=1, le=50)
    until: Literal["retrieve", "read", "write", "critic"] = "critic"
    fulltext: bool = True


@dataclass
class Run:
    id: str
    request: RunRequest
    status: str = "queued"  # queued | running | done | error
    events: list[dict] = field(default_factory=list)
    state: dict | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    finished: bool = False
    cond: threading.Condition = field(default_factory=threading.Condition)

    def set_status(self, status: str) -> None:
        with self.cond:
            self.status = status

    def emit(self, event: dict) -> None:
        with self.cond:
            self.events.append(event)
            self.cond.notify_all()

    def finish(self, status: str, state: dict | None, error: str | None = None) -> None:
        with self.cond:
            self.status, self.state, self.error = status, state, error
            self.events.append({"type": "done", "status": status, "error": error})
            self.finished = True
            self.cond.notify_all()


class RunManager:
    def __init__(
        self,
        factory: Callable[..., Any],
        *,
        max_concurrent: int = 2,
        max_runs: int = 50,
        max_queue: int = 10,
        out_dir: Path | None = None,
    ) -> None:
        self.factory, self.max_runs, self.max_queue, self.out_dir = factory, max_runs, max_queue, out_dir
        self.runs: dict[str, Run] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=max_concurrent, thread_name_prefix="run")

    def submit(self, req: RunRequest) -> Run:
        with self._lock:
            if sum(not r.finished for r in self.runs.values()) >= self.max_queue:
                raise HTTPException(429, "too many runs queued; try again shortly")
            run = Run(id=uuid.uuid4().hex[:12], request=req)
            self.runs[run.id] = run
            for old in [r for r in self.runs.values() if r.finished][: max(0, len(self.runs) - self.max_runs)]:
                del self.runs[old.id]  # oldest finished first (dict keeps insertion order)
        self._pool.submit(self._work, run)
        return run

    def get(self, run_id: str) -> Run | None:
        return self.runs.get(run_id)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _work(self, run: Run) -> None:
        req, final = run.request, None
        run.set_status("running")
        try:
            copilot = self.factory(req.until, req.fulltext)
            filters = SearchFilters(year_from=req.year_from, year_to=req.year_to, max_papers=req.max_papers)
            # progress events are pushed live (also from parallel Readers), not batched per graph step
            states = deque(
                copilot.stream(
                    req.question,
                    filters,
                    on_event=lambda ev: run.emit({"type": "trace", **ev.model_dump(mode="json")}),
                ),
                maxlen=1,
            )
            final = states[0] if states else None
            if self.out_dir and final:
                try:
                    save_run(self.out_dir / run.id, final)
                except OSError:
                    pass  # persistence is best-effort
            run.finish("done", final)
        except Exception as exc:  # surface any failure to the client
            run.finish("error", final, f"{type(exc).__name__}: {exc}")


def summarize(run: Run) -> dict[str, Any]:
    st: dict = run.state or {}
    papers = st.get("papers", {})
    crit, usage, draft = st.get("critique"), st.get("usage"), st.get("draft")
    return {
        "id": run.id,
        "status": run.status,
        "error": run.error,
        "question": run.request.question,
        "events": len(run.events),
        "shortlist": [
            {
                "id": pid,
                "title": papers[pid].title,
                "year": papers[pid].year,
                "score": papers[pid].score,
                "url": papers[pid].url,
            }
            for pid in st.get("shortlist", [])
            if pid in papers
        ],
        "draft_version": draft.version if draft else None,
        "support_rate": crit.support_rate if crit else None,
        "iterations": st.get("iteration", 0),
        "tokens": usage.total_tokens if usage else 0,
        "errors": st.get("errors", []),
        "report": f"/runs/{run.id}/report.md" if run.status == "done" and draft else None,
    }


def create_app(
    factory: Callable[..., Any] | None = None,
    *,
    settings: Settings | None = None,
    api_key: Any = _UNSET,
    max_concurrent: int = 2,
    max_runs: int = 50,
    max_queue: int = 10,
    out_dir: Path | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    key = os.environ.get("COPILOT_API_KEY") or None if api_key is _UNSET else api_key
    manager = RunManager(
        factory or CopilotFactory(settings),
        max_concurrent=max_concurrent,
        max_runs=max_runs,
        max_queue=max_queue,
        out_dir=out_dir if out_dir is not None else Path(settings.traces_dir) / "runs",
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        manager.shutdown()

    app = FastAPI(title="Research Copilot", version="1.0.0", lifespan=lifespan)
    app.state.manager = manager

    def auth(x_api_key: str | None = Header(None), api_key_q: str | None = Query(None, alias="api_key")) -> None:
        if key and not hmac.compare_digest((x_api_key or api_key_q or "").encode(), key.encode()):
            raise HTTPException(401, "missing or invalid API key")

    def find(run_id: str) -> Run:
        run = manager.get(run_id)
        if run is None:
            raise HTTPException(404, "unknown run")
        return run

    def finished_with_draft(run_id: str) -> Run:
        run = find(run_id)
        if not run.finished:
            raise HTTPException(409, f"run is {run.status}; report not ready")
        if run.status == "error" or not (run.state or {}).get("draft"):
            raise HTTPException(404, "this run produced no report (failed, or stopped before the writer)")
        return run

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "runs": len(manager.runs), "auth": bool(key)}

    @app.post("/runs", status_code=202, dependencies=[Depends(auth)])
    def start_run(req: RunRequest) -> dict[str, Any]:
        run = manager.submit(req)
        return {
            "id": run.id,
            "status": run.status,
            "links": {
                "self": f"/runs/{run.id}",
                "events": f"/runs/{run.id}/events",
                "report": f"/runs/{run.id}/report.md",
            },
        }

    @app.get("/runs/{run_id}", dependencies=[Depends(auth)])
    def get_run(run_id: str) -> dict[str, Any]:
        return summarize(find(run_id))

    @app.get("/runs/{run_id}/events", dependencies=[Depends(auth)])
    def stream_events(run_id: str, request: Request) -> StreamingResponse:
        run = find(run_id)
        try:
            start = int(request.headers.get("last-event-id", -1)) + 1
        except ValueError:
            start = 0

        def gen():
            i = max(start, 0)
            while True:
                with run.cond:
                    if i >= len(run.events) and not run.finished:
                        run.cond.wait(timeout=15)
                    batch, finished = run.events[i:], run.finished
                for ev in batch:
                    yield f"id: {i}\nevent: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
                    i += 1
                if not batch:
                    if finished:
                        return
                    yield ": keepalive\n\n"

        return StreamingResponse(
            gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )

    @app.get("/runs/{run_id}/report.md", dependencies=[Depends(auth)], response_class=PlainTextResponse)
    def report_md(run_id: str) -> str:
        return render_markdown(finished_with_draft(run_id).state)  # type: ignore[arg-type]

    @app.get("/runs/{run_id}/report.html", dependencies=[Depends(auth)], response_class=HTMLResponse)
    def report_html(run_id: str) -> str:
        run = finished_with_draft(run_id)
        return render_html(render_markdown(run.state), run.request.question)  # type: ignore[arg-type]

    @app.get("/runs/{run_id}/run.json", dependencies=[Depends(auth)])
    def run_json(run_id: str) -> Response:
        run = find(run_id)
        if not run.finished:
            raise HTTPException(409, f"run is {run.status}")
        return Response(dumps_state(run.state or {}), media_type="application/json")

    return app
