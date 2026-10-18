# Using the HTTP API

```bash
docker run -p 8000:8000 -e APP=api -e ANTHROPIC_API_KEY=... -e COPILOT_API_KEY=secret research-copilot

# start a run
curl -s -X POST localhost:8000/runs -H 'x-api-key: secret' -H 'content-type: application/json' \
  -d '{"question":"gloss-free sign language production","year_from":2021,"max_papers":12}'
# -> {"id":"3f9c...","status":"queued","links":{...}}

# watch the agents work (Server-Sent Events; add -N for no buffering)
curl -N -H 'x-api-key: secret' localhost:8000/runs/3f9c.../events

# summary, then the report
curl -H 'x-api-key: secret' localhost:8000/runs/3f9c...
curl -H 'x-api-key: secret' localhost:8000/runs/3f9c.../report.md
```
Body fields: `question` (10-500 chars), `year_from`, `year_to`, `max_papers` (1-50),
`until` (`retrieve|read|write|critic`, default `critic`), `fulltext` (default true).

Operational behaviour: at most 2 runs execute concurrently (others queue; 429 when 10 are pending),
the 50 most recent runs are kept in memory (all are also saved under `$COPILOT_TRACES_DIR/runs/<id>/`),
and a dropped SSE connection can resume with the `Last-Event-ID` header.

## Event format
Each SSE `trace` event is JSON: `{"type":"trace","agent","kind","message","ts","data"}`. `kind` is one of
`thinking | plan | search | papers | read | notes | evidence | draft | check | decision | warn | info`;
`data` holds the details (e.g. `plan` -> `queries`; `papers` -> the ranked shortlist with scores and sources;
`check` -> `verdict`, `claim`, `reason`; `decision` -> `action`, `reason`). Events are pushed as they happen,
including from parallel Readers.
