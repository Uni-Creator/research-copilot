# Agent Navigation & Operating Instructions

Persistent instructions for coding agents working within `research-copilot`.

## Core Navigation Rules

1. **Consult documentation first**: Before exploring the repository, read [docs/codebase/README.md](docs/codebase/README.md) and the relevant architecture or dependency documentation.
2. **Never scan the entire repository**: Do not read the entire codebase by default or dump directories into context.
3. **Start from task-relevant entry points**: Begin investigation from the specific files, symbols, interfaces, and tests directly implicated in the user's request.
4. **Inspect dependency impact**: Use [docs/codebase/dependency-graph.md](docs/codebase/dependency-graph.md) and [docs/codebase/change-impact.md](docs/codebase/change-impact.md) to identify direct dependencies and reverse dependencies before altering public interfaces or shared contracts.
5. **Inspect code on-demand only**: Open source code only when necessary to verify concrete implementation details, resolve ambiguity, identify side effects, or modify behavior.
6. **Bound exploration outward**: Follow imports, function calls, API contracts, and data flows outward only as far as required to solve the task.
7. **Do not reread unchanged files**: Avoid rereading unchanged, unrelated files merely to reconstruct peripheral project context.
8. **Use targeted search**: Rely on targeted grep/AST searches, symbol lookups, Git diffs, and targeted test executions instead of mass file dumps.
9. **Documentation is a navigation aid**: Treat persistent documentation as an index and map, not as a replacement for executable source code.
10. **Source code is authoritative**: If existing documentation contradicts active source code, trust the source code and correct the documentation.
11. **Keep documentation synchronized**: After structural changes, run `python scripts/codebase_map.py --incremental` or `make map` to refresh dependency intelligence and update affected documentation.
12. **Keep persistent instructions concise**: Store high-level navigation rules here; keep in-depth technical references and architecture details modularized under `docs/codebase/`.

## Common Workflows

- **Run unit & integration tests**: `.venv/bin/pytest -q`
- **Run targeted test suite**: `.venv/bin/pytest tests/test_<module>.py -q`
- **Lint & style check**: `.venv/bin/ruff check .`
- **Incremental impact check**: `python scripts/codebase_map.py --incremental`
- **Full documentation & link validation**: `python scripts/codebase_map.py --check`
