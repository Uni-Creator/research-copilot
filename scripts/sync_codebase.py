#!/usr/bin/env python3
"""scripts/sync_codebase.py - All-in-one post-change automation script.

Executes after code modifications to:
1. Detect modified files and calculate blast radius via reverse dependencies.
2. Run relevant targeted tests (or full suite) via pytest.
3. Run style/lint verification via ruff.
4. Update dependency graph, Mermaid diagram, and reverse dependency tables in docs/codebase/.
5. Update file index if files/symbols were added/modified.
6. Rebuild dependency-inventory.json.
7. Validate documentation links and verify acyclic DAG structure (0 cycles).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.codebase_map import (  # noqa: E402
    CodebaseAnalyzer,
    get_git_status,
    validate_documentation_links,
)

DEP_GRAPH_PATH = REPO_ROOT / "docs" / "codebase" / "dependency-graph.md"
FILE_INDEX_PATH = REPO_ROOT / "docs" / "codebase" / "file-index.md"
INVENTORY_PATH = REPO_ROOT / "docs" / "codebase" / "dependency-inventory.json"


def run_command(cmd: list[str], desc: str) -> tuple[int, str]:
    """Run a command and return exit code and combined output."""
    print(f"▶ {desc}...")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    out = (proc.stdout + "\n" + proc.stderr).strip()
    if proc.returncode != 0:
        print(f"✖ Failed: {desc}\n{out}", file=sys.stderr)
    else:
        print(f"✔ Success: {desc}")
    return proc.returncode, out


def update_dependency_graph_doc(analyzer: CodebaseAnalyzer) -> bool:
    """Update Mermaid diagram and Reverse Dependencies table in dependency-graph.md."""
    if not DEP_GRAPH_PATH.exists():
        print(f"Warning: {DEP_GRAPH_PATH} not found", file=sys.stderr)
        return False

    content = DEP_GRAPH_PATH.read_text(encoding="utf-8")

    # 1. Generate updated Mermaid graph
    new_mermaid = analyzer.generate_mermaid_graph()

    # Replace existing Mermaid block
    mermaid_pattern = re.compile(r"```mermaid\ngraph TD[\s\S]*?```")
    if mermaid_pattern.search(content):
        content = mermaid_pattern.sub(new_mermaid, content, count=1)
    else:
        print("Note: Could not match Mermaid block in dependency-graph.md", file=sys.stderr)

    # 2. Generate updated Reverse Dependencies table
    table_lines = [
        "| Module | Fan-In (Reverse Dependencies) | Fan-Out | Reverse Dependencies (Importers) |",
        "|---|---|---|---|",
    ]

    # Sort modules by fan-in descending, then module name
    sorted_mods = sorted(
        analyzer.modules.items(),
        key=lambda item: (len(item[1].reverse_dependencies), item[0]),
        reverse=True,
    )

    for mod_name, info in sorted_mods:
        if mod_name.startswith("tests"):
            continue
        fan_in = len(info.reverse_dependencies)
        fan_out = len(info.direct_internal_imports)
        revs = ", ".join(info.reverse_dependencies) if info.reverse_dependencies else "(None - runtime entry point)"
        table_lines.append(f"| `{mod_name}` | {fan_in} | {fan_out} | {revs} |")

    new_table = "\n".join(table_lines)

    # Replace existing reverse dependencies table
    table_pattern = re.compile(
        r"\| Module \| Fan-In \(Reverse Dependencies\) \| Fan-Out \| Reverse Dependencies \(Importers\) \|[\s\S]*?(?=\n---|\Z)"
    )
    if table_pattern.search(content):
        content = table_pattern.sub(new_table, content, count=1)
    else:
        print("Note: Could not match Reverse Dependencies table in dependency-graph.md", file=sys.stderr)

    DEP_GRAPH_PATH.write_text(content, encoding="utf-8")
    print(f"✔ Updated {DEP_GRAPH_PATH.relative_to(REPO_ROOT)}")
    return True


def check_missing_files_in_index(analyzer: CodebaseAnalyzer) -> list[str]:
    """Check if any Python files in copilot/, eval/, ui/ are missing from file-index.md."""
    if not FILE_INDEX_PATH.exists():
        return []

    index_text = FILE_INDEX_PATH.read_text(encoding="utf-8")
    missing = []
    for mod_name, info in analyzer.modules.items():
        if mod_name.startswith("tests") or info.file_path.endswith("__init__.py"):
            continue
        if info.file_path not in index_text:
            missing.append(info.file_path)
    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sync codebase intelligence, run tests/lints, and update architectural documentation."
    )
    parser.add_argument("--full-test", action="store_true", help="Always run the full test suite instead of targeted tests")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running pytest")
    parser.add_argument("--skip-lint", action="store_true", help="Skip running ruff linter")
    args = parser.parse_args(argv)

    print("=================================================================")
    print("      RESEARCH-COPILOT: POST-CHANGE SYNC & VERIFICATION          ")
    print("=================================================================\n")

    # 1. AST Analysis
    analyzer = CodebaseAnalyzer(REPO_ROOT)
    cycles = analyzer.detect_cycles()
    if cycles:
        print(f"❌ ARCHITECTURAL VIOLATION: {len(cycles)} circular dependencies detected!", file=sys.stderr)
        for c in cycles:
            print(f"   Cycle: {' -> '.join(c)}", file=sys.stderr)
        return 1
    print("✔ Architectural graph verified: Strict acyclic DAG (0 cycles).")

    # 2. Git Change Detection & Blast Radius
    git_status = get_git_status(REPO_ROOT)
    dirty_files = git_status["added"] + git_status["modified"] + git_status["deleted"]
    impact = analyzer.compute_blast_radius(dirty_files)

    print(f"\n[Changes Detected]: {len(dirty_files)} file(s)")
    if dirty_files:
        print(f"  Directly modified modules: {impact['directly_modified_modules']}")
        print(f"  Downstream affected modules: {impact['downstream_affected_modules']}")
        print(f"  Targeted tests: {impact['recommended_tests']}")
    else:
        print("  Working directory clean or no python code modifications detected.")

    # 3. Run Linter
    if not args.skip_lint:
        print("\n--- [Step 1: Code Lint & Style Check] ---")
        code, _ = run_command([sys.executable, "-m", "ruff", "check", "."], "Ruff Lint Verification")
        if code != 0:
            return code

    # 4. Run Tests
    if not args.skip_tests:
        print("\n--- [Step 2: Targeted or Full Test Suite] ---")
        # Check if critical core contracts modified
        critical_mods = {"copilot.graph.state", "copilot.graph.workflow", "copilot.config"}
        touched_critical = any(m in critical_mods for m in impact["directly_modified_modules"])

        if args.full_test or touched_critical or not impact["recommended_tests"]:
            test_target = ["tests"]
            desc = "Full Test Suite (tests/)"
        else:
            test_target = impact["recommended_tests"]
            desc = f"Targeted Test Suite ({', '.join(test_target)})"

        code, _ = run_command([sys.executable, "-m", "pytest", "-q"] + test_target, desc)
        if code != 0:
            return code

    # 5. Documentation Updates
    print("\n--- [Step 3: Update Dependency Graph & Architecture Docs] ---")
    update_dependency_graph_doc(analyzer)

    # Rebuild inventory JSON
    inv = analyzer.generate_inventory_json()
    INVENTORY_PATH.write_text(json.dumps(inv, indent=2), encoding="utf-8")
    print(f"✔ Rebuilt {INVENTORY_PATH.relative_to(REPO_ROOT)}")

    # Check for missing files in file-index.md
    missing_in_index = check_missing_files_in_index(analyzer)
    if missing_in_index:
        print(f"ℹ Note: The following new files should be added to file-index.md: {missing_in_index}")

    # 6. Documentation Link Validation
    print("\n--- [Step 4: Documentation Link & DAG Integrity Check] ---")
    link_issues = validate_documentation_links(REPO_ROOT)
    if link_issues:
        print("❌ Broken documentation links detected:", file=sys.stderr)
        for iss in link_issues:
            print(f"   {iss}", file=sys.stderr)
        return 1
    print("✔ All documentation links verified successfully.")

    print("\n=================================================================")
    print("  ALL CHECKS PASSED: Codebase, tests, and documentation in sync! ")
    print("=================================================================\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
