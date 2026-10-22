#!/usr/bin/env python3
"""scripts/codebase_map.py - Persistent Codebase Map & Dependency Intelligence System.

AST-based dependency analyzer, cycle detector, reverse-dependency tracer,
git-aware impact analyzer, and documentation synchronization validator.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

# Codebase scan roots
SOURCE_DIRS = ["copilot", "eval", "ui"]
TEST_DIRS = ["tests"]
ALL_DIRS = SOURCE_DIRS + TEST_DIRS


@dataclass
class ModuleInfo:
    file_path: str  # relative to repo root, e.g. "copilot/graph/state.py"
    module_name: str  # dotted, e.g. "copilot.graph.state"
    docstring: str = ""
    direct_internal_imports: list[str] = field(default_factory=list)
    direct_external_imports: list[str] = field(default_factory=list)
    reverse_dependencies: list[str] = field(default_factory=list)
    public_classes: list[str] = field(default_factory=list)
    public_functions: list[str] = field(default_factory=list)
    associated_tests: list[str] = field(default_factory=list)


def file_to_module_name(file_path: Path) -> str:
    """Convert relative Path to dotted module name."""
    parts = list(file_path.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def resolve_relative_import(current_module: str, current_is_init: bool, level: int, module: str | None) -> str:
    """Resolve Python relative import using level and target module."""
    parts = current_module.split(".")
    if not current_is_init and parts:
        parts.pop()
    # Each level > 1 pops an additional parent
    for _ in range(level - 1):
        if parts:
            parts.pop()
    base = ".".join(parts)
    if module:
        return f"{base}.{module}" if base else module
    return base


def parse_module(path: Path) -> ModuleInfo:
    rel_path = path.relative_to(REPO_ROOT).as_posix()
    is_init = path.name == "__init__.py"
    mod_name = file_to_module_name(path.relative_to(REPO_ROOT))

    try:
        content = path.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(path))
    except Exception as exc:
        print(f"Warning: Failed to parse {rel_path}: {exc}", file=sys.stderr)
        return ModuleInfo(file_path=rel_path, module_name=mod_name)

    docstring = ast.get_docstring(tree) or ""
    first_doc_line = docstring.strip().split("\n")[0] if docstring else ""

    direct_internal: set[str] = set()
    direct_external: set[str] = set()
    classes: list[str] = []
    functions: list[str] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if not node.name.startswith("_"):
                classes.append(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                functions.append(node.name)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                root = name.split(".")[0]
                if root in ALL_DIRS:
                    direct_internal.add(name)
                else:
                    direct_external.add(name)
        elif isinstance(node, ast.ImportFrom):
            target_mod = node.module or ""
            if node.level > 0:
                resolved = resolve_relative_import(mod_name, is_init, node.level, node.module)
                root = resolved.split(".")[0]
                if root in ALL_DIRS:
                    direct_internal.add(resolved)
                else:
                    direct_external.add(resolved)
            else:
                root = target_mod.split(".")[0]
                if root in ALL_DIRS:
                    direct_internal.add(target_mod)
                else:
                    direct_external.add(target_mod)

    return ModuleInfo(
        file_path=rel_path,
        module_name=mod_name,
        docstring=first_doc_line,
        direct_internal_imports=sorted(direct_internal),
        direct_external_imports=sorted(direct_external),
        public_classes=classes,
        public_functions=functions,
    )


class CodebaseAnalyzer:
    def __init__(self, root: Path = REPO_ROOT) -> None:
        self.root = root
        self.modules: dict[str, ModuleInfo] = {}  # mod_name -> ModuleInfo
        self.path_to_mod: dict[str, str] = {}  # rel_path -> mod_name
        self._scan()

    def _scan(self) -> None:
        for d in ALL_DIRS:
            dir_path = self.root / d
            if not dir_path.is_dir():
                continue
            for py_path in sorted(dir_path.rglob("*.py")):
                if py_path.name.startswith("."):
                    continue
                info = parse_module(py_path)
                self.modules[info.module_name] = info
                self.path_to_mod[info.file_path] = info.module_name

        self._compute_reverse_dependencies()
        self._compute_test_mappings()

    def _match_module(self, imported_name: str) -> str | None:
        """Match an imported name (which might be a symbol or submodule) to a canonical known module."""
        if imported_name in self.modules:
            return imported_name
        # Check prefixes
        candidates = [m for m in self.modules if imported_name == m or imported_name.startswith(f"{m}.")]
        if candidates:
            return max(candidates, key=len)
        return None

    def _compute_reverse_dependencies(self) -> None:
        rev_deps: dict[str, set[str]] = defaultdict(set)
        for mod_name, info in self.modules.items():
            for imp in info.direct_internal_imports:
                matched = self._match_module(imp)
                if matched and matched != mod_name:
                    rev_deps[matched].add(mod_name)
        for mod_name, callers in rev_deps.items():
            if mod_name in self.modules:
                self.modules[mod_name].reverse_dependencies = sorted(callers)

    def _compute_test_mappings(self) -> None:
        for info in self.modules.values():
            if not info.file_path.startswith("tests/"):
                continue
            test_file = info.file_path
            for imp in info.direct_internal_imports:
                matched = self._match_module(imp)
                if matched and not matched.startswith("tests") and matched in self.modules:
                    if test_file not in self.modules[matched].associated_tests:
                        self.modules[matched].associated_tests.append(test_file)
                        self.modules[matched].associated_tests.sort()

    def detect_cycles(self) -> list[list[str]]:
        """Return list of dependency cycles within the scanned modules."""
        visited: dict[str, int] = {}  # 0=unvisited, 1=visiting, 2=visited
        cycles: list[list[str]] = []

        def dfs(node: str, path: list[str]) -> None:
            visited[node] = 1
            info = self.modules.get(node)
            if info:
                for imp in info.direct_internal_imports:
                    tgt = self._match_module(imp)
                    if not tgt or tgt == node:
                        continue
                    if visited.get(tgt) == 1:
                        cycle_start = path.index(tgt)
                        cycles.append(path[cycle_start:] + [tgt])
                    elif visited.get(tgt, 0) == 0:
                        dfs(tgt, path + [tgt])
            visited[node] = 2

        for mod in sorted(self.modules):
            if visited.get(mod, 0) == 0:
                dfs(mod, [mod])
        return cycles

    def compute_blast_radius(self, modified_files: list[str]) -> dict[str, Any]:
        """Given a list of modified files, compute downstream affected modules and tests."""
        affected_mods: set[str] = set()
        affected_tests: set[str] = set()

        dirty_mods: set[str] = set()
        for f in modified_files:
            norm_f = Path(f).as_posix()
            if norm_f in self.path_to_mod:
                dirty_mods.add(self.path_to_mod[norm_f])

        # BFS downstream traverse via reverse_dependencies
        queue = deque(dirty_mods)
        visited: set[str] = set(dirty_mods)

        while queue:
            curr = queue.popleft()
            affected_mods.add(curr)
            info = self.modules.get(curr)
            if info:
                for t in info.associated_tests:
                    affected_tests.add(t)
                for rev in info.reverse_dependencies:
                    if rev not in visited:
                        visited.add(rev)
                        queue.append(rev)

        return {
            "directly_modified_modules": sorted(dirty_mods),
            "downstream_affected_modules": sorted(affected_mods - dirty_mods),
            "recommended_tests": sorted(affected_tests),
        }

    def generate_inventory_json(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "total_modules": len(self.modules),
            "cycles": self.detect_cycles(),
            "modules": {m: asdict(info) for m, info in sorted(self.modules.items())},
        }

    def generate_mermaid_graph(self, top_subsystems_only: bool = False) -> str:
        """Generate a Mermaid dependency diagram."""
        lines = ["```mermaid", "graph TD"]
        # Subgraphs by package
        packages: dict[str, list[str]] = defaultdict(list)
        for m in self.modules.keys():
            if m.startswith("tests"):
                continue
            pkg = m.split(".")[0]
            if len(m.split(".")) > 1 and pkg == "copilot":
                sub = m.split(".")[1]
                if sub in ["agents", "graph", "tools"]:
                    packages[f"copilot.{sub}"].append(m)
                else:
                    packages["copilot.core"].append(m)
            else:
                packages[pkg].append(m)

        for pkg, mods in sorted(packages.items()):
            lines.append(f"    subgraph {pkg.replace('.', '_')}[\"{pkg}\"]")
            for m in sorted(mods):
                short = m.split(".")[-1]
                lines.append(f"        {m.replace('.', '_')}[\"{short}\"]")
            lines.append("    end")

        # Edges
        added_edges = set()
        for m, info in self.modules.items():
            if m.startswith("tests"):
                continue
            src = m.replace(".", "_")
            for imp in info.direct_internal_imports:
                tgt_mod = self._match_module(imp)
                if tgt_mod and tgt_mod != m and not tgt_mod.startswith("tests"):
                    tgt = tgt_mod.replace(".", "_")
                    edge = (src, tgt)
                    if edge not in added_edges:
                        added_edges.add(edge)
                        lines.append(f"    {src} --> {tgt}")

        lines.append("```")
        return "\n".join(lines)


def get_git_status(root: Path) -> dict[str, list[str]]:
    """Inspect git status for modified, untracked, or staged files."""
    res = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True)
    added, modified, deleted = [], [], []
    for line in res.stdout.splitlines():
        if not line.strip():
            continue
        status = line[:2]
        filepath = line[3:].strip()
        # Handle rename syntax: "R  old -> new"
        if "->" in filepath:
            filepath = filepath.split("->")[-1].strip()
        if "D" in status:
            deleted.append(filepath)
        elif "?" in status or "A" in status:
            added.append(filepath)
        else:
            modified.append(filepath)
    return {"added": added, "modified": modified, "deleted": deleted}


def validate_documentation_links(root: Path) -> list[str]:
    """Verify that all markdown file links referenced in docs/codebase exist."""
    issues = []
    docs_dir = root / "docs" / "codebase"
    if not docs_dir.exists():
        return [f"Missing documentation directory: {docs_dir}"]

    link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    md_files = list(docs_dir.glob("*.md"))
    agents_md = root / "AGENTS.md"
    if agents_md.exists():
        md_files.append(agents_md)

    for md_file in md_files:
        content = md_file.read_text(encoding="utf-8")
        for match in link_pattern.finditer(content):
            link = match.group(2)
            if link.startswith("http://") or link.startswith("https://") or link.startswith("#"):
                continue
            if link.startswith("file://"):
                path_part = link[7:].split("#")[0]
                target_path = Path(path_part)
            else:
                path_part = link.split("#")[0]
                target_path = (md_file.parent / path_part).resolve()

            if not target_path.exists():
                issues.append(f"{md_file.name}: Broken link '{link}' -> resolved to missing path {target_path}")

    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Codebase Map & Dependency Intelligence System")
    parser.add_argument("--check", action="store_true", help="Validate documentation links, verify DAG acyclicity")
    parser.add_argument("--incremental", action="store_true", help="Check git status and report downstream impact")
    parser.add_argument("--rebuild", action="store_true", help="Full rebuild of dependency inventory JSON")
    parser.add_argument("--json", action="store_true", help="Output dependency map as JSON to stdout")
    parser.add_argument("--mermaid", action="store_true", help="Output Mermaid dependency graph")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "codebase" / "dependency-inventory.json",
                        help="Path to output dependency inventory JSON")

    args = parser.parse_args(argv)
    analyzer = CodebaseAnalyzer(REPO_ROOT)

    cycles = analyzer.detect_cycles()
    if cycles:
        print(f"Warning: {len(cycles)} circular dependencies detected!", file=sys.stderr)
        for c in cycles:
            print(f"  Cycle: {' -> '.join(c)}", file=sys.stderr)
    else:
        print("Dependency graph verified: 0 circular dependencies (acyclic DAG).")

    if args.json:
        print(json.dumps(analyzer.generate_inventory_json(), indent=2))
        return 0

    if args.mermaid:
        print(analyzer.generate_mermaid_graph())
        return 0

    if args.incremental:
        git_changes = get_git_status(REPO_ROOT)
        all_dirty = git_changes["added"] + git_changes["modified"] + git_changes["deleted"]
        print("\n--- Incremental Git Change Detection ---")
        print(f"Added files: {git_changes['added']}")
        print(f"Modified files: {git_changes['modified']}")
        print(f"Deleted files: {git_changes['deleted']}")

        impact = analyzer.compute_blast_radius(all_dirty)
        print("\n--- Blast Radius & Change Impact ---")
        print(f"Directly modified modules: {impact['directly_modified_modules']}")
        print(f"Downstream affected modules: {impact['downstream_affected_modules']}")
        print(f"Targeted tests to run: {impact['recommended_tests']}")
        if impact['recommended_tests']:
            test_cmd = f"pytest {' '.join(impact['recommended_tests'])}"
            print(f"Command: {test_cmd}")
        return 0

    if args.rebuild or args.check:
        inventory = analyzer.generate_inventory_json()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
        print(f"Dependency inventory written to {args.out.relative_to(REPO_ROOT)}")

    if args.check:
        link_issues = validate_documentation_links(REPO_ROOT)
        if link_issues:
            print("\nDocumentation Link Issues Found:", file=sys.stderr)
            for iss in link_issues:
                print(f"  - {iss}", file=sys.stderr)
            return 1
        else:
            print("All documentation links verified successfully.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
