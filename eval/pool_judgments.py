"""Pool candidates from saved run artifacts to create a human relevance-labeling file.

Example:
  python -m eval.pool_judgments --artifacts-dir eval/results/slp-smoke/artifacts \
      --topics eval/datasets/slp_v2.json --out eval/annotation_pool.json

This script never assigns relevance grades automatically.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _normal_title(value: Any) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).split())


def _normal_id(value: Any) -> str:
    text = str(value or "").lower().strip()
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = re.sub(r"^https?://arxiv\.org/(abs|pdf)/", "", text)
    text = re.sub(r"^arxiv:", "", text)
    text = re.sub(r"\.pdf$", "", text)
    text = re.sub(r"v\d+$", "", text)
    return text


def _identity_keys(paper: dict[str, Any]) -> list[tuple[str, str]]:
    """Return all usable identity keys so cross-source records can be merged."""
    keys: list[tuple[str, str]] = []
    arxiv = _normal_id(paper.get("arxiv_id") or paper.get("arxiv"))
    doi = _normal_id(paper.get("doi"))
    title = _normal_title(paper.get("title"))
    if arxiv:
        keys.append(("arxiv", arxiv))
    if doi:
        keys.append(("doi", doi))
    if title:
        keys.append(("title", title))
    return keys


def build_annotation_pool(artifacts_dir: Path, dataset: dict[str, Any]) -> dict[str, Any]:
    artifacts = sorted(artifacts_dir.glob("*.json"))
    pools: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    identity_indexes: dict[str, dict[tuple[str, str], tuple[str, str]]] = {}
    questions = {topic["id"]: topic for topic in dataset.get("topics", [])}

    for path in artifacts:
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        topic_info = artifact.get("topic", {})
        topic_id = topic_info.get("id")
        variant = artifact.get("variant")
        state = artifact.get("state", {})
        if not topic_id or not variant:
            continue
        paper_map = state.get("papers", {})
        shortlist = state.get("shortlist", []) or []
        shortlist_rank = {str(pid): rank for rank, pid in enumerate(shortlist, 1)}
        ranked = sorted(
            paper_map.items(),
            key=lambda item: (-float(item[1].get("score", 0) or 0),
                              -int(item[1].get("citation_count", 0) or 0),
                              str(item[1].get("title", ""))),
        )
        pool_rank = {str(pid): rank for rank, (pid, _) in enumerate(ranked, 1)}
        topic_pool = pools.setdefault(topic_id, {})
        identity_index = identity_indexes.setdefault(topic_id, {})
        for pid, raw in paper_map.items():
            paper = dict(raw) if isinstance(raw, dict) else {}
            keys = _identity_keys(paper)
            if not keys:
                continue
            existing = next((identity_index[key] for key in keys if key in identity_index), None)
            key = existing or keys[0]
            row = topic_pool.setdefault(key, {
                "title": paper.get("title", ""),
                "arxiv_id": paper.get("arxiv_id"),
                "doi": paper.get("doi"),
                "year": paper.get("year"),
                "url": paper.get("url"),
                "abstract": paper.get("abstract", ""),
                "sources": paper.get("sources", []),
                "grade": None,
                "aspects": [],
                "rationale": "",
                "pool_rank_by_variant": {},
                "shortlist_rank_by_variant": {},
            })
            for field in ("arxiv_id", "doi", "year", "url", "abstract"):
                if not row.get(field) and paper.get(field):
                    row[field] = paper[field]
            row["sources"] = sorted(set((row.get("sources") or []) + (paper.get("sources") or [])))
            for identity_key in keys:
                identity_index.setdefault(identity_key, key)
            row["pool_rank_by_variant"][variant] = pool_rank.get(str(pid))
            row["shortlist_rank_by_variant"][variant] = shortlist_rank.get(str(pid))

    topics_out = []
    for topic_id, topic_pool in pools.items():
        topic = questions.get(topic_id, {})
        candidates = sorted(topic_pool.values(), key=lambda p: min(
            [rank for rank in p["pool_rank_by_variant"].values() if rank is not None] or [10**9]
        ))
        topics_out.append({
            "id": topic_id,
            "question": topic.get("question", ""),
            "topic_aspects": topic.get("topic_aspects", []),
            "candidates": candidates,
        })
    return {
        "schema_version": 1,
        "dataset_name": dataset.get("name", "Research Copilot annotation pool"),
        "source_artifacts_dir": str(artifacts_dir),
        "annotation_instructions": {
            "grade": {"0": "not relevant", "1": "partially relevant/background", "2": "directly relevant"},
            "aspects": "Select topic_aspect IDs that this paper provides useful evidence for.",
            "rationale": "Briefly explain the label based on title/abstract/full text. Do not use must-cite membership as the relevance label.",
            "blinding": "For less biased labeling, hide pool rank and variant fields from annotators if practical.",
        },
        "topics": topics_out,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--topics", default="eval/datasets/slp_v2.json")
    parser.add_argument("--out", default="eval/annotation_pool.json")
    args = parser.parse_args(argv)
    artifacts_dir = Path(args.artifacts_dir)
    dataset = json.loads(Path(args.topics).read_text(encoding="utf-8"))
    pool = build_annotation_pool(artifacts_dir, dataset)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pool, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    count = sum(len(topic["candidates"]) for topic in pool["topics"])
    print(f"Wrote {count} candidate records across {len(pool['topics'])} topic(s) to {out}")
    print("No relevance grades were assigned; annotate the pool before calculating judged relevance metrics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
