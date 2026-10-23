"""Validate the offline structure of a Research Copilot evaluation dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def validate_dataset(data: dict[str, Any]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    topics = data.get("topics")
    if not isinstance(topics, list) or not topics:
        return ["top-level 'topics' must be a non-empty list"], warnings

    seen_topic_ids: set[str] = set()
    for index, topic in enumerate(topics):
        where = f"topics[{index}]"
        if not isinstance(topic, dict):
            errors.append(f"{where} must be an object")
            continue
        topic_id = topic.get("id")
        if not isinstance(topic_id, str) or not topic_id.strip():
            errors.append(f"{where}.id must be a non-empty string")
        elif topic_id in seen_topic_ids:
            errors.append(f"duplicate topic id: {topic_id}")
        else:
            seen_topic_ids.add(topic_id)
        if not isinstance(topic.get("question"), str) or not topic["question"].strip():
            errors.append(f"{where}.question must be a non-empty string")

        aspects = topic.get("topic_aspects", [])
        if not isinstance(aspects, list):
            errors.append(f"{where}.topic_aspects must be a list")
            aspects = []
        aspect_ids: set[str] = set()
        for j, aspect in enumerate(aspects):
            if not isinstance(aspect, dict) or not isinstance(aspect.get("id"), str) or not aspect["id"].strip():
                errors.append(f"{where}.topic_aspects[{j}] must contain a non-empty id")
                continue
            if aspect["id"] in aspect_ids:
                errors.append(f"{where} has duplicate topic aspect id {aspect['id']}")
            aspect_ids.add(aspect["id"])

        must_cite = topic.get("must_cite", [])
        if not isinstance(must_cite, list):
            errors.append(f"{where}.must_cite must be a list")
            must_cite = []
        for j, entry in enumerate(must_cite):
            if not isinstance(entry, dict) or not str(entry.get("title", "")).strip():
                errors.append(f"{where}.must_cite[{j}] must contain a title")

        judgments = topic.get("relevance_judgments", [])
        if not isinstance(judgments, list):
            errors.append(f"{where}.relevance_judgments must be a list")
            continue
        if not judgments:
            warnings.append(f"{where} has no relevance judgments; judged relevance metrics will be null")
        seen_judgments: set[tuple[str, str]] = set()
        for j, entry in enumerate(judgments):
            entry_where = f"{where}.relevance_judgments[{j}]"
            if not isinstance(entry, dict) or not str(entry.get("title", "")).strip():
                errors.append(f"{entry_where} must contain a title")
                continue
            try:
                grade = int(entry.get("grade"))
            except (TypeError, ValueError):
                errors.append(f"{entry_where}.grade must be 0, 1, or 2")
                continue
            if grade not in (0, 1, 2):
                errors.append(f"{entry_where}.grade must be 0, 1, or 2")
            tagged = entry.get("aspects", [])
            if not isinstance(tagged, list):
                errors.append(f"{entry_where}.aspects must be a list")
                tagged = []
            unknown = sorted(set(tagged) - aspect_ids)
            if unknown:
                errors.append(f"{entry_where}.aspects contains unknown IDs: {', '.join(unknown)}")
            identity = str(entry.get("arxiv_id") or entry.get("doi") or entry.get("title")).lower().strip()
            key = ("id", identity) if entry.get("arxiv_id") or entry.get("doi") else ("title", identity)
            if key in seen_judgments:
                errors.append(f"{entry_where} duplicates a prior relevance judgment ({identity})")
            seen_judgments.add(key)
    return errors, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topics", default="eval/datasets/slp_v2.json")
    args = parser.parse_args(argv)
    path = Path(args.topics)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: could not read {path}: {exc}")
        return 2
    errors, warnings = validate_dataset(data)
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        print(f"Dataset invalid: {len(errors)} error(s), {len(warnings)} warning(s)")
        return 1
    print(f"Dataset valid: {len(data['topics'])} topic(s), {len(warnings)} warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
