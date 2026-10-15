"""Deduplication of paper records coming from different sources."""
from __future__ import annotations

from copilot.graph.state import Paper


class PaperIndex:
    """Incrementally merges records that refer to the same paper.

    Two records match if they share an arXiv id, a DOI, or a normalised title.
    ``add`` returns a stable integer handle (position) so callers can attach
    per-record data (e.g. rank-fusion scores) before final ids are known.
    """

    def __init__(self) -> None:
        self._papers: list[Paper] = []
        self._by_key: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._papers)

    @property
    def papers(self) -> list[Paper]:
        return self._papers

    def add(self, paper: Paper) -> int:
        idx = next((self._by_key[k] for k in paper.keys() if k in self._by_key), None)
        if idx is None:
            idx = len(self._papers)
            self._papers.append(paper)
        else:
            self._papers[idx] = self._papers[idx].merged_with(paper)
        for k in self._papers[idx].keys():   # merging may reveal new identity keys
            self._by_key.setdefault(k, idx)
        return idx
