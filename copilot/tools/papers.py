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
        matching_indices = list(dict.fromkeys(self._by_key[k] for k in paper.keys() if k in self._by_key))
        if not matching_indices:
            idx = len(self._papers)
            self._papers.append(paper)
        else:
            idx = matching_indices[0]
            self._papers[idx] = self._papers[idx].merged_with(paper)
            for other_idx in matching_indices[1:]:
                self._papers[idx] = self._papers[idx].merged_with(self._papers[other_idx])
                for k, target in list(self._by_key.items()):
                    if target == other_idx:
                        self._by_key[k] = idx
        for k in self._papers[idx].keys():  # merging may reveal new identity keys
            self._by_key[k] = idx
        return idx
