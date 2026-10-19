"""Small text helpers shared by tools and agents."""

from __future__ import annotations

import hashlib
import re

_WORD = re.compile(r"\w+", re.UNICODE)

STOPWORDS = frozenset(
    """a an the of for in on to and or with via using based from by is are as at into
    its their we our how what which that this these those towards toward approach
    approaches method methods recent novel new study""".split()
)


def normalize_title(title: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace (used for dedupe)."""
    return " ".join(_WORD.findall(title.lower()))


def tokenize(text: str) -> list[str]:
    """Significant lowercase tokens (stopwords and 1-char tokens removed)."""
    return [t for t in _WORD.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


def short_hash(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]
