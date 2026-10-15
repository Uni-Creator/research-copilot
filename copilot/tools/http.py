"""HTTP plumbing: polite rate limiting, retry with backoff, and an on-disk response cache."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import httpx


class RateLimiter:
    """Enforces a minimum interval between calls (thread-safe)."""

    def __init__(self, min_interval: float) -> None:
        self._min = min_interval
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            delay = self._next - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._next = time.monotonic() + self._min


class DiskCache:
    """Tiny JSON file cache keyed by (namespace, params). Saves API calls and money."""

    def __init__(self, root: Path, ttl_seconds: int = 7 * 86400, enabled: bool = True) -> None:
        self.root = Path(root)
        self.ttl = ttl_seconds
        self.enabled = enabled

    def _path(self, namespace: str, key: dict[str, Any]) -> Path:
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:32]
        return self.root / namespace / f"{digest}.json"

    def get(self, namespace: str, key: dict[str, Any]) -> Any | None:
        if not self.enabled:
            return None
        p = self._path(namespace, key)
        if not p.exists() or time.time() - p.stat().st_mtime > self.ttl:
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def set(self, namespace: str, key: dict[str, Any], value: Any) -> None:
        if not self.enabled:
            return
        p = self._path(namespace, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value), encoding="utf-8")


def _retry_delay(resp: httpx.Response | None, attempt: int, backoff: float) -> float:
    if resp is not None:
        try:
            return min(float(resp.headers["Retry-After"]), 60.0)
        except (KeyError, ValueError):
            pass
    return backoff**attempt


def get_with_retry(
    client: httpx.Client,
    url: str,
    *,
    params: dict[str, Any],
    headers: dict[str, str] | None = None,
    limiter: RateLimiter | None = None,
    retries: int = 4,
    backoff: float = 2.0,
) -> httpx.Response:
    """GET with exponential backoff on 429 / 5xx / network errors."""
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        resp: httpx.Response | None = None
        if limiter:
            limiter.wait()
        try:
            resp = client.get(url, params=params, headers=headers)
            if resp.status_code != 429 and resp.status_code < 500:
                resp.raise_for_status()
                return resp
            last_exc = httpx.HTTPStatusError(
                f"{resp.status_code} from {url}", request=resp.request, response=resp
            )
        except httpx.TransportError as exc:
            last_exc = exc
        if attempt < retries:
            time.sleep(_retry_delay(resp, attempt, backoff))
    assert last_exc is not None
    raise last_exc
