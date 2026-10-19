
"""HTTP plumbing: shared rate limiting, retry with backoff, and disk caching."""
from __future__ import annotations

import email.utils
import hashlib
import json
import logging
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

MAX_BACKOFF_SECONDS = 60.0


class RateLimiter:
    """Thread-safe minimum request interval with shared cooldown support."""

    def __init__(self, min_interval: float) -> None:
        if min_interval < 0:
            raise ValueError("min_interval must be non-negative")

        self._min = min_interval
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        """Wait until this request may start."""
        with self._lock:
            delay = self._next - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._next = time.monotonic() + self._min

    def defer(self, seconds: float) -> None:
        """Prevent all users of this limiter from starting during a cooldown."""
        with self._lock:
            self._next = max(
                self._next,
                time.monotonic() + max(0.0, seconds),
            )


class DiskCache:
    """JSON response cache keyed by namespace and request parameters."""

    def __init__(
        self,
        root: Path,
        ttl_seconds: int = 7 * 86400,
        enabled: bool = True,
    ) -> None:
        self.root = Path(root)
        self.ttl = ttl_seconds
        self.enabled = enabled

    def _path(self, namespace: str, key: dict[str, Any]) -> Path:
        encoded = json.dumps(
            key, sort_keys=True, separators=(",", ":"), default=str
        )
        digest = hashlib.sha256(encoded.encode()).hexdigest()[:32]
        return self.root / namespace / f"{digest}.json"

    def get(self, namespace: str, key: dict[str, Any]) -> Any | None:
        if not self.enabled:
            return None

        path = self._path(namespace, key)
        try:
            if time.time() - path.stat().st_mtime > self.ttl:
                return None
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def set(self, namespace: str, key: dict[str, Any], value: Any) -> None:
        if not self.enabled:
            return

        path = self._path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)

        # A per-thread temporary file avoids concurrent writers sharing
        # the same temporary path. replace() publishes a complete JSON file.
        temp = path.with_name(
            f".{path.name}.{threading.get_ident()}.tmp"
        )
        try:
            temp.write_text(
                json.dumps(value, ensure_ascii=False),
                encoding="utf-8",
            )
            temp.replace(path)
        finally:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass


def _retry_after_seconds(resp: httpx.Response | None) -> float | None:
    """Parse Retry-After as seconds or an HTTP date."""
    if resp is None:
        return None

    value = resp.headers.get("Retry-After")
    if not value:
        return None

    try:
        return max(0.0, float(value))
    except ValueError:
        pass

    try:
        retry_at = email.utils.parsedate_to_datetime(value)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(
            0.0,
            (retry_at - datetime.now(timezone.utc)).total_seconds(),
        )
    except (TypeError, ValueError, OverflowError):
        return None


def _retry_delay(
    resp: httpx.Response | None,
    attempt: int,
    backoff: float,
) -> float:
    """Return a delay with positive jitter; never undercut Retry-After."""
    retry_after = _retry_after_seconds(resp)

    if retry_after is not None:
        base = retry_after
    else:
        base = min(backoff**attempt, MAX_BACKOFF_SECONDS)

    # Positive jitter spreads retries without violating Retry-After.
    jitter = random.uniform(0.0, min(1.0, max(0.1, base * 0.1)))
    return base + jitter


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
    """GET with bounded retries for 429, 5xx, and transport errors.

    A shared limiter propagates 429 cooldowns to other requests using
    the same limiter. Non-retryable HTTP errors fail immediately.
    """
    if retries < 0:
        raise ValueError("retries must be non-negative")
    if backoff <= 1:
        raise ValueError("backoff must be greater than 1")

    last_exc: Exception | None = None

    for attempt in range(retries + 1):
        if limiter is not None:
            limiter.wait()

        resp: httpx.Response | None = None

        try:
            resp = client.get(url, params=params, headers=headers)

            if resp.status_code == 429 or resp.status_code >= 500:
                last_exc = httpx.HTTPStatusError(
                    f"{resp.status_code} from {url}",
                    request=resp.request,
                    response=resp,
                )
            else:
                # Other 4xx responses are not retried.
                resp.raise_for_status()
                return resp

        except httpx.TransportError as exc:
            last_exc = exc

        if attempt >= retries:
            break

        delay = _retry_delay(resp, attempt, backoff)
        status = resp.status_code if resp is not None else "transport error"

        logger.warning(
            "HTTP retry %d/%d: status=%s url=%s delay=%.2fs",
            attempt + 1,
            retries,
            status,
            url,
            delay,
        )

        if resp is not None and resp.status_code == 429 and limiter is not None:
            # Share the server's cooldown with all callers using this limiter.
            limiter.defer(delay)
        else:
            time.sleep(delay)

    assert last_exc is not None
    raise last_exc
