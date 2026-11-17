"""A small in-memory sliding-window rate limiter for ``POST /analyses``.

Deliberately simple: one process, one lock, a deque of timestamps per client key.
On the Hugging Face Space there is a single Uvicorn worker, which is exactly the
deployment this protects.  It is not a security boundary, only abuse protection.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class SlidingWindowLimiter:
    """Allow at most ``limit`` calls per ``window`` seconds per key."""

    def __init__(self, limit: int, window_seconds: float = 60.0) -> None:
        self.limit = max(1, limit)
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> tuple[bool, int]:
        """Return (allowed, retry_after_seconds)."""
        now = time.monotonic()
        with self._lock:
            bucket = self._hits[key]
            while bucket and now - bucket[0] > self.window:
                bucket.popleft()
            if len(bucket) >= self.limit:
                retry = int(self.window - (now - bucket[0])) + 1
                return False, retry
            bucket.append(now)
            return True, 0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()
