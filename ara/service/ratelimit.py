"""Rate limiting: per-key token bucket (in-process; Redis-backed in production
compose via the same interface)."""
from __future__ import annotations

import threading
import time

from ara.core.errors import RateLimitError


class TokenBucketLimiter:
    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.capacity = float(per_minute)
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last_refill)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * (self.per_minute / 60.0))
            if tokens < 1.0:
                raise RateLimitError(f"rate limit exceeded ({self.per_minute}/min)")
            self._buckets[key] = (tokens - 1.0, now)
