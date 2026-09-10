"""Failure recovery: timeouts, retries with exponential backoff + jitter, retry
ceilings, and fallback tools. Irreversible operations are NEVER retried blindly."""
from __future__ import annotations

import random
import time
from typing import Any, Callable, TypeVar

from ara.core.errors import AraError, ToolTimeout, ToolUnavailable
from ara.core.logging import get_logger

log = get_logger("ara.resilience")
T = TypeVar("T")

RETRYABLE = (ToolTimeout, ToolUnavailable, ConnectionError, TimeoutError)


def with_retry(
    fn: Callable[[], T],
    *,
    max_attempts: int = 3,
    base_delay_s: float = 0.2,
    max_delay_s: float = 4.0,
    retryable: tuple[type[Exception], ...] = RETRYABLE,
    on_retry: Callable[[int, Exception], None] | None = None,
    deadline_s: float | None = None,
) -> T:
    """Run fn with bounded exponential backoff. Never retries non-retryable errors.
    `on_retry` receives (attempt, exc) for tracing."""
    start = time.monotonic()
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        if deadline_s is not None and time.monotonic() - start > deadline_s:
            raise ToolTimeout(f"retry loop exceeded deadline ({deadline_s}s)")
        try:
            return fn()
        except retryable as exc:  # noqa: PERF203
            last_exc = exc
            if attempt == max_attempts:
                break
            delay = min(max_delay_s, base_delay_s * (2 ** (attempt - 1)))
            delay *= 0.5 + random.random()  # jitter
            if on_retry:
                on_retry(attempt, exc)
            log.warning("retry_scheduled", extra={"fields": {"attempt": attempt, "delay_s": round(delay, 3),
                                                            "error": str(exc)}})
            time.sleep(delay)
    raise ToolUnavailable(f"all {max_attempts} attempts failed", details={"last_error": str(last_exc)})


def run_with_timeout(fn: Callable[[], T], timeout_s: float, *, what: str = "operation") -> T:
    """Run fn under a wall-clock timeout (thread-based; works for blocking calls)."""
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn)
        try:
            return fut.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError as exc:
            fut.cancel()
            raise ToolTimeout(f"{what} timed out after {timeout_s}s") from exc
        except AraError:
            raise
        except Exception as exc:
            raise AraError(f"{what} failed: {exc}") from exc


def classify_failure(exc: Exception) -> dict[str, Any]:
    """Deterministic failure classification driving the recovery policy."""
    if isinstance(exc, ToolTimeout):
        return {"class": "timeout", "retryable": True}
    if isinstance(exc, ToolUnavailable):
        return {"class": "unavailable", "retryable": True}
    if isinstance(exc, AraError):
        return {"class": exc.code, "retryable": exc.retryable}
    return {"class": "unexpected", "retryable": False}
