"""Client for the colpali-service, with an in-process mock twin.

`ColPaliClient` (HTTP) is used in production: the backend never loads models
itself. `InProcessMockColPali` runs the identical mock embedding in-process so
unit/integration tests need no network. Retry/backoff is inherited from the
resilience layer.
"""
from __future__ import annotations

from typing import Protocol

from ara.agent.resilience import with_retry
from ara.core.errors import RetrievalError
from ara.core.logging import get_logger
from ara.retrieval import mockcolpali

log = get_logger("ara.colpali")


class ColPaliLike(Protocol):
    mode: str

    def embed_queries(self, queries: list[str]) -> list[list[list[float]]]: ...
    def embed_pages(self, pages: list[dict]) -> list[list[list[float]]]: ...


class InProcessMockColPali:
    mode = "mock"

    def embed_queries(self, queries: list[str]) -> list[list[list[float]]]:
        return [mockcolpali.embed_query(q) for q in queries]

    def embed_pages(self, pages: list[dict]) -> list[list[list[float]]]:
        return [mockcolpali.embed_page_chunks(p.get("text") or p.get("id", "empty")) for p in pages]


class ColPaliClient:
    def __init__(self, base_url: str, timeout_s: float = 30.0, max_attempts: int = 3):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        self.mode = "service"

    def _post(self, path: str, payload: dict) -> dict:
        def call() -> dict:
            import httpx

            try:
                resp = httpx.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout_s)
            except httpx.HTTPError as exc:
                raise RetrievalError(f"colpali-service unreachable: {exc}") from exc
            if resp.status_code == 503:
                raise RetrievalError("colpali-service model unavailable (503)")
            if resp.status_code >= 500:
                raise RetrievalError(f"colpali-service error {resp.status_code}")
            if resp.status_code >= 400:
                raise RetrievalError(f"colpali-service rejected request ({resp.status_code}): {resp.text[:200]}")
            return resp.json()

        return with_retry(call, max_attempts=self.max_attempts, base_delay_s=0.3,
                          retryable=(RetrievalError, ConnectionError))

    def health(self) -> dict:
        import httpx

        resp = httpx.get(f"{self.base_url}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def embed_queries(self, queries: list[str]) -> list[list[list[float]]]:
        data = self._post("/embed/queries", {"queries": queries})
        return data["embeddings"]

    def embed_pages(self, pages: list[dict]) -> list[list[list[float]]]:
        data = self._post("/embed/pages", {"pages": pages})
        return data["embeddings"]


def make_colpali(settings) -> ColPaliLike:
    if settings.colpali_mode == "real" and settings.colpali_service_url:
        log.info("colpali_client_http", extra={"fields": {"url": settings.colpali_service_url, "mode": "real"}})
        return ColPaliClient(settings.colpali_service_url)
    if settings.colpali_service_url:
        # service may still be running in mock mode; use HTTP when available at runtime
        log.info("colpali_mode_mock", extra={"fields": {"url": settings.colpali_service_url}})
    return InProcessMockColPali()
