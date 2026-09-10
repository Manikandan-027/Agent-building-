"""Vector stores: multi-vector (ColPali-style) points with metadata filtering.

- `InMemoryMultiVectorStore`: exact MaxSim search; used for dev/CI (deterministic).
- `QdrantMultiVectorStore`: production adapter. Candidate retrieval uses a
  mean-pooled named vector; final scoring recomputes TRUE MaxSim client-side from
  full multi-vectors kept in the payload. Requires `pip install .[prod]` + QDRANT_URL.

Both stores implement the same protocol; selection is purely configuration-driven.
Tenant/permission filters are applied inside the store AND re-checked by the
retrieval engine (defense in depth).
"""
from __future__ import annotations

import threading
from typing import Any, Protocol

from ara.core.logging import get_logger
from ara.retrieval import mockcolpali

log = get_logger("ara.vectors")


class AccessFilter(Protocol):
    tenant_id: str
    role: str
    user_id: str

    def payload_filter(self) -> dict: ...


def make_access_filter(tenant_id: str, role: str, user_id: str) -> dict:
    """Canonical payload filter. Documents MUST carry tenant_id; access_json holds
    optional role/user restrictions ('*' = unrestricted within tenant)."""
    return {"tenant_id": tenant_id, "role": role, "user_id": user_id}


def _passes(payload: dict, flt: dict) -> bool:
    if payload.get("tenant_id") != flt["tenant_id"]:
        return False
    access = payload.get("access", {})
    roles = access.get("roles") or ["*"]
    users = access.get("user_ids") or ["*"]
    return (flt["role"] in roles or "*" in roles) and (flt["user_id"] in users or "*" in users)


class InMemoryMultiVectorStore:
    name = "inmemory"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._points: dict[str, dict[str, Any]] = {}

    def upsert(self, point_id: str, vectors: list[list[float]], payload: dict) -> None:
        with self._lock:
            self._points[point_id] = {"vectors": vectors, "payload": payload}

    def delete(self, filter_payload: dict) -> int:
        with self._lock:
            doomed = [pid for pid, p in self._points.items()
                      if all(p["payload"].get(k) == v for k, v in filter_payload.items())]
            for pid in doomed:
                del self._points[pid]
            return len(doomed)

    def search(self, query_vectors: list[list[float]], flt: dict, k: int = 5,
               min_score: float = 0.0) -> list[dict]:
        with self._lock:
            hits = []
            for pid, point in self._points.items():
                if not _passes(point["payload"], flt):
                    continue
                score = mockcolpali.maxsim(query_vectors, point["vectors"])
                if score >= min_score:
                    hits.append({"id": pid, "score": round(score, 4), "payload": point["payload"]})
            hits.sort(key=lambda h: -h["score"])
            return hits[:k]

    def count(self) -> int:
        with self._lock:
            return len(self._points)


class QdrantMultiVectorStore:
    """Production adapter (see module docstring for the multi-vector strategy)."""

    name = "qdrant"

    def __init__(self, url: str, collection: str = "ara_pages", dim: int = mockcolpali.DIM):
        from qdrant_client import QdrantClient  # type: ignore
        from qdrant_client import models  # type: ignore

        self.models = models
        self.client = QdrantClient(url=url, timeout=10)
        self.collection = collection
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config={
                    "pooled": models.VectorParams(size=dim, distance=models.Distance.COSINE),
                },
            )

    def upsert(self, point_id: str, vectors: list[list[float]], payload: dict) -> None:
        m = self.models
        pooled = [sum(col) / len(col) for col in zip(*vectors)]
        self.client.upsert(
            collection_name=self.collection,
            points=[m.PointStruct(id=point_id, vector={"pooled": pooled},
                                  payload={**payload, "_multivec": vectors})],
        )

    def delete(self, filter_payload: dict) -> int:
        m = self.models
        cond = [m.FieldCondition(key=k, match=m.MatchValue(value=v)) for k, v in filter_payload.items()]
        before = self.client.count(self.collection).count
        self.client.delete(collection_name=self.collection,
                           points_selector=m.FilterSelector(filter=m.Filter(must=cond)))
        return before - self.client.count(self.collection).count

    def search(self, query_vectors: list[list[float]], flt: dict, k: int = 5,
               min_score: float = 0.0) -> list[dict]:
        m = self.models
        pooled = [sum(col) / len(col) for col in zip(*query_vectors)]
        cond = [m.FieldCondition(key="tenant_id", match=m.MatchValue(value=flt["tenant_id"]))]
        candidates = self.client.query_points(collection=self.collection, query=pooled,
                                              using="pooled", limit=max(k * 6, 30)).points
        out = []
        for p in candidates:
            payload = p.payload or {}
            if not _passes(payload, flt):
                continue
            score = mockcolpali.maxsim(query_vectors, payload.get("_multivec", []))
            if score >= min_score:
                out.append({"id": p.id, "score": round(score, 4), "payload": payload})
        out.sort(key=lambda h: -h["score"])
        return out[:k]

    def count(self) -> int:
        return self.client.count(self.collection).count


def make_vector_store(settings) -> InMemoryMultiVectorStore | QdrantMultiVectorStore:
    """Configuration-driven backend selection with graceful fallback."""
    if settings.qdrant_url:
        try:
            store = QdrantMultiVectorStore(settings.qdrant_url)
            log.info("vector_store_qdrant", extra={"fields": {"url": settings.qdrant_url}})
            return store
        except ImportError:
            log.warning("qdrant_client_missing_fallback_inmemory")
        except Exception as exc:  # noqa: BLE001
            log.warning("qdrant_unavailable_fallback_inmemory", extra={"fields": {"error": str(exc)[:200]}})
    return InMemoryMultiVectorStore()
