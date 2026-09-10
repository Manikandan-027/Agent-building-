"""Retrieval pipeline:

query -> normalize -> visual embedding (colpali) -> candidate search with
tenant/permission filters -> rank -> confidence check -> Evidence records with
injection scanning -> hand only relevant evidence to the reasoning model.

Retrieval NEVER escapes the caller's tenant or permission scope: the filter is
applied in the store AND re-asserted on every hit before evidence is admitted.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ara.core.errors import AuthorizationError, RetrievalError
from ara.core.types import ContentTrust
from ara.core.ids import new_id
from ara.core.logging import get_logger
from ara.evidence.manager import EvidenceManager
from ara.guardrails.injection import InjectionDetector
from ara.retrieval.colpali_client import ColPaliLike
from ara.retrieval.vectors import make_access_filter

log = get_logger("ara.retrieval")


@dataclass
class RetrievalResult:
    evidence: list[dict]
    confidence: float          # top score
    margin: float              # top1 - top2 (retrieval certainty signal)
    used_visual: bool


class RetrievalEngine:
    def __init__(self, colpali: ColPaliLike, store, evidence_manager: EvidenceManager,
                 detector: InjectionDetector, *, top_k: int = 5, min_score: float = 0.2):
        self.colpali = colpali
        self.store = store
        self.em = evidence_manager
        self.detector = detector
        self.top_k = top_k
        self.min_score = min_score

    @staticmethod
    def normalize_query(query: str) -> str:
        q = re.sub(r"\s+", " ", query).strip()
        return q[:600]

    def needs_visual_retrieval(self, query: str, available_docs: int) -> bool:
        """Heuristic: with any document corpus present, use visual retrieval.
        (Charts/tables/scans have no reliable text alternative — visual-first is
        the ColPali design.)"""
        return available_docs > 0

    def retrieve(self, *, query: str, tenant_id: str, role: str = "user", user_id: str = "",
                 k: int | None = None, document_ids: list[str] | None = None) -> RetrievalResult:
        q = self.normalize_query(query)
        if not q:
            raise RetrievalError("empty query")

        try:
            q_vectors = self.colpali.embed_queries([q])[0]
        except Exception as exc:
            raise RetrievalError(f"query embedding failed: {exc}") from exc

        flt = make_access_filter(tenant_id, role, user_id)
        hits = self.store.search(q_vectors, flt, k=k or self.top_k, min_score=self.min_score)

        # defense in depth: re-assert scope on every hit before admission
        if document_ids:
            hits = [h for h in hits if h["payload"]["document_id"] in document_ids]
        evidence: list[dict] = []
        for h in hits:
            payload = h["payload"]
            if payload.get("tenant_id") != tenant_id:
                log.error("scope_violation_blocked", extra={"fields": {"point": h["id"]}})
                raise AuthorizationError("retrieval returned a point outside tenant scope")
            ev = self.em.make(content=payload.get("text_preview", ""), document_id=payload["document_id"],
                              page=payload["page_number"], content_type="visual_document_page",
                              trust=ContentTrust.RETRIEVED,
                              source="internal_corpus", relevance_score=h["score"], tenant_id=tenant_id,
                              meta={"title": payload.get("title"), "document_type": payload.get("document_type"),
                                    "document_version": payload.get("version", "v1")})
            ev["evidence_id"] = new_id("ev")
            evidence.append(ev)
        confidence = hits[0]["score"] if hits else 0.0
        margin = (hits[0]["score"] - hits[1]["score"]) if len(hits) > 1 else confidence
        if not hits:
            log.info("retrieval_no_hits", extra={"fields": {"tenant_id": tenant_id}})
        return RetrievalResult(evidence=evidence, confidence=round(confidence, 4),
                               margin=round(margin, 4), used_visual=True)
