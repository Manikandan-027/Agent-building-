"""Evidence manager: explicit, citable evidence records.

Every piece of information the agent may ground a claim on becomes an Evidence
object with provenance, trust level, injection scan results, and tenant scope.
Model-generated text is NEVER evidence. Untrusted sources (retrieved docs, web,
tool output) are marked as such and injection-flagged items are quarantined
from grounding.
"""
from __future__ import annotations

from typing import Iterable

from ara.core.ids import iso_now
from ara.core.logging import get_logger
from ara.core.types import ContentTrust, SourceAuthority
from ara.guardrails.injection import InjectionDetector, InjectionScan

log = get_logger("ara.evidence")

# Trust levels allowed to ground factual claims.
GROUNDAUTH_TRUST = {ContentTrust.APP_DATA.value, ContentTrust.RETRIEVED.value,
                    ContentTrust.TOOL_OUTPUT.value, ContentTrust.WEB.value}
QUARANTINE_TRUST = {ContentTrust.MODEL.value}


def as_evidence_dict(ev) -> dict:
    """Accept pydantic Evidence or plain dict (state round-trips through JSON)."""
    return ev.model_dump() if hasattr(ev, "model_dump") else ev


class EvidenceManager:
    def __init__(self, detector: InjectionDetector, source_authority: dict[str, str] | None = None):
        self.detector = detector
        # deterministic authority assignment per source type
        self.source_authority = {"internal_corpus": "high", "uploaded_document": "medium",
                                 "web": "low", "tool_output": "medium", "calculation": "high",
                                 **(source_authority or {})}

    def make(self, *, content: str, document_id: str, page: int = 0, content_type: str = "visual_document_page",
             trust: ContentTrust = ContentTrust.RETRIEVED, source: str = "internal_corpus",
             relevance_score: float = 0.0, tenant_id: str = "", meta: dict | None = None) -> dict:
        scan = self.detector.scan(content)
        authority = self.source_authority.get(source, SourceAuthority.UNKNOWN.value)
        ev = {
            "evidence_id": None,  # assigned by caller (stable across persistence)
            "document_id": document_id,
            "page": page,
            "content_type": content_type,
            "content": content,
            "relevance_score": round(relevance_score, 4),
            "source_authority": authority,
            "source_trust": trust.value,
            "retrieval_timestamp": iso_now(),
            "tenant_id": tenant_id,
            "injection_scan": scan.to_dict(),
            "meta": {"source": source, **(meta or {})},
        }
        if scan.flagged:
            log.warning("evidence_injection_flagged", extra={"fields": {"reasons": scan.reasons}})
        return ev

    @staticmethod
    def is_groundedable(ev) -> bool:
        """Flagged injections and model text can never support claims."""
        ev = as_evidence_dict(ev)
        if ev.get("injection_scan", {}).get("flagged"):
            return False
        return ev.get("source_trust") in GROUNDAUTH_TRUST

    @staticmethod
    def by_id(evidence: Iterable, evidence_id: str) -> dict | None:
        for ev in evidence:
            d = as_evidence_dict(ev)
            if d.get("evidence_id") == evidence_id:
                return d
        return None
