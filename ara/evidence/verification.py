"""Claim verification: NO EVIDENCE -> NO FACTUAL CLAIM.

Pipeline:
  1. Claims are extracted from the draft (LLM CLAIMER proposes; heuristic fallback).
  2. Runtime deterministically validates each claim:
     - cited evidence exists, is groundedable (not injection-flagged / not model text)
     - token support between claim and cited evidence meets threshold
     - NUMBERS in claims must literally appear in cited evidence or in verified
       calculator results  (forbid_unverified_numbers)
     - DATES likewise    (forbid_unverified_dates)
  3. Contradictions are detected across evidence and resolved deterministically
     (authority -> date -> version) or explicitly reported.
  4. Unsupported claims are stripped; if the answer cannot satisfy the contract's
     verification requirements, it is REFUSED.

The verification layer never asks an LLM whether a claim is true — it checks
against evidence deterministically.
"""
from __future__ import annotations

import re
from typing import Any

from ara.core.logging import get_logger
from ara.evidence.manager import EvidenceManager

log = get_logger("ara.verification")

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
DATE_RE = re.compile(r"\b(19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b|\b(19|20)\d{2}\b")
SUPPORT_THRESHOLD = 0.22
TOKEN_RE = re.compile(r"[a-z0-9$%.]{2,}")


def tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


class VerificationEngine:
    def __init__(self, evidence_manager: EvidenceManager):
        self.em = evidence_manager

    # ---------------------------------------------------------------- claims
    def extract_claims(self, draft: str, llm_proposal: dict | None = None) -> list[dict]:
        """LLM proposes claim->evidence mapping; runtime validates. Fallback: split
        sentences, attribute the citations physically present in each sentence."""
        claims: list[dict] = []
        if llm_proposal and isinstance(llm_proposal.get("claims"), list):
            for c in llm_proposal["claims"][:24]:
                if isinstance(c, dict) and isinstance(c.get("text"), str) and len(c["text"]) >= 8:
                    claims.append({
                        "text": c["text"],
                        "evidence_ids": [str(e) for e in (c.get("evidence_ids") or [])][:8],
                        "kind": c.get("kind", "factual"),
                    })
            return claims
        for sentence in re.split(r"(?<=[.!?])\s+", draft):
            refs = re.findall(r"\b(?:ev|call)_[a-z0-9_]+\b", sentence)
            clean = re.sub(r"\b(?:ev|call)_[a-z0-9_]+\b", "", sentence).strip(" ,.")
            if len(clean) >= 8:
                nums = bool(NUM_RE.search(clean))
                claims.append({"text": clean, "evidence_ids": refs,
                               "kind": "numeric" if nums else "factual"})
        return claims[:24]

    # ---------------------------------------------------------------- verify
    def verify(self, draft: str, evidence: list[dict], requirements, *,
               calculator_results: list[dict] | None = None,
               llm_proposal: dict | None = None) -> dict:
        claims = self.extract_claims(draft, llm_proposal)
        calculator_numbers = self._calc_numbers(calculator_results or [])
        verified: list[dict] = []
        unsupported: list[str] = []

        for claim in claims:
            verdict = self._verify_claim(claim, evidence, calculator_numbers)
            verified.append(verdict)
            if not verdict["supported"]:
                unsupported.append(claim["text"])

        conflicts = self.detect_conflicts(evidence)
        cited_any = any(v["evidence_ids"] for v in verified if v["supported"])
        numbers_ok = all(v["supported"] for v in verified if v["kind"] == "numeric") if requirements.forbid_unverified_numbers else True
        dates_ok = all(v["supported"] for v in verified if v["kind"] == "date") if requirements.forbid_unverified_dates else True

        refused, reason = False, None
        any_citation_attempt = claims and any(c.get("evidence_ids") for c in claims)
        if requirements.require_citations and not any_citation_attempt:
            refused, reason = True, "answer contains no citations to admissible evidence"
        elif requirements.refuse_on_unsupported_claims and unsupported and not any(v["supported"] for v in verified):
            refused, reason = True, "no claim in the draft is supported by admissible evidence"
        elif not numbers_ok:
            refused, reason = True, "numeric claims not verifiable against evidence or verified calculations"
        elif not dates_ok:
            refused, reason = True, "date claims not verifiable against evidence"

        return {
            "answer_supported": bool(verified) and not refused and all(v["supported"] for v in verified),
            "claims": verified,
            "unsupported_claims": unsupported,
            "conflicts": conflicts,
            "refused": refused,
            "refusal_reason": reason,
            "checks": {"numbers_ok": numbers_ok, "dates_ok": dates_ok,
                       "citations_present": cited_any, "claim_count": len(claims)},
        }

    def _verify_claim(self, claim: dict, evidence: list[dict], calculator_numbers: set[str]) -> dict:
        text = claim["text"]
        cited = claim.get("evidence_ids") or []
        problems: list[str] = []
        supporting: list[str] = []

        if not cited:
            problems.append("no citation")

        claim_tokens = tokens(text)
        for ev_id in cited:
            ev = EvidenceManager.by_id(evidence, ev_id)
            if ev is None:
                problems.append(f"cited evidence {ev_id} does not exist")
                continue
            if not self.em.is_groundedable(ev):
                problems.append(f"cited evidence {ev_id} is inadmissible "
                                f"(trust={ev.get('source_trust')}, injection={ev.get('injection_scan', {}).get('flagged')})")
                continue
            support = len(claim_tokens & tokens(ev.get("content", ""))) / (len(claim_tokens) or 1)
            if support >= SUPPORT_THRESHOLD:
                supporting.append(ev_id)
        if cited and not supporting:
            problems.append("no admissible supporting evidence")

        # deterministic numeric grounding
        missing_numbers = []
        for num in NUM_RE.findall(text):
            n = num.lstrip("-").rstrip(".")
            in_ev = any(n in ev.get("content", "") for ev in evidence
                        if ev.get("evidence_id") in (supporting or cited))
            if not in_ev and n not in calculator_numbers and not DATE_RE.fullmatch(num):
                missing_numbers.append(num)
        if missing_numbers:
            problems.append(f"numbers not found in cited evidence/calculations: {missing_numbers}")

        missing_dates = []
        for d in DATE_RE.findall(text) and DATE_RE.finditer(text):
            ds = d.group(0)
            in_ev = any(ds in ev.get("content", "") for ev in evidence
                        if ev.get("evidence_id") in (supporting or cited))
            if not in_ev:
                missing_dates.append(ds)
        if missing_dates:
            problems.append(f"dates not found in cited evidence: {missing_dates}")

        return {"text": text, "evidence_ids": supporting, "kind": claim.get("kind", "factual"),
                "supported": not problems, "problems": problems}

    @staticmethod
    def _calc_numbers(calc_results: list[dict]) -> set[str]:
        out = set()
        for r in calc_results:
            res = r.get("result")
            if isinstance(res, (int, float)):
                out.add(f"{res:g}")
                out.add(str(res))
        return out

    # ---------------------------------------------------------------- conflicts
    def detect_conflicts(self, evidence: list[dict]) -> list[dict]:
        """Numeric conflicts: same contextual phrase, different numbers, different
        documents. Deterministic resolution: authority > date > version; else report."""
        conflicts: list[dict] = []
        usable = [ev for ev in evidence if self.em.is_groundedable(ev)]
        for i in range(len(usable)):
            for j in range(i + 1, len(usable)):
                a, b = usable[i], usable[j]
                if a["document_id"] == b["document_id"] and a.get("page") == b.get("page"):
                    continue
                conflicts.extend(self._pair_conflicts(a, b))
        return conflicts

    def _pair_conflicts(self, a: dict, b: dict) -> list[dict]:
        ta, tb = tokens(a.get("content", "")), tokens(b.get("content", ""))
        overlap = ta & tb
        if len(overlap) < 4:
            return []
        nums_a = set(NUM_RE.findall(a.get("content", "")))
        nums_b = set(NUM_RE.findall(b.get("content", "")))
        differing = nums_a.symmetric_difference(nums_b)
        # only flag when both sides mention numbers and contexts align strongly
        if not (nums_a and nums_b and differing and len(overlap) >= max(4, 0.5 * min(len(ta), len(tb)))):
            return []
        shared = {n for n in nums_a & nums_b}
        conflict = {
            "evidence_a": a["evidence_id"], "evidence_b": b["evidence_id"],
            "values_a": sorted(nums_a), "values_b": sorted(nums_b),
            "shared_values": sorted(shared),
            "resolution": self._resolve(a, b),
        }
        return [conflict]

    def _resolve(self, a: dict, b: dict) -> dict:
        """Deterministic resolution preference; 'unresolved' must be reported, never hidden."""
        rank = {"high": 3, "medium": 2, "low": 1, "unknown": 0}
        ra, rb = rank.get(a.get("source_authority"), 0), rank.get(b.get("source_authority"), 0)
        if ra != rb:
            winner = a if ra > rb else b
            return {"strategy": "source_authority", "prefer": winner["evidence_id"]}
        da, db = a.get("retrieval_timestamp", ""), b.get("retrieval_timestamp", "")
        va, vb = str(a.get("meta", {}).get("document_version", "")), str(b.get("meta", {}).get("document_version", ""))
        if va != vb:
            return {"strategy": "document_version", "prefer": a["evidence_id"] if va >= vb else b["evidence_id"]}
        if da != db:
            return {"strategy": "recency", "prefer": a["evidence_id"] if da >= db else b["evidence_id"]}
        return {"strategy": "unresolved", "prefer": None}
