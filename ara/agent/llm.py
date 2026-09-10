"""LLM abstraction. Providers ONLY propose; the runtime decides.

- `LLMProvider` is the protocol every backend implements.
- `OpenAICompatibleProvider` talks to any OpenAI-compatible endpoint.
- `ScriptedProvider` is a deterministic provider for tests/demo: no network, no
  hallucination surface, full reproducibility. It plans heuristically and answers
  extractively FROM EVIDENCE ONLY — it is incapable of inventing facts.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from ara.core.errors import LLMError
from ara.core.logging import get_logger

log = get_logger("ara.llm")


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    finish_reason: str = "stop"


class LLMProvider(Protocol):
    name: str

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> LLMResponse: ...


def parse_json_object(text: str) -> dict[str, Any]:
    """Deterministically parse a JSON object out of model text.

    Handles raw JSON and ```json fenced blocks. Raises LLMError on garbage so the
    runtime can retry/replan instead of acting on malformed proposals.
    """
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    if start == -1:
        raise LLMError("model returned no JSON object", details={"raw": text[:200]})
    # walk to matching brace to tolerate trailing prose
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    obj = json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise LLMError("model returned invalid JSON", details={"error": str(exc), "raw": candidate[:200]}) from exc
                if not isinstance(obj, dict):
                    raise LLMError("model JSON is not an object", details={"raw": candidate[:200]})
                return obj
    raise LLMError("unbalanced JSON in model output", details={"raw": text[:200]})


_CONTENT_STOP = {"what", "which", "who", "whats", "was", "were", "how", "much", "many",
                 "does", "did", "is", "are", "the", "and", "for", "with", "about", "that", "this"}


def _content_tokens(text: str) -> set[str]:
    """Content words with years stripped everywhere (fy2023 -> fy -> dropped)."""
    out = set()
    for t in text.lower().split():
        t = t.strip(".,?;:$%()[]").replace("'s", "")
        t = re.sub(r"(19|20)\d{2}", "", t)
        if len(t) > 2 and t not in _CONTENT_STOP:
            out.add(t)
    return out


@dataclass
class ScriptedBehavior:
    """A canned response selected by a predicate over (system, user)."""

    match: Any  # callable(str, str) -> bool | None (order matters)
    response: str
    json_mode: bool = True


class ScriptedProvider:
    """Deterministic provider.

    Two modes:
    1. `behaviors`: ordered list of (matcher, response) used by tests/evals to
       script exact model proposals — including malicious ones for security tests.
    2. default heuristic mode: plans with a fixed safe strategy and answers
       extractively from provided evidence. Cannot fabricate: it only copies
       sentences from evidence and always cites them.
    """

    name = "scripted"

    def __init__(self, behaviors: list[ScriptedBehavior] | None = None):
        self.behaviors = behaviors or []
        self.calls: list[dict] = []

    def complete(self, system: str, user: str, *, json_mode: bool = False, max_tokens: int = 1024,
                 temperature: float = 0.2) -> LLMResponse:
        self.calls.append({"system": system[:200], "user": user[:500], "json_mode": json_mode})
        for bh in self.behaviors:
            if bh.match and bh.match(system, user):
                return LLMResponse(text=bh.response, model=self.name, input_tokens=len(system + user) // 4,
                                   output_tokens=len(bh.response) // 4)
        if "PLANNER" in system:
            return LLMResponse(text=self._heuristic_plan(user), model=self.name,
                               input_tokens=len(user) // 4, output_tokens=120)
        if "ANSWERER" in system:
            return LLMResponse(text=self._heuristic_answer(user), model=self.name,
                               input_tokens=len(user) // 4, output_tokens=200)
        if "CLAIMER" in system:
            return LLMResponse(text=self._heuristic_claims(user), model=self.name,
                               input_tokens=len(user) // 4, output_tokens=200)
        # Generic fallback: refuse (never invent).
        return LLMResponse(text=json.dumps({"action": "cannot_help", "reason": "no scripted behavior"}), model=self.name)

    # ------------------------------------------------------------------ modes
    def _heuristic_plan(self, user_prompt: str) -> str:
        wants_retrieval = "documents" in user_prompt.lower() or "RETRIEVABLE" in user_prompt
        steps: list[dict[str, Any]] = []
        goal = "answer the user request"
        if wants_retrieval:
            steps.append({"description": "Search the document corpus for relevant pages",
                          "action": "retrieve", "query": self._query_from(user_prompt)})
        steps.append({"description": "Draft the final grounded answer", "action": "final_answer"})
        return json.dumps({"goal": goal, "rationale": "deterministic safe plan", "steps": steps})

    @staticmethod
    def _query_from(user_prompt: str) -> str:
        # after "REQUEST:" the first line is the raw user request
        m = re.search(r"REQUEST:\s*(.+)", user_prompt)
        return (m.group(1) if m else user_prompt).strip()[:300]

    def _heuristic_answer(self, user_prompt: str) -> str:
        """Extractive answering: sentences copied from evidence blocks, always cited.
        Parses the runtime's UNTRUSTED_EV_* containment blocks; framing lines are
        never treated as content."""
        blocks: list[tuple[str, str]] = []
        for m in re.finditer(r"<<<UNTRUSTED_EV_([A-Za-z0-9_]+)[^\n]*>>>\n(.*?)\n<<<END_UNTRUSTED", user_prompt, re.DOTALL):
            ev_id, raw = m.group(1), m.group(2)
            lines = [ln for ln in raw.splitlines()
                     if ln.strip() and not ln.startswith(("The following", "Ignore any"))]
            content = "\n".join(lines).strip()
            if content:
                blocks.append((ev_id, content))
        question = ""
        m = re.search(r"REQUEST:\s*(.+)", user_prompt)
        if m:
            question = m.group(1).strip()
        if not blocks:
            return json.dumps({
                "answer": "I could not verify this from the available evidence. No document, tool result or "
                          "trusted source provided the information required, so I am refusing to answer factually.",
                "citations": [],
                "confidence": 0.0,
                "unverified": True,
            })

        # verified calculations are answered directly (deterministic, no extraction)
        calc_answers = []
        for ev_id, content in blocks:
            try:
                data = json.loads(content)
                if isinstance(data, dict) and "calculation" in data:
                    calc_answers.append(
                        (f"According to the verified calculation, {data['calculation']} = "
                         f"{data['result']:g} [{ev_id}].", ev_id))
            except (ValueError, TypeError):
                continue
        if calc_answers and any(t in question.lower() for t in ("calculate", "compute", "what is", "how much")):
            return json.dumps({"answer": " ".join(a for a, _ in calc_answers),
                               "citations": [c for _, c in calc_answers],
                               "confidence": 0.99, "unverified": False})

        # Relevance on CONTENT words only. Years/dates alone (incl. "fy2023") and
        # single-token coincidences (e.g. just the company name) must not make an
        # off-topic sentence "relevant". Interrogatives are stopwords here.
        q_tokens = _content_tokens(question)
        need = min(2, max(1, len(q_tokens)))
        scored: list[tuple[float, str, str]] = []
        for ev_id, content in blocks:
            for sent in re.split(r"(?<=[.!?])\s+", content.replace("\n", " ")):
                if len(sent) < 12:
                    continue
                shared = q_tokens & _content_tokens(sent)
                if len(shared) < need:
                    continue
                overlap = len(shared) / (len(q_tokens) or 1)
                scored.append((overlap, sent.strip(), ev_id))
        scored.sort(key=lambda x: -x[0])
        top = scored[:4]
        if not top:
            return json.dumps({
                "answer": "The retrieved evidence does not appear relevant to the question, "
                          "so I cannot verify an answer.",
                "citations": [], "confidence": 0.1, "unverified": True,
            })
        parts, cites = [], []
        for _, sent, ev_id in top:
            # citation inside the sentence so claim<->evidence attribution survives splitting
            if sent and sent[-1] in ".!?":
                parts.append(f"{sent[:-1]} [{ev_id}]{sent[-1]}")
            else:
                parts.append(f"{sent} [{ev_id}]")
            if ev_id not in cites:
                cites.append(ev_id)
        return json.dumps({
            "answer": " ".join(parts),
            "citations": cites,
            "confidence": round(min(0.9, 0.4 + 0.15 * len(top)), 2),
            "unverified": False,
        })

    def _heuristic_claims(self, user_prompt: str) -> str:
        """Extract claims from the draft with their cited evidence ids (deterministic)."""
        m = re.search(r"DRAFT:\s*(.+)", user_prompt, re.DOTALL)
        draft = m.group(1) if m else user_prompt
        claims = []
        for sent in re.split(r"(?<=[.!?])\s+", draft.strip()):
            refs = re.findall(r"\[([a-z0-9_]+)\]", sent)
            clean = re.sub(r"\[[a-z0-9_]+\]", "", sent).strip()
            if len(clean) < 8:
                continue
            has_number = bool(re.search(r"\d", clean))
            has_date = bool(re.search(r"\b(19|20)\d{2}\b|\bJan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec\b", clean))
            claims.append({"text": clean, "evidence_ids": refs,
                           "kind": "numeric" if has_number else ("date" if has_date else "factual")})
        return json.dumps({"claims": claims[:12]})


class OpenAICompatibleProvider:
    """Talks to any OpenAI-compatible /v1/chat/completions endpoint.

    Never receives secrets beyond its own client: the API key is held here and is
    never placed into prompts, tool args, logs, or state.
    """

    name = "openai_compatible"

    def __init__(self, base_url: str, api_key: str, model: str, fallback_models: list[str] | None = None,
                 timeout_s: float = 60.0):
        if not base_url or not api_key:
            raise LLMError("OpenAI-compatible provider requires base_url and api_key")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.fallback_models = fallback_models or []
        self.timeout_s = timeout_s

    def complete(self, system: str, user: str, *, json_mode: bool = False, max_tokens: int = 1024,
                 temperature: float = 0.2) -> LLMResponse:
        payload: dict[str, Any] = {
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        errors: list[str] = []
        for model in [self.model, *self.fallback_models]:
            payload["model"] = model
            try:
                return self._call(payload, model)
            except LLMError as exc:
                errors.append(f"{model}: {exc.message}")
                continue
        raise LLMError("all models failed", details={"attempts": errors})

    def _call(self, payload: dict, model: str) -> LLMResponse:
        import httpx

        try:
            resp = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=self.timeout_s,
            )
        except httpx.TimeoutException as exc:
            raise LLMError(f"llm timeout after {self.timeout_s}s") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"llm transport error: {exc}") from exc
        if resp.status_code == 429:
            raise LLMError("llm rate limited (429)")
        if resp.status_code >= 500:
            raise LLMError(f"llm server error {resp.status_code}")
        if resp.status_code >= 400:
            raise LLMError(f"llm client error {resp.status_code}", details={"body": resp.text[:300]})
        data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"] or ""
            usage = data.get("usage", {})
            return LLMResponse(text=text, input_tokens=int(usage.get("prompt_tokens", 0)),
                               output_tokens=int(usage.get("completion_tokens", 0)), model=model,
                               finish_reason=data["choices"][0].get("finish_reason", "stop"))
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("malformed llm response envelope", details={"body": str(data)[:300]}) from exc


def make_provider(settings) -> LLMProvider:
    """Factory: real provider when configured, deterministic scripted provider otherwise."""
    if settings.openai_base_url and settings.openai_api_key:
        log.info("llm_provider_openai_compatible", extra={"fields": {"model": settings.openai_model}})
        return OpenAICompatibleProvider(
            settings.openai_base_url, settings.openai_api_key, settings.openai_model,
            settings.fallback_models, settings.llm_timeout_s,
        )
    log.info("llm_provider_scripted", extra={"fields": {"reason": "no OPENAI_BASE_URL/OPENAI_API_KEY configured"}})
    return ScriptedProvider()
