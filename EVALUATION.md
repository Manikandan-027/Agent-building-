# ARA Evaluation

Run: `python scripts/run_evals.py` (writes `evals/latest_report.md` / `.json`) or
`make evals`. Regression gate in CI: `tests/integration/test_eval_regression.py`
(representative scenarios + full-dataset pass + refusal precision == 1.0).

Latest measured run (deterministic scripted provider, mock ColPali):

```
Scenarios: 17/17 passed · refusal precision 1.00 · avg latency 5 ms
```

| Category | Scenarios | What is exercised |
|---|---|---|
| normal | 2 | grounded retrieval answer; deterministic arithmetic |
| ambiguous | 2 | vague reference; subject missing from corpus ⇒ refuse |
| retrieval | 2 | relevant hit among distractors; irrelevant-only corpus ⇒ refuse |
| multimodal | 1 | PDF ingestion pipeline; placeholder corpus must NOT be invented |
| contradiction | 1 | conflicting figures ⇒ deterministic resolution + CONFLICT notice, never both-as-consistent |
| tool_failure | 1 | permanently failing tool ⇒ graceful refusal, no fabricated output |
| security | 4 | injected document; injected web tool; forbidden tool; HIGH-risk approval flow |
| memory | 2 | semantic promotion policy; attribute replacement |
| autonomy | 2 | budget exhaustion ⇒ safe stop; multi-step retrieve→calculate→answer |

## How to read the metrics

- **task_success_rate** — scenarios meeting *all* deterministic behavioral checks
  (status, must/must-not contain, refusal, citations, approval flow, injection
  flags, tool usage). Checks live in `evals/runner.py`.
- **refusal_precision** — of scenarios where evidence is insufficient, the fraction
  that *refused* rather than answered. Currently **1.0**: the system has never
  fabricated an answer on insufficient evidence in the suite.
- **unsupported_claim_rate** — claims *flagged and stripped by the verifier* per
  judged answer. This is the verification layer working, not hallucinations
  reaching users: stripped claims never appear in final answers.
- **citation_coverage** — share of scenarios whose answers cite admissible
  evidence (refusals intentionally cite nothing).
- **guardrail_activations** — count of guardrail/verification interventions.

## Adversarial coverage (also enforced as unit/integration tests)

- Instruction override, role hijack, secret lure, base64/zero-width obfuscation
  (incl. homoglyph attacks) — detected *and* contained at sentence level.
- Poisoned document mixed with legitimate content: attack quarantined, clean
  sentence still groundable; answer never contains injected directives.
- Hostile web/MCP tool content: marked untrusted, contained, never executed.
- Hallucinated citations, fabricated tool-call claims, secret leakage: blocked by
  output guardrail (tested).
- Approval bypass attempts: HIGH/CRITICAL tools suspend durably; approve/reject
  paths tested end-to-end over HTTP; rejected actions never execute.
- Budget exhaustion at every dimension: safe-stop messages, no partial answers.
- Audit chain tampering: detected by `verify_chain()`.

## Regressions

Every production bug found during development became a regression test — e.g.
the citation-after-punctuation splitting bug, the stale semantic-memory slot bug,
the MCP blocking-readline timeout bug, and the unresolved-conflict fallback bug
all have named tests. `python -m pytest tests/ -q` runs **138 tests**.
