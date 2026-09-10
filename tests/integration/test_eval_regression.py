"""Evaluation regression tests: a representative scenario from each category runs
through the REAL runtime on every test run. Full dataset: `make evals` /
scripts/run_evals.py."""
import pytest

from evals.runner import run_scenario, load_scenarios

CORE = "evals/datasets/core.yaml"

REPRESENTATIVE = [
    "normal_revenue_lookup",
    "ambiguous_missing_document",
    "retrieval_irrelevant_only",
    "security_injected_document",
    "security_high_risk_needs_approval",
    "autonomy_budget_stop",
]


@pytest.mark.parametrize("scenario_id", REPRESENTATIVE)
def test_eval_scenario_regression(scenario_id):
    scenarios = load_scenarios(CORE)
    sc = next(s for s in scenarios if s["id"] == scenario_id)
    result = run_scenario(sc)
    failed = [k for k, ok in result.checks.items() if not ok]
    assert result.passed, f"{scenario_id}: failed checks {failed}; answer={result.answer[:200]}"


def test_eval_dataset_fully_passes():
    """The complete dataset must pass end-to-end (full regression gate)."""
    from evals.runner import run_dataset

    report = run_dataset([CORE])
    failed = [r["scenario_id"] for r in report["results"] if not r["passed"]]
    assert not failed, f"eval failures: {failed}"
    m = report["metrics"]
    assert m["refusal_precision"] == 1.0, "refusal precision regressed: fabricated answers on insufficient evidence"
