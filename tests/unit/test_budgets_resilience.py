"""Budget enforcement + resilience (retry/backoff/timeout/fallback) tests."""
import time

import pytest

from ara.agent.budgets import BudgetState
from ara.agent.resilience import run_with_timeout, with_retry
from ara.core.errors import BudgetExceeded, ToolTimeout, ToolUnavailable
from ara.core.tracing import CostMeter


def make_budget(**kw):
    defaults = dict(max_iterations=3, max_tool_calls=2, max_llm_calls=2, max_tokens=1000,
                    max_cost_usd=10.0, deadline=time.monotonic() + 60, cost=CostMeter(0.001, 0.001))
    defaults.update(kw)
    return BudgetState(**defaults)


def test_iteration_budget_stops_loops():
    b = make_budget()
    for _ in range(3):
        b.check_iteration()
    with pytest.raises(BudgetExceeded) as ei:
        b.check_iteration()
    assert b.stop_reason == "max_iterations" and "iteration" in str(ei.value)


def test_tool_call_budget():
    b = make_budget()
    b.pre_tool(); b.pre_tool()
    with pytest.raises(BudgetExceeded):
        b.pre_tool()


def test_token_and_cost_budget():
    b = make_budget(max_tokens=100, max_cost_usd=1000)
    b.after_llm(30, 20)  # 50 tokens: fine
    with pytest.raises(BudgetExceeded) as ei:
        b.after_llm(60, 50)  # 110 cumulative > 100
    assert b.stop_reason == "max_tokens" and "token" in str(ei.value)


def test_time_budget_deadline():
    b = make_budget(deadline=time.monotonic() - 0.01)
    with pytest.raises(BudgetExceeded):
        b.check_iteration()
    assert b.stop_reason == "time_budget"


def test_retry_succeeds_on_second_attempt():
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise ConnectionError("transient")
        return "ok"

    assert with_retry(flaky, max_attempts=3, base_delay_s=0.01) == "ok"
    assert attempts["n"] == 2


def test_retry_gives_up_after_ceiling():
    with pytest.raises(ToolUnavailable):
        with_retry(lambda: (_ for _ in ()).throw(ConnectionError("down")), max_attempts=2, base_delay_s=0.01)


def test_retry_never_retries_non_retryable():
    attempts = {"n": 0}

    def bad():
        attempts["n"] += 1
        raise ValueError("logic error")

    with pytest.raises(Exception):
        with_retry(bad, max_attempts=3, base_delay_s=0.01)
    assert attempts["n"] == 1  # no blind retries of programming errors


def test_run_with_timeout_enforces_wall_clock():
    def slow():
        time.sleep(2)
        return "late"

    with pytest.raises(ToolTimeout):
        run_with_timeout(slow, timeout_s=0.05, what="slow-tool")


def test_run_with_timeout_passes_through_result():
    assert run_with_timeout(lambda: 42, timeout_s=1) == 42
