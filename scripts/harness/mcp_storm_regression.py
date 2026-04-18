"""phase-3.7 step 3.7.6: Tool-call storm + output-size regression test."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backend.agents.mcp_guardrails import (  # noqa: E402
    DebounceExceeded, sliding_window_debounce, cap_output_size,
)


def _test_storm_guard_fires() -> dict:
    t = [0.0]

    def clock():
        return t[0]

    @sliding_window_debounce(max_calls=3, window_s=10.0, clock=clock)
    def fake_tool(ticker: str) -> dict:
        return {"ok": True, "ticker": ticker}

    calls_ok = 0
    debounce_raised = False
    for _ in range(3):
        fake_tool("AAPL")
        calls_ok += 1
    try:
        fake_tool("AAPL")
    except DebounceExceeded:
        debounce_raised = True

    t[0] = 11.0
    reset_ok = False
    try:
        fake_tool("AAPL")
        reset_ok = True
    except DebounceExceeded:
        reset_ok = False

    return {
        "test": "storm_guard_fires_on_4th_identical_call",
        "calls_ok_before_trip": calls_ok,
        "debounce_raised_on_4th": debounce_raised,
        "window_reset_ok": reset_ok,
        "pass": calls_ok == 3 and debounce_raised and reset_ok,
    }


def _test_different_args_independent() -> dict:
    t = [0.0]

    def clock():
        return t[0]

    @sliding_window_debounce(max_calls=3, window_s=10.0, clock=clock)
    def fake_tool(ticker: str) -> dict:
        return {"ok": True, "ticker": ticker}

    successes = 0
    any_raised = False
    for t_str in ("AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"):
        for _ in range(3):
            try:
                fake_tool(t_str)
                successes += 1
            except DebounceExceeded:
                any_raised = True
    return {
        "test": "different_args_are_not_debounced_together",
        "successes": successes,
        "any_raised": any_raised,
        "pass": successes == 15 and not any_raised,
    }


def _test_cap_output_size_truncates() -> dict:
    big_list = [{"ticker": f"SYM_{i:04d}", "price": 100.0 + i * 0.01,
                 "blob": "x" * 100}
                 for i in range(2000)]
    result = {"status": "ok", "items": big_list}
    raw = json.dumps(result)
    original_size = len(raw.encode("utf-8"))

    capped = cap_output_size(result, max_bytes=100_000)
    capped_size = len(json.dumps(capped, default=repr).encode("utf-8"))

    return {
        "test": "cap_output_size_truncates_over_100kb",
        "original_size_bytes": original_size,
        "capped_size_bytes": capped_size,
        "truncated_flag": capped.get("_truncated"),
        "truncated_field": capped.get("_truncated_field"),
        "pass": (
            original_size > 100_000
            and capped_size <= 100_000
            and capped.get("_truncated") is True
            and capped.get("_truncated_field") == "items"
        ),
    }


def _test_small_payload_passthrough() -> dict:
    small = {"status": "ok", "value": 42, "tags": ["a", "b", "c"]}
    out = cap_output_size(small, max_bytes=100_000)
    return {
        "test": "small_payload_passthrough",
        "equal": out == small,
        "pass": out == small and "_truncated" not in out,
    }


def main() -> int:
    tests = [
        _test_storm_guard_fires(),
        _test_different_args_independent(),
        _test_cap_output_size_truncates(),
        _test_small_payload_passthrough(),
    ]
    all_pass = all(t["pass"] for t in tests)
    result = {
        "step": "3.7.6",
        "verdict": "PASS" if all_pass else "FAIL",
        "tests": tests,
        "tests_passed": sum(1 for t in tests if t["pass"]),
        "tests_total": len(tests),
    }
    out = REPO / "handoff" / "mcp_storm_regression.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(out),
        "verdict": result["verdict"],
        "tests_passed": result["tests_passed"],
        "tests_total": result["tests_total"],
    }))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
