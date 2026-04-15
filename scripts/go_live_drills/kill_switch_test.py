"""
Go-Live drill test: -15% drawdown kill-switch (Phase 4.4.4.1).

Standalone, stdlib-only drill. Exercises
`SignalsServer.risk_check` for the four canonical drawdown scenarios
defined by `docs/GO_LIVE_CHECKLIST.md` section 4.4.4.1:

  1. drawdown = -15.5%, BUY  -> blocked (drawdown_circuit_breaker)
  2. drawdown = -14.5%, BUY  -> allowed
  3. drawdown = -15.0%, BUY  -> blocked (inclusive boundary)
  4. drawdown = -15.5%, SELL -> allowed (de-risking is never blocked)

Run from the repo root:

    python scripts/go_live_drills/kill_switch_test.py

Exit code 0 on PASS, 1 on any failure. The script is deliberately
stdlib-only and loads `signals_server.py` directly by file path via
`importlib.util`, bypassing the `backend.agents.mcp_servers` package
__init__ (which eagerly imports `data_server` and `backtest_server`
and would pull in FastAPI / GCP libs that the drill intentionally
does not require). This keeps the drill runnable in the remote
agent environment without the full backend .venv.

The `-15.0` drawdown threshold is hardcoded in
`SignalsServer.get_risk_constraints` (Phase 4.4.4.4 evidence,
commit 9b0e943), so the drill exercises the same literal that
flows into `risk_check` on every call.
"""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SIGNALS_SERVER_PATH = REPO_ROOT / "backend" / "agents" / "mcp_servers" / "signals_server.py"


def load_signals_server():
    """Load signals_server.py as an isolated module by file path."""
    spec = importlib.util.spec_from_file_location(
        "signals_server_drill", str(SIGNALS_SERVER_PATH)
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not create module spec for signals_server.py at "
            + str(SIGNALS_SERVER_PATH)
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_portfolio(drawdown_pct: float, positions=None):
    return {
        "total_value": 10000.0,
        "cash": 10000.0,
        "positions": positions or {},
        "trades_today": [],
        "current_drawdown_pct": drawdown_pct,
    }


def scenario_1_deep_drawdown_blocks_buy(server):
    portfolio = _base_portfolio(-15.5)
    trade = {"ticker": "AAPL", "action": "BUY", "shares": 1, "price": 100.0}
    resp = server.risk_check(portfolio, trade)
    assert resp["allowed"] is False, (
        "S1 expected allowed=False at dd=-15.5 BUY, got " + repr(resp)
    )
    assert "drawdown_circuit_breaker" in resp["conflicts"], (
        "S1 expected drawdown_circuit_breaker in conflicts, got "
        + repr(resp["conflicts"])
    )
    return "S1 dd=-15.5 BUY -> blocked (drawdown_circuit_breaker)"


def scenario_2_shallow_drawdown_allows_buy(server):
    portfolio = _base_portfolio(-14.5)
    trade = {"ticker": "AAPL", "action": "BUY", "shares": 1, "price": 100.0}
    resp = server.risk_check(portfolio, trade)
    assert resp["allowed"] is True, (
        "S2 expected allowed=True at dd=-14.5 BUY, got " + repr(resp)
    )
    assert resp["conflicts"] == [], (
        "S2 expected empty conflicts, got " + repr(resp["conflicts"])
    )
    return "S2 dd=-14.5 BUY -> allowed"


def scenario_3_exact_boundary_blocks_buy(server):
    # Inclusive boundary: risk_check uses `current_dd <= max_drawdown_pct`
    # (signals_server.py line 896). dd == -15.0 must block.
    portfolio = _base_portfolio(-15.0)
    trade = {"ticker": "AAPL", "action": "BUY", "shares": 1, "price": 100.0}
    resp = server.risk_check(portfolio, trade)
    assert resp["allowed"] is False, (
        "S3 expected allowed=False at inclusive boundary dd=-15.0 BUY, got "
        + repr(resp)
    )
    assert "drawdown_circuit_breaker" in resp["conflicts"], (
        "S3 expected drawdown_circuit_breaker in conflicts at boundary, got "
        + repr(resp["conflicts"])
    )
    return "S3 dd=-15.0 BUY -> blocked (inclusive boundary pin)"


def scenario_4_deep_drawdown_allows_sell(server):
    # De-risking is never blocked by the drawdown breaker
    # (see signals_server.py line 895 comment: "SELLs are still allowed").
    positions = {"AAPL": {"shares": 10, "price": 100.0}}
    portfolio = _base_portfolio(-15.5, positions=positions)
    trade = {"ticker": "AAPL", "action": "SELL", "shares": 5, "price": 100.0}
    resp = server.risk_check(portfolio, trade)
    assert resp["allowed"] is True, (
        "S4 expected allowed=True at dd=-15.5 SELL, got " + repr(resp)
    )
    return "S4 dd=-15.5 SELL -> allowed (de-risking always permitted)"


def main():
    module = load_signals_server()
    server = module.SignalsServer()

    # Sanity check: the hardcoded threshold is still -15.0.
    limits = server.get_risk_constraints()
    threshold = limits.get("max_drawdown_pct")
    assert threshold == -15.0, (
        "pre-drill sanity: expected max_drawdown_pct=-15.0 (Phase 4.4.4.4 "
        "evidence), got " + repr(threshold)
    )

    scenarios = [
        scenario_1_deep_drawdown_blocks_buy,
        scenario_2_shallow_drawdown_allows_buy,
        scenario_3_exact_boundary_blocks_buy,
        scenario_4_deep_drawdown_allows_sell,
    ]

    failures = []
    for scenario in scenarios:
        try:
            line = scenario(server)
            print("PASS " + line)
        except AssertionError as exc:
            failures.append((scenario.__name__, str(exc)))
            print("FAIL " + scenario.__name__ + ": " + str(exc))

    if failures:
        print("DRILL FAIL: " + str(len(failures)) + " scenario(s) failed")
        return 1
    print("DRILL PASS: 4/4 kill-switch scenarios verified against "
          + "SignalsServer.risk_check (threshold=" + repr(threshold) + ")")
    return 0


if __name__ == "__main__":
    sys.exit(main())
