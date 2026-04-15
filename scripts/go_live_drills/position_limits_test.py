"""
Go-Live drill test: position limits (Phase 4.4.4.2).

Standalone, stdlib-only drill. Exercises `SignalsServer.risk_check`
for the three position-limit thresholds named in
`docs/GO_LIVE_CHECKLIST.md` section 4.4.4.2:

  * per-ticker exposure cap (`max_exposure_per_ticker_pct = 10.0`)
  * total exposure cap     (`max_total_exposure_pct     = 100.0`)
  * daily trade count cap  (`max_daily_trades           = 5`)

Six canonical scenarios:

  S1. per-ticker breach      BUY AAPL 15 sh @ $100 -> 15%  -> blocked
                             (max_exposure_per_ticker)
  S2. per-ticker boundary    BUY AAPL 10 sh @ $100 -> 10%  -> allowed
                             (strict-greater pin: 10.00% must pass)
  S3. per-ticker aggregation existing AAPL 5 sh @ $100 + BUY 6 sh
                             -> 11% -> blocked (max_exposure_per_ticker)
  S4. total-exposure breach  existing MSFT 95 sh @ $100 (95% of eq)
                             + BUY AAPL 6 sh @ $100 -> 101% -> blocked
                             (max_total_exposure)
  S5. daily-trade cap breach trades_today=[5 stubs], BUY 1 sh @ $100
                             -> blocked (max_daily_trades)
  S6. daily-trade cap ok     trades_today=[4 stubs], BUY 1 sh @ $100
                             -> allowed

Run from the repo root:

    python scripts/go_live_drills/position_limits_test.py

Exit code 0 on PASS, 1 on any failure. The script is deliberately
stdlib-only and loads `signals_server.py` directly by file path via
`importlib.util`, bypassing the `backend.agents.mcp_servers` package
__init__ (which eagerly imports `data_server` and `backtest_server`
and would pull in FastAPI / GCP libs that the drill intentionally
does not require). Mirrors the Phase 4.4.4.1 kill-switch drill in
`kill_switch_test.py` -- copy-pasteable, independent, re-runnable.

The three limit literals are pinned by Phase 4.4.4.4 evidence
(commit 9b0e943) in `SignalsServer.get_risk_constraints`. The drill
runs a pre-drill sanity check confirming the literals are still
wired as the single source of truth for `risk_check`, so any future
drift is caught loudly here before the scenarios run.
"""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SIGNALS_SERVER_PATH = REPO_ROOT / "backend" / "agents" / "mcp_servers" / "signals_server.py"


def load_signals_server():
    """Load signals_server.py as an isolated module by file path."""
    spec = importlib.util.spec_from_file_location(
        "signals_server_drill_poslimits", str(SIGNALS_SERVER_PATH)
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not create module spec for signals_server.py at "
            + str(SIGNALS_SERVER_PATH)
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _portfolio(total_value=10000.0, cash=10000.0, positions=None, trades_today=None):
    return {
        "total_value": total_value,
        "cash": cash,
        "positions": positions or {},
        "trades_today": trades_today if trades_today is not None else [],
        "current_drawdown_pct": 0.0,
    }


def scenario_1_per_ticker_breach_blocks_buy(server):
    # 15 shares * $100 = $1500 notional on $10000 equity = 15% > 10%
    portfolio = _portfolio()
    trade = {"ticker": "AAPL", "action": "BUY", "shares": 15, "price": 100.0}
    resp = server.risk_check(portfolio, trade)
    assert resp["allowed"] is False, (
        "S1 expected allowed=False at per-ticker 15%, got " + repr(resp)
    )
    assert "max_exposure_per_ticker" in resp["conflicts"], (
        "S1 expected max_exposure_per_ticker in conflicts, got "
        + repr(resp["conflicts"])
    )
    return "S1 per-ticker 15% BUY -> blocked (max_exposure_per_ticker)"


def scenario_2_per_ticker_boundary_allows_buy(server):
    # 10 shares * $100 = $1000 on $10000 equity = exactly 10.00%.
    # risk_check uses strict `> max_per_ticker_pct` at line 872, so 10.00%
    # is allowed. This pins the boundary semantics against silent regressions
    # to `>=`.
    portfolio = _portfolio()
    trade = {"ticker": "AAPL", "action": "BUY", "shares": 10, "price": 100.0}
    resp = server.risk_check(portfolio, trade)
    assert resp["allowed"] is True, (
        "S2 expected allowed=True at exact 10.00% boundary, got " + repr(resp)
    )
    assert resp["conflicts"] == [], (
        "S2 expected empty conflicts at boundary, got " + repr(resp["conflicts"])
    )
    return "S2 per-ticker 10.00% boundary BUY -> allowed (strict-greater pin)"


def scenario_3_per_ticker_aggregation_blocks_buy(server):
    # Existing position AAPL 5 @ $100 = $500 (already 5% of equity).
    # Propose BUY 6 @ $100 = $600. Projected = $1100 = 11% > 10%.
    positions = {"AAPL": {"shares": 5, "price": 100.0}}
    portfolio = _portfolio(positions=positions)
    trade = {"ticker": "AAPL", "action": "BUY", "shares": 6, "price": 100.0}
    resp = server.risk_check(portfolio, trade)
    assert resp["allowed"] is False, (
        "S3 expected allowed=False at aggregated 11%, got " + repr(resp)
    )
    assert "max_exposure_per_ticker" in resp["conflicts"], (
        "S3 expected max_exposure_per_ticker in conflicts via aggregation, got "
        + repr(resp["conflicts"])
    )
    return "S3 per-ticker aggregation 5%+6% -> 11% BUY -> blocked"


def scenario_4_total_exposure_breach_blocks_buy(server):
    # Existing MSFT 95 @ $100 = $9500 (95% of equity, per-ticker limit OK
    # for MSFT but we are proposing a DIFFERENT ticker, so the MSFT 95%
    # does NOT trip the per-ticker check for AAPL).
    # Propose BUY AAPL 6 @ $100 = $600. AAPL per-ticker = 6% OK.
    # Total = 9500 + 600 = 10100 > 10000 -> 101% > 100% -> blocked.
    # Cash must be >= $600 to survive the cash floor check (set to 1000).
    positions = {"MSFT": {"shares": 95, "price": 100.0}}
    portfolio = _portfolio(cash=1000.0, positions=positions)
    trade = {"ticker": "AAPL", "action": "BUY", "shares": 6, "price": 100.0}
    resp = server.risk_check(portfolio, trade)
    assert resp["allowed"] is False, (
        "S4 expected allowed=False at total 101%, got " + repr(resp)
    )
    assert "max_total_exposure" in resp["conflicts"], (
        "S4 expected max_total_exposure in conflicts, got "
        + repr(resp["conflicts"])
    )
    return "S4 total exposure 95%+6% -> 101% BUY -> blocked (max_total_exposure)"


def scenario_5_daily_trade_cap_blocks_buy(server):
    # trades_today list of length 5 -- production shape. risk_check line 846
    # uses `>= max_daily_trades`, so count == 5 blocks.
    # Use a trivially-sized trade (1 share @ $100 = $100) so the daily-cap
    # branch fires before any exposure branch.
    trades_today = [
        {"ticker": "T1", "action": "BUY"},
        {"ticker": "T2", "action": "BUY"},
        {"ticker": "T3", "action": "SELL"},
        {"ticker": "T4", "action": "BUY"},
        {"ticker": "T5", "action": "SELL"},
    ]
    portfolio = _portfolio(trades_today=trades_today)
    trade = {"ticker": "AAPL", "action": "BUY", "shares": 1, "price": 100.0}
    resp = server.risk_check(portfolio, trade)
    assert resp["allowed"] is False, (
        "S5 expected allowed=False at trades_today=5, got " + repr(resp)
    )
    assert "max_daily_trades" in resp["conflicts"], (
        "S5 expected max_daily_trades in conflicts, got "
        + repr(resp["conflicts"])
    )
    return "S5 daily trade count 5 BUY -> blocked (max_daily_trades)"


def scenario_6_daily_trade_cap_allows_buy(server):
    # trades_today list of length 4 -- one trade under the cap. Same
    # trivially-sized trade as S5. Should pass every check.
    trades_today = [
        {"ticker": "T1", "action": "BUY"},
        {"ticker": "T2", "action": "BUY"},
        {"ticker": "T3", "action": "SELL"},
        {"ticker": "T4", "action": "BUY"},
    ]
    portfolio = _portfolio(trades_today=trades_today)
    trade = {"ticker": "AAPL", "action": "BUY", "shares": 1, "price": 100.0}
    resp = server.risk_check(portfolio, trade)
    assert resp["allowed"] is True, (
        "S6 expected allowed=True at trades_today=4, got " + repr(resp)
    )
    assert resp["conflicts"] == [], (
        "S6 expected empty conflicts under cap, got " + repr(resp["conflicts"])
    )
    return "S6 daily trade count 4 BUY -> allowed (under cap)"


def main():
    module = load_signals_server()
    server = module.SignalsServer()

    # Pre-drill sanity: pin all 4 limit literals against the 4.4.4.4 evidence.
    # Any drift fails loudly before the scenarios run.
    limits = server.get_risk_constraints()
    expected = {
        "max_exposure_per_ticker_pct": 10.0,
        "max_total_exposure_pct": 100.0,
        "max_drawdown_pct": -15.0,
        "max_daily_trades": 5,
    }
    for key, want in expected.items():
        got = limits.get(key)
        assert got == want, (
            "pre-drill sanity: expected " + key + "=" + repr(want)
            + " (Phase 4.4.4.4 evidence), got " + repr(got)
        )

    scenarios = [
        scenario_1_per_ticker_breach_blocks_buy,
        scenario_2_per_ticker_boundary_allows_buy,
        scenario_3_per_ticker_aggregation_blocks_buy,
        scenario_4_total_exposure_breach_blocks_buy,
        scenario_5_daily_trade_cap_blocks_buy,
        scenario_6_daily_trade_cap_allows_buy,
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
    print(
        "DRILL PASS: 6/6 position-limit scenarios verified against "
        "SignalsServer.risk_check (per-ticker=10.0, total=100.0, "
        "daily_trades=5)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
