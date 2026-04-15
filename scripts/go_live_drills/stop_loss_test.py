"""
Go-Live drill test: -8% stop-loss SELL emission (Phase 4.4.4.3).

Standalone, stdlib-only drill. Exercises
`portfolio_manager.decide_trades` for the canonical stop-loss exit
scenarios defined by `docs/GO_LIVE_CHECKLIST.md` section 4.4.4.3:

  1. entry=100, stop=92, current=91.5 (-8.5%)  -> SELL reason=stop_loss
  2. entry=100, stop=92, current=92.0 (-8.0%)  -> SELL (inclusive boundary)
  3. entry=100, stop=92, current=93.0 (-7.0%)  -> no stop SELL (above stop)
  4. stop_loss_price=None, current=50.0        -> no stop SELL (stop unset)
  5. current=91 with re-eval rec=BUY           -> SELL (stop precedence)
  6. current=91.5 with holding_analyses=[]     -> SELL (no re-eval needed)

Run from the repo root:

    python scripts/go_live_drills/stop_loss_test.py

Exit code 0 on PASS, 1 on any failure. The script is deliberately
stdlib-only. It loads `portfolio_manager.py` directly by file path via
`importlib.util`, pre-registering stub modules for `backend`,
`backend.config`, and `backend.config.settings` in `sys.modules` so the
`from backend.config.settings import Settings` line inside
`portfolio_manager.py` resolves without requiring the real
`pydantic_settings` dependency. This keeps the drill runnable in the
remote agent environment without the full backend .venv.

The stop-loss exit logic lives in `portfolio_manager.decide_trades`
at lines 73-85: for every current position, if
`stop_loss_price is truthy and current_price <= stop_loss_price`,
the function appends a `TradeOrder(ticker, action="SELL",
reason="stop_loss", price=current)` and `continue`s past the
signal-classification block. This is the canonical stop-loss exit
path consumed by `autonomous_loop.py` -> `paper_trader.execute_sell`.

Stop prices are absolute dollar values, set at BUY time from
`risk_assessment.risk_limits.stop_loss` or `.stop_loss_pct`. A
position opened at `entry=100.0` with an 8% stop therefore has
`stop_loss_price=92.0`, and a -8.5% move puts
`current_price=91.5 <= 92.0`, triggering the SELL. The drill
exercises the same branch the paper-trading loop executes on every
tick.
"""

import importlib.util
import inspect
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PORTFOLIO_MANAGER_PATH = (
    REPO_ROOT / "backend" / "services" / "portfolio_manager.py"
)


def _register_backend_settings_stub():
    """Register minimal backend.config.settings stub in sys.modules.

    portfolio_manager.py does `from backend.config.settings import Settings`.
    The real backend.config.settings module imports pydantic_settings, which
    is not always present in the remote agent environment. The drill only
    needs Settings as a type hint; it passes a SimpleNamespace with the 3
    required attributes as the actual settings arg to decide_trades().
    """

    class Settings:  # minimal stub; only used as a type-hint target
        pass

    backend_mod = types.ModuleType("backend")
    backend_mod.__path__ = []  # mark as package
    config_mod = types.ModuleType("backend.config")
    config_mod.__path__ = []
    settings_mod = types.ModuleType("backend.config.settings")
    settings_mod.Settings = Settings

    sys.modules.setdefault("backend", backend_mod)
    sys.modules.setdefault("backend.config", config_mod)
    sys.modules.setdefault("backend.config.settings", settings_mod)


def load_portfolio_manager():
    """Load portfolio_manager.py as an isolated module by file path."""
    _register_backend_settings_stub()
    spec = importlib.util.spec_from_file_location(
        "portfolio_manager_drill", str(PORTFOLIO_MANAGER_PATH)
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not create module spec for portfolio_manager.py at "
            + str(PORTFOLIO_MANAGER_PATH)
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stub_settings():
    """Minimal settings object -- only the 3 fields decide_trades reads."""
    ns = types.SimpleNamespace()
    ns.paper_starting_capital = 10000.0
    ns.paper_min_cash_reserve_pct = 5.0
    ns.paper_max_positions = 10
    return ns


def _base_portfolio_state(cash=10000.0, nav=10000.0):
    return {
        "nav": nav,
        "cash": cash,
        "positions_value": 0.0,
        "position_count": 0,
    }


def _position(current_price, stop_loss_price, ticker="AAPL"):
    """Build a current position dict with the fields decide_trades reads."""
    return {
        "ticker": ticker,
        "quantity": 10.0,
        "avg_entry_price": 100.0,
        "cost_basis": 1000.0,
        "current_price": current_price,
        "market_value": round(10.0 * current_price, 2),
        "unrealized_pnl": round(10.0 * (current_price - 100.0), 2),
        "unrealized_pnl_pct": round((current_price - 100.0), 2),
        "entry_date": "2026-04-01T00:00:00+00:00",
        "last_analysis_date": "2026-04-15T00:00:00+00:00",
        "recommendation": "BUY",
        "risk_judge_position_pct": 10.0,
        "stop_loss_price": stop_loss_price,
    }


def scenario_1_stop_breach_emits_sell(pm):
    """-8.5% from entry with 8% stop -> SELL reason=stop_loss."""
    position = _position(current_price=91.5, stop_loss_price=92.0)
    orders = pm.decide_trades(
        current_positions=[position],
        candidate_analyses=[],
        holding_analyses=[],
        portfolio_state=_base_portfolio_state(),
        settings=_stub_settings(),
    )
    sells = [o for o in orders if o.action == "SELL"]
    buys = [o for o in orders if o.action == "BUY"]
    assert len(sells) == 1, (
        "S1 expected exactly 1 SELL, got "
        + repr([(o.action, o.reason) for o in orders])
    )
    assert sells[0].ticker == "AAPL", (
        "S1 expected ticker=AAPL, got " + repr(sells[0].ticker)
    )
    assert sells[0].reason == "stop_loss", (
        "S1 expected reason=stop_loss, got " + repr(sells[0].reason)
    )
    assert sells[0].price == 91.5, (
        "S1 expected price=91.5, got " + repr(sells[0].price)
    )
    assert buys == [], (
        "S1 expected zero BUY orders, got " + repr(buys)
    )
    return (
        "S1 entry=100 stop=92 current=91.5 (-8.5%) "
        "-> SELL reason=stop_loss price=91.5"
    )


def scenario_2_inclusive_boundary_emits_sell(pm):
    """current == stop -> SELL triggered (<= semantic pin)."""
    position = _position(current_price=92.0, stop_loss_price=92.0)
    orders = pm.decide_trades(
        current_positions=[position],
        candidate_analyses=[],
        holding_analyses=[],
        portfolio_state=_base_portfolio_state(),
        settings=_stub_settings(),
    )
    stop_sells = [o for o in orders if o.reason == "stop_loss"]
    assert len(stop_sells) == 1, (
        "S2 expected 1 stop_loss SELL at inclusive boundary, got "
        + repr([(o.action, o.reason) for o in orders])
    )
    assert stop_sells[0].action == "SELL"
    assert stop_sells[0].price == 92.0
    return (
        "S2 current=92.0 == stop=92.0 "
        "-> SELL reason=stop_loss (inclusive boundary pin)"
    )


def scenario_3_above_stop_no_sell(pm):
    """Price above stop -> no stop-loss SELL."""
    position = _position(current_price=93.0, stop_loss_price=92.0)
    orders = pm.decide_trades(
        current_positions=[position],
        candidate_analyses=[],
        holding_analyses=[],
        portfolio_state=_base_portfolio_state(),
        settings=_stub_settings(),
    )
    stop_sells = [o for o in orders if o.reason == "stop_loss"]
    assert stop_sells == [], (
        "S3 expected zero stop_loss SELLs at current=93 > stop=92, got "
        + repr([(o.action, o.reason) for o in orders])
    )
    return "S3 current=93.0 > stop=92.0 -> no stop-loss SELL"


def scenario_4_no_stop_price_is_safe(pm):
    """stop_loss_price=None -> no trigger even at -50%."""
    position = _position(current_price=50.0, stop_loss_price=None)
    orders = pm.decide_trades(
        current_positions=[position],
        candidate_analyses=[],
        holding_analyses=[],
        portfolio_state=_base_portfolio_state(),
        settings=_stub_settings(),
    )
    stop_sells = [o for o in orders if o.reason == "stop_loss"]
    assert stop_sells == [], (
        "S4 expected zero stop_loss SELLs with stop_loss_price=None, got "
        + repr([(o.action, o.reason) for o in orders])
    )
    return (
        "S4 stop_loss_price=None current=50.0 (-50%) "
        "-> no stop-loss SELL (no stop set)"
    )


def scenario_5_stop_precedes_re_eval_signal(pm):
    """Stop fires even when re-eval says BUY (precedence pin)."""
    position = _position(current_price=91.0, stop_loss_price=92.0)
    # Re-eval says BUY, the strongest pro-hold signal.
    analysis = {
        "ticker": "AAPL",
        "recommendation": "BUY",
        "analysis_date": "2026-04-15",
    }
    orders = pm.decide_trades(
        current_positions=[position],
        candidate_analyses=[],
        holding_analyses=[analysis],
        portfolio_state=_base_portfolio_state(),
        settings=_stub_settings(),
    )
    stop_sells = [o for o in orders if o.reason == "stop_loss"]
    signal_sells = [
        o for o in orders if o.reason in ("sell_signal", "signal_downgrade")
    ]
    assert len(stop_sells) == 1, (
        "S5 expected 1 stop_loss SELL with re-eval BUY, got "
        + repr([(o.action, o.reason) for o in orders])
    )
    assert stop_sells[0].action == "SELL"
    assert signal_sells == [], (
        "S5 expected zero signal-based sells (stop precedence), got "
        + repr(signal_sells)
    )
    return (
        "S5 current=91 stop=92 re-eval=BUY "
        "-> SELL reason=stop_loss (precedence over re-eval)"
    )


def scenario_6_stop_without_re_eval(pm):
    """Stop fires with zero holding_analyses (no re-eval needed)."""
    position = _position(current_price=91.5, stop_loss_price=92.0)
    orders = pm.decide_trades(
        current_positions=[position],
        candidate_analyses=[],
        holding_analyses=[],  # no re-evaluation this tick
        portfolio_state=_base_portfolio_state(),
        settings=_stub_settings(),
    )
    stop_sells = [o for o in orders if o.reason == "stop_loss"]
    assert len(stop_sells) == 1, (
        "S6 expected 1 stop_loss SELL with zero holding_analyses, got "
        + repr([(o.action, o.reason) for o in orders])
    )
    return (
        "S6 current=91.5 stop=92 holding_analyses=[] "
        "-> SELL reason=stop_loss (no re-eval required)"
    )


def main():
    pm = load_portfolio_manager()

    # Pre-drill sanity check 1: decide_trades signature hasn't drifted.
    sig = inspect.signature(pm.decide_trades)
    expected_params = {
        "current_positions",
        "candidate_analyses",
        "holding_analyses",
        "portfolio_state",
        "settings",
    }
    actual_params = set(sig.parameters.keys())
    assert actual_params == expected_params, (
        "pre-drill sanity: decide_trades params drifted from "
        + repr(sorted(expected_params))
        + " to "
        + repr(sorted(actual_params))
    )

    # Pre-drill sanity check 2: TradeOrder.reason field exists and is str.
    assert "reason" in pm.TradeOrder.__dataclass_fields__, (
        "pre-drill sanity: TradeOrder.reason dataclass field missing"
    )
    assert "action" in pm.TradeOrder.__dataclass_fields__, (
        "pre-drill sanity: TradeOrder.action dataclass field missing"
    )
    assert "ticker" in pm.TradeOrder.__dataclass_fields__, (
        "pre-drill sanity: TradeOrder.ticker dataclass field missing"
    )
    assert "price" in pm.TradeOrder.__dataclass_fields__, (
        "pre-drill sanity: TradeOrder.price dataclass field missing"
    )

    scenarios = [
        scenario_1_stop_breach_emits_sell,
        scenario_2_inclusive_boundary_emits_sell,
        scenario_3_above_stop_no_sell,
        scenario_4_no_stop_price_is_safe,
        scenario_5_stop_precedes_re_eval_signal,
        scenario_6_stop_without_re_eval,
    ]

    failures = []
    for scenario in scenarios:
        try:
            line = scenario(pm)
            print("PASS " + line)
        except AssertionError as exc:
            failures.append((scenario.__name__, str(exc)))
            print("FAIL " + scenario.__name__ + ": " + str(exc))

    if failures:
        print("DRILL FAIL: " + str(len(failures)) + " scenario(s) failed")
        return 1
    print(
        "DRILL PASS: 6/6 stop-loss scenarios verified against "
        "portfolio_manager.decide_trades "
        "(stop semantic: current_price <= stop_loss_price)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
