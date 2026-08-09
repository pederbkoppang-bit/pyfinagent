"""phase-36.17 -- a halted cycle must still enforce PRE-EXISTING stop-losses.

`run_daily_cycle` halts at Step 5.5 and `return summary`, and Step 5.6 -- which calls
`trader.backfill_missing_stops()` then `trader.check_stop_losses()` -- begins AFTER
that return. So on any halted cycle the protective-exit machinery never runs.

LINE ANCHORS MOVE. Do not trust the numbers below; re-derive them with
`grep -n "Step 5.6: Stop-loss enforcement" backend/services/autonomous_loop.py`.
  * masterplan step text  : :1334/:1336  -- STALE, two generations old
  * pre-fix tree          : :1437/:1439
  * post-fix (this change adds +70 lines above Step 5.6): :1507/:1509
The first revision of this docstring reproduced the pre-fix pair while claiming to be
"re-derived", which the Q/A caught -- the same class of defect this step's own
criterion 6 exists to prevent.

On the `breach` path that is nearly moot: `check_and_enforce_kill_switch` has already
called `flatten_all`, so there is little left to stop out. On the `paused` and
`blocked` paths there is NO flatten -- `kill_switch.py:14` documents "Pause = halt new
entries; existing positions kept" -- so the book keeps full exposure with its stop-loss
enforcement switched off, every cycle until the halt clears. `blocked` is
NON-LATCHING, so it can recur indefinitely with no operator resume required.

SEVERITY: BEFORE this change `check_stop_losses` had exactly ONE production caller --
the Step 5.6 block. No API route, no scheduler job, no MCP tool; the cycle was the sole
means of stop enforcement, so a halted cycle meant none at all. THIS FIX ADDS THE SECOND
CALLER, so the count is now two. Re-derive rather than trusting this line:
`grep -rn check_stop_losses backend --include="*.py" | grep -v /tests/`.

OPERATOR DECISION (2026-08-09): option (b) -- a stop-loss-only, SELL-only pass inside
the halt branch, scoped to `paused` and `blocked`, EXCLUDING `backfill_missing_stops`,
with `return summary` still last. This is NYSE Pillar's "Block only -- accept cancels"
and ESMA para 93's middle rung. `backfill_missing_stops` is excluded because
SYNTHESIZING a stop level is a new risk decision (ESMA para 11(5)) that could land above
the current mark and flatten a paused book by side effect.

Contract: handoff/current/contract_36.17.md
Research: handoff/current/research_brief_36.17.md

ISOLATION IS MANDATORY IN THIS MODULE -- ported in intent from
test_phase_36_12_kill_switch_trading_path_block.py:
  * `_live_audit_file_is_write_protected` -- `_append_audit` writes to the LIVE
    safety file `handoff/kill_switch_audit.jsonl` and SWALLOWS write errors, so a
    byte comparison is the only reliable detector.
  * `captured_alerts` -- `raise_cron_alert_sync` posts to the operator's REAL Slack
    (alerting.py:167) with no test guard. Measured on 36.12's first runs: an unstubbed
    module pages a false "Kill-switch DISARMED" P1.
  * `cycle_health.get_log` stub -- `handoff/.cycle_heartbeat.json` and
    `handoff/cycle_history.jsonl` are GIT-TRACKED.
  * `settings.news_screen_enabled = False` + mocked `AnalysisOrchestrator` -- an
    unstubbed run makes a REAL ~150s LLM call and bills the operator.
"""
from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


# ── fixtures (isolation -- see module docstring) ────────────────────────────


@pytest.fixture(autouse=True)
def _live_audit_file_is_write_protected():
    live = REPO_ROOT / "handoff" / "kill_switch_audit.jsonl"
    before = live.read_bytes() if live.exists() else None
    yield
    after = live.read_bytes() if live.exists() else None
    assert after == before, (
        f"phase-36.17: a test in this module wrote to the LIVE audit trail {live}. "
        "Redirect ks._AUDIT_PATH to tmp_path."
    )


@pytest.fixture(autouse=True)
def captured_alerts(monkeypatch):
    calls: list[dict] = []

    def _capture(**kwargs):
        calls.append(kwargs)
        return True

    import backend.services.observability.alerting as alerting

    monkeypatch.setattr(alerting, "raise_cron_alert_sync", _capture)
    return calls


def _detached_state(ks):
    """A KillSwitchState with no boot replay and no baselines."""
    fresh = object.__new__(ks.KillSwitchState)
    fresh._lock = threading.Lock()
    fresh._paused = False
    fresh._pause_reason = None
    fresh._sod_nav = None
    fresh._sod_date = None
    fresh._peak_nav = None
    fresh._paused_at = None
    return fresh


@pytest.fixture
def ks_isolated(tmp_path, monkeypatch):
    import backend.services.kill_switch as ks

    live = tmp_path / "handoff" / "kill_switch_audit.jsonl"
    live.parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "handoff" / "audit").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(ks, "_AUDIT_PATH", live)

    state = _detached_state(ks)
    monkeypatch.setattr(ks, "_state", state)
    monkeypatch.setattr(ks, "get_state", lambda: state)
    monkeypatch.setattr(ks, "_disarmed_logged", False)
    yield ks, state, live


# ── the harness ────────────────────────────────────────────────────────────

# The position that is ALREADY below its stop. `check_stop_losses` is the seam that
# decides this; we stub its return so the test can never be VACUOUS. The brief
# measured that a bare MagicMock yields `list(MagicMock()) == []` -- 0 iterations --
# which is exactly why 36.12's `execute_sell.called is False` assertions stay green
# even under option (b), and why they must NOT be relied on as coverage here.
BREACHED_TICKER = "WDC"


def _drive_cycle(monkeypatch, ks_check, *, paused: bool, ks_state):
    """Drive the REAL `run_daily_cycle` to the halt branch and return (summary, trader).

    Mock surface copied from test_phase_36_12...py:248-289.
    """
    import backend.services.autonomous_loop as al
    import backend.services.cycle_health as cycle_health
    from backend.config.settings import Settings

    if paused:
        ks_state._paused = True
        ks_state._pause_reason = "manual"

    portfolio = {
        "total_nav": 18_000.0,
        "starting_capital": 20_000.0,
        "current_cash": 18_000.0,
    }
    open_position = {
        "ticker": BREACHED_TICKER,
        "quantity": 100,
        "avg_entry_price": 50.0,
        "stop_loss_price": 46.0,
        "current_price": 41.0,  # BELOW the stop -- this position should be exited
    }

    trader = MagicMock()
    trader.check_and_enforce_kill_switch.return_value = ks_check
    trader.mark_to_market.return_value = {"nav": 18_000.0, "positions": [open_position]}
    trader.get_or_create_portfolio.return_value = portfolio
    # The seam under test: this position IS below its stop.
    trader.check_stop_losses.return_value = [BREACHED_TICKER]
    trader.execute_sell.return_value = {"ticker": BREACHED_TICKER, "price": 41.0}

    bq = MagicMock()
    bq.get_paper_positions.return_value = [open_position]
    bq.get_paper_portfolio.return_value = portfolio

    settings = Settings()
    settings.news_screen_enabled = False  # COST hazard -- see module docstring

    monkeypatch.setattr(al, "BigQueryClient", lambda *a, **k: bq)
    monkeypatch.setattr(al, "PaperTrader", lambda *a, **k: trader)
    monkeypatch.setattr(al, "screen_universe", lambda *a, **k: [])
    monkeypatch.setattr(al, "rank_candidates", lambda *a, **k: [])
    monkeypatch.setattr(al, "get_sp500_tickers", lambda *a, **k: [])
    monkeypatch.setattr(al, "get_russell1000_tickers", lambda *a, **k: [])
    monkeypatch.setattr(al, "_log_cycle_signals_to_bq", lambda *a, **k: 0)
    monkeypatch.setattr(al, "AnalysisOrchestrator", MagicMock())
    # decide_trades returns a REAL TradeOrder BUY, not [] and not a dict, so that
    # `execute_buy.called is False` is INDEPENDENTLY FALSIFIABLE (criterion 4).
    #
    # Q/A cycle 1 caught this as vacuity shape #11: with the original
    # `return_value=[]`, execute_buy was structurally unreachable even on a full
    # fall-through, so the no-BUY assertion could never fail and the invariant was
    # actually carried by its sibling `decide.called is False`. MEASURED during the
    # cycle-2 fix: a dict-shaped order does NOT restore falsifiability either --
    # autonomous_loop.py:1711 accesses `order.action` as an ATTRIBUTE, so a dict is
    # skipped before ever reaching execute_buy. Only the real dataclass works.
    #
    # `_get_live_price` must be patched too: :1713 fetches a LIVE price via yfinance
    # before the buy, which would make this test do real network I/O.
    from backend.services.portfolio_manager import TradeOrder
    import backend.services.paper_trader as _pt

    monkeypatch.setattr(_pt, "_get_live_price", lambda *a, **k: 100.0)
    decide = MagicMock(return_value=[
        TradeOrder(
            ticker="NVDA",
            action="BUY",
            amount_usd=1_000.0,
            reason="test_probe_must_never_execute",
            price=100.0,
            price_at_analysis=100.0,
        )
    ])
    monkeypatch.setattr(al, "decide_trades", decide)
    monkeypatch.setattr(al, "_running", False)
    fake_log = MagicMock()
    fake_log.record_cycle_start.return_value = "2026-01-01T00:00:00+00:00"
    monkeypatch.setattr(cycle_health, "get_log", lambda *a, **k: fake_log)

    summary = asyncio.run(al.run_daily_cycle(settings=settings))
    return summary, trader, decide


KS_BLOCKED = {
    "triggered": False,
    "blocked": True,
    "block_reason": "kill_switch_disarmed_lost_history",
    "pre_armed": False,
    "breach": {"any_breached": False, "armed": True},
    "auto_resume": {"action": "no_op"},
}

KS_QUIET = {
    "triggered": False,
    "blocked": False,
    "breach": {"any_breached": False, "armed": True},
    "auto_resume": {"action": "no_op"},
}

KS_TRIGGERED = {
    "triggered": True,
    "blocked": False,
    "breach": {"any_breached": True, "armed": True},
    "auto_resume": {"action": "no_op"},
}


# ── REPRODUCE FIRST (criteria 1 + 2) ───────────────────────────────────────
#
# These two tests assert the DEFECT as it exists BEFORE the fix. They are the
# "record it verbatim" half of criteria 1 and 2. After the fix lands they are
# INVERTED (see the *_enforces_stops tests below) -- this block is kept in the
# file's history, not in the file, so the suite never carries a test that
# asserts the broken behaviour is correct.
#
# VERBATIM REPRODUCTION, recorded 2026-08-09 BEFORE the fix (criteria 1 + 2).
# Two temporary tests asserted the defect directly:
#   test_REPRODUCE_paused_cycle_never_checks_stops   -- KS_QUIET + state._paused
#   test_REPRODUCE_blocked_cycle_never_checks_stops  -- KS_BLOCKED
# each asserting `trader.check_stop_losses.called is False` and
# `trader.execute_sell.called is False` with a position priced 41.0 against a
# 46.0 stop. Measured result on the pre-fix tree:
#     2 failed, 4 passed   <- the 2 REPRODUCE tests PASSED (defect real);
#                             the 2 *_enforces_stops tests FAILED with
#                             "AssertionError: a blocked cycle did not check
#                              stop-losses (phase-36.17) / assert False is True"
# and on the post-fix tree the SAME file inverts exactly:
#     2 failed, 4 passed   <- the 2 REPRODUCE tests now FAIL, enforcement passes.
# Full output is transcribed in handoff/current/experiment_results_36.17.md.


# ── POST-FIX BEHAVIOUR (criteria 1, 2, 4) ──────────────────────────────────


def test_phase_36_17_paused_cycle_enforces_preexisting_stops(ks_isolated, monkeypatch):
    """PAUSED path: the book keeps its positions, so it must keep its stops.

    Before the fix this asserted the inverse (`check_stop_losses.called is False`)
    and PASSED -- that is the reproduction recorded in experiment_results_36.17.md.
    """
    ks, state, _live = ks_isolated
    summary, trader, decide = _drive_cycle(
        monkeypatch, KS_QUIET, paused=True, ks_state=state
    )

    assert summary["halted"] is True
    assert summary["halt_reason"] == "paused"

    # THE FIX: pre-existing stops are enforced even though the cycle halted.
    assert trader.check_stop_losses.called is True, (
        "a paused cycle did not check stop-losses -- the book is holding exposure "
        "with its protective exits switched off (phase-36.17)"
    )
    sold = [c for c in trader.execute_sell.call_args_list
            if c.kwargs.get("reason") == "stop_loss_trigger"]
    assert sold, "the breached position was not exited on a paused cycle"
    assert sold[0].kwargs["ticker"] == BREACHED_TICKER
    assert summary.get("halt_stop_loss_triggered") == [BREACHED_TICKER]

    # THE INVARIANT (criterion 4): a halted book still places NO new BUYs and
    # never reaches the decide/execute phase.
    assert trader.execute_buy.called is False, "a BUY was placed on a halted cycle"
    assert decide.called is False, "decide_trades ran on a halted cycle"

    # The halt must NOT be cleared, and the backfill must NOT run (a synthesized
    # stop is a new risk decision that could flatten a paused book by side effect).
    assert trader.backfill_missing_stops.called is False, (
        "backfill_missing_stops ran during a halt -- synthesizing a stop level is a "
        "NEW risk decision and can flatten a paused book by side effect"
    )
    assert state._paused is True, "the halt was cleared by the stop-loss pass"


def test_phase_36_17_blocked_cycle_enforces_preexisting_stops(ks_isolated, monkeypatch):
    """BLOCKED path (phase-36.12's third halt reason) -- proves the fix covers all
    three halt reasons, not only the one that prompted it (criterion 2).

    This path matters MORE than `paused`: the switch is not paused, so
    `execute_buy`'s own pause gate returns None and BUY suppression comes ONLY from
    the `return summary`. And `blocked` is NON-LATCHING, so it recurs every cycle.
    """
    ks, state, _live = ks_isolated
    summary, trader, decide = _drive_cycle(
        monkeypatch, KS_BLOCKED, paused=False, ks_state=state
    )

    assert summary["halted"] is True
    assert summary["halt_reason"] == "kill_switch_disarmed_lost_history"

    assert trader.check_stop_losses.called is True, (
        "a blocked cycle did not check stop-losses (phase-36.17)"
    )
    sold = [c for c in trader.execute_sell.call_args_list
            if c.kwargs.get("reason") == "stop_loss_trigger"]
    assert sold, "the breached position was not exited on a blocked cycle"
    assert summary.get("halt_stop_loss_triggered") == [BREACHED_TICKER]

    assert trader.execute_buy.called is False, "a BUY was placed on a blocked cycle"
    assert decide.called is False, "decide_trades ran on a blocked cycle"
    assert trader.backfill_missing_stops.called is False

    # The block must still SURVIVE into the published summary.
    published = summary.get("kill_switch") or {}
    assert published.get("blocked") is True


# ── criterion 5: the `triggered` path is observably UNCHANGED ──────────────


def test_phase_36_17_triggered_path_is_unchanged(ks_isolated, monkeypatch):
    """`breach` already called flatten_all, so the exit-only pass must NOT run there.

    Asserted against a fixture rather than reasoned about (criterion 5). Running a
    stop pass after a flatten would duplicate exits, fee events and learn-loop rows
    over positions that no longer exist.
    """
    ks, state, _live = ks_isolated
    summary, trader, decide = _drive_cycle(
        monkeypatch, KS_TRIGGERED, paused=False, ks_state=state
    )

    assert summary["halted"] is True
    assert summary["halt_reason"] == "breach"
    assert trader.check_stop_losses.called is False, (
        "the exit-only pass ran on the `breach` path, which has already flattened -- "
        "that duplicates exits and fee/learn-loop events (criterion 5)"
    )
    assert "halt_stop_loss_triggered" not in summary
    assert trader.execute_buy.called is False
    assert decide.called is False


# ── criterion 4 / collision guard: the halted summary shape is preserved ───


def test_phase_36_17_halt_summary_shape_is_preserved(ks_isolated, monkeypatch):
    """The exit-only pass must not disturb the halt's own bookkeeping.

    MEASURED COLLISION (research brief): appending to `summary["steps"]` inside the
    halt branch turns test_phase_36_12...:298 and :374 RED, because both assert
    `summary["steps"][-1] == "kill_switch_halted"`. The pass therefore records under
    a DISTINCT key. phase-85.4 also depends on `status` and `halt_reason`.
    """
    ks, state, _live = ks_isolated
    summary, trader, _decide = _drive_cycle(
        monkeypatch, KS_BLOCKED, paused=False, ks_state=state
    )

    assert summary["steps"][-1] == "kill_switch_halted", (
        f"the exit-only pass appended to summary['steps']: {summary['steps']}"
    )
    assert "stop_loss_enforcement" not in summary["steps"]
    assert summary["status"] == "halted_kill_switch"  # phase-85.4 depends on this
