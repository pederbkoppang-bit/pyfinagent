"""phase-36.17 -- a halted cycle must still enforce PRE-EXISTING stop-losses.

`run_daily_cycle` halts at Step 5.5 and `return summary`, and Step 5.6 -- which calls
`trader.backfill_missing_stops()` then `trader.check_stop_losses()` -- begins AFTER
that return. So on any halted cycle the protective-exit machinery never runs.

LINE ANCHORS MOVE, SO THIS DOCSTRING DELIBERATELY CITES NO CURRENT ONES.
Re-derive with:
    grep -n "Step 5.6: Stop-loss enforcement" backend/services/autonomous_loop.py
    grep -n "return summary" backend/services/autonomous_loop.py | head -1
The halt's `return summary` immediately precedes the Step 5.6 header; that RELATION
is the invariant, and it is asserted behaviourally by the tests below rather than by
any number.

Historical anchors, kept only to show how far they have drifted -- all STALE:
the masterplan step text says :1334/:1336; the pre-fix tree was :1437/:1439.
Two successive Q/A cycles FAILED this module's artifacts on exactly this defect --
numbers derived before the final edit and shipped without re-derivation. Hence the
rule now applied here: cite the command, never the number.

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
    grep -rn "trader.check_stop_losses" backend --include="*.py" | grep -v /tests/

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


# ── criterion 7 gaps found by Q/A cycle 3 -- three mutants SURVIVED the ──────
# author's 6-cell matrix. These close them. Each was written by taking the
# surviving mutant and asking "what assertion would have killed this?".


def test_phase_36_17_a_failed_sell_is_not_recorded_as_enforced(ks_isolated, monkeypatch):
    """Mutant M-D: `if sl_trade:` -> `if True:` SURVIVED with 4 passed.

    `execute_sell` returns None when the position is already gone (it is naturally
    idempotent). Without this test, a mutant that drops the truthiness check records
    the ticker in `halt_stop_loss_triggered` even though NO sell happened -- i.e. the
    cycle summary would claim a stop was enforced when the book still holds the
    exposure. That is the worst possible lie for this particular defect to tell.
    """
    ks, state, _live = ks_isolated

    import backend.services.autonomous_loop as al
    import backend.services.cycle_health as cycle_health
    from backend.config.settings import Settings
    from backend.services.portfolio_manager import TradeOrder
    import backend.services.paper_trader as _pt

    state._paused = True
    state._pause_reason = "manual"
    portfolio = {"total_nav": 18_000.0, "starting_capital": 20_000.0, "current_cash": 18_000.0}

    trader = MagicMock()
    trader.check_and_enforce_kill_switch.return_value = KS_QUIET
    trader.mark_to_market.return_value = {"nav": 18_000.0, "positions": []}
    trader.get_or_create_portfolio.return_value = portfolio
    trader.check_stop_losses.return_value = [BREACHED_TICKER]
    trader.execute_sell.return_value = None          # <-- the position was already gone

    bq = MagicMock()
    bq.get_paper_positions.return_value = []
    bq.get_paper_portfolio.return_value = portfolio

    settings = Settings()
    settings.news_screen_enabled = False
    monkeypatch.setattr(_pt, "_get_live_price", lambda *a, **k: 100.0)
    monkeypatch.setattr(al, "BigQueryClient", lambda *a, **k: bq)
    monkeypatch.setattr(al, "PaperTrader", lambda *a, **k: trader)
    monkeypatch.setattr(al, "screen_universe", lambda *a, **k: [])
    monkeypatch.setattr(al, "rank_candidates", lambda *a, **k: [])
    monkeypatch.setattr(al, "get_sp500_tickers", lambda *a, **k: [])
    monkeypatch.setattr(al, "get_russell1000_tickers", lambda *a, **k: [])
    monkeypatch.setattr(al, "_log_cycle_signals_to_bq", lambda *a, **k: 0)
    monkeypatch.setattr(al, "AnalysisOrchestrator", MagicMock())
    monkeypatch.setattr(al, "decide_trades", MagicMock(return_value=[
        TradeOrder(ticker="NVDA", action="BUY", amount_usd=1_000.0,
                   reason="probe", price=100.0, price_at_analysis=100.0)]))
    monkeypatch.setattr(al, "_running", False)
    fake_log = MagicMock()
    fake_log.record_cycle_start.return_value = "2026-01-01T00:00:00+00:00"
    monkeypatch.setattr(cycle_health, "get_log", lambda *a, **k: fake_log)

    summary = asyncio.run(al.run_daily_cycle(settings=settings))

    assert summary["halted"] is True
    assert trader.check_stop_losses.called is True
    assert trader.execute_sell.called is True, "the exit was never attempted"
    # THE POINT: attempted != enforced. A sell that returned nothing must NOT be
    # reported as a stop-out.
    assert summary.get("halt_stop_loss_triggered") == [], (
        "a stop was recorded as ENFORCED although execute_sell returned None -- "
        "the summary claims an exit that did not happen (phase-36.17, mutant M-D)"
    )


def test_phase_36_17_a_raising_stop_pass_stays_loud_and_still_halts(
    ks_isolated, monkeypatch, caplog
):
    """Mutants M-E and M-F: dropping `summary["halt_stop_loss_error"]` and
    downgrading `logger.exception` -> `logger.debug` both SURVIVED with 4 passed,
    because NO test drove check_stop_losses to raise.

    The 'Failure handling' paragraph of experiment_results §1 positively claims the
    pass is fail-safe AND loud. Until this test existed that claim had zero covering
    assertion, so a regression to a silent swallow in a STOP-LOSS path would not have
    been noticed.
    """
    ks, state, _live = ks_isolated

    import backend.services.autonomous_loop as al
    import backend.services.cycle_health as cycle_health
    from backend.config.settings import Settings
    import backend.services.paper_trader as _pt

    portfolio = {"total_nav": 18_000.0, "starting_capital": 20_000.0, "current_cash": 18_000.0}
    boom = RuntimeError("BQ unavailable during halt")

    trader = MagicMock()
    trader.check_and_enforce_kill_switch.return_value = KS_BLOCKED
    trader.mark_to_market.return_value = {"nav": 18_000.0, "positions": []}
    trader.get_or_create_portfolio.return_value = portfolio
    trader.check_stop_losses.side_effect = boom     # <-- the pass explodes

    bq = MagicMock()
    bq.get_paper_positions.return_value = []
    bq.get_paper_portfolio.return_value = portfolio

    settings = Settings()
    settings.news_screen_enabled = False
    monkeypatch.setattr(_pt, "_get_live_price", lambda *a, **k: 100.0)
    monkeypatch.setattr(al, "BigQueryClient", lambda *a, **k: bq)
    monkeypatch.setattr(al, "PaperTrader", lambda *a, **k: trader)
    monkeypatch.setattr(al, "screen_universe", lambda *a, **k: [])
    monkeypatch.setattr(al, "rank_candidates", lambda *a, **k: [])
    monkeypatch.setattr(al, "get_sp500_tickers", lambda *a, **k: [])
    monkeypatch.setattr(al, "get_russell1000_tickers", lambda *a, **k: [])
    monkeypatch.setattr(al, "_log_cycle_signals_to_bq", lambda *a, **k: 0)
    monkeypatch.setattr(al, "AnalysisOrchestrator", MagicMock())
    decide = MagicMock(return_value=[])
    monkeypatch.setattr(al, "decide_trades", decide)
    monkeypatch.setattr(al, "_running", False)
    fake_log = MagicMock()
    fake_log.record_cycle_start.return_value = "2026-01-01T00:00:00+00:00"
    monkeypatch.setattr(cycle_health, "get_log", lambda *a, **k: fake_log)

    import logging

    caplog.set_level(logging.DEBUG, logger="backend.services.autonomous_loop")
    summary = asyncio.run(al.run_daily_cycle(settings=settings))

    # FAIL-SAFE: the halt still completes and keeps the terminal status that the
    # phase-85.4 loudness guards depend on.
    assert summary["halted"] is True
    assert summary["status"] == "halted_kill_switch"
    assert summary["halt_reason"] == "kill_switch_disarmed_lost_history"
    assert decide.called is False, "a raising stop pass let the cycle fall through"
    # LOUD: the failure is recorded where an operator can see it. A silent swallow
    # in a stop-loss path is exactly what must not happen.
    assert "halt_stop_loss_error" in summary, (
        "the exit-only pass raised and recorded NOTHING -- a stop-loss failure was "
        "swallowed silently (phase-36.17, mutants M-E/M-F)"
    )
    assert repr(boom) == summary["halt_stop_loss_error"]
    # ...and LOUD AT ERROR LEVEL, not quietly at debug. Mutant M-F
    # (`logger.exception` -> `logger.debug`) SURVIVED until this assertion existed:
    # the summary key alone cannot distinguish a logged failure from a whispered
    # one, and "stay loud" is a claim the artifacts make positively.
    loud = [
        r for r in caplog.records
        if r.levelno >= logging.ERROR
        and "exit-only stop-loss pass FAILED" in r.getMessage()
    ]
    assert loud, (
        "the exit-only pass failed but nothing was logged at ERROR+ -- a stop-loss "
        "failure was downgraded to a whisper (phase-36.17, mutant M-F). "
        f"records seen: {[(r.levelname, r.getMessage()[:60]) for r in caplog.records]}"
    )


# ── criterion 7 gaps found by Q/A cycle 5 -- two MONEY-PATH mutants survived ──
# Both were confirmed non-equivalent at source before a guard was written for
# them, and both are the same shape as M-D: the summary reporting a stop as
# ENFORCED when the book is still exposed.


def test_phase_36_17_the_halt_exit_is_a_FULL_exit_not_a_partial(ks_isolated, monkeypatch):
    """Mutant MUT-B: `quantity=None` -> `quantity=1` in the halt pass SURVIVED.

    `paper_trader.execute_sell` documents "If quantity is None, sells entire
    position", and does `sell_qty = quantity or position["quantity"]`. So a mutant
    passing `quantity=1` liquidates ONE SHARE, still returns a truthy trade record,
    and the ticker is STILL appended to `halt_stop_loss_triggered` -- the book keeps
    essentially full exposure while the cycle summary reports the stop as enforced.

    The M-D guard closed the sell-returned-None case; it cannot see a partial fill,
    because a partial fill returns a trade record just like a full one. This asserts
    the ARGUMENT rather than the outcome, which is the only thing that separates
    them.
    """
    ks, state, _live = ks_isolated
    summary, trader, _decide = _drive_cycle(
        monkeypatch, KS_QUIET, paused=True, ks_state=state
    )

    sold = [c for c in trader.execute_sell.call_args_list
            if c.kwargs.get("reason") == "stop_loss_trigger"]
    assert sold, "no stop-loss exit was attempted"
    assert sold[0].kwargs["quantity"] is None, (
        "the halt exit passed quantity=%r -- paper_trader.py does "
        "`sell_qty = quantity or position['quantity']`, so anything but None is a "
        "PARTIAL exit that still gets recorded as an enforced stop while the book "
        "stays exposed (phase-36.17, mutant MUT-B)" % (sold[0].kwargs["quantity"],)
    )
    assert summary.get("halt_stop_loss_triggered") == [BREACHED_TICKER]


def test_phase_36_17_stops_are_checked_against_FRESH_marks(ks_isolated, monkeypatch):
    """Mutant MUT-C: deleting the halt-path `mark_to_market` SURVIVED.

    `check_stop_losses` compares `pos.get("current_price", 0)` against the stop, and
    `current_price` is what `mark_to_market` refreshes. Remove the halt-path
    mark_to_market and the stop comparison silently runs on STALE prices -- so a
    position that has crossed its stop since the last mark would not be exited.

    experiment_results section 1 makes this a positive, load-bearing claim ("Runs
    AFTER mark_to_market so the stop comparison sees fresh prices") and it had ZERO
    covering assertion: the trader is a MagicMock, so the mocked
    `check_stop_losses` cannot represent price staleness. The only observable that
    distinguishes the two is CALL ORDER, so that is what this asserts.
    """
    ks, state, _live = ks_isolated
    summary, trader, _decide = _drive_cycle(
        monkeypatch, KS_BLOCKED, paused=False, ks_state=state
    )

    names = [c[0] for c in trader.method_calls]
    assert "check_stop_losses" in names, "the halt path never checked stops"
    csl = names.index("check_stop_losses")

    # MEASURED call order on the halt path:
    #   roll_daily_anchor, get_positions, mark_to_market, check_scale_out_fires,
    #   check_and_enforce_kill_switch, mark_to_market, check_stop_losses, ...
    # There are TWO mark_to_market calls. A naive `names.index("mark_to_market")
    # < names.index("check_stop_losses")` finds the FIRST (the Step-5 one) and
    # therefore PASSES even when the halt-path mark is deleted -- measured: that
    # is exactly how MUT-C survived its first guard. The load-bearing property is
    # that a mark happens AFTER the kill-switch evaluation, so assert the
    # IMMEDIATE PREDECESSOR.
    assert names[csl - 1] == "mark_to_market", (
        "the call immediately before check_stop_losses was %r, not mark_to_market "
        "-- stops were compared against STALE prices, so a position that crossed "
        "its stop since the last mark would not be exited (phase-36.17, MUT-C). "
        "order was: %r" % (names[csl - 1], names)
    )
    ks_eval = names.index("check_and_enforce_kill_switch")
    assert any(n == "mark_to_market" for n in names[ks_eval:csl]), (
        "no mark_to_market between the kill-switch evaluation and the stop check "
        "-- the halt-path refresh is missing (phase-36.17, MUT-C)"
    )
