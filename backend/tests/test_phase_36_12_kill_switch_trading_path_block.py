"""phase-36.12 -- the ORDER-PLACING path must not forgive a drawdown silently.

`paper_trader.check_and_enforce_kill_switch` used to call `state.update_peak(nav)`
and `state.update_sod_nav(nav)` BEFORE `evaluate_breach`, and then branch only on
`any_breached`. With both baselines lost (audit-log rotation), that anchored both
to today's NAV, forgave the entire real drawdown, and reported ARMED and HEALTHY --
`armed` was structurally always True on the one path that actually places orders.

Contract: handoff/current/contract_36.12.md
Research: handoff/current/research_brief_36.12.md

ISOLATION IS MANDATORY IN THIS MODULE. `_append_audit` writes to the module-level
`_AUDIT_PATH`, which in production is the LIVE safety file
`handoff/kill_switch_audit.jsonl`. The autouse fixture below is ported verbatim in
intent from `test_phase_36_7_kill_switch_rotation_rearm.py` -- without it, that
brace would exist only in the 36.7 module (research brief section G).
"""
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

OLD_PROMISE_PHRASES = (
    "re-anchors both baselines",
    "re-anchors them",
    "next cycle re-anchors",
)


# ── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _live_audit_file_is_write_protected():
    """Nothing in THIS module may append to the real audit trail.

    Belt to the per-test `_AUDIT_PATH` redirects: if a future test forgets to
    redirect, this fails it loudly instead of quietly writing fabricated rows into
    live safety state. `_append_audit` swallows write errors, so a raise inside the
    code under test could not surface -- the byte comparison is the only reliable
    detector.
    """
    live = REPO_ROOT / "handoff" / "kill_switch_audit.jsonl"
    before = live.read_bytes() if live.exists() else None
    yield
    after = live.read_bytes() if live.exists() else None
    assert after == before, (
        f"phase-36.12: a test in this module wrote to the LIVE audit trail {live}. "
        "Redirect ks._AUDIT_PATH to tmp_path."
    )


@pytest.fixture(autouse=True)
def captured_alerts(monkeypatch):
    """Intercept the P1 pager instead of firing it, and make it assertable.

    NOT merely a silencer. `raise_cron_alert_sync` posts to the operator's REAL
    Slack channel via `chat.postMessage` (alerting.py:167) -- there is no test/env
    guard on that path, so an unstubbed run of this module pages the operator with a
    false "kill switch DISARMED" P1. Measured, not assumed: the first runs of this
    suite did exactly that (`alert bot-token fallback delivered=True
    source=kill_switch title='Kill-switch DISARMED at cycle start -- new orders
    BLOCKED'`).

    Capturing rather than no-op'ing also lets the block test assert the alert IS
    raised, which is part of "fail LOUD" -- a silenced pager would make a
    dropped-alert regression invisible.

    NOTE the same hazard pre-exists on `KillSwitchState.pause()` (alerting on any
    non-manual trigger, kill_switch.py:318-330), which several older test modules
    reach; that is queued as its own masterplan step, not fixed here.
    """
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
    fresh._auto_resume_alerted_at = None
    return fresh


@pytest.fixture
def ks_isolated(tmp_path, monkeypatch):
    """Isolated kill_switch: tmp audit path + a detached module-global `_state`.

    Yields (ks_module, state, live_audit_path). The archive dir is derived from
    `_AUDIT_PATH`, so redirecting the one path isolates the whole replay surface.
    """
    import backend.services.kill_switch as ks

    live = tmp_path / "handoff" / "kill_switch_audit.jsonl"
    live.parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "handoff" / "audit").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(ks, "_AUDIT_PATH", live)
    assert ks._audit_archive_dir() == tmp_path / "handoff" / "audit"

    state = _detached_state(ks)
    monkeypatch.setattr(ks, "_state", state)
    monkeypatch.setattr(ks, "get_state", lambda: state)
    monkeypatch.setattr(ks, "_disarmed_logged", False)
    yield ks, state, live


def _make_trader(nav: float, starting_capital: float):
    from backend.config.settings import Settings
    from backend.services.paper_trader import PaperTrader

    s = Settings()
    s.paper_daily_loss_limit_pct = 4.0
    s.paper_trailing_dd_limit_pct = 10.0
    bq = MagicMock()
    bq.get_paper_portfolio.return_value = {
        "current_cash": nav,
        "starting_capital": starting_capital,
        "total_nav": nav,
        "portfolio_id": "p1",
    }
    bq.get_paper_positions.return_value = []
    return PaperTrader(settings=s, bq_client=bq), bq


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _write_rows(path: Path, *rows: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


# ── criterion 1 + 2: the defect, and the refusal that replaced it ───────────


def test_phase_36_12_lost_history_cycle_blocks_instead_of_reporting_armed_and_healthy(
    ks_isolated, captured_alerts
):
    """THE DEFECT. Both baselines unrecoverable, NAV materially below the true
    historical peak, and an audit trail that proves the process ran before
    (pause/resume rows survived the rotation; the baseline rows did not -- the
    exact 2026-07-26 shape).

    Pre-fix this returned triggered=False with a freshly anchored peak and
    `armed: true`: ARMED AND HEALTHY on a book whose drawdown had just been
    forgiven. Post-fix the cycle must refuse to place new orders.
    """
    ks, state, live = ks_isolated
    # Rotation survivors: pause/resume only. NO sod_snapshot / peak_update anywhere.
    _write_rows(
        live,
        {"ts": "2026-07-20T10:00:00+00:00", "event": "pause", "trigger": "limit_breach"},
        {"ts": "2026-07-20T12:00:00+00:00", "event": "resume", "trigger": "manual"},
    )
    trader, _bq = _make_trader(nav=18_000.0, starting_capital=20_000.0)

    result = trader.check_and_enforce_kill_switch()

    assert result["pre_armed"] is False, (
        "the armed state must be measured BEFORE the baselines are anchored"
    )
    assert result["blocked"] is True, (
        "a book with prior history that comes up disarmed must not place new orders"
    )
    assert result["block_reason"] == "kill_switch_disarmed_lost_history"
    assert result["triggered"] is False, "block is not a flatten -- positions are kept"
    assert state.is_paused() is False, "block is NON-LATCHING -- it must not pause"
    # FAIL LOUD: the refusal must page the operator, not just log.
    assert len(captured_alerts) == 1, captured_alerts
    assert captured_alerts[0]["severity"] == "P1"
    assert captured_alerts[0]["error_type"] == "disarmed_lost_history_block"


def test_phase_36_12_blocked_cycle_halts_the_autonomous_loop():
    """BEHAVIOURAL, not a source scan. The refusal only matters if the caller
    honours it: `autonomous_loop` used to halt on `triggered or is_paused()`, and a
    blocked cycle is neither.

    This replaces a substring scan that the cycle-1 Q/A defeated -- it kept the
    literal `ks_check.get("blocked")` and neutered it with `and False`, and the scan
    stayed green (qa.md 4c vacuity shape #3: literal kept, behaviour stripped).
    """
    from backend.services.autonomous_loop import cycle_halt_reason

    blocked = {
        "triggered": False,
        "blocked": True,
        "block_reason": "kill_switch_disarmed_lost_history",
        "breach": {"any_breached": False},
    }
    assert cycle_halt_reason(blocked, False) == "kill_switch_disarmed_lost_history"
    # ... and the cycle still runs when nothing is wrong
    assert cycle_halt_reason({"triggered": False, "blocked": False}, False) is None


def test_phase_36_12_a_blocked_cycle_really_places_no_orders(ks_isolated, monkeypatch):
    """THE COMPOSITION, EXECUTED. Drives the real `run_daily_cycle` with the
    kill-switch stubbed to `blocked: True` and asserts the cycle halts and no order
    is placed.

    This is the guard three prior cycles kept failing to build. Cycle 1 had a
    substring scan (defeated by keeping the literal and adding `and False`); cycle 2
    had a behavioural test of the PREDICATE but only a scan of the WIRING; cycle 3's
    AST guard constrained the branch's SHAPE but never its BODY, so deleting
    `return summary` survived everything and a halted cycle fell through into Step
    5.6 and then decide/execute. Guarding structure kept relocating the hole one
    level up. This executes it instead.

    TWO HAZARDS this test had to defuse, both measured rather than assumed:
      * COST -- the cycle reaches Step 5.5 only if the analysis pipeline is stubbed.
        An unstubbed run makes a REAL 150s LLM call; `news_screen_enabled` must be
        off and `AnalysisOrchestrator` mocked, or this test bills the operator.
      * TRACKED STATE -- `run_daily_cycle` records a heartbeat and a history row via
        `cycle_health.get_log()`, which write `handoff/.cycle_heartbeat.json` and
        `handoff/cycle_history.jsonl`. Both are GIT-TRACKED. The first draft of this
        test dirtied them on every run (caught by `git status`, restored from HEAD);
        the cycle-log stub below is what keeps a test from mutating operator state.
    """
    import backend.services.autonomous_loop as al
    import backend.services.cycle_health as cycle_health
    from backend.config.settings import Settings

    ks, _state, _live = ks_isolated

    ks_blocked = {
        "triggered": False,
        "blocked": True,
        "block_reason": "kill_switch_disarmed_lost_history",
        "pre_armed": False,
        "breach": {"any_breached": False, "armed": True},
        "auto_resume": {"action": "no_op"},
    }
    portfolio = {"total_nav": 18_000.0, "starting_capital": 20_000.0, "current_cash": 18_000.0}

    trader = MagicMock()
    trader.check_and_enforce_kill_switch.return_value = ks_blocked
    trader.mark_to_market.return_value = {"nav": 18_000.0, "positions": []}
    trader.get_or_create_portfolio.return_value = portfolio
    bq = MagicMock()
    bq.get_paper_positions.return_value = []
    bq.get_paper_portfolio.return_value = portfolio

    settings = Settings()
    settings.news_screen_enabled = False

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
    # Keep the cycle's own bookkeeping off the GIT-TRACKED handoff files.
    fake_log = MagicMock()
    fake_log.record_cycle_start.return_value = "2026-01-01T00:00:00+00:00"
    monkeypatch.setattr(cycle_health, "get_log", lambda *a, **k: fake_log)

    summary = asyncio.run(al.run_daily_cycle(settings=settings))

    assert summary["halted"] is True, (
        "a blocked cycle must halt -- if this fails, the halt branch no longer "
        "returns and the cycle continues into decide/execute"
    )
    assert "kill_switch_halted" in summary["steps"]
    assert summary["steps"][-1] == "kill_switch_halted", (
        f"the cycle continued past the halt: {summary['steps']}"
    )
    # The money assertions: nothing was decided and nothing was traded.
    assert decide.called is False, "decide_trades ran on a halted cycle"
    assert trader.execute_buy.called is False, "a BUY was placed on a halted cycle"
    assert trader.execute_sell.called is False, "a SELL was placed on a halted cycle"


def test_phase_36_12_an_already_paused_cycle_still_halts(ks_isolated, monkeypatch):
    """The `is_paused` ARGUMENT must be wired to the live state, not a constant.

    Found in cycle 4 by re-measuring the whole matrix at the current baseline: mutant
    M8 (`cycle_halt_reason(ks_check, _ks_state().is_paused())` ->
    `cycle_halt_reason(ks_check, False)`) had started SURVIVING. The AST guard
    constrains the call's SHAPE and the branch that follows it, but never its
    ARGUMENTS -- so a paused book would have gone on trading, and nothing anywhere
    would have failed. Same relocation family as QA-X6/QA-Y1/QA-Z1, one level
    sideways: the arguments rather than the body.

    This is the pre-36.12 halt term, so the test also pins that 36.12 did not weaken
    it while adding `blocked`.
    """
    import backend.services.autonomous_loop as al
    import backend.services.cycle_health as cycle_health
    from backend.config.settings import Settings

    ks, state, _live = ks_isolated
    state.pause(trigger="test")  # isolated audit path; alert captured by the fixture
    assert state.is_paused() is True

    # Not triggered and NOT blocked -- `paused` is the only thing that can halt it.
    ks_quiet = {
        "triggered": False, "blocked": False, "pre_armed": True,
        "breach": {"any_breached": False, "armed": True},
        "auto_resume": {"action": "no_op"},
    }
    portfolio = {"total_nav": 18_000.0, "starting_capital": 20_000.0, "current_cash": 18_000.0}
    trader = MagicMock()
    trader.check_and_enforce_kill_switch.return_value = ks_quiet
    trader.mark_to_market.return_value = {"nav": 18_000.0, "positions": []}
    trader.get_or_create_portfolio.return_value = portfolio
    bq = MagicMock()
    bq.get_paper_positions.return_value = []
    bq.get_paper_portfolio.return_value = portfolio

    settings = Settings()
    settings.news_screen_enabled = False
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

    summary = asyncio.run(al.run_daily_cycle(settings=settings))

    assert summary["halted"] is True, "an already-paused book must not trade"
    assert summary["steps"][-1] == "kill_switch_halted"
    assert decide.called is False
    assert trader.execute_buy.called is False
    assert trader.execute_sell.called is False


def test_phase_36_12_a_quiet_cycle_actually_proceeds_and_trades(ks_isolated, monkeypatch):
    """THE PROCEED DIRECTION, executed. A healthy, unpaused, unblocked cycle must run
    decide/execute -- not just "not halt".

    Found in cycle 5 as QA-P2, the FIFTH hole in this one piece of wiring: forcing
    `cycle_halt_reason(ks_check, True)` makes EVERY cycle halt, and nothing in the
    repo failed. Both existing behavioural composition tests assert the HALT
    direction only, and the older `run_daily_cycle` suites never reach Step 5.5, so
    they are blind to it. Direction is fail-safe (over-halt, no orders) and loud, but
    an unguarded permanent halt is exactly the engine-stop failure mode phase-69.1
    exists to prevent.

    This also guards criterion 3 at the level where a first-ever-boot deadlock would
    actually manifest: the unit test proves the boot ANCHORS, this proves a quiet
    cycle TRADES.
    """
    import backend.services.autonomous_loop as al
    import backend.services.cycle_health as cycle_health
    from backend.config.settings import Settings

    ks, state, _live = ks_isolated
    assert state.is_paused() is False, "precondition: the state must be unpaused"

    ks_quiet = {
        "triggered": False, "blocked": False, "pre_armed": True,
        "breach": {"any_breached": False, "armed": True},
        "auto_resume": {"action": "no_op"},
    }
    portfolio = {"total_nav": 18_000.0, "starting_capital": 20_000.0, "current_cash": 18_000.0}
    trader = MagicMock()
    trader.check_and_enforce_kill_switch.return_value = ks_quiet
    trader.mark_to_market.return_value = {"nav": 18_000.0, "positions": []}
    trader.get_or_create_portfolio.return_value = portfolio
    trader.check_stop_losses.return_value = []
    bq = MagicMock()
    bq.get_paper_positions.return_value = []
    bq.get_paper_portfolio.return_value = portfolio

    settings = Settings()
    settings.news_screen_enabled = False
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

    summary = asyncio.run(al.run_daily_cycle(settings=settings))

    assert not summary.get("halted"), (
        f"a healthy unpaused cycle must NOT halt -- steps={summary.get('steps')}"
    )
    assert "kill_switch_halted" not in summary.get("steps", [])
    assert decide.called is True, (
        "a healthy cycle must reach decide_trades -- if this fails the engine is "
        "permanently halted and no order will ever be placed"
    )


def test_phase_36_12_halt_precedence_is_breach_then_block_then_paused():
    """The reason is what the operator reads in the log line, so precedence is part
    of the contract: a real breach must not be reported as a lost-history block."""
    from backend.services.autonomous_loop import cycle_halt_reason

    assert cycle_halt_reason({"triggered": True, "blocked": True}, True) == "breach"
    assert cycle_halt_reason({"triggered": False, "blocked": True}, True) == "blocked"
    assert cycle_halt_reason({"triggered": False, "blocked": False}, True) == "paused"
    # A pre-36.12 payload (no `blocked` key at all) must still behave exactly as before.
    assert cycle_halt_reason({"triggered": False}, False) is None
    assert cycle_halt_reason({"triggered": True}, False) == "breach"


def test_phase_36_12_the_loop_actually_calls_the_halt_predicate():
    """Belt to the behavioural tests above: the predicate is only load-bearing if
    Step 5.5 calls it AND branches on what it returned.

    STRUCTURAL (AST), not a substring scan, and labelled as structural rather than
    behavioural. Cycle 2's version asserted two literals were present; the cycle-2 Q/A
    defeated it by KEEPING both and inserting `halt_reason = None` between them
    (qa.md 4c vacuity shape #3, relocated one level up). Walking the AST and requiring
    the `if` to be the STATEMENT IMMEDIATELY FOLLOWING the assignment closes that: any
    intervening statement, and any rebinding of the name, is now visible.

    THE COMPOSITION IS ALSO TESTED BEHAVIOURALLY, 95 lines above, by
    `test_phase_36_12_a_blocked_cycle_really_places_no_orders` -- it drives the real
    `run_daily_cycle` and asserts no order is placed. This AST guard is the cheap,
    fast belt on the branch's SHAPE; that test is the brace on its BEHAVIOUR. Keep
    both: the AST guard catches an edit in milliseconds without mocking the cycle's
    whole I/O surface, and it fails on shapes the behavioural test would also catch
    but more slowly.

    (Cycle-4 correction: this docstring used to say a behavioural test "is queued
    rather than faked". The cycle-3 Q/A named that claim as false in TWO places;
    only the artifact was fixed and this one survived -- and it had since become
    false a second way, because the test it called deferred already exists in this
    file. Found by the cycle-4 Q/A.)
    """
    import ast

    src = (REPO_ROOT / "backend" / "services" / "autonomous_loop.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(src)

    def _calls_predicate(node) -> bool:
        return (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "cycle_halt_reason"
        )

    found = []
    for parent in ast.walk(tree):
        body = getattr(parent, "body", None)
        if not isinstance(body, list):
            continue
        for i, stmt in enumerate(body[:-1]):
            if not _calls_predicate(stmt):
                continue
            target = stmt.targets[0]
            assert isinstance(target, ast.Name), "predicate result must bind a plain name"
            nxt = body[i + 1]
            assert isinstance(nxt, ast.If), (
                "the statement immediately after the cycle_halt_reason call must be the "
                f"branch on it, not {type(nxt).__name__} -- something was inserted between "
                "the halt decision and the halt"
            )
            assert isinstance(nxt.test, ast.Name) and nxt.test.id == target.id, (
                "the branch must test the predicate's own result, unmodified"
            )
            found.append(target.id)

    assert found, "run_daily_cycle no longer calls cycle_halt_reason -- the halt is unwired"


# ── criterion 3: a genuinely new book must still trade ──────────────────────


def test_phase_36_12_first_ever_boot_still_anchors_and_trades(ks_isolated):
    """No audit history at all AND nav == starting_capital: indistinguishable from a
    brand-new book. It must anchor both baselines and trade normally -- a fix that
    blocks here deadlocks every new book."""
    ks, state, live = ks_isolated
    assert not live.exists() or _rows(live) == []
    trader, _bq = _make_trader(nav=100_000.0, starting_capital=100_000.0)

    result = trader.check_and_enforce_kill_switch()

    assert result["blocked"] is False
    assert result["triggered"] is False
    snap = state.snapshot()
    assert snap["sod_nav"] == 100_000.0
    assert snap["peak_nav"] == 100_000.0
    assert result["breach"]["armed"] is True


@pytest.mark.parametrize(
    "bad_total_nav, label",
    [
        (_ABSENT := object(), "key absent entirely"),
        (0.0, "zero float -- the `or` fallback's other half"),
        (0, "zero int"),
        (None, "explicit null"),
        ([], "container-typed NAV (float() raises TypeError)"),
        ({}, "dict-typed NAV (float() raises TypeError)"),
    ],
)
def test_phase_36_12_a_degraded_nav_read_is_not_evidence_of_an_untraded_book(
    ks_isolated, bad_total_nav, label
):
    """`nav` falls back to `starting_capital` when `total_nav` is falsy, so a degraded
    BQ read MANUFACTURES nav == starting_capital. Without a raw-NAV check that reads a
    real traded book as a first-ever boot and skips the block -- the UNSAFE direction.

    PARAMETRIZED IN CYCLE 3, and the parameters are the point. Cycle 2's version tested
    only the key-ABSENT case, so two mutants survived with proven differentials: dropping
    the `> 0` clause (a zero-valued NAV, which this test's own code comment named) and
    inverting the `except` fallback to True (a container-typed NAV). Each row below is a
    mutant that used to live. Both were found by the cycle-2 Q/A.
    """
    ks, _state, live = ks_isolated
    trader, bq = _make_trader(nav=0.0, starting_capital=20_000.0)
    portfolio = {"current_cash": 0.0, "starting_capital": 20_000.0, "portfolio_id": "p1"}
    if bad_total_nav is not _ABSENT:
        portfolio["total_nav"] = bad_total_nav
    bq.get_paper_portfolio.return_value = portfolio

    result = trader.check_and_enforce_kill_switch()

    assert result["blocked"] is True, (
        f"an unreadable NAV ({label}) is not evidence of an untraded book -- it must "
        "not unlock the first-ever-boot exemption"
    )


def test_phase_36_12_baseline_history_probe_fails_closed(ks_isolated, monkeypatch):
    """The probe's docstring promises 'on a probe failure this returns True ... the
    conservative direction'. The cycle-1 Q/A inverted that fallback to `return False`
    and the whole suite stayed green -- a safety claim with zero coverage. This is
    the test that makes the claim falsifiable."""
    ks, _state, _live = ks_isolated

    def _explode():
        raise OSError("audit tree unreadable")

    monkeypatch.setattr(ks.KillSwitchState, "_read_audit_rows", staticmethod(_explode))
    assert ks.baseline_history_exists() is True, "an unreadable audit tree must read as 'has history'"

    # ...and the trading path must therefore BLOCK a book it cannot vouch for,
    # even one sitting exactly at starting capital (which would otherwise look new).
    trader, _bq = _make_trader(nav=100_000.0, starting_capital=100_000.0)
    assert trader.check_and_enforce_kill_switch()["blocked"] is True


def test_phase_36_12_traded_book_at_starting_capital_with_history_is_not_first_boot(
    ks_isolated,
):
    """D1 half of the discriminator, isolated: baseline rows HAVE existed before, so
    even a NAV that coincidentally equals starting_capital is not a new book."""
    ks, state, live = ks_isolated
    _write_rows(
        live,
        {"ts": "2026-07-01T10:00:00+00:00", "event": "peak_update", "nav": 130_000.0},
    )
    # The row exists in the file but this detached state never replayed it, so the
    # in-memory baselines are still None -- exactly the lost-in-memory shape.
    trader, _bq = _make_trader(nav=100_000.0, starting_capital=100_000.0)

    result = trader.check_and_enforce_kill_switch()

    assert result["blocked"] is True
    assert result["block_reason"] == "kill_switch_disarmed_lost_history"


# ── criterion 6: per-leg independence, and the breach keeps precedence ──────


def test_phase_36_12_real_breach_on_the_surviving_leg_still_flattens(ks_isolated):
    """One baseline present, one missing: `armed` is False, but the surviving daily
    leg registers a REAL breach. The flatten must still happen -- putting the
    disarmed block ahead of the breach branch would suppress it."""
    ks, state, live = ks_isolated
    state.update_sod_nav(100_000.0)  # daily leg present; peak still None
    trader, bq = _make_trader(nav=90_000.0, starting_capital=100_000.0)
    bq.get_paper_positions.return_value = []

    with patch("backend.services.paper_trader.ExecutionRouter") as Router:
        Router.return_value = MagicMock()
        result = trader.check_and_enforce_kill_switch()

    assert result["triggered"] is True, (
        "a real breach on the surviving leg must win over the disarmed block"
    )
    assert result["breach"]["any_breached"] is True
    assert state.is_paused() is True


def test_phase_36_12_evaluate_breach_still_evaluates_the_surviving_leg(ks_isolated):
    """Criterion 6 as a property of `evaluate_breach` itself: the fix must not add a
    wholesale `if not armed: return` short-circuit."""
    ks, state, _live = ks_isolated
    state.update_sod_nav(100_000.0)  # peak stays None
    breach = ks.evaluate_breach(
        current_nav=90_000.0, daily_loss_limit_pct=4.0, trailing_dd_limit_pct=10.0
    )
    assert breach["armed"] is False
    assert breach["trailing_baseline_missing"] is True
    assert breach["daily_baseline_missing"] is False
    assert breach["daily_loss_pct"] == pytest.approx(10.0)
    assert breach["any_breached"] is True


# ── criterion 5: no route to reset_peak from the trading path ───────────────


def test_phase_36_12_block_path_writes_no_peak_reset_row_and_never_calls_reset_peak(
    ks_isolated, monkeypatch
):
    """TWO assertions, deliberately. `reset_peak` is DARK today
    (`kill_switch_peak_reset_enabled=False`), so it returns None without writing --
    the zero-rows assertion ALONE would pass even if the trading path called it.
    The monkeypatched raiser is what makes the guard able to fail."""
    ks, state, live = ks_isolated
    _write_rows(
        live, {"ts": "2026-07-20T10:00:00+00:00", "event": "pause", "trigger": "x"}
    )

    def _boom(*_a, **_k):
        raise AssertionError(
            "phase-36.12: the trading path called reset_peak -- that routes around "
            "the owed KS-PEAK-RESET operator token (step 79.6)"
        )

    monkeypatch.setattr(ks.KillSwitchState, "reset_peak", _boom)
    trader, _bq = _make_trader(nav=18_000.0, starting_capital=20_000.0)

    result = trader.check_and_enforce_kill_switch()

    assert result["blocked"] is True
    assert [r for r in _rows(live) if r.get("event") == "peak_reset"] == []


# ── CWE-223: the anchor must be distinguishable in the audit trail ──────────


def test_phase_36_12_lost_history_anchor_writes_a_distinguishable_audit_row(
    ks_isolated,
):
    """A bare `peak_update` row is forensically identical to a legitimate ratchet.
    The lost-history anchor must additionally emit a row that says what happened."""
    ks, state, live = ks_isolated
    _write_rows(
        live, {"ts": "2026-07-20T10:00:00+00:00", "event": "pause", "trigger": "x"}
    )
    trader, _bq = _make_trader(nav=18_000.0, starting_capital=20_000.0)

    trader.check_and_enforce_kill_switch()

    marks = [
        r for r in _rows(live) if r.get("event") == "baseline_anchor_on_lost_history"
    ]
    assert len(marks) == 1, "exactly one distinguishable row per lost-history anchor"
    row = marks[0]
    assert row["prior_sod_nav"] is None
    assert row["prior_peak_nav"] is None
    assert row["anchored_nav"] == 18_000.0
    assert row["blocked_orders"] is True


def test_phase_36_12_the_new_event_is_replay_inert_for_baselines_but_restores_provenance(
    ks_isolated,
):
    """The new row must NEVER become a baseline source -- `peak_reset` stays the only
    assignment-semantics event. It must, however, restore the PROVENANCE flag, or a
    restart would launder a lost-history anchor into a clean-looking ACTIVE badge."""
    ks, _state, live = ks_isolated
    _write_rows(
        live,
        {
            "ts": "2026-07-26T10:00:00+00:00",
            "event": "baseline_anchor_on_lost_history",
            "prior_sod_nav": None,
            "prior_peak_nav": None,
            "anchored_nav": 18_000.0,
            "blocked_orders": True,
        },
    )
    replayed = ks.KillSwitchState()
    snap = replayed.snapshot()
    assert snap["sod_nav"] is None
    assert snap["peak_nav"] is None
    assert snap["baseline_provenance"] == "lost_history_anchor"


def test_phase_36_12_an_operator_peak_reset_supersedes_the_lost_history_flag(
    ks_isolated,
):
    """The flag latches, but not forever: a later token-gated `peak_reset` is the
    operator choosing the mark, so the baselines stop being an unsigned fiction."""
    ks, _state, live = ks_isolated
    _write_rows(
        live,
        {"ts": "2026-07-26T10:00:00+00:00", "event": "baseline_anchor_on_lost_history",
         "prior_sod_nav": None, "prior_peak_nav": None,
         "anchored_nav": 18_000.0, "blocked_orders": True},
        {"ts": "2026-07-26T11:00:00+00:00", "event": "peak_reset", "old_peak": 18_000.0,
         "new_peak": 19_000.0, "trigger": "resume:manual", "operator": "peder"},
    )
    snap = ks.KillSwitchState().snapshot()
    assert snap["peak_nav"] == 19_000.0
    assert snap["baseline_provenance"] is None


def test_phase_36_12_the_state_endpoint_stops_reporting_a_clean_armed_book(ks_isolated):
    """The immutable live_check's demand, as a test. After the blocked cycle the
    switch IS armed again (the anchor happened), so `armed: true` alone would tell the
    operator nothing is wrong. GET /kill-switch must carry the provenance, or the
    post-anchor payload is byte-indistinguishable from a healthy book."""
    import backend.api.paper_trading as api

    ks, _state, live = ks_isolated
    _write_rows(
        live, {"ts": "2026-07-20T10:00:00+00:00", "event": "pause", "trigger": "x"}
    )
    trader, _bq = _make_trader(nav=18_000.0, starting_capital=20_000.0)
    assert trader.check_and_enforce_kill_switch()["blocked"] is True

    bq = MagicMock()
    bq.get_paper_portfolio.return_value = {
        "total_nav": 18_000.0, "starting_capital": 20_000.0,
    }
    with patch.object(api, "get_bq_client", lambda: bq):
        payload = asyncio.run(api.get_kill_switch_state())

    assert payload["breach"]["armed"] is True, (
        "precondition: the anchor DID re-arm the switch -- that is exactly why the "
        "provenance key has to exist"
    )
    assert payload["baseline_provenance"] == "lost_history_anchor"


# ── criterion 8: the operator-facing strings ────────────────────────────────


def test_phase_36_12_no_operator_string_still_promises_an_automatic_re_anchor():
    """Source scan half. Comments count: a comment-carried token has tripped this
    project's scans three separate times, so the phrase must be gone from prose too,
    not merely from the emitted string."""
    targets = [
        REPO_ROOT / "backend" / "api" / "paper_trading.py",
        REPO_ROOT / "frontend" / "src" / "components" / "KillSwitchPanel.tsx",
    ]
    offenders = []
    for path in targets:
        text = path.read_text(encoding="utf-8")
        for phrase in OLD_PROMISE_PHRASES:
            if phrase in text:
                offenders.append(f"{path.name}: {phrase!r}")
    assert offenders == [], (
        "these files still tell the operator to wait for the next cycle to "
        f"re-anchor the baselines: {offenders}"
    )


def test_phase_36_12_disarmed_resume_409_describes_the_block(ks_isolated):
    """Behavioural half -- a bare source scan is the weak guard shape. Drive the real
    endpoint with a disarmed state and read the message it actually emits."""
    from fastapi import HTTPException

    import backend.api.paper_trading as api

    ks, _state, _live = ks_isolated  # module-global _state is detached + disarmed

    bq = MagicMock()
    bq.get_paper_portfolio.return_value = {"total_nav": 18_000.0}
    monkey = patch.object(api, "get_bq_client", lambda: bq)

    with monkey:
        with pytest.raises(HTTPException) as exc:
            asyncio.run(
                api.resume_trading(api.KillSwitchActionRequest(confirmation="RESUME"))
            )

    detail = str(exc.value.detail)
    assert exc.value.status_code == 409
    assert "DISARMED" in detail
    for phrase in OLD_PROMISE_PHRASES:
        assert phrase not in detail, f"the 409 body still promises {phrase!r}"
    assert "block" in detail.lower(), (
        "the 409 must tell the operator what the next cycle actually does now "
        "(block new orders + write an audited row), not just refuse"
    )
