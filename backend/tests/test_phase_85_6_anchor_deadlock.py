"""phase-85.6 -- the book must be un-pausable.

THE DEADLOCK (measured, four links):
  cycle dies in `analyzing` -> the start-of-day anchor never rolls -> the anchor
  goes stale -> the daily-loss leg DISARMS -> POST /resume returns 409 -> the
  switch stays paused -> even a COMPLETING cycle logs "kill-switch active
  (paused) -- skipping decide/execute" and trades nothing.

The roll had exactly ONE production call site (`paper_trader.py:1298`) inside a
method with exactly ONE production caller (`services/autonomous_loop.py:1375`,
Step 5.5 of 10, behind the analysis phase). phase-85.6 moves it to Step 0.

Criterion 1 demands the proof be a CYCLE that dies mid-analysis, not an
inspection -- so `test_c1_*` drives the real `run_daily_cycle` with a fault
injected into `_run_single_analysis` and asserts the anchor is fresh afterwards.

ISOLATION: every test injects its own kill-switch state and redirects
cycle_history / heartbeat / lockfile into tmp_path. Nothing here reads or writes
the operator's live `handoff/kill_switch_audit.jsonl` -- that is the phase-36.28
coupling class, which phase-85.4's mutation M9 caught in my own tests.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from backend.config.settings import Settings
from backend.services import autonomous_loop, cycle_health, cycle_lock
from backend.services.paper_trader import PaperTrader, sod_anchor_needs_reroll


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _yesterday() -> str:
    return (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()


class FakeKillSwitchState:
    """A real-enough kill-switch state: records rolls, never touches disk."""

    def __init__(self, sod_nav=None, sod_date=None, peak_nav=None, paused=False):
        self._sod_nav = sod_nav
        self._sod_date = sod_date
        self._peak_nav = peak_nav
        self._paused = paused
        self.roll_calls: list[tuple[float, str]] = []
        # Every method name the subject invokes, in order. Lets a test pin the
        # roller's INTERACTION SURFACE rather than only its visible effects.
        self.calls: list[str] = []

    def snapshot(self) -> dict:
        self.calls.append("snapshot")
        return {
            "sod_nav": self._sod_nav,
            "sod_date": self._sod_date,
            "peak_nav": self._peak_nav,
            "paused": self._paused,
        }

    def update_sod_nav(self, nav: float, date: str | None = None) -> None:
        self.calls.append("update_sod_nav")
        date = date or _today()
        # Mirror the phase-36.9 F3 production refusal: a non-positive anchor is
        # REFUSED and leaves sod_nav None so a later call can retry.
        self.roll_calls.append((nav, date))
        if nav is None or float(nav) <= 0:
            return
        self._sod_nav = float(nav)
        self._sod_date = date

    def update_peak(self, nav: float) -> None:
        self.calls.append("update_peak")
        if nav and (self._peak_nav is None or nav > self._peak_nav):
            self._peak_nav = nav

    def is_paused(self) -> bool:
        self.calls.append("is_paused")
        return self._paused


class _FakeBQ:
    def __getattr__(self, name):
        def _noop(*a, **k):
            return []

        return _noop


def _trader(ks: FakeKillSwitchState, total_nav=10_000.0) -> PaperTrader:
    t = PaperTrader(Settings(), _FakeBQ(), kill_switch_state=ks)
    t.get_or_create_portfolio = lambda: {  # type: ignore[method-assign]
        "total_nav": total_nav,
        "starting_capital": 10_000.0,
        "current_cash": total_nav,
    }
    return t


# ─────────────────────────────────────────────────────────────────────────────
# The roller itself
# ─────────────────────────────────────────────────────────────────────────────


def test_roll_anchors_a_stale_date_to_today():
    ks = FakeKillSwitchState(sod_nav=23_830.46, sod_date="2026-08-05", peak_nav=24_666.57)
    out = _trader(ks, total_nav=23_500.0).roll_daily_anchor()
    assert out["rolled"] is True, out
    assert out["prior_sod_date"] == "2026-08-05"
    assert ks.snapshot()["sod_date"] == _today()
    assert ks.snapshot()["sod_nav"] == 23_500.0


def test_roll_anchors_an_absent_anchor():
    ks = FakeKillSwitchState(sod_nav=None, sod_date=None, peak_nav=10_000.0)
    out = _trader(ks).roll_daily_anchor()
    assert out["rolled"] is True, out
    assert ks.snapshot()["sod_date"] == _today()


def test_roll_is_a_same_day_noop():
    """Idempotence is what lets the :1298 roll stay in place unchanged."""
    ks = FakeKillSwitchState(sod_nav=9_000.0, sod_date=_today(), peak_nav=10_000.0)
    out = _trader(ks, total_nav=8_000.0).roll_daily_anchor()
    assert out["rolled"] is False
    assert out["reason"] == "anchor_already_current"
    assert ks.roll_calls == [], "a same-day call must not touch the anchor at all"
    assert ks.snapshot()["sod_nav"] == 9_000.0, "the morning anchor must survive"


def test_roll_refuses_a_non_positive_nav_and_says_so():
    """phase-36.9 F3: a 0.0 anchor is a semipredicate that blocks its own repair.
    The roller must surface the refusal rather than report success."""
    ks = FakeKillSwitchState(sod_nav=None, sod_date=None)
    t = _trader(ks)
    t.get_or_create_portfolio = lambda: {"total_nav": 0.0, "starting_capital": 0.0}
    out = t.roll_daily_anchor()
    assert out["rolled"] is False
    assert out["reason"] == "refused_non_positive_nav"
    assert ks.snapshot()["sod_nav"] is None, "a refused anchor must stay None"


def test_roll_never_raises_and_fails_safe():
    """A roll failure must not break the cycle. Failing to roll leaves the anchor
    stale, which DISARMS the daily leg -- i.e. it refuses to trade. Fail-open
    here cannot enable trading."""
    ks = FakeKillSwitchState(sod_nav=100.0, sod_date=_yesterday())
    t = _trader(ks)

    def _boom():
        raise RuntimeError("BQ down")

    t.get_or_create_portfolio = _boom  # type: ignore[method-assign]
    out = t.roll_daily_anchor()
    assert out["rolled"] is False
    assert out["reason"].startswith("error:")
    assert ks.snapshot()["sod_date"] == _yesterday(), "state must be untouched"


# ─────────────────────────────────────────────────────────────────────────────
# Criterion 5 -- no loosening
# ─────────────────────────────────────────────────────────────────────────────


def test_c5_on_a_falling_book_the_early_anchor_is_never_more_forgiving():
    """THE LOAD-BEARING SAFETY CLAIM.

    The early roll sees the PREVIOUS CLOSE's NAV; the old roll saw today's
    freshly-marked NAV. The dangerous direction is anchoring LOWER during a
    drawdown, because a lower anchor makes the measured daily loss SMALLER and
    the switch fires LATER. On a falling book that cannot happen: the previous
    close is the higher number.
    """
    prev_close, marked_down = 10_000.0, 9_000.0

    early = FakeKillSwitchState(sod_date=_yesterday(), sod_nav=1.0)
    _trader(early, total_nav=prev_close).roll_daily_anchor()

    late = FakeKillSwitchState(sod_date=_yesterday(), sod_nav=1.0)
    _trader(late, total_nav=marked_down).roll_daily_anchor()

    early_anchor = early.snapshot()["sod_nav"]
    late_anchor = late.snapshot()["sod_nav"]
    assert early_anchor >= late_anchor

    # Translated into what the operator actually cares about: measured daily loss.
    early_loss_pct = (early_anchor - marked_down) / early_anchor * 100
    late_loss_pct = (late_anchor - marked_down) / late_anchor * 100
    assert early_loss_pct >= late_loss_pct, (
        f"LOOSENED: early anchor reports {early_loss_pct:.2f}% loss vs "
        f"{late_loss_pct:.2f}% under the old anchor -- the switch would fire LATER"
    )
    assert early_loss_pct == pytest.approx(10.0)
    assert late_loss_pct == pytest.approx(0.0)


def test_c5_the_roller_changes_no_threshold_and_disarms_nothing():
    """The roller's whole surface is: read NAV, check the date guard, call
    update_sod_nav. It must not touch peak, pause state, or any limit.

    Asserting only that `peak_nav` did not MOVE was too weak -- the mutation
    matrix (M6) injected `update_peak(0.01)` and survived, because the fake's
    peak is monotonic so a low value is silently ignored. So pin the CALL SET,
    not the resulting values: any extra method the roller starts calling on the
    kill-switch state fails this test whether or not it happens to change a
    number today.
    """
    ks = FakeKillSwitchState(sod_nav=100.0, sod_date=_yesterday(), peak_nav=12_345.0,
                             paused=True)
    _trader(ks, total_nav=11_000.0).roll_daily_anchor()

    assert set(ks.calls) <= {"snapshot", "update_sod_nav"}, (
        f"the roller touched more of the kill-switch state than it may: {ks.calls}"
    )
    assert "update_sod_nav" in ks.calls, "PRECONDITION: the roller did not roll"
    snap = ks.snapshot()
    assert snap["peak_nav"] == 12_345.0, "the trailing peak must be untouched"
    assert snap["paused"] is True, "the roller must not un-pause anything"


def test_c5_a_paused_book_still_rolls():
    """Pausedness must not block the roll -- otherwise the deadlock survives the
    fix. Live-corroborated: a sod_snapshot dated 2026-08-05 was written on an
    already-paused book."""
    ks = FakeKillSwitchState(sod_nav=1.0, sod_date=_yesterday(), paused=True)
    assert _trader(ks).roll_daily_anchor()["rolled"] is True


def test_c5_the_shared_reroll_predicate_is_the_one_in_production():
    """The roller must gate on the REAL predicate, not a hand-copied duplicate --
    phase-36.9 already lost a guard to exactly that drift."""
    assert sod_anchor_needs_reroll({"sod_nav": 100.0, "sod_date": _yesterday()}, _today())
    assert not sod_anchor_needs_reroll({"sod_nav": 100.0, "sod_date": _today()}, _today())
    assert sod_anchor_needs_reroll({"sod_nav": 0.0, "sod_date": _today()}, _today())
    assert sod_anchor_needs_reroll({"sod_nav": None, "sod_date": _today()}, _today())


# ─────────────────────────────────────────────────────────────────────────────
# Criterion 1 -- a CYCLE that dies mid-analysis still leaves a fresh anchor
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def cycle_env(monkeypatch, tmp_path):
    monkeypatch.setattr(cycle_health, "_HISTORY_PATH", tmp_path / "cycle_history.jsonl")
    monkeypatch.setattr(cycle_health, "_HEARTBEAT_PATH", tmp_path / ".cycle_heartbeat.json")
    monkeypatch.setattr(cycle_lock, "_LOCK_PATH", tmp_path / ".autonomous_loop.lock")
    monkeypatch.setattr(autonomous_loop, "_running", False, raising=False)

    import backend.services.kill_switch as ks_mod

    monkeypatch.setattr(ks_mod, "get_state", lambda: FakeKillSwitchState(paused=False))

    import backend.services.observability.alerting as alerting

    monkeypatch.setattr(alerting, "raise_cron_alert_sync", lambda **k: True)

    rows = [{"ticker": t, "score": 9.0, "price": 100.0} for t in ("AAA", "BBB")]
    monkeypatch.setattr(autonomous_loop, "screen_universe", lambda *a, **k: list(rows))
    monkeypatch.setattr(autonomous_loop, "rank_candidates", lambda *a, **k: list(rows))
    monkeypatch.setattr(autonomous_loop, "get_sp500_tickers", lambda *a, **k: ["AAA", "BBB"])
    monkeypatch.setattr(autonomous_loop, "get_russell1000_tickers", lambda *a, **k: ["AAA"])
    monkeypatch.setattr(autonomous_loop, "BigQueryClient", lambda *a, **k: _FakeBQ())

    # SPEND GUARD -- keep this. Emptying `screen_universe` is NOT enough to empty
    # the funnel, because `rank_candidates` above is also stubbed to a constant
    # 2-row list; a test that only overrode `screen_universe` still reached the
    # real analysis and spawned real `claude` CLI subprocesses via
    # claude_code_client.claude_code_invoke. That happened once while writing
    # this file and was caught by a stack dump, not by a test.
    #
    # So: (1) a benign default analysis, and (2) a tripwire that makes any future
    # escape FAIL LOUDLY instead of quietly spending. An individual test may
    # override _run_single_analysis; nothing may re-enable the rail.
    async def _stub_analysis(ticker, settings, **kwargs):
        return {"ticker": ticker, "_path": "lite", "total_cost_usd": 0.0,
                "recommendation": "HOLD", "final_weighted_score": 5.0}

    monkeypatch.setattr(autonomous_loop, "_run_single_analysis", _stub_analysis)
    monkeypatch.setattr(autonomous_loop, "_persist_analysis", _stub_analysis)

    import backend.agents.claude_code_client as ccc

    def _no_spend(*a, **k):
        raise AssertionError(
            "SPEND GUARD TRIPPED: a phase-85.6 test reached the real Claude Code "
            "rail. Stub the analysis path; never let a unit test spawn the CLI."
        )

    monkeypatch.setattr(ccc, "claude_code_invoke", _no_spend)
    return tmp_path


class _CycleTrader:
    """Trader stub for the cycle: the ONLY real logic is the anchor roll, which
    delegates to the production PaperTrader.roll_daily_anchor bound to our state."""

    def __init__(self, ks: FakeKillSwitchState):
        self.bq = _FakeBQ()
        self._real = _trader(ks, total_nav=23_500.0)
        self.ks = ks

    def roll_daily_anchor(self):
        return self._real.roll_daily_anchor()

    def mark_to_market(self):
        return {"nav": 23_500.0, "cash": 0.0, "pnl_pct": 0.0, "positions": []}

    def check_and_enforce_kill_switch(self):
        return {"paused": False, "any_breached": False}

    def get_positions(self):
        return []

    def save_daily_snapshot(self, **k):
        return None

    def __getattr__(self, name):
        def _noop(*a, **k):
            return []

        return _noop


def _settings(**over) -> Settings:
    base = {
        "lite_mode": True,
        "paper_analyze_top_n": 2,
        "paper_screen_top_n": 4,
        "paper_max_daily_cost_usd": 100.0,
        "macro_regime_filter_enabled": False,
        "pead_signal_enabled": False,
        "news_screen_enabled": False,
        "meta_scorer_enabled": False,
    }
    base.update(over)
    return Settings().model_copy(update=base)


def test_c1_a_cycle_that_dies_mid_analysis_still_leaves_a_fresh_anchor(
    cycle_env, monkeypatch
):
    """CRITERION 1, PROVEN BY A CYCLE -- not by inspection.

    The anchor starts stale (2026-08-05, the real production value). The cycle
    is made to die in `analyzing`, exactly as the 08-06 and 08-07 cycles did.
    Afterwards the anchor must be TODAY's.
    """
    ks = FakeKillSwitchState(sod_nav=23_830.46, sod_date="2026-08-05",
                             peak_nav=24_666.57, paused=False)
    trader = _CycleTrader(ks)
    monkeypatch.setattr(autonomous_loop, "PaperTrader", lambda *a, **k: trader)

    entered_analysis: list[str] = []

    async def _hang(ticker, settings, **kwargs):
        entered_analysis.append(ticker)
        await asyncio.sleep(300)

    monkeypatch.setattr(autonomous_loop, "_run_single_analysis", _hang)

    summary = asyncio.run(
        autonomous_loop.run_daily_cycle(settings=_settings(paper_cycle_max_seconds=10.0))
    )

    assert entered_analysis, (
        "PRECONDITION FAILED: the cycle never reached the analysis phase, so it "
        "did not die where the deadlock happens and this test proves nothing"
    )
    assert summary["status"] == "timeout", summary
    assert "analyzing" in summary["steps"]

    snap = ks.snapshot()
    assert snap["sod_date"] == _today(), (
        f"DEADLOCK INTACT: the cycle died in analyzing and left the anchor at "
        f"{snap['sod_date']!r}"
    )
    assert snap["sod_nav"] == 23_500.0
    assert summary["sod_anchor_roll"]["rolled"] is True


def test_c1_the_roll_runs_before_screening_not_after_analysis(cycle_env, monkeypatch):
    """Ordering is the entire fix. If the roll ever drifts back behind the
    analysis phase the deadlock returns, so pin the ORDER, not just the outcome."""
    ks = FakeKillSwitchState(sod_nav=100.0, sod_date="2026-08-05")
    trader = _CycleTrader(ks)
    monkeypatch.setattr(autonomous_loop, "PaperTrader", lambda *a, **k: trader)

    summary = asyncio.run(autonomous_loop.run_daily_cycle(settings=_settings()))

    # Read the ORDER THE CYCLE ITSELF RECORDED. An earlier version of this test
    # timed when `screen_universe` was CALLED, which the mutation matrix (M2)
    # walked straight through: moving the roll to just after
    # summary["steps"].append("screening") still puts it before screen_universe
    # runs. summary["steps"] is the cycle's own account of its phase order and
    # cannot be gamed that way.
    steps = summary["steps"]
    assert "sod_anchor_roll" in steps, f"the roll never ran: {steps}"
    assert "screening" in steps, f"PRECONDITION: screening never ran: {steps}"
    assert steps.index("sod_anchor_roll") < steps.index("screening"), steps
    assert steps[0] == "sod_anchor_roll", (
        f"the roll must be the FIRST thing the cycle does, got {steps[0]!r}"
    )


def test_c1_the_anchor_survives_a_crash_in_analysis_too(cycle_env, monkeypatch):
    """Not only the timeout shape: an exception escaping the analysis phase must
    also leave a fresh anchor."""
    ks = FakeKillSwitchState(sod_nav=100.0, sod_date="2026-08-05")
    trader = _CycleTrader(ks)
    monkeypatch.setattr(autonomous_loop, "PaperTrader", lambda *a, **k: trader)

    async def _boom(*a, **k):
        raise RuntimeError("injected: analysis dispatch exploded")

    monkeypatch.setattr(autonomous_loop, "dispatch_analyses", _boom)

    summary = asyncio.run(autonomous_loop.run_daily_cycle(settings=_settings()))
    assert summary["status"] == "error", summary
    assert ks.snapshot()["sod_date"] == _today()


def test_c1_the_cycle_records_the_roll_outcome_for_the_operator(cycle_env, monkeypatch):
    """A roll that silently refused would re-create the deadlock invisibly."""
    ks = FakeKillSwitchState(sod_nav=100.0, sod_date="2026-08-05")
    trader = _CycleTrader(ks)
    trader._real.get_or_create_portfolio = lambda: {"total_nav": 0.0, "starting_capital": 0.0}
    monkeypatch.setattr(autonomous_loop, "PaperTrader", lambda *a, **k: trader)

    summary = asyncio.run(autonomous_loop.run_daily_cycle(settings=_settings()))
    assert summary["sod_anchor_roll"]["rolled"] is False
    assert summary["sod_anchor_roll"]["reason"] == "refused_non_positive_nav"
    assert "sod_anchor_roll" in summary["steps"]


# ─────────────────────────────────────────────────────────────────────────────
# Criterion 2 -- the 409 message must match the real mechanism
# ─────────────────────────────────────────────────────────────────────────────

BANNED = [
    "NO operator action is required",
    "this refusal clears itself",
    "at the top of the "
    "next paper-trading cycle",
]


def test_c2_the_409_no_longer_makes_the_two_false_promises():
    import backend.api.paper_trading as pt

    src = __import__("inspect").getsource(pt.resume_trading)
    # Strip comments so the explanatory block that QUOTES the old text (to record
    # why it was wrong) is not mistaken for the message itself.
    live = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    for phrase in BANNED:
        assert phrase not in live, (
            f"the 409 still promises {phrase!r} -- it was false and cost days"
        )


def test_c2_the_409_names_the_actual_unblock_mechanism():
    import backend.api.paper_trading as pt

    src = __import__("inspect").getsource(pt.resume_trading)
    live = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    assert "UNBLOCK CONDITION" in live
    assert "roll_daily_anchor" in live, "the message must name the real roller"
    assert "weekday-only" in live, "the weekend case must be stated"
    assert "will \\nNOT clear on its own" in live or "NOT clear on its own" in live


def test_c2_the_mechanism_the_message_names_actually_exists():
    """CLAIM AUDIT, not a string check. The message tells the operator the roll
    runs at Step 0 via PaperTrader.roll_daily_anchor -> update_sod_nav. Verify
    each named thing is real, so the message cannot go stale silently."""
    import inspect

    from backend.services import autonomous_loop as loop
    from backend.services.paper_trader import PaperTrader

    assert hasattr(PaperTrader, "roll_daily_anchor")
    roll_src = inspect.getsource(PaperTrader.roll_daily_anchor)
    assert "update_sod_nav" in roll_src, "the roller does not call update_sod_nav"

    cycle_src = inspect.getsource(loop.run_daily_cycle)
    # Anchor on the CALL EXPRESSION, not the bare name: the explanatory comment
    # block above the call also says "roll_daily_anchor", so a bare-name index()
    # matches the comment and reports the right order even after the call has
    # been moved. Mutation M2 walked through exactly that.
    call = "await asyncio.to_thread(trader.roll_daily_anchor)"
    assert call in cycle_src, "the cycle does not call the roller"
    assert cycle_src.index(call) < cycle_src.index(
        'summary["steps"].append("screening")'
    ), "the roll is no longer ahead of screening -- the deadlock is back"
