"""phase-85.4 -- a non-completing cycle must be LOUD.

Criterion 3 (verbatim from .claude/masterplan.json):

    a non-completing cycle becomes LOUD: a terminal row is always written to
    cycle_history.jsonl (including on timeout/crash) AND an alert fires naming
    the phase it died in -- proven by a fault-injected cycle that dies
    mid-analysis, not by inspection

So these tests do NOT read source code. They drive the REAL
`autonomous_loop.run_daily_cycle` with a fault injected into
`_run_single_analysis` -- i.e. inside the analysis phase, after the cycle has
appended "analyzing" to summary["steps"] -- and then assert against the
artifacts the cycle actually produced: the JSONL row and the dispatched alert.

Three fault modes are injected, because the three real failures observed in
production were not the same failure:

  * TIMEOUT   -- the 2026-08-06 and 2026-08-07 shape (asyncio.timeout fires
                 while tickers are still in flight)
  * CRASH     -- an unhandled exception escaping the analysis phase
  * KILL-SWITCH HALT -- the 2026-08-05 shape, the one this step FIXES: the
                 cycle reaches the mark/trade region, the switch is paused,
                 and the early return at autonomous_loop.py:~1327 used to
                 leave summary["status"] at the ":362" placeholder "running"

Isolation: `cycle_history.jsonl`, the cycle heartbeat, and the cycle LOCKFILE
are all redirected into tmp_path, and `raise_cron_alert_sync` is replaced by a
recorder. Nothing here touches the operator's live handoff/ tree or Slack.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from backend.config.settings import Settings
from backend.services import autonomous_loop, cycle_health, cycle_lock


# ─────────────────────────────────────────────────────────────────────────────
# Isolation
# ─────────────────────────────────────────────────────────────────────────────


class _StubKillSwitchState:
    """Injected replacement for the module singleton `kill_switch.get_state()`.

    WHY THIS EXISTS -- and it is not cosmetic. `cycle_halt_reason` is called
    with `_ks_state().is_paused()`, which reads the module singleton, which
    replays the OPERATOR'S REAL on-disk audit journal
    (handoff/kill_switch_audit.jsonl). Without this stub these tests pass only
    because the operator's live book happens to be paused, and would flip to
    red the moment it is resumed -- the phase-36.28 coupling class, in a test
    written to prove a fix for a different defect.

    Caught by scripts/qa/mutation_matrix_85_4.py M9: neutering the FAKE
    TRADER's kill-switch verdict did not turn the tests red, because the live
    singleton was answering instead.
    """

    def __init__(self, paused: bool) -> None:
        self._paused = paused

    def is_paused(self) -> bool:
        return self._paused


@pytest.fixture(autouse=True)
def _isolate_everything(monkeypatch, tmp_path):
    """Redirect every durable artifact the cycle writes into tmp_path, and cut
    every read of live operator state."""
    monkeypatch.setattr(cycle_health, "_HISTORY_PATH", tmp_path / "cycle_history.jsonl")
    monkeypatch.setattr(cycle_health, "_HEARTBEAT_PATH", tmp_path / ".cycle_heartbeat.json")
    monkeypatch.setattr(cycle_lock, "_LOCK_PATH", tmp_path / ".autonomous_loop.lock")
    # cycle_health.get_log() memoises a singleton whose writes go through the
    # module-level path, so resetting the singleton is not required -- but the
    # module-level _running flag IS process state and must be reset.
    monkeypatch.setattr(autonomous_loop, "_running", False, raising=False)

    # Default: a HEALTHY (not paused) switch, so a test that wants the halt
    # path must ASK for it via `paused_kill_switch`. Never the live journal.
    import backend.services.kill_switch as ks

    monkeypatch.setattr(ks, "get_state", lambda: _StubKillSwitchState(False))
    yield


@pytest.fixture
def paused_kill_switch(monkeypatch):
    """Inject a PAUSED switch, explicitly and locally."""
    import backend.services.kill_switch as ks

    monkeypatch.setattr(ks, "get_state", lambda: _StubKillSwitchState(True))
    return True


@pytest.fixture
def alerts(monkeypatch):
    """Capture every alert the cycle raises instead of posting to Slack."""
    captured: list[dict] = []

    def _recorder(source, error_type, severity, title, details):
        captured.append(
            {
                "source": source,
                "error_type": error_type,
                "severity": severity,
                "title": title,
                "details": details,
            }
        )
        return True

    import backend.services.observability.alerting as alerting

    monkeypatch.setattr(alerting, "raise_cron_alert_sync", _recorder)
    return captured


class _FakeBQ:
    """Every BigQuery call the cycle makes, neutered."""

    def __getattr__(self, name):
        def _noop(*a, **k):
            return []

        return _noop


class _FakeTrader:
    """Minimal PaperTrader surface for the phases this test reaches."""

    def __init__(self, *a, **k):
        self.bq = _FakeBQ()
        self.kill_switch_paused = False
        self.mark_calls = 0

    def get_portfolio_state(self):
        return {"nav": 10_000.0, "cash": 10_000.0, "pnl_pct": 0.0, "positions": []}

    def mark_to_market(self):
        self.mark_calls += 1
        return {"nav": 10_000.0, "cash": 10_000.0, "pnl_pct": 0.0, "positions": []}

    def check_and_enforce_kill_switch(self):
        return {"paused": self.kill_switch_paused, "any_breached": False}

    def save_daily_snapshot(self, **k):
        return None

    def get_positions(self):
        return []

    def get_open_positions(self):
        return []

    def __getattr__(self, name):
        # Unknown trader methods return an empty list: the cycle iterates over
        # most of them, and an empty list is the "nothing to do" answer for
        # every one it iterates. Anything that needs a dict is defined above.
        def _noop(*a, **k):
            return []

        return _noop


def _settings() -> Settings:
    """Real Settings so every attribute the cycle touches exists, with the
    feature flags forced to the shape this test needs."""
    s = Settings()
    return s.model_copy(
        update={
            "lite_mode": True,
            "paper_analyze_top_n": 2,
            "paper_screen_top_n": 4,
            "paper_max_daily_cost_usd": 100.0,
            # Every optional screen OFF so the mock surface stays honest.
            "macro_regime_filter_enabled": False,
            "pead_signal_enabled": False,
            "news_screen_enabled": False,
            "meta_scorer_enabled": False,
            "paper_merged_analysis_dispatch_enabled": False,
        }
    )


def _install_screening(monkeypatch, tickers: list[str]):
    """Short-circuit Step 1 so the cycle reaches Step 3 without network."""
    rows = [{"ticker": t, "score": 9.0, "price": 100.0} for t in tickers]
    monkeypatch.setattr(autonomous_loop, "screen_universe", lambda *a, **k: list(rows))
    monkeypatch.setattr(autonomous_loop, "rank_candidates", lambda *a, **k: list(rows))
    monkeypatch.setattr(autonomous_loop, "get_sp500_tickers", lambda *a, **k: list(tickers))
    monkeypatch.setattr(autonomous_loop, "get_russell1000_tickers", lambda *a, **k: list(tickers))
    monkeypatch.setattr(autonomous_loop, "BigQueryClient", lambda *a, **k: _FakeBQ())


def _read_rows(tmp_path: Path) -> list[dict]:
    p = tmp_path / "cycle_history.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _terminal_rows(tmp_path: Path) -> list[dict]:
    return [r for r in _read_rows(tmp_path) if r.get("status") != "started"]


def _run(settings) -> dict:
    return asyncio.run(autonomous_loop.run_daily_cycle(settings=settings))


# ─────────────────────────────────────────────────────────────────────────────
# Criterion 3 -- fault-injected cycles
# ─────────────────────────────────────────────────────────────────────────────


def test_c3_timeout_midanalysis_writes_terminal_row_and_names_the_phase(
    monkeypatch, tmp_path, alerts
):
    """The 2026-08-06 / 2026-08-07 shape: the wall-clock budget expires while
    per-ticker analyses are still in flight."""
    trader = _FakeTrader()
    _install_screening(monkeypatch, ["AAA", "BBB"])
    monkeypatch.setattr(autonomous_loop, "PaperTrader", lambda *a, **k: trader)

    entered_analysis: list[str] = []

    async def _hang(ticker, settings, **kwargs):
        entered_analysis.append(ticker)
        await asyncio.sleep(300)  # far beyond the injected budget
        return {"ticker": ticker}

    monkeypatch.setattr(autonomous_loop, "_run_single_analysis", _hang)

    # The budget must outlast the (all-fake, sub-second) screening phase and
    # expire while analyses are in flight. The `entered_analysis` assertion
    # below is the harness asserting its OWN precondition: if screening had
    # eaten the budget the fault would have landed in the wrong phase and this
    # test would be proving nothing.
    s = _settings().model_copy(update={"paper_cycle_max_seconds": 10.0})
    summary = _run(s)

    assert entered_analysis, (
        "PRECONDITION FAILED: the cycle never reached the analysis phase, so "
        "the timeout was not injected mid-analysis and this test proves nothing"
    )
    assert summary["status"] == "timeout", summary

    rows = _terminal_rows(tmp_path)
    assert len(rows) == 1, f"expected exactly one terminal row, got {rows}"
    assert rows[0]["status"] == "timeout"
    assert rows[0]["completed_at"], "terminal row must carry completed_at"

    cycle_alerts = [a for a in alerts if a["source"] == "autonomous_loop"]
    assert cycle_alerts, f"no cycle alert raised; captured={alerts}"
    a = cycle_alerts[-1]
    assert a["severity"] == "P1"
    # The phase must be in the TITLE, not only buried in the details body.
    assert "analyzing" in a["title"], a["title"]
    assert a["details"]["died_in_phase"] == "analyzing", a["details"]


def test_c3_crash_midanalysis_writes_terminal_row_and_names_the_phase(
    monkeypatch, tmp_path, alerts
):
    """An unhandled exception escaping the analysis phase must still produce a
    terminal row plus a phase-named alert."""
    trader = _FakeTrader()
    _install_screening(monkeypatch, ["AAA", "BBB"])
    monkeypatch.setattr(autonomous_loop, "PaperTrader", lambda *a, **k: trader)

    # Raise from the gather itself, not from inside _run_and_persist_one (whose
    # per-ticker try/except would swallow it) -- this models a fault in the
    # dispatch machinery rather than in one ticker.
    async def _boom(*a, **k):
        raise RuntimeError("injected fault: analysis dispatch exploded")

    monkeypatch.setattr(autonomous_loop, "dispatch_analyses", _boom)

    summary = _run(_settings())

    assert summary["status"] == "error", summary
    assert "injected fault" in summary.get("error", "")

    rows = _terminal_rows(tmp_path)
    assert len(rows) == 1, rows
    assert rows[0]["status"] == "error"

    cycle_alerts = [a for a in alerts if a["source"] == "autonomous_loop"]
    assert cycle_alerts, alerts
    a = cycle_alerts[-1]
    assert "analyzing" in a["title"], a["title"]
    assert a["details"]["died_in_phase"] == "analyzing", a["details"]


def test_c3_killswitch_halt_records_a_real_terminal_status_not_running(
    monkeypatch, tmp_path, alerts, paused_kill_switch
):
    """THE REGRESSION THIS STEP EXISTS TO KILL.

    The kill-switch early return used to leave summary["status"] at the
    initializer's placeholder "running", so:
      * cycle_history.jsonl got a terminal row claiming the cycle was RUNNING,
      * the completed-age clock could not tell a halt from a success,
      * and the P1 was titled "Autonomous trading cycle running".

    Measured on the 2026-08-05 cycle -- the one day all six tickers finished.
    """
    trader = _FakeTrader()
    trader.kill_switch_paused = True
    _install_screening(monkeypatch, ["AAA", "BBB"])
    monkeypatch.setattr(autonomous_loop, "PaperTrader", lambda *a, **k: trader)

    async def _ok(ticker, settings, **kwargs):
        return {
            "ticker": ticker,
            "_path": "lite",
            "total_cost_usd": 0.0,
            "recommendation": "HOLD",
            "final_weighted_score": 5.0,
        }

    monkeypatch.setattr(autonomous_loop, "_run_single_analysis", _ok)
    monkeypatch.setattr(autonomous_loop, "_persist_analysis", _ok)

    summary = _run(_settings())

    assert summary.get("halted") is True, summary
    assert summary["status"] != "running", (
        "REGRESSION: the kill-switch halt leaked the ':362' placeholder status"
    )
    assert summary["status"] == "halted_kill_switch", summary["status"]

    rows = _terminal_rows(tmp_path)
    assert len(rows) == 1, rows
    assert rows[0]["status"] == "halted_kill_switch", rows[0]

    cycle_alerts = [a for a in alerts if a["source"] == "autonomous_loop"]
    assert cycle_alerts, alerts
    a = cycle_alerts[-1]
    assert a["severity"] == "P1"
    assert "running" not in a["title"], a["title"]
    assert a["details"]["died_in_phase"] == "kill_switch_halted", a["details"]


def test_c3_halted_status_is_not_counted_as_a_completion(
    monkeypatch, tmp_path, alerts, paused_kill_switch
):
    """A halted cycle must NOT reset the completed-age clock.

    This is the join between criterion 3 and criterion 4: status fidelity is
    only worth having if the health signal reads it.
    """
    trader = _FakeTrader()
    trader.kill_switch_paused = True
    _install_screening(monkeypatch, ["AAA"])
    monkeypatch.setattr(autonomous_loop, "PaperTrader", lambda *a, **k: trader)

    async def _ok(ticker, settings, **kwargs):
        return {"ticker": ticker, "_path": "lite", "total_cost_usd": 0.0}

    monkeypatch.setattr(autonomous_loop, "_run_single_analysis", _ok)
    monkeypatch.setattr(autonomous_loop, "_persist_analysis", _ok)

    _run(_settings())

    verdict = cycle_health.cycle_heartbeat_alarm()
    assert verdict["last_terminal_status"] == "halted_kill_switch"
    assert verdict["last_success_at"] is None, (
        "a kill-switch halt must not register as the last successful cycle"
    )
    assert verdict["success_stale"] is True
