"""phase-66.1 tests: cc_rail guard -- probe gate + circuit breaker + single P1 page.

Immutable verification command:
`python -m pytest backend/tests/test_phase_66_1_rail_guard.py -q`

Covers the 66.1 criteria:
1. probe gate: forced probe failure -> ZERO cc_rail invocation attempts
2. breaker: threshold trip -> exactly ONE P1 page (transition latch); no cycle
   can log >threshold consecutive failed calls without a page
4. import-path regression: the four autonomous_loop P1 sites import a module
   that actually exists (the away-window zero-pages root cause)
(criterion 3 is live scheduled-cycle BQ evidence -- not unit-testable.)

Isolation: every test runs against a fresh guard; the BQ log writer and the
Slack alerter are monkeypatched autouse so tests can never write llm_call_log
rows or page the real channel (no .env-bleed: nothing here reads backend/.env).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import backend.agents.claude_code_client as ccc  # noqa: E402
import backend.services.observability.alerting as alerting  # noqa: E402
import backend.services.observability.api_call_log as acl  # noqa: E402


def _envelope(text="ok-text"):
    return {
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "session_id": "sess-t",
        "result": text,
    }


@pytest.fixture(autouse=True)
def _isolated_guard(monkeypatch):
    """Fresh guard per test; BQ writer + Slack pager stubbed; threshold=5."""
    pages: list[dict] = []
    monkeypatch.setattr(acl, "log_llm_call", lambda **kw: None)
    monkeypatch.setattr(
        alerting, "raise_cron_alert_sync",
        lambda **kw: pages.append(kw) or True,
    )
    monkeypatch.setattr(ccc, "_rail_breaker_threshold", lambda: 5)
    ccc.rail_guard_reset("test-cycle")
    yield pages
    ccc.rail_guard_reset(None)


# ── criterion 4 regression: the zero-pages import-path bug ───────────────


def test_rail_guard_import_path_bug_fixed():
    """backend.services.alerting DOES NOT EXIST -- importing it was the
    away-window zero-pages root cause. The four P1 sites must now import
    backend.services.observability.alerting (which must resolve)."""
    assert importlib.util.find_spec("backend.services.alerting") is None
    assert importlib.util.find_spec("backend.services.observability.alerting") is not None
    src = (REPO_ROOT / "backend/services/autonomous_loop.py").read_text()
    assert "from backend.services.alerting import" not in src
    assert src.count(
        "from backend.services.observability.alerting import raise_cron_alert"
    ) >= 4
    assert callable(alerting.raise_cron_alert) and callable(alerting.raise_cron_alert_sync)


# ── criterion 1: probe gate skips ALL invocations ────────────────────────


def test_rail_guard_probe_gate_zero_invocations(monkeypatch, _isolated_guard):
    attempts = []
    monkeypatch.setattr(
        ccc, "claude_code_invoke",
        lambda *a, **k: attempts.append(1) or _envelope(),
    )
    ccc.rail_guard_disable("forced probe failure (test)")

    client = ccc.ClaudeCodeClient("claude-sonnet-4-6")
    for _ in range(10):
        resp = client.generate_content("hi")
        assert resp.text == ""
        assert resp.thoughts.startswith("rail_guard_skipped: probe gate")

    assert attempts == []  # ZERO subprocess-path attempts
    st = ccc.rail_guard_status()
    assert st["rail_skipped"] is True and st["skipped_calls"] == 10
    # the loop's probe site owns the page; the guard must not add one
    assert _isolated_guard == []


# ── criterion 2: breaker trips at threshold, pages exactly once ──────────


def test_rail_guard_breaker_trips_and_pages_exactly_once(monkeypatch, _isolated_guard):
    attempts = []

    def _boom(*a, **k):
        attempts.append(1)
        raise ccc.ClaudeCodeError("simulated 401 auth failure")

    monkeypatch.setattr(ccc, "claude_code_invoke", _boom)
    client = ccc.ClaudeCodeClient("claude-sonnet-4-6")

    for _ in range(9):
        resp = client.generate_content("hi")
        assert resp.text == ""

    # threshold=5: exactly 5 real attempts, then the breaker eats calls 6-9
    assert len(attempts) == 5
    st = ccc.rail_guard_status()
    assert st["breaker_tripped"] is True
    assert st["consecutive_failures"] == 5
    assert st["skipped_calls"] == 4
    # exactly ONE P1, on the closed->open transition
    assert len(_isolated_guard) == 1
    page = _isolated_guard[0]
    assert page["severity"] == "P1"
    assert page["source"] == "claude_code_rail"
    assert page["error_type"] == "breaker_open"
    assert page["details"]["consecutive_failures"] == 5


def test_rail_guard_no_page_when_probe_already_paged(_isolated_guard):
    """Probe-gate consumes the latch: breaker opening later in the same
    cycle must not page the same rail-down incident twice."""
    ccc.rail_guard_disable("probe down (test)")
    for _ in range(7):
        ccc._rail_guard_record_failure("late failure")
    assert _isolated_guard == []


def test_rail_guard_success_resets_consecutive_count(monkeypatch, _isolated_guard):
    monkeypatch.setattr(ccc, "claude_code_invoke", lambda *a, **k: _envelope())
    client = ccc.ClaudeCodeClient("claude-sonnet-4-6")

    for _ in range(4):  # threshold-1 failures...
        ccc._rail_guard_record_failure("blip")
    assert client.generate_content("hi").text == "ok-text"  # ...then a success
    for _ in range(4):
        ccc._rail_guard_record_failure("blip")

    st = ccc.rail_guard_status()
    assert st["breaker_tripped"] is False  # 4+4 non-consecutive never trips
    assert _isolated_guard == []


def test_rail_guard_reset_reenables_rail_next_cycle(monkeypatch, _isolated_guard):
    for _ in range(6):
        ccc._rail_guard_record_failure("down")
    assert ccc.rail_guard_status()["breaker_tripped"] is True

    ccc.rail_guard_reset("next-cycle")  # per-cycle window reset
    st = ccc.rail_guard_status()
    assert st["breaker_tripped"] is False and st["consecutive_failures"] == 0
    assert st["cycle_id"] == "next-cycle"

    monkeypatch.setattr(ccc, "claude_code_invoke", lambda *a, **k: _envelope())
    client = ccc.ClaudeCodeClient("claude-sonnet-4-6")
    assert client.generate_content("hi").text == "ok-text"


def test_rail_guard_healthy_path_unchanged(monkeypatch, _isolated_guard):
    """Guard closed -> byte-identical behavior: invoke called, text returned,
    ok=True row logged, no pages, nothing skipped."""
    monkeypatch.setattr(ccc, "claude_code_invoke", lambda *a, **k: _envelope())
    logged = []
    monkeypatch.setattr(
        ccc.ClaudeCodeClient, "_log_cc_call",
        staticmethod(lambda env, **kw: logged.append(kw)),
    )
    client = ccc.ClaudeCodeClient("claude-sonnet-4-6")
    resp = client.generate_content("hi", generation_config={"_role": "Market"})
    assert resp.text == "ok-text"
    assert len(logged) == 1 and logged[0]["ok"] is True
    st = ccc.rail_guard_status()
    assert st["skipped_calls"] == 0 and st["consecutive_failures"] == 0
    assert _isolated_guard == []


# ── cycle_health wiring ──────────────────────────────────────────────────


def test_rail_guard_cycle_history_row_carries_flags(tmp_path, monkeypatch):
    import backend.services.cycle_health as ch

    monkeypatch.setattr(ch, "_HISTORY_PATH", tmp_path / "cycle_history.jsonl")
    log = ch.CycleHealthLog()
    log.record_cycle_end(
        cycle_id="c1", started_at="2026-07-06T18:00:00+00:00",
        status="completed", rail_skipped=True, breaker_tripped=True,
    )
    import json

    row = json.loads((tmp_path / "cycle_history.jsonl").read_text().splitlines()[-1])
    assert row["rail_skipped"] is True and row["breaker_tripped"] is True
