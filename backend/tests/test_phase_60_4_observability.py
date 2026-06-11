"""phase-60.4 tests: observability + ops residuals (AW-7, AW-1/2 residuals, AW-10, hygiene).

The immutable selector is
`-k 'cc_rail_log or ingestion_silence or ticket_failure or redact or 60_4'`;
test names embed those terms.
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ── criterion 1: cc_rail_log writer ──────────────────────────────────────


def _envelope(in_tok=4248, out_tok=4768):
    return {
        "usage": {
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cache_read_input_tokens": 100,
            "cache_creation_input_tokens": 50,
        },
        "session_id": "sess-abc123",
        "result": "ok-text",
    }


def test_cc_rail_log_writer_success(monkeypatch):
    from backend.agents.claude_code_client import ClaudeCodeClient
    import backend.services.observability.api_call_log as acl

    captured = {}
    monkeypatch.setattr(acl, "log_llm_call", lambda **kw: captured.update(kw))

    ClaudeCodeClient._log_cc_call(
        _envelope(), agent="lite_trader", ticker="MU",
        latency_ms=88862.0, model="claude-sonnet-4-6", ok=True,
    )
    assert captured["provider"] == "anthropic"
    assert captured["model"] == "claude-sonnet-4-6"
    assert captured["agent"] == "cc_rail:lite_trader"
    assert captured["ticker"] == "MU"
    assert captured["input_tok"] == 4248 and captured["output_tok"] == 4768
    assert captured["latency_ms"] == pytest.approx(88862.0)
    assert captured["ok"] is True
    assert captured["request_id"] == "sess-abc123"


def test_cc_rail_log_writer_error_path(monkeypatch):
    from backend.agents.claude_code_client import ClaudeCodeClient
    import backend.services.observability.api_call_log as acl

    captured = {}
    monkeypatch.setattr(acl, "log_llm_call", lambda **kw: captured.update(kw))
    ClaudeCodeClient._log_cc_call(
        None, agent=None, ticker=None, latency_ms=120000.0,
        model="claude-sonnet-4-6", ok=False,
    )
    assert captured["ok"] is False and captured["agent"] == "cc_rail"
    assert captured["input_tok"] == 0


def test_cc_rail_log_generate_content_wires_writer(monkeypatch):
    import backend.agents.claude_code_client as ccc
    from backend.agents.claude_code_client import ClaudeCodeClient

    monkeypatch.setattr(ccc, "claude_code_invoke", lambda *a, **k: _envelope())
    calls = []
    monkeypatch.setattr(
        ClaudeCodeClient, "_log_cc_call",
        staticmethod(lambda env, **kw: calls.append((env, kw))),
    )
    client = ClaudeCodeClient("claude-sonnet-4-6")
    resp = client.generate_content(
        "hi", generation_config={"_role": "Market", "_ticker": "MU"},
    )
    assert resp.text == "ok-text"
    assert len(calls) == 1
    env, kw = calls[0]
    assert kw["agent"] == "Market" and kw["ticker"] == "MU" and kw["ok"] is True


# ── criterion 2: ingestion_silence + ticket_failure ─────────────────────


def _mk_tickets_db(tmp_path, last_created_at: str | None):
    """Real TicketsDB schema (its __init__ creates it), then a raw row."""
    from backend.db.tickets_db import TicketsDB

    db = TicketsDB(db_path=str(tmp_path / "tickets.db"))
    if last_created_at:
        conn = sqlite3.connect(db.db_path)
        conn.execute(
            "INSERT INTO tickets (ticket_number, source, sender_id, message_text, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (5100, "slack", "U1", "test", last_created_at),
        )
        conn.commit()
        conn.close()
    return db


def test_ingestion_silence_age_computation(tmp_path):
    ten_days_ago = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    db = _mk_tickets_db(tmp_path, ten_days_ago)
    age = db.get_last_ticket_age_days()
    assert age == pytest.approx(10.0, abs=0.1)


def test_ingestion_silence_fresh_ticket_under_threshold(tmp_path):
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    db = _mk_tickets_db(tmp_path, yesterday)
    age = db.get_last_ticket_age_days()
    assert age is not None and age < 7


def test_ingestion_silence_empty_table_returns_none(tmp_path):
    db = _mk_tickets_db(tmp_path, None)
    assert db.get_last_ticket_age_days() is None


def test_ticket_failure_notice_posts_to_channel(monkeypatch):
    from backend.services.queue_notification import QueueNotificationService

    svc = QueueNotificationService()
    posted = {}

    class _FakeSlack:
        def chat_postMessage(self, **kw):
            posted.update(kw)

    svc.slack_client = _FakeSlack()
    svc._slack_client_initialized = True

    ticket = {
        "ticket_number": 5101, "source": "slack", "channel_id": "C0FORD",
        "slack_thread_id": "1781111.0001",
        "message_text": "please check the optimizer results",
    }
    ok = asyncio.run(svc.send_ticket_failure_notification(ticket, "Max retries (3) exceeded"))
    assert ok is True
    assert posted["channel"] == "C0FORD"
    assert posted["thread_ts"] == "1781111.0001"
    assert "FAILED" in posted["text"] and "Max retries (3) exceeded" in posted["text"]
    assert "#5101" in posted["text"]


def test_ticket_failure_notice_fail_open_without_client():
    from backend.services.queue_notification import QueueNotificationService

    svc = QueueNotificationService()
    svc.slack_client = None
    svc._slack_client_initialized = True
    ok = asyncio.run(svc.send_ticket_failure_notification(
        {"ticket_number": 1, "source": "slack", "channel_id": "C1"}, "err",
    ))
    assert ok is False  # never raises


# ── criterion 3: event-loop hygiene + watchdog busy-vs-down ─────────────


def test_60_4_no_naked_yfinance_in_async_analyzers():
    """Structural: the lite analyzers must not call .info / .history on the
    event loop -- the fetch goes through _fetch_yf_market_data via
    asyncio.to_thread (the away-week watchdog ReadTimeouts were this)."""
    import inspect

    import backend.services.autonomous_loop as al

    for fn in (al._run_claude_analysis, al._run_gemini_analysis):
        src = inspect.getsource(fn)
        assert "stock.info" not in src and "stock.history(" not in src, fn.__name__
        assert "asyncio.to_thread(_fetch_yf_market_data" in src, fn.__name__


def test_60_4_watchdog_cycle_state_line_busy(monkeypatch):
    import backend.slack_bot.scheduler as sched
    import backend.services.cycle_lock as cl

    monkeypatch.setattr(cl, "inspect_lock", lambda: {
        "is_stale": False, "started_at": "2026-06-11T18:00:00Z",
        "age_sec": 1200.0, "pid": 123, "pid_alive": True,
    })
    line = sched._cycle_state_line()
    assert "IN PROGRESS" in line and "BUSY" in line


def test_60_4_watchdog_cycle_state_line_down(monkeypatch):
    import backend.slack_bot.scheduler as sched
    import backend.services.cycle_lock as cl

    monkeypatch.setattr(cl, "inspect_lock", lambda: None)
    line = sched._cycle_state_line()
    assert "DOWN" in line and "lockfile absent" in line


# ── criterion 4: meta-scorer fallback surfaced ──────────────────────────


def test_60_4_meta_scorer_state_line_degraded(monkeypatch):
    import backend.slack_bot.scheduler as sched
    import backend.services.cycle_health as ch

    monkeypatch.setattr(ch, "get_log", lambda: SimpleNamespace(
        last_cycles=lambda n: [{"meta_scorer_degraded": True}],
    ))
    line = sched._meta_scorer_state_line()
    assert "DEGRADED" in line and "no-LLM fallbacks" in line


def test_60_4_meta_scorer_state_line_healthy(monkeypatch):
    import backend.slack_bot.scheduler as sched
    import backend.services.cycle_health as ch

    monkeypatch.setattr(ch, "get_log", lambda: SimpleNamespace(
        last_cycles=lambda n: [{"meta_scorer_degraded": False}],
    ))
    assert sched._meta_scorer_state_line() == ""  # digest byte-identical


# ── criterion 5: redaction ───────────────────────────────────────────────


def test_redact_synthetic_key_through_handler():
    from backend.services.observability.log_redaction import SecretRedactionFilter

    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    lg = logging.getLogger("httpx.test60_4")
    lg.propagate = False
    h = _Capture()
    h.addFilter(SecretRedactionFilter())
    lg.addHandler(h)
    try:
        lg.warning(
            'HTTP Request: GET https://api.stlouisfed.org/fred/series?'
            'series_id=DGS10&api_key=synthetic1234567890abcdef&file_type=json "200 OK"'
        )
    finally:
        lg.removeHandler(h)
    assert len(records) == 1
    assert "synthetic1234567890abcdef" not in records[0]
    assert "api_key=***REDACTED***" in records[0]
    assert "series_id=DGS10" in records[0]  # non-secret params untouched


def test_redact_function_classes():
    from backend.services.observability.log_redaction import redact_secrets

    s = redact_secrets("token=abcd1234efgh5678 and access_token=zyxw9876zyxw9876")
    assert "abcd1234efgh5678" not in s and "zyxw9876zyxw9876" not in s
    # Short values untouched (no false positives on key=1)
    assert redact_secrets("retry key=1 ok") == "retry key=1 ok"


def test_redact_filter_never_drops_records():
    from backend.services.observability.log_redaction import SecretRedactionFilter

    rec = logging.LogRecord("x", logging.INFO, "f", 1, "clean message", (), None)
    assert SecretRedactionFilter().filter(rec) is True
    assert rec.getMessage() == "clean message"
