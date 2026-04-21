"""phase-9.2 tests for backend.slack_bot.jobs.daily_price_refresh."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.slack_bot.jobs.daily_price_refresh import run, JOB_NAME
from backend.slack_bot.job_runtime import IdempotencyStore, IdempotencyKey


def test_run_writes_rows_via_injected_fns():
    store = IdempotencyStore()
    fetched = run(
        tickers=["AAPL", "MSFT"],
        fetch_fn=lambda ts: {t: {"close": 1.0} for t in ts},
        write_fn=lambda rows: len(rows),
        store=store,
        day="2026-04-20",
    )
    assert fetched["written"] == 2
    assert not fetched["skipped"]


def test_idempotency_dedups_same_day():
    store = IdempotencyStore()
    key = IdempotencyKey.daily(JOB_NAME, day="2026-04-20")
    run(tickers=["AAPL"], fetch_fn=lambda ts: {}, write_fn=lambda r: 0, store=store, day="2026-04-20")
    assert store.seen(key)
    # Second call: skipped
    out2 = run(tickers=["AAPL"], fetch_fn=lambda ts: {"AAPL": {}}, write_fn=lambda r: 99, store=store, day="2026-04-20")
    assert out2["skipped"] is True
    assert out2["written"] == 0


def test_no_live_yfinance_call(monkeypatch):
    """run() must NOT import yfinance when fetch_fn is injected."""
    import sys as _s
    _s.modules.pop("yfinance", None)
    store = IdempotencyStore()
    run(
        tickers=["AAPL"],
        fetch_fn=lambda ts: {"AAPL": {}},
        write_fn=lambda r: 1,
        store=store,
        day="2026-04-20",
    )
    assert "yfinance" not in _s.modules
