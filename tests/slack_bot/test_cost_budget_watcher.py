"""phase-9.8 tests."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.slack_bot.jobs.cost_budget_watcher import run
from backend.slack_bot.job_runtime import IdempotencyStore


def test_under_budget_no_trip():
    alerts: list = []
    store = IdempotencyStore()
    out = run(
        daily_spend_usd=1.0,
        monthly_spend_usd=5.0,
        daily_cap_usd=5.0,
        monthly_cap_usd=50.0,
        alert_fn=lambda r, s: alerts.append((r, s)),
        store=store,
        day="2026-04-20",
    )
    assert not out["tripped"]
    assert alerts == []


def test_daily_over_budget_trips():
    alerts: list = []
    store = IdempotencyStore()
    out = run(
        daily_spend_usd=10.0,  # > 5 daily cap
        monthly_spend_usd=20.0,
        daily_cap_usd=5.0,
        monthly_cap_usd=50.0,
        alert_fn=lambda r, s: alerts.append((r, s)),
        store=store,
        day="2026-04-20",
    )
    assert out["tripped"] is True
    assert out["reason"] == "daily"
    assert len(alerts) == 1


def test_monthly_over_budget_trips():
    alerts: list = []
    store = IdempotencyStore()
    out = run(
        daily_spend_usd=1.0,
        monthly_spend_usd=100.0,  # > 50 monthly cap
        daily_cap_usd=5.0,
        monthly_cap_usd=50.0,
        alert_fn=lambda r, s: alerts.append((r, s)),
        store=store,
        day="2026-04-20",
    )
    assert out["tripped"] is True
    # Whichever scope alerts first wins; both daily and monthly checked.
    assert len(alerts) == 1


def test_alert_fn_injectable():
    out = run(
        daily_spend_usd=100.0,
        monthly_spend_usd=1.0,
        daily_cap_usd=5.0,
        monthly_cap_usd=50.0,
        alert_fn=None,  # default logger-warning path
        store=IdempotencyStore(),
        day="2026-04-20",
    )
    assert out["tripped"] is True
