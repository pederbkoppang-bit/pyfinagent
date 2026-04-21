"""phase-9.7 tests."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.slack_bot.jobs.weekly_data_integrity import run, DRIFT_THRESHOLD
from backend.slack_bot.job_runtime import IdempotencyStore


def test_drift_above_threshold_alerts():
    alerts: list = []
    store = IdempotencyStore()
    # 50% drop = 0.50 delta > 0.20 threshold
    out = run(
        current_counts={"alt_congress_trades": 5000},
        prior_counts={"alt_congress_trades": 10000},
        alert_fn=lambda drifts: alerts.extend(drifts),
        store=store,
        iso_year_week="2026-W17",
    )
    assert len(alerts) == 1
    assert alerts[0]["table"] == "alt_congress_trades"
    assert alerts[0]["delta_pct"] > DRIFT_THRESHOLD


def test_drift_below_threshold_no_alert():
    alerts: list = []
    store = IdempotencyStore()
    # 5% change < 20%
    out = run(
        current_counts={"foo": 9500},
        prior_counts={"foo": 10000},
        alert_fn=lambda drifts: alerts.extend(drifts),
        store=store,
        iso_year_week="2026-W17",
    )
    assert alerts == []


def test_missing_prior_baseline_skipped():
    """Tables without a prior baseline are not alerted (first-scan tolerance)."""
    alerts: list = []
    store = IdempotencyStore()
    out = run(
        current_counts={"new_table": 100},
        prior_counts={},
        alert_fn=lambda drifts: alerts.extend(drifts),
        store=store,
        iso_year_week="2026-W17",
    )
    assert alerts == []
