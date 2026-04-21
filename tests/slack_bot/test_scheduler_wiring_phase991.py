"""phase-9.9.1 regression tests — scheduler-wiring fix for cost_budget_watcher
and weekly_data_integrity.

Guards against the two runtime bugs qa_99_remediation_v1 reproduced:
- cost_budget_watcher.run() TypeError on zero-arg APScheduler fire
- weekly_data_integrity.run() inert with empty-dict defaults
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.slack_bot.jobs import cost_budget_watcher, weekly_data_integrity
from backend.slack_bot.job_runtime import IdempotencyStore


def test_cost_budget_watcher_zero_args_uses_fetch_fn():
    """Regression: APScheduler fires run() with no spend args; fetch_fn stub supplies them."""
    calls: list[int] = []

    def stub_fetch() -> tuple[float, float]:
        calls.append(1)
        return 1.0, 10.0

    out = cost_budget_watcher.run(
        fetch_fn=stub_fetch,
        store=IdempotencyStore(),
        day="2026-04-20",
    )
    assert calls == [1], "fetch_fn must be invoked when no explicit spend values passed"
    assert out["daily"] == 1.0
    assert out["monthly"] == 10.0
    assert not out["tripped"]


def test_cost_budget_watcher_fetch_fn_over_cap_trips():
    """fetch_fn returning spend > daily cap trips the watcher."""
    out = cost_budget_watcher.run(
        fetch_fn=lambda: (100.0, 5.0),
        store=IdempotencyStore(),
        day="2026-04-20",
    )
    assert out["tripped"] is True
    assert out["reason"] == "daily"


def test_cost_budget_watcher_bq_unreachable_fail_open(monkeypatch):
    """phase-9.9.2: BQ client failure returns (0.0, 0.0), no raise.

    Swapped from Anthropic Cost API (billable-API-only) to BQ
    INFORMATION_SCHEMA.JOBS_BY_PROJECT for Max-subscription deployments.
    """
    import google.cloud.bigquery as bq

    def boom(*_a, **_kw):
        raise RuntimeError("BQ unreachable")

    monkeypatch.setattr(bq, "Client", boom)
    daily, monthly = cost_budget_watcher._default_fetch_spend()
    assert (daily, monthly) == (0.0, 0.0)


def test_cost_budget_watcher_explicit_value_beats_fetch():
    """If an explicit value is passed, fetch_fn is not invoked for that scope."""
    calls: list[int] = []

    def boom() -> tuple[float, float]:
        calls.append(1)
        return 999.0, 999.0

    out = cost_budget_watcher.run(
        daily_spend_usd=1.0,
        monthly_spend_usd=1.0,
        fetch_fn=boom,
        store=IdempotencyStore(),
        day="2026-04-20",
    )
    assert calls == [], "fetch_fn must NOT be invoked when both explicit values supplied"
    assert out["daily"] == 1.0
    assert not out["tripped"]


def test_weekly_data_integrity_zero_args_uses_fetch_fn_and_snapshot():
    """Regression: run() with no count args invokes fetch_fn + snapshot load/save."""
    with tempfile.TemporaryDirectory() as td:
        snap = str(Path(td) / "snapshot.json")
        # Seed prior-week snapshot
        Path(snap).write_text(json.dumps({"my_table": 10000}))

        out = weekly_data_integrity.run(
            fetch_fn=lambda: {"my_table": 5000},  # 50% drop — above 20% threshold
            snapshot_path=snap,
            alert_fn=None,
            store=IdempotencyStore(),
            iso_year_week="2026-W17",
        )
        assert len(out["drifts"]) == 1
        assert out["drifts"][0]["table"] == "my_table"
        assert out["drifts"][0]["delta_pct"] == 0.5

        # After run, snapshot was overwritten with current counts
        saved = json.loads(Path(snap).read_text())
        assert saved == {"my_table": 5000}


def test_weekly_data_integrity_snapshot_missing_first_run():
    """First run (no prior snapshot) returns empty drifts, writes current as baseline."""
    with tempfile.TemporaryDirectory() as td:
        snap = str(Path(td) / "snapshot.json")
        # No prior file

        out = weekly_data_integrity.run(
            fetch_fn=lambda: {"fresh_table": 100},
            snapshot_path=snap,
            alert_fn=None,
            store=IdempotencyStore(),
            iso_year_week="2026-W17",
        )
        assert out["drifts"] == []
        assert json.loads(Path(snap).read_text()) == {"fresh_table": 100}


def test_weekly_data_integrity_fetch_fail_open(monkeypatch):
    """_default_fetch_counts fails open to {} if BQ client unavailable."""
    import backend.slack_bot.jobs.weekly_data_integrity as wdi

    def boom(*_a, **_kw):
        raise RuntimeError("BQ unreachable")

    monkeypatch.setattr("backend.db.bigquery_client.BigQueryClient", boom, raising=False)
    # If BQ import itself fails, the except branch catches — assert no raise
    result = wdi._default_fetch_counts()
    assert isinstance(result, dict)


def test_scheduler_wiring_cost_budget_watcher_fires_zero_args(monkeypatch):
    """End-to-end regression: APScheduler invokes run() with zero args -> no TypeError.

    Forces BQ fetch to fail-open so the test is hermetic (no real BQ credentials
    required in CI). This is THE regression guard for the original phase-9.9
    TypeError bug — if run() required any positional/kw args, a bare call here
    would raise before the monkeypatch took effect.
    """
    monkeypatch.setattr(cost_budget_watcher, "_default_fetch_spend", lambda: (0.0, 0.0))
    out = cost_budget_watcher.run()
    assert "daily" in out
    assert "tripped" in out
    assert out["daily"] == 0.0
    assert out["monthly"] == 0.0
    assert not out["tripped"]


def test_scheduler_wiring_weekly_data_integrity_fires_zero_args(monkeypatch, tmp_path):
    """End-to-end: weekly_data_integrity.run() with no args succeeds (fail-open on BQ)."""
    import backend.slack_bot.jobs.weekly_data_integrity as wdi

    snap = tmp_path / "snap.json"
    monkeypatch.setattr(wdi, "_DEFAULT_SNAPSHOT_PATH", str(snap))
    # Force BQ fetch to fail-open so we don't hit real BQ in CI
    monkeypatch.setattr(wdi, "_default_fetch_counts", lambda: {})

    out = wdi.run()
    assert out["drifts"] == []
    assert out["skipped"] is False
