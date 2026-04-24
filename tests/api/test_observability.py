"""phase-15.10 observability wiring tests.

Covers:
- `/api/observability/latency` returns the bare `p50/p95/p99` keys.
- `structured_log` helper emits the stable JSON envelope.
- `/api/cost-budget/today` carries the phase-15.10 cost-per-call rollup
  fields even when BQ is unreachable (fail-open to None).
- `/api/jobs/status` returns the canonical 7 jobs.
- `_read_audit_tail` fails open to empty + truncated=False on missing
  paths.
"""
from __future__ import annotations

import io
import json
import logging
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fastapi.testclient import TestClient

from backend.main import app
from backend.services.perf_tracker import get_perf_tracker
from backend.api.cost_budget_api import (
    structured_log as cb_structured_log,
    CostBudgetToday,
)
from backend.api.job_status_api import get_job_status
from backend.api.harness_autoresearch import (
    structured_log as ha_structured_log,
    _read_audit_tail,
)


client = TestClient(app)


def test_latency_endpoint_keys_present():
    tracker = get_perf_tracker()
    # Seed three synthetic entries so summarize has data.
    for ms in (10.0, 50.0, 100.0):
        tracker.record(
            endpoint="/test",
            method="GET",
            status_code=200,
            latency_ms=ms,
            cache_hit=False,
        )
    r = client.get("/api/observability/latency")
    assert r.status_code == 200
    d = r.json()
    for k in ("p50", "p95", "p99"):
        assert k in d, d
    # Numeric + non-negative.
    for k in ("p50", "p95", "p99"):
        assert isinstance(d[k], (int, float))
        assert d[k] >= 0


def test_structured_log_emits_required_json_fields(caplog):
    caplog.set_level(logging.INFO)
    # One call per module to exercise both helpers.
    cb_structured_log("/api/cost-budget/today", 12.3, "ok", foo="bar")
    ha_structured_log("/api/harness/sprint-state", 4.5, "empty", week_iso="2026-W17")

    parsed: list[dict] = []
    for rec in caplog.records:
        msg = rec.getMessage()
        if msg.startswith("{") and msg.endswith("}"):
            try:
                parsed.append(json.loads(msg))
            except json.JSONDecodeError:
                continue
    # At least two JSON records.
    assert len(parsed) >= 2, [r.getMessage() for r in caplog.records]
    # Every record has the four stable keys.
    required = {"endpoint", "duration_ms", "status", "ts"}
    for row in parsed:
        assert required.issubset(row.keys()), row


def test_cost_budget_today_has_observability_rollup_fields():
    r = client.get("/api/cost-budget/today")
    assert r.status_code == 200
    d = r.json()
    # phase-15.10 rollup fields (may be None under local dev when BQ is
    # unavailable, but the keys must be present).
    assert "llm_tokens_today" in d, d
    assert "cost_per_llm_call_usd" in d, d


def test_get_job_status_returns_seven_jobs():
    resp = get_job_status()
    assert len(resp.jobs) == 7
    # All canonical names show up.
    names = {j.name for j in resp.jobs}
    expected = {
        "daily_price_refresh",
        "weekly_fred_refresh",
        "nightly_mda_retrain",
        "hourly_signal_warmup",
        "nightly_outcome_rebuild",
        "weekly_data_integrity",
        "cost_budget_watcher",
    }
    assert names == expected


def test_read_audit_tail_missing_path_fail_open(tmp_path):
    missing = tmp_path / "nonexistent.jsonl"
    events, truncated = _read_audit_tail(missing, 200)
    assert events == []
    assert truncated is False
