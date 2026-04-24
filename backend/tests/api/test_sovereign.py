"""phase-10.5.0 unit tests for the Sovereign read endpoints.

Stubs out BQ-touching helpers so the tests are deterministic and BQ-free:
- Forward-fill calendar density (verifies the >=25 floor for 30d).
- Window param validation (rejects unknown values via Literal).
- Leaderboard fallback to results.tsv when the strategy_deployments
  view is missing.
- compute-cost always carries the 5 provider keys, even on empty BQ.
- Cache-hit second call returns immediately from the in-memory cache.
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.api import sovereign_api as sov  # noqa: E402
from backend.main import app  # noqa: E402
from backend.services.api_cache import get_api_cache  # noqa: E402

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clear_cache():
    """Ensure each test starts with a clean cache so cached responses
    from earlier tests don't shadow the current one's stub."""
    get_api_cache().clear() if hasattr(get_api_cache(), "clear") else None
    # APICache exposes _store directly; clear it.
    cache = get_api_cache()
    try:
        cache._store.clear()
    except Exception:
        pass
    yield


def test_red_line_window_param_validates():
    r = client.get("/api/sovereign/red-line", params={"window": "999d"})
    assert r.status_code == 422


def test_red_line_returns_at_least_25_points_with_forward_fill(monkeypatch):
    """Synthesise 7 distinct snapshot dates; forward-fill must produce
    >= 25 calendar points for the 30d window."""
    # Build 7 distinct dates ending today.
    today = datetime.now(timezone.utc).date()
    snapshots = [
        {"d": (today - timedelta(days=offset)).isoformat(), "nav": 9500.0 + offset}
        for offset in (0, 5, 10, 15, 20, 25, 28)
    ]
    monkeypatch.setattr(sov, "_fetch_snapshots", lambda window_days: snapshots)
    r = client.get("/api/sovereign/red-line", params={"window": "30d"})
    assert r.status_code == 200
    body = r.json()
    assert "series" in body
    assert len(body["series"]) >= 25, len(body["series"])
    # Every point should carry source.
    for p in body["series"]:
        assert p["source"] in ("actual", "filled", "pre_inception")


def test_red_line_empty_when_no_snapshots(monkeypatch):
    monkeypatch.setattr(sov, "_fetch_snapshots", lambda window_days: [])
    r = client.get("/api/sovereign/red-line", params={"window": "7d"})
    body = r.json()
    assert body["series"] == []
    assert body["note"] is not None


def test_leaderboard_falls_back_to_results_tsv_when_view_missing(monkeypatch):
    monkeypatch.setattr(sov, "_fetch_strategy_deployments", lambda: None)
    r = client.get("/api/sovereign/leaderboard")
    assert r.status_code == 200
    body = r.json()
    assert body["source"] in ("results_tsv", "empty")
    if body["entries"]:
        # results.tsv has at least the seed_0000 row.
        assert any(e["strategy_id"] == "seed_0000" for e in body["entries"])


def test_compute_cost_always_includes_all_five_provider_keys(monkeypatch):
    monkeypatch.setattr(sov, "_fetch_bq_daily_bytes", lambda window_days: [])
    r = client.get("/api/sovereign/compute-cost", params={"window": "7d"})
    body = r.json()
    assert "totals" in body
    for k in ("anthropic", "vertex", "openai", "bigquery", "altdata"):
        assert k in body["totals"], f"missing {k}: {body['totals']}"


def test_endpoints_use_cache_on_repeat_call(monkeypatch):
    """Second call within TTL must serve from cache.
    Validated by counting fetcher invocations."""
    calls: list[int] = []

    def stub(window_days):
        calls.append(window_days)
        return [{"d": "2026-04-22", "nav": 9500.0}]

    monkeypatch.setattr(sov, "_fetch_snapshots", stub)
    client.get("/api/sovereign/red-line", params={"window": "7d"})
    client.get("/api/sovereign/red-line", params={"window": "7d"})
    assert len(calls) == 1, calls


def test_forward_fill_backfills_pre_inception():
    """Pre-window days (before the first actual snapshot) are backfilled
    with the first-actual NAV and labelled `pre_inception` so the chart
    line is continuous across the full window."""
    snapshots = [{"d": (date.today() - timedelta(days=2)).isoformat(), "nav": 100.0}]
    out = sov._forward_fill_calendar(snapshots, window_days=10, today=date.today())
    # 11 points = 10 + today.
    assert len(out) == 11
    sources = {p.source for p in out}
    assert sources == {"actual", "filled", "pre_inception"}
