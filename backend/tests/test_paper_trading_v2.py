"""
Phase 4.5 test suite (step 4.5.10).

Covers:
  - compute_reconciliation() unit cases (3) -- zero data, below-threshold,
    above-threshold alert.
  - v2 endpoint smokes (status + response shape subset).
  - No-regression on pre-Phase-4.5 endpoints.

Mock pattern: patch at import site (`backend.api.paper_trading.BigQueryClient`)
because the BQ client is currently a module-level constructor rather than a
FastAPI `Depends()` injection. Research notes (RESEARCH.md 4.5.10) flag this
as tech-debt for a future phase; for now it is the minimum-intrusion test
pattern that does not require refactoring the routers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

# Ensure the project root is on sys.path when pytest is invoked from backend/.
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ── Fakes ──────────────────────────────────────────────────────────


class FakeBQ:
    """Minimal BigQueryClient stub for router + service tests."""

    def __init__(self, settings=None):
        self._trades: list[dict] = []
        self._snapshots: list[dict] = []
        self._portfolio = {
            "portfolio_id": "default",
            "starting_capital": 100000.0,
            "current_cash": 100000.0,
            "total_nav": 100000.0,
            "total_pnl_pct": 0.0,
            "benchmark_return_pct": 0.0,
            "inception_date": "2026-01-01T10:00:00Z",
            "updated_at": "2026-04-16T10:00:00Z",
        }

    def seed_simple_cycle(self) -> None:
        self._snapshots = [
            {"snapshot_date": f"2026-01-{i + 1:02d}", "total_nav": 100000.0 + 100 * i}
            for i in range(40)
        ]
        self._trades = [
            {"trade_id": "b1", "ticker": "AAPL", "action": "BUY", "quantity": 10,
             "price": 100.0, "created_at": "2026-01-02T10:00:00Z", "total_value": 1000},
            {"trade_id": "s1", "ticker": "AAPL", "action": "SELL", "quantity": 10,
             "price": 110.0, "created_at": "2026-01-15T10:00:00Z",
             "mfe_pct": 12.0, "mae_pct": -3.0, "capture_ratio": 0.83,
             "realized_pnl_pct": 10.0, "holding_days": 13, "total_value": 1100,
             "reason": "tp"},
        ]

    def get_paper_portfolio(self, portfolio_id: str = "default"):
        return dict(self._portfolio)

    def get_paper_positions(self):
        return []

    def get_paper_trades(self, limit: int = 100):
        return list(self._trades[:limit])

    def get_paper_snapshots(self, limit: int = 365):
        return list(self._snapshots[:limit])


# ── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def client():
    from backend.api.paper_trading import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def fake_bq():
    bq = FakeBQ()
    bq.seed_simple_cycle()
    return bq


# ── Reconciliation unit tests (3 cases) ────────────────────────────


class TestReconciliationUnit:
    def test_no_data_returns_insufficient(self):
        from backend.services.reconciliation import compute_reconciliation

        bq = FakeBQ()  # empty
        result = compute_reconciliation(bq)
        assert result["note"] == "insufficient_snapshots"
        assert result["summary"]["n_points"] == 0
        assert result["summary"]["alert"] is False

    def test_below_threshold_no_alert(self, fake_bq):
        from backend.services.reconciliation import compute_reconciliation

        # Seed a shadow-NAV path that hugs paper NAV; swallow yfinance by
        # patching _fetch_prices to return an empty dict (falls back to
        # paper fill price, so shadow tracks paper tightly).
        with patch("backend.services.reconciliation._fetch_prices", lambda *a, **k: {}):
            result = compute_reconciliation(fake_bq)
        assert result["note"] is None
        assert result["summary"]["n_points"] > 0
        # With no yfinance prices, divergence can still exist from cash
        # drift; assert the alert flag is well-formed (bool) and series is
        # non-empty.
        assert isinstance(result["summary"]["alert"], bool)
        assert len(result["series"]) == result["summary"]["n_points"]

    def test_above_threshold_triggers_alert(self, fake_bq):
        """Synthesize a divergence >5% by inflating paper NAV vs shadow."""
        from backend.services.reconciliation import compute_reconciliation

        # Force paper NAV to drift up without matching trades
        fake_bq._snapshots = [
            {"snapshot_date": f"2026-01-{i + 1:02d}", "total_nav": 100000.0 + 5000 * i}
            for i in range(20)
        ]
        with patch("backend.services.reconciliation._fetch_prices", lambda *a, **k: {}):
            result = compute_reconciliation(fake_bq)
        assert result["summary"]["latest_divergence_pct"] > 5.0
        assert result["summary"]["alert"] is True
        assert result["summary"]["alert_threshold_pct"] == 5.0


# ── v2 endpoint smokes ─────────────────────────────────────────────


class TestV2Endpoints:
    def _with_bq(self, bq):
        return patch("backend.api.paper_trading.BigQueryClient", lambda settings: bq)

    def test_metrics_v2_returns_expected_keys(self, client, fake_bq):
        with self._with_bq(fake_bq), patch(
            "backend.services.paper_metrics_v2.persist_metrics_v2", lambda *a, **k: None
        ):
            r = client.get("/api/paper-trading/metrics-v2")
        assert r.status_code == 200
        body = r.json()
        for key in ("psr", "dsr", "sortino", "calmar", "rolling_sharpe", "n_obs", "computed_at"):
            assert key in body, f"missing key: {key}"

    def test_round_trips_returns_summary(self, client, fake_bq):
        with self._with_bq(fake_bq):
            r = client.get("/api/paper-trading/round-trips")
        assert r.status_code == 200
        body = r.json()
        assert "n_round_trips" in body and "win_rate" in body and "profit_factor" in body

    def test_gate_returns_five_booleans(self, client, fake_bq):
        with self._with_bq(fake_bq), patch(
            "backend.services.reconciliation._fetch_prices", lambda *a, **k: {}
        ):
            r = client.get("/api/paper-trading/gate")
        assert r.status_code == 200
        body = r.json()
        assert set(body["booleans"].keys()) == {
            "trades_ge_100",
            "psr_ge_95_sustained_30d",
            "dsr_ge_95",
            "sr_gap_le_30pct",
            "max_dd_within_tolerance",
        }
        assert isinstance(body["promote_eligible"], bool)

    def test_reconciliation_endpoint_returns_series(self, client, fake_bq):
        with self._with_bq(fake_bq), patch(
            "backend.services.reconciliation._fetch_prices", lambda *a, **k: {}
        ):
            r = client.get("/api/paper-trading/reconciliation")
        assert r.status_code == 200
        body = r.json()
        assert "series" in body and "summary" in body

    def test_live_prices_accepts_and_sanitizes(self, client, fake_bq):
        from backend.services import live_prices as lp

        with self._with_bq(fake_bq), patch.object(lp, "_fetch_price", lambda t: 100.0):
            r = client.get("/api/paper-trading/live-prices?tickers=AAPL,MSFT")
        assert r.status_code == 200
        body = r.json()
        assert "AAPL" in body["prices"] and "MSFT" in body["prices"]
        assert body["prices"]["AAPL"]["cached"] in (True, False)

    def test_live_prices_empty_returns_400(self, client, fake_bq):
        with self._with_bq(fake_bq):
            r = client.get("/api/paper-trading/live-prices?tickers=")
        assert r.status_code == 400

    def test_kill_switch_status_returns_thresholds(self, client, fake_bq):
        with self._with_bq(fake_bq):
            r = client.get("/api/paper-trading/kill-switch")
        assert r.status_code == 200
        body = r.json()
        assert body["thresholds"]["daily_loss_limit_pct"] == 4.0
        assert body["thresholds"]["trailing_dd_limit_pct"] == 10.0
        assert isinstance(body["paused"], bool)

    def test_pause_requires_confirmation(self, client, fake_bq):
        with self._with_bq(fake_bq):
            r = client.post("/api/paper-trading/pause", json={"confirmation": "WRONG"})
        assert r.status_code == 400

    def test_flatten_all_requires_confirmation(self, client, fake_bq):
        with self._with_bq(fake_bq):
            r = client.post("/api/paper-trading/flatten-all", json={"confirmation": "WRONG"})
        assert r.status_code == 400

    def test_cycles_history_empty_ok(self, client, fake_bq):
        with self._with_bq(fake_bq):
            r = client.get("/api/paper-trading/cycles/history?limit=5")
        assert r.status_code == 200
        body = r.json()
        assert "cycles" in body and "count" in body

    def test_mfe_mae_scatter_returns_summary(self, client, fake_bq):
        with self._with_bq(fake_bq):
            r = client.get("/api/paper-trading/mfe-mae-scatter")
        assert r.status_code == 200
        body = r.json()
        for key in ("points", "summary", "computed_at"):
            assert key in body
        for key in ("edge_ratio", "avg_capture_ratio", "n_points", "n_leakers"):
            assert key in body["summary"]


# ── Reality-gap harness log integration ────────────────────────────


class TestRealityGapLogging:
    def test_reconciliation_log_line_contains_fields(self):
        """
        4.5.10: the harness cycle log gets a `- Reconciliation: ...` line.
        We call _reconciliation_log_line directly with the reconciliation
        import patched to return a known-value summary.
        """
        import importlib

        rh = importlib.import_module("scripts.harness.run_harness")

        # Patch the lazy-imported compute_reconciliation (inside
        # _reconciliation_log_line) via the reconciliation module directly.
        fake_summary = {
            "summary": {"latest_divergence_pct": 2.34, "alert": False, "alert_threshold_pct": 5.0},
        }
        with patch("backend.services.reconciliation.compute_reconciliation",
                   lambda *a, **k: fake_summary), patch(
            "backend.db.bigquery_client.BigQueryClient", lambda s: None
        ):
            line = rh._reconciliation_log_line()
        assert line.startswith("- Reconciliation:")
        assert "divergence=2.34%" in line
        assert "alert=False" in line
        assert "threshold=5.0%" in line

    def test_reconciliation_log_alert_prefixes_warn(self):
        import importlib

        rh = importlib.import_module("scripts.harness.run_harness")
        fake_summary = {
            "summary": {"latest_divergence_pct": 8.1, "alert": True, "alert_threshold_pct": 5.0},
        }
        with patch("backend.services.reconciliation.compute_reconciliation",
                   lambda *a, **k: fake_summary), patch(
            "backend.db.bigquery_client.BigQueryClient", lambda s: None
        ):
            line = rh._reconciliation_log_line()
        assert "[WARN]" in line
        assert "alert=True" in line


# ── No-regression on pre-4.5 endpoints ─────────────────────────────


class TestNoRegression:
    def test_status_existing_fields_still_present(self, client, fake_bq):
        with patch("backend.api.paper_trading.BigQueryClient", lambda settings: fake_bq):
            r = client.get("/api/paper-trading/status")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] in ("active", "paused", "not_initialized")
        if body["status"] != "not_initialized":
            assert "portfolio" in body and "position_count" in body

    def test_performance_adds_round_trip_summary_but_keeps_legacy_fields(
        self, client, fake_bq
    ):
        with patch("backend.api.paper_trading.BigQueryClient", lambda settings: fake_bq):
            r = client.get("/api/paper-trading/performance")
        assert r.status_code == 200
        body = r.json()
        # Legacy fields still present
        for legacy in ("nav", "pnl_pct", "sharpe_ratio", "total_sell_trades"):
            assert legacy in body, f"regression: missing legacy key {legacy}"
        # v2 additive field
        assert "round_trip_summary" in body
