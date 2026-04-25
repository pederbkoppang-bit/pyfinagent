"""phase-10.7.1 unit tests for the Alpha Velocity metric.

6 cases per research brief:
1. Positive velocity basic (SR 1.0 -> 1.5 over 30 days)
2. Negative velocity decay (SR 1.5 -> 0.8)
3. Insufficient observations returns None
4. Zero/negative window_days raises ValueError
5. Compute + insert via FakeBQ stub
6. Migration script --dry-run exits 0 + prints CREATE TABLE SQL

Tests do NOT touch live BigQuery; FakeBQ stub mirrors the
`backend/tests/test_paper_trading_v2.py` pattern.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

# Ensure repo root on sys.path so `backend.meta_evolution` resolves
# to OUR package, not anything in site-packages.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.meta_evolution.alpha_velocity import (  # noqa: E402
    MIN_OBSERVATIONS,
    compute_alpha_velocity,
    persist_sample,
)


class FakeBQ:
    """Minimal stub matching `bigquery.Client.insert_rows_json` shape."""

    def __init__(self):
        self.calls: list[tuple[str, list[dict[str, Any]]]] = []

    def insert_rows_json(self, table_fqn: str, rows: list[dict[str, Any]]):
        self.calls.append((table_fqn, rows))
        return []  # no errors


def _ts(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


def test_positive_velocity_basic():
    """SR 1.0 -> 1.5 over 30 days = +0.0167/day."""
    sample = compute_alpha_velocity(
        strategy_id="seed_0000",
        window_start=_ts(2026, 1, 1),
        window_end=_ts(2026, 1, 31),
        n_obs=30,
        sharpe_start=1.0,
        sharpe_end=1.5,
        macro_regime="EASING",
    )
    assert sample.window_days == 30
    score = sample.alpha_velocity_score
    assert score is not None
    assert abs(score - (0.5 / 30.0)) < 1e-9
    row = sample.to_bq_row()
    assert row["strategy_id"] == "seed_0000"
    assert row["alpha_velocity_score"] == score
    assert row["macro_regime"] == "EASING"


def test_negative_velocity_decay():
    """SR 1.5 -> 0.8 over 30 days = negative score."""
    sample = compute_alpha_velocity(
        strategy_id="decay_test",
        window_start=_ts(2026, 2, 1),
        window_end=_ts(2026, 3, 3),
        n_obs=30,
        sharpe_start=1.5,
        sharpe_end=0.8,
        macro_regime="HIKING",
    )
    score = sample.alpha_velocity_score
    assert score is not None
    assert score < 0
    assert abs(score - (-0.7 / 30.0)) < 1e-9


def test_insufficient_observations_returns_null():
    """n_obs < MIN_OBSERVATIONS -> alpha_velocity_score is None."""
    sample = compute_alpha_velocity(
        strategy_id="too_few",
        window_start=_ts(2026, 4, 1),
        window_end=_ts(2026, 4, 30),
        n_obs=MIN_OBSERVATIONS - 5,
        sharpe_start=1.0,
        sharpe_end=1.5,
    )
    assert sample.alpha_velocity_score is None
    row = sample.to_bq_row()
    assert row["alpha_velocity_score"] is None


def test_zero_window_days_raises():
    """window_start == window_end => ValueError when score is requested."""
    sample = compute_alpha_velocity(
        strategy_id="zero_window",
        window_start=_ts(2026, 5, 1),
        window_end=_ts(2026, 5, 1),
        n_obs=30,
        sharpe_start=1.0,
        sharpe_end=1.2,
    )
    with pytest.raises(ValueError, match="window_days"):
        _ = sample.alpha_velocity_score


def test_compute_and_insert_mocked_bq():
    """End-to-end: compute -> persist -> FakeBQ records the call."""
    sample = compute_alpha_velocity(
        strategy_id="seed_0000",
        window_start=_ts(2026, 6, 1),
        window_end=_ts(2026, 6, 30),
        n_obs=29,
        sharpe_start=1.10,
        sharpe_end=1.25,
        macro_regime="NEUTRAL",
        components={"capture_ratio_start": 0.45, "capture_ratio_end": 0.52},
    )
    bq = FakeBQ()
    persist_sample(bq, sample)
    assert len(bq.calls) == 1
    table_fqn, rows = bq.calls[0]
    assert table_fqn.endswith("pyfinagent_pms.alpha_velocity_samples")
    assert len(rows) == 1
    row = rows[0]
    assert row["strategy_id"] == "seed_0000"
    assert row["n_obs"] == 29
    assert row["macro_regime"] == "NEUTRAL"
    components = json.loads(row["components_json"])
    assert components["capture_ratio_start"] == 0.45
    assert row["alpha_velocity_score"] is not None
    assert row["window_days"] == 29


def test_migration_script_dry_run():
    """`--dry-run` exits 0 and emits CREATE TABLE SQL with the canonical FQN."""
    script = REPO_ROOT / "scripts" / "migrations" / "create_alpha_velocity_table.py"
    assert script.exists(), f"migration script missing: {script}"
    result = subprocess.run(
        [sys.executable, str(script), "--dry-run"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, f"dry-run exit {result.returncode}: {result.stderr}"
    out = result.stdout
    assert "DRY RUN" in out
    assert "CREATE TABLE IF NOT EXISTS" in out
    assert "alpha_velocity_samples" in out
    assert "PARTITION BY DATE(window_start)" in out
    assert "CLUSTER BY strategy_id, macro_regime" in out
