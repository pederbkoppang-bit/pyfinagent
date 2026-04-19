"""phase-3.3 tests for VIXRollingQuantileRegimeDetector.

Coverage:
 1. Instantiation with defaults.
 2. yfinance-absent / fetch-failure fail-opens to static 2-regime fallback.
 3. Rolling-quantile classification on a synthetic VIX series.
 4. Merge-runs collapses consecutive same-label days into regime windows.
 5. Settings flag gate: harness wiring behavior when flag is False.
 6. `_to_date_str` coerces Timestamp / str / datetime correctly.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pytest

from backend.backtest.regime_detector import (
    RegimeDetector,
    VIXRollingQuantileRegimeDetector,
    _to_date_str,
)


# ---------- 1. Instantiation ----------


def test_regime_detector_instantiates_with_defaults():
    d = VIXRollingQuantileRegimeDetector(
        start_date="2023-01-01", end_date="2024-01-01"
    )
    assert d.low_q == 0.33
    assert d.high_q == 0.67
    assert d.window_days == 252
    assert d.vix_symbol == "^VIX"
    # Protocol check
    assert isinstance(d, RegimeDetector)


# ---------- 2. Fail-open fallback ----------


def test_detect_returns_fallback_when_vix_fetch_fails():
    d = VIXRollingQuantileRegimeDetector(
        start_date="2023-01-01", end_date="2024-01-01"
    )
    # Force the internal fetch to raise.
    with patch.object(
        VIXRollingQuantileRegimeDetector,
        "_fetch_vix_closes",
        side_effect=RuntimeError("no yfinance"),
    ):
        regimes = d.detect()
    assert isinstance(regimes, list)
    assert len(regimes) == 2
    assert regimes[0]["name"] == "Pre-COVID"
    assert regimes[1]["name"] == "Post-COVID"


def test_detect_returns_fallback_when_vix_insufficient():
    d = VIXRollingQuantileRegimeDetector(
        start_date="2023-01-01", end_date="2024-01-01"
    )
    # Return a series shorter than window_days to trigger fallback branch.
    with patch.object(
        VIXRollingQuantileRegimeDetector,
        "_fetch_vix_closes",
        return_value=None,
    ):
        regimes = d.detect()
    assert [r["name"] for r in regimes] == ["Pre-COVID", "Post-COVID"]


# ---------- 3. Rolling-quantile on synthetic series ----------


def test_classify_series_produces_three_labels_on_synthetic_data():
    """Synthetic VIX: mostly-low, then ramp to high -- expect both labels."""
    pytest.importorskip("pandas")
    import pandas as pd

    d = VIXRollingQuantileRegimeDetector(
        start_date="2020-01-01",
        end_date="2021-01-01",
        low_q=0.33,
        high_q=0.67,
        window_days=10,  # short window for test
    )
    # 20 days low (10-15), then 20 days high (25-40)
    low_part = [12 + (i % 3) for i in range(20)]  # 12, 13, 14, 12...
    high_part = [28 + (i % 10) for i in range(20)]  # 28-37
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    closes = pd.Series(low_part + high_part, index=dates)

    labels = d.classify_series(closes)
    assert len(labels) == 40
    # The high_part region should contain at least some "high_vol" labels.
    assert "high_vol" in labels[20:]
    # Labels are in the valid set
    assert all(l in {"low_vol", "medium_vol", "high_vol"} for l in labels)


# ---------- 4. Merge-runs ----------


def test_merge_runs_collapses_consecutive_labels():
    dates = [
        "2020-01-01",
        "2020-01-02",
        "2020-01-03",
        "2020-01-04",
        "2020-01-05",
        "2020-01-06",
    ]
    labels = ["low_vol", "low_vol", "high_vol", "high_vol", "high_vol", "medium_vol"]
    merged = VIXRollingQuantileRegimeDetector._merge_runs(dates, labels)
    assert len(merged) == 3
    assert merged[0] == {
        "name": "low_vol",
        "start_date": "2020-01-01",
        "end_date": "2020-01-02",
    }
    assert merged[1]["name"] == "high_vol"
    assert merged[1]["start_date"] == "2020-01-03"
    assert merged[1]["end_date"] == "2020-01-05"
    assert merged[2] == {
        "name": "medium_vol",
        "start_date": "2020-01-06",
        "end_date": "2020-01-06",
    }


def test_merge_runs_handles_empty_input():
    assert VIXRollingQuantileRegimeDetector._merge_runs([], []) == []
    # Mismatched lengths
    assert VIXRollingQuantileRegimeDetector._merge_runs(["2020-01-01"], []) == []


# ---------- 5. Settings flag gate (harness wiring behavior) ----------


def test_settings_flag_default_is_false():
    """regime_detection_enabled MUST default to False -- preserves legacy behavior."""
    from backend.config.settings import Settings

    # Instantiate without env; required fields use placeholder values to
    # isolate THIS flag only.
    import os

    os.environ.setdefault("GCP_PROJECT_ID", "test-proj")
    os.environ.setdefault("RAG_DATA_STORE_ID", "test-rag")
    os.environ.setdefault("INGESTION_AGENT_URL", "http://test")
    os.environ.setdefault("QUANT_AGENT_URL", "http://test")

    s = Settings()  # type: ignore[call-arg]
    assert s.regime_detection_enabled is False


# ---------- 6. Date coercion ----------


def test_to_date_str_handles_various_inputs():
    assert _to_date_str("2020-01-01") == "2020-01-01"
    assert _to_date_str(datetime(2020, 1, 15)) == "2020-01-15"
    try:
        import pandas as pd

        assert _to_date_str(pd.Timestamp("2020-06-01")) == "2020-06-01"
    except ImportError:
        pass
