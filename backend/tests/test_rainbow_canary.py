"""phase-12.3 tests for Rainbow canary SLO diff.

Zero live cluster / BQ dependencies. Synthetic-injection test harness
matches the MVP scope locked in the research brief.
"""
from __future__ import annotations

import pytest

from backend.services.observability.rainbow_canary import (
    DEFAULT_MIN_SAMPLES,
    DEFAULT_THRESHOLD,
    SLODiff,
    canary_snapshot_from_buffer,
    compute_slo_diff,
    percentile,
)


# ---------- percentile ----------


def test_percentile_basic():
    vals = list(range(1, 101))  # 1..100
    assert percentile(vals, 50) == 50.5  # median via linear interpolation
    assert percentile(vals, 95) == 95.05
    assert percentile(vals, 99) == 99.01
    assert percentile(vals, 0) == 1
    assert percentile(vals, 100) == 100


def test_percentile_empty_returns_zero():
    assert percentile([], 95) == 0.0


def test_percentile_single_value():
    assert percentile([42.0], 50) == 42.0
    assert percentile([42.0], 95) == 42.0


def test_percentile_rejects_out_of_range_p():
    with pytest.raises(ValueError):
        percentile([1, 2, 3], 101)
    with pytest.raises(ValueError):
        percentile([1, 2, 3], -1)


# ---------- compute_slo_diff ----------


def test_compute_slo_diff_no_regression():
    # Identical distributions -> ratio ~1.0 -> no regression.
    blue = [100.0 + i for i in range(20)]  # 100..119
    green = [100.0 + i for i in range(20)]
    result = compute_slo_diff(blue, green)
    assert isinstance(result, SLODiff)
    assert result.reason == "ok"
    assert result.ratio == pytest.approx(1.0, abs=0.01)
    assert result.regression is False


def test_compute_slo_diff_regression_when_green_2x_slower():
    blue = [100.0 + i for i in range(20)]
    green = [200.0 + i for i in range(20)]  # ~2x slower
    result = compute_slo_diff(blue, green)
    assert result.reason == "ok"
    assert result.ratio > 1.2
    assert result.regression is True


def test_compute_slo_diff_green_faster_not_regression():
    blue = [200.0 + i for i in range(20)]
    green = [100.0 + i for i in range(20)]  # green faster
    result = compute_slo_diff(blue, green)
    assert result.reason == "ok"
    assert result.ratio < 1.0
    assert result.regression is False


def test_compute_slo_diff_empty_samples_fail_open():
    result = compute_slo_diff([], [100.0] * 20)
    assert result.reason == "empty"
    assert result.regression is False
    result = compute_slo_diff([100.0] * 20, [])
    assert result.reason == "empty"
    assert result.regression is False


def test_compute_slo_diff_below_min_samples_fail_open():
    # Only 5 samples on each side — below DEFAULT_MIN_SAMPLES (10).
    result = compute_slo_diff([100.0] * 5, [300.0] * 5)
    assert result.reason == "insufficient_samples"
    assert result.regression is False
    assert result.blue_samples == 5
    assert result.green_samples == 5


def test_compute_slo_diff_threshold_tunable():
    blue = [100.0] * 20
    green = [125.0] * 20  # ratio = 1.25
    # Default threshold 1.2 -> regression
    result = compute_slo_diff(blue, green)
    assert result.regression is True
    # Raise threshold to 1.3 -> no regression
    result2 = compute_slo_diff(blue, green, threshold=1.3)
    assert result2.regression is False


# ---------- canary_snapshot_from_buffer ----------


def test_canary_snapshot_from_buffer_partitions_by_source():
    """Inject 20 blue + 20 green rows into api_call_log buffer; verify partition + diff."""
    from backend.services.observability.api_call_log import (
        log_api_call,
        reset_buffer_for_test,
    )

    reset_buffer_for_test()
    try:
        # 20 blue rows @ 100ms each
        for _ in range(20):
            log_api_call(
                source="pyfinagent-blue",
                endpoint="/api/x",
                http_status=200,
                latency_ms=100.0,
                response_bytes=42,
                ok=True,
            )
        # 20 green rows @ 250ms each (ratio 2.5 -> regression)
        for _ in range(20):
            log_api_call(
                source="pyfinagent-green",
                endpoint="/api/x",
                http_status=200,
                latency_ms=250.0,
                response_bytes=42,
                ok=True,
            )

        result = canary_snapshot_from_buffer(
            is_blue=lambda r: r["source"] == "pyfinagent-blue",
            is_green=lambda r: r["source"] == "pyfinagent-green",
        )
        assert result.reason == "ok"
        assert result.blue_samples == 20
        assert result.green_samples == 20
        assert result.ratio == pytest.approx(2.5, rel=0.01)
        assert result.regression is True
    finally:
        reset_buffer_for_test()


def test_canary_snapshot_from_buffer_empty_returns_fail_open():
    from backend.services.observability.api_call_log import reset_buffer_for_test

    reset_buffer_for_test()
    result = canary_snapshot_from_buffer(
        is_blue=lambda r: False,
        is_green=lambda r: False,
    )
    assert result.reason == "empty"
    assert result.regression is False


# ---------- Constants sanity ----------


def test_defaults():
    assert DEFAULT_THRESHOLD == 1.2
    assert DEFAULT_MIN_SAMPLES == 10
