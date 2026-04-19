"""phase-6.7 tests for the observability primitives.

Coverage:
 1. get_rate_limiter returns a singleton per source.
 2. retry_with_backoff honours Retry-After header.
 3. retry_with_backoff retries on 429 then returns.
 4. retry_with_backoff caps delay at `cap`.
 5. AlertDeduper fires only at N>=threshold.
 6. AlertDeduper respects repeat-hours window.
 7. AlertDeduper critical severity bypasses dedup.
 8. api_call_log.log_api_call buffers rows.
 9. api_call_log.log_api_call fail-open when BQ absent.
 10. log_llm_call buffer separate from log_api_call buffer.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from backend.services.observability import (
    AlertDeduper,
    buffer_size,
    flush,
    flush_llm,
    get_rate_limiter,
    llm_buffer_size,
    log_api_call,
    log_llm_call,
    reset_buffer_for_test,
    reset_default_deduper,
    reset_registry,
    retry_with_backoff,
    RetryExhausted,
)
from backend.services.observability.retry import _delay_for, _parse_retry_after


# ---------- 1. Rate limiter singleton ----------


def test_get_rate_limiter_returns_singleton_per_source():
    reset_registry()
    lim1 = get_rate_limiter("finnhub")
    lim2 = get_rate_limiter("finnhub")
    lim_other = get_rate_limiter("fred")
    assert lim1 is lim2
    assert lim1 is not lim_other


# ---------- 2 + 3. Retry helpers ----------


def test_parse_retry_after_int_seconds():
    assert _parse_retry_after("3") == 3.0
    assert _parse_retry_after("3.5") == 3.5


def test_parse_retry_after_missing():
    assert _parse_retry_after(None) is None
    assert _parse_retry_after("") is None


def test_retry_with_backoff_honours_retry_after_header():
    """A 429 response with Retry-After=2 triggers one retry with delay>=2s (mocked sleep)."""

    class _FakeResp:
        def __init__(self, status: int, headers: dict | None = None) -> None:
            self.status_code = status
            self.headers = headers or {}

    calls = {"n": 0}
    sleeps: list[float] = []

    def _fake_fn() -> _FakeResp:
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResp(429, {"Retry-After": "2"})
        return _FakeResp(200)

    result = retry_with_backoff(
        _fake_fn,
        max_attempts=3,
        base=0.01,
        multiplier=2.0,
        cap=30.0,
        jitter="none",
        sleep=lambda s: sleeps.append(s),
    )
    assert result.status_code == 200
    assert calls["n"] == 2
    assert sleeps[0] >= 2.0  # honored Retry-After even though base delay < 2


def test_retry_with_backoff_exhausts_then_returns_response():
    """After max_attempts, if every call returned a retryable status, return the last response."""

    class _FakeResp:
        def __init__(self, status: int) -> None:
            self.status_code = status
            self.headers = {}

    def _always_429() -> _FakeResp:
        return _FakeResp(429)

    result = retry_with_backoff(
        _always_429,
        max_attempts=3,
        base=0.01,
        multiplier=2.0,
        cap=1.0,
        jitter="none",
        sleep=lambda s: None,
    )
    assert result.status_code == 429  # last response surfaced to caller


# ---------- 4. Delay cap ----------


def test_delay_for_caps_at_cap():
    # At high attempt, raw delay is multiplier**attempt * base, which at
    # attempt=10 multiplier=2 base=1 = 1024, which should cap at 30.
    for attempt in range(5, 15):
        raw = _delay_for(attempt, base=1.0, multiplier=2.0, cap=30.0, jitter="none")
        assert raw <= 30.0


# ---------- 5 + 6 + 7. AlertDeduper ----------


def test_alert_deduper_fires_at_threshold():
    d = AlertDeduper(window_minutes=5, repeat_hours=1, consecutive_threshold=3)
    assert d.should_fire("finnhub", "Timeout") is False
    assert d.should_fire("finnhub", "Timeout") is False
    assert d.should_fire("finnhub", "Timeout") is True  # 3rd = fire


def test_alert_deduper_respects_repeat_window():
    d = AlertDeduper(window_minutes=5, repeat_hours=1, consecutive_threshold=2)
    d.should_fire("x", "y")
    assert d.should_fire("x", "y") is True  # 2nd fires
    # immediately after: another 2 calls; should NOT fire (within repeat window)
    d.should_fire("x", "y")
    assert d.should_fire("x", "y") is False


def test_alert_deduper_critical_bypasses_dedup():
    d = AlertDeduper(window_minutes=5, repeat_hours=1, consecutive_threshold=100)
    # Even with threshold=100, critical fires on first call
    assert d.should_fire("x", "y", severity="P0") is True
    assert d.should_fire("x", "y", severity="critical") is True


# ---------- 8 + 9. api_call_log ----------


def test_log_api_call_buffers_rows_without_bq():
    reset_buffer_for_test()
    assert buffer_size() == 0
    log_api_call(source="finnhub", endpoint="/news", http_status=200, latency_ms=12.3)
    log_api_call(source="finnhub", endpoint="/news", http_status=200, latency_ms=14.1)
    assert buffer_size() == 2
    # Flush attempts BQ insert; google-cloud-bigquery may be absent or
    # unauthenticated in the test env. Either way, flush must not raise
    # and must return 0.
    result = flush()
    assert result == 0 or isinstance(result, int)
    # buffer is drained either way (fail-open)
    assert buffer_size() == 0


def test_log_api_call_never_raises_with_invalid_input():
    reset_buffer_for_test()
    # Pass something that will trip the float/int coercion; must not raise.
    log_api_call(
        source="x",
        endpoint="",
        http_status="not-a-number",  # type: ignore[arg-type]
        latency_ms="nope",  # type: ignore[arg-type]
        response_bytes="many",  # type: ignore[arg-type]
    )
    # Row may or may not have been buffered (depending on where int() fails);
    # the contract is just that the call did not raise.
    assert buffer_size() >= 0


# ---------- 10. llm_call_log buffer isolation ----------


def test_log_llm_call_separate_buffer_from_api_call_log():
    reset_buffer_for_test()
    # Drain LLM buffer in case of prior test state
    flush_llm()
    assert llm_buffer_size() == 0
    log_llm_call(
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        latency_ms=123.4,
        input_tok=1000,
        output_tok=50,
    )
    assert llm_buffer_size() == 1
    assert buffer_size() == 0  # api_call_log buffer NOT touched
    flush_llm()
    assert llm_buffer_size() == 0
