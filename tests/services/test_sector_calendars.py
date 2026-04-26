"""Unit tests for sector_calendars — schema, RTTNews parser, score apply, cache."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone, timedelta

import pytest

from backend.services.sector_calendars import (
    SectorEvent,
    apply_sector_events_to_score,
    _parse_rttnews_html,
    _save_cache,
    _load_cache,
    _CACHE_DIR,
)


def _mk_event(
    ticker="REGN",
    event_type="fda_pdufa",
    days_to_event=5,
    direction="positive_catalyst",
) -> SectorEvent:
    today = date.today()
    sched = today + timedelta(days=days_to_event)
    return SectorEvent(
        ticker=ticker,
        event_type=event_type,
        scheduled_date=sched,
        days_to_event=days_to_event,
        sector="Health Care" if event_type == "fda_pdufa" else "Multi",
        signal_direction=direction,
        source="rttnews" if event_type == "fda_pdufa" else "bq_calendar",
    )


def test_schema_extras_forbidden():
    with pytest.raises(Exception):
        SectorEvent.model_validate({
            "ticker": "AAPL", "event_type": "earnings",
            "scheduled_date": "2026-04-29", "days_to_event": 2,
            "sector": "Multi", "signal_direction": "positive_catalyst",
            "source": "bq_calendar", "unexpected_field": "boom",
        })


def test_apply_no_events_passes_through():
    assert apply_sector_events_to_score(10.0, "AAPL", "Tech", None) == 10.0
    assert apply_sector_events_to_score(10.0, "AAPL", "Tech", {}) == 10.0


def test_apply_unmatched_ticker_passes_through():
    ev = _mk_event(ticker="REGN")
    assert apply_sector_events_to_score(10.0, "AAPL", "Tech", {"REGN": ev}) == 10.0


def test_apply_binary_risk_filters_out():
    ev = _mk_event(ticker="REGN", days_to_event=0, direction="binary_risk")
    out = apply_sector_events_to_score(10.0, "REGN", "Health Care", {"REGN": ev})
    assert out is None


def test_apply_fda_positive_catalyst_boosts_20pct():
    ev = _mk_event(ticker="REGN", days_to_event=5, direction="positive_catalyst")
    out = apply_sector_events_to_score(10.0, "REGN", "Health Care", {"REGN": ev})
    assert out == pytest.approx(12.0)


def test_apply_earnings_within_3_days_boosts_10pct():
    ev = _mk_event(
        ticker="AAPL", event_type="earnings",
        days_to_event=2, direction="positive_catalyst",
    )
    out = apply_sector_events_to_score(10.0, "AAPL", "Multi", {"AAPL": ev})
    assert out == pytest.approx(11.0)


def test_apply_fda_far_catalyst_no_boost():
    ev = _mk_event(ticker="REGN", days_to_event=20, direction="positive_catalyst")
    # Beyond 7-day window: no boost
    out = apply_sector_events_to_score(10.0, "REGN", "Health Care", {"REGN": ev})
    assert out == pytest.approx(10.0)


def test_apply_neutral_event_no_change():
    ev = _mk_event(ticker="REGN", days_to_event=10, direction="neutral")
    out = apply_sector_events_to_score(10.0, "REGN", "Health Care", {"REGN": ev})
    assert out == pytest.approx(10.0)


def test_parse_rttnews_handles_empty_html():
    assert _parse_rttnews_html("") == []
    assert _parse_rttnews_html("<html></html>") == []


def test_parse_rttnews_extracts_table_row():
    today = date(2026, 4, 27)
    html = """
    <html><body>
      <table>
        <tr><th>Date</th><th>Company</th><th>Ticker</th><th>Drug</th><th>Indication</th></tr>
        <tr>
          <td>5/15/2026</td>
          <td>Regeneron Pharmaceuticals</td>
          <td>REGN</td>
          <td>SuperDrug</td>
          <td>Migraine</td>
        </tr>
      </table>
    </body></html>
    """
    events = _parse_rttnews_html(html, today=today)
    assert len(events) == 1
    ev = events[0]
    assert ev.ticker == "REGN"
    assert ev.event_type == "fda_pdufa"
    assert ev.scheduled_date == date(2026, 5, 15)
    assert ev.days_to_event == 18
    assert ev.signal_direction == "positive_catalyst"  # 18 days, between 1-30
    assert ev.source == "rttnews"


def test_parse_rttnews_binary_risk_within_one_day():
    today = date(2026, 4, 27)
    # FDA event tomorrow = binary risk
    html = """
    <table><tr>
      <td>4/28/2026</td><td>Acme Pharma</td><td>ACME</td><td>BetaDrug</td><td>Cancer</td>
    </tr></table>
    """
    events = _parse_rttnews_html(html, today=today)
    assert len(events) == 1
    assert events[0].signal_direction == "binary_risk"
    assert events[0].days_to_event == 1


def test_parse_rttnews_skips_rows_without_ticker():
    today = date(2026, 4, 27)
    html = """
    <table>
      <tr><td>5/15/2026</td><td>Some Company</td><td>This Is Not A Ticker</td></tr>
    </table>
    """
    events = _parse_rttnews_html(html, today=today)
    assert events == []


def test_parse_rttnews_skips_rows_without_date():
    today = date(2026, 4, 27)
    html = """
    <table>
      <tr><td>TBD</td><td>Acme Pharma</td><td>ACME</td></tr>
    </table>
    """
    events = _parse_rttnews_html(html, today=today)
    assert events == []


def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.services.sector_calendars._CACHE_DIR", tmp_path)
    events = {"REGN": _mk_event(ticker="REGN")}
    _save_cache(events)
    loaded = _load_cache()
    assert loaded is not None
    assert "REGN" in loaded
    assert loaded["REGN"].event_type == "fda_pdufa"


def test_cache_returns_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.services.sector_calendars._CACHE_DIR", tmp_path)
    assert _load_cache() is None


def test_cache_unreadable_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.services.sector_calendars._CACHE_DIR", tmp_path)
    bucket = datetime.now(timezone.utc).strftime("%Y%m%d")
    p = tmp_path / f"sector_calendars_{bucket}.json"
    p.write_text("not json", encoding="utf-8")
    assert _load_cache() is None
