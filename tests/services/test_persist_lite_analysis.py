"""phase-23.1.11: _persist_lite_analysis writes lite-Claude rows to analysis_results
so the Reports History tab surfaces paper-trading candidates."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from backend.services.autonomous_loop import _persist_lite_analysis


def _lite_analysis(ticker="COHR") -> dict:
    """Mirrors what `_run_claude_analysis` returns post phase-23.1.11."""
    return {
        "ticker": ticker,
        "recommendation": "BUY",
        "final_score": 7,
        "risk_assessment": {"reason": "Strong momentum + reasonable valuation"},
        "price_at_analysis": 320.91,
        "analysis_date": "2026-04-27T18:00:21+00:00",
        "total_cost_usd": 0.01,
        "full_report": {
            "source": "claude-sonnet-4-6",
            "analysis": {"action": "BUY", "confidence": 72, "score": 7, "reason": "..."},
            "market_data": {
                "name": "Coherent Corp.",
                "price": 320.91,
                "market_cap": 50_000_000_000,
                "pe_ratio": 317.7,
                "sector": "Information Technology",
                "industry": "Semiconductors",
                "momentum_20d": 46.1,
                "momentum_60d": 51.2,
            },
        },
    }


def test_persist_calls_save_report_with_correct_fields():
    bq = MagicMock()
    asyncio.run(_persist_lite_analysis(_lite_analysis(), bq))
    bq.save_report.assert_called_once()
    kwargs = bq.save_report.call_args.kwargs
    assert kwargs["ticker"] == "COHR"
    assert kwargs["company_name"] == "Coherent Corp."
    assert kwargs["recommendation"] == "BUY"
    assert kwargs["final_score"] == 7.0
    assert kwargs["summary"] == "Strong momentum + reasonable valuation"
    assert kwargs["price_at_analysis"] == 320.91
    assert kwargs["market_cap"] == 50_000_000_000
    assert kwargs["pe_ratio"] == 317.7
    assert kwargs["sector"] == "Information Technology"
    assert kwargs["industry"] == "Semiconductors"
    assert kwargs["recommendation_confidence"] == 72
    assert kwargs["total_cost_usd"] == 0.01
    assert kwargs["standard_model"] == "claude-sonnet-4-6"


def test_persist_handles_missing_market_data_gracefully():
    """Lean analysis with no market_data → company_name falls back to ticker."""
    bq = MagicMock()
    a = _lite_analysis()
    del a["full_report"]["market_data"]
    asyncio.run(_persist_lite_analysis(a, bq))
    bq.save_report.assert_called_once()
    kwargs = bq.save_report.call_args.kwargs
    assert kwargs["company_name"] == "COHR"  # falls back to ticker
    assert kwargs["sector"] == ""
    assert kwargs["industry"] == ""


def test_persist_handles_bq_exception_without_propagating():
    """BQ outage → log warning, but cycle continues."""
    bq = MagicMock()
    bq.save_report.side_effect = RuntimeError("BQ is down")
    # Must NOT raise
    asyncio.run(_persist_lite_analysis(_lite_analysis(), bq))


def test_persist_skips_when_no_ticker():
    bq = MagicMock()
    asyncio.run(_persist_lite_analysis({"final_score": 5}, bq))
    bq.save_report.assert_not_called()


def test_persist_skips_when_ticker_empty_string():
    bq = MagicMock()
    asyncio.run(_persist_lite_analysis({"ticker": "", "final_score": 5}, bq))
    bq.save_report.assert_not_called()


def test_persist_handles_missing_full_report():
    """No full_report → still writes with defaults."""
    bq = MagicMock()
    asyncio.run(_persist_lite_analysis({"ticker": "AAPL", "recommendation": "HOLD"}, bq))
    bq.save_report.assert_called_once()
    kwargs = bq.save_report.call_args.kwargs
    assert kwargs["ticker"] == "AAPL"
    assert kwargs["company_name"] == "AAPL"
    assert kwargs["recommendation"] == "HOLD"
    assert kwargs["final_score"] == 0.0
    assert kwargs["summary"] == ""
    assert kwargs["standard_model"] == ""


def test_persist_preserves_full_report_for_audit():
    """The complete full_report dict is passed through to save_report."""
    bq = MagicMock()
    a = _lite_analysis()
    asyncio.run(_persist_lite_analysis(a, bq))
    kwargs = bq.save_report.call_args.kwargs
    assert kwargs["full_report"] == a["full_report"]
    # Check audit-trail fields are inside full_report
    assert kwargs["full_report"]["source"] == "claude-sonnet-4-6"
    assert kwargs["full_report"]["analysis"]["confidence"] == 72


def test_persist_defaults_recommendation_to_HOLD():
    bq = MagicMock()
    asyncio.run(_persist_lite_analysis({"ticker": "AAPL", "final_score": 5}, bq))
    kwargs = bq.save_report.call_args.kwargs
    assert kwargs["recommendation"] == "HOLD"
