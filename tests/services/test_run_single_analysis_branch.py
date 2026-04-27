"""phase-23.1.12: _run_single_analysis branches on settings.lite_mode.

The hardcoded `settings.lite_mode = True` in run_paper_trading_cycle Step 3
was overriding the operator's choice. This test suite confirms the fix:
the function path now follows the operator-configured flag.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.services.autonomous_loop import _run_single_analysis


def _make_settings(lite_mode: bool, gemini_model: str = "claude-sonnet-4-6") -> MagicMock:
    s = MagicMock()
    s.lite_mode = lite_mode
    s.gemini_model = gemini_model
    s.deep_think_model = "claude-opus-4-6"
    s.anthropic_api_key = "sk-ant-test"
    return s


def test_lite_mode_true_calls_only_claude_lite_path():
    """When operator opts into lite_mode, only the lite Claude analyzer runs."""
    fake_lite_result = {"ticker": "AAPL", "_path": "lite", "recommendation": "BUY"}
    with patch(
        "backend.services.autonomous_loop._run_claude_analysis",
        new=AsyncMock(return_value=fake_lite_result),
    ) as lite_mock, patch(
        "backend.services.autonomous_loop.AnalysisOrchestrator",
    ) as orch_cls:
        out = asyncio.run(_run_single_analysis("AAPL", _make_settings(lite_mode=True)))
        assert out == fake_lite_result
        lite_mock.assert_called_once()
        orch_cls.assert_not_called()


def test_lite_mode_false_runs_full_orchestrator_with_operator_models():
    """When operator chose lite_mode=False, the full pipeline runs WITHOUT
    forcing Gemini fallback models — the operator's gemini_model and
    deep_think_model are honored."""
    s = _make_settings(lite_mode=False, gemini_model="claude-sonnet-4-6")

    fake_report = {
        "final_synthesis": {
            "recommendation": {"action": "BUY"},
            "final_score": 9,
            "risk_assessment": {"reason": "looks good"},
        },
        "quant": {"yf_data": {"valuation": {"currentPrice": 200.0}}},
        "cost_summary": {"total_cost_usd": 1.5},
    }
    fake_orch = MagicMock()
    fake_orch.run_full_analysis = AsyncMock(return_value=fake_report)

    with patch(
        "backend.services.autonomous_loop.AnalysisOrchestrator",
        return_value=fake_orch,
    ) as orch_cls, patch(
        "backend.services.autonomous_loop._run_claude_analysis",
        new=AsyncMock(),
    ) as lite_mock:
        out = asyncio.run(_run_single_analysis("AAPL", s))

        # Orchestrator called with the operator's settings (NOT a fallback copy)
        orch_cls.assert_called_once_with(s)
        # Lite path NOT touched — full orchestrator succeeded
        lite_mock.assert_not_called()

        assert out["recommendation"] == "BUY"
        assert out["final_score"] == 9
        assert out["price_at_analysis"] == 200.0
        assert out["total_cost_usd"] == 1.5
        # phase-23.1.12: full path does NOT have the _path=lite marker
        assert out.get("_path") != "lite"


def test_lite_mode_false_falls_back_to_lite_when_orchestrator_fails():
    """If the full orchestrator raises or returns empty, lite Claude takes over."""
    fake_orch = MagicMock()
    fake_orch.run_full_analysis = AsyncMock(side_effect=RuntimeError("vertex down"))

    fake_lite_result = {"ticker": "AAPL", "_path": "lite", "recommendation": "HOLD"}
    with patch(
        "backend.services.autonomous_loop.AnalysisOrchestrator",
        return_value=fake_orch,
    ), patch(
        "backend.services.autonomous_loop._run_claude_analysis",
        new=AsyncMock(return_value=fake_lite_result),
    ) as lite_mock:
        out = asyncio.run(_run_single_analysis("AAPL", _make_settings(lite_mode=False)))
        assert out == fake_lite_result
        lite_mock.assert_called_once()


def test_lite_mode_false_returns_none_when_both_paths_fail():
    """Both full and lite fail → None returned (cycle keeps going for other tickers)."""
    fake_orch = MagicMock()
    fake_orch.run_full_analysis = AsyncMock(side_effect=RuntimeError("vertex down"))

    with patch(
        "backend.services.autonomous_loop.AnalysisOrchestrator",
        return_value=fake_orch,
    ), patch(
        "backend.services.autonomous_loop._run_claude_analysis",
        new=AsyncMock(side_effect=RuntimeError("anthropic down")),
    ):
        out = asyncio.run(_run_single_analysis("AAPL", _make_settings(lite_mode=False)))
        assert out is None


def test_lite_mode_false_returns_none_when_orchestrator_returns_empty():
    """Empty report from orchestrator → fall back to lite."""
    fake_orch = MagicMock()
    fake_orch.run_full_analysis = AsyncMock(return_value=None)

    fake_lite_result = {"ticker": "AAPL", "_path": "lite", "recommendation": "HOLD"}
    with patch(
        "backend.services.autonomous_loop.AnalysisOrchestrator",
        return_value=fake_orch,
    ), patch(
        "backend.services.autonomous_loop._run_claude_analysis",
        new=AsyncMock(return_value=fake_lite_result),
    ):
        out = asyncio.run(_run_single_analysis("AAPL", _make_settings(lite_mode=False)))
        # Falls back to lite when orchestrator returns empty/None
        assert out == fake_lite_result


def test_lite_mode_true_returns_none_when_lite_fails():
    """In lite_mode=True, no Gemini fallback — failure propagates as None."""
    with patch(
        "backend.services.autonomous_loop._run_claude_analysis",
        new=AsyncMock(side_effect=RuntimeError("anthropic 401")),
    ):
        out = asyncio.run(_run_single_analysis("AAPL", _make_settings(lite_mode=True)))
        assert out is None
