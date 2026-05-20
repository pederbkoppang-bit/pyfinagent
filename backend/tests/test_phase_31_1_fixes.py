"""phase-31.1 tests for the two fixes surfaced by the morning smoketest.

Fix 1: settings.gemini_model misnomer + Anthropic credit dependency.
       Verified via main.py lifespan log -- not unit-testable without
       starting the FastAPI app. Skipped (covered by manual operator
       observation of the startup log line).

Fix 2: OutcomeTracker model-injection -> agent_memories writes fire.
       Previously `OutcomeTracker(settings)` was constructed with no
       model so the production write path was dormant
       (phase-30.0 Stage 12 FAIL; known separate-step issue disclosed
       in phase-30.3 experiment_results.md).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def test_phase_31_1_learn_from_closed_trades_passes_model_to_outcome_tracker():
    """phase-31.1: _learn_from_closed_trades must construct a model client
    via make_client and pass it to OutcomeTracker.

    This test verifies the wiring: when make_client returns a non-None
    client, OutcomeTracker is constructed with `model=<client>` so the
    `if self._model:` guard at outcome_tracker.py:147 is truthy and the
    production save_agent_memory write path is exercised.
    """
    mock_bq = MagicMock()
    mock_bq.get_paper_trades.return_value = []  # No SELL trades found -> early exit fine
    mock_settings = SimpleNamespace(gemini_model="claude-sonnet-4-6")

    constructed_with_model = {"value": None}

    def fake_outcome_tracker(settings, model=None):
        # Capture the model arg passed by _learn_from_closed_trades
        constructed_with_model["value"] = model
        tracker = MagicMock()
        tracker.evaluate_recommendation = MagicMock()
        return tracker

    fake_client = MagicMock(name="GeminiClientStub")

    with patch(
        "backend.services.outcome_tracker.OutcomeTracker",
        side_effect=fake_outcome_tracker,
    ), patch(
        "backend.agents.llm_client.make_client",
        return_value=fake_client,
    ):
        from backend.services.autonomous_loop import _learn_from_closed_trades
        asyncio.run(_learn_from_closed_trades(["WDC"], mock_bq, mock_settings))

    # The OutcomeTracker constructor MUST receive the model client.
    assert constructed_with_model["value"] is fake_client, (
        f"phase-31.1 fix BROKEN: OutcomeTracker received "
        f"model={constructed_with_model['value']!r}, expected fake_client. "
        f"agent_memories writes would stay dormant in production."
    )


def test_phase_31_1_fail_open_when_make_client_raises():
    """phase-31.1: if make_client raises (e.g., missing API keys),
    _learn_from_closed_trades MUST log a warning and proceed with
    `model=None` -- preserving the legacy behavior of NOT writing
    agent_memories rather than crashing the cycle. The substantive
    cycle (paper_trades / paper_positions writes) must NOT be blocked
    by an observability-layer failure.
    """
    mock_bq = MagicMock()
    mock_bq.get_paper_trades.return_value = []
    mock_settings = SimpleNamespace(gemini_model="claude-sonnet-4-6")

    constructed_with_model = {"value": "<sentinel-not-set>"}

    def fake_outcome_tracker(settings, model=None):
        constructed_with_model["value"] = model
        tracker = MagicMock()
        return tracker

    with patch(
        "backend.services.outcome_tracker.OutcomeTracker",
        side_effect=fake_outcome_tracker,
    ), patch(
        "backend.agents.llm_client.make_client",
        side_effect=ValueError("simulated missing API key"),
    ):
        from backend.services.autonomous_loop import _learn_from_closed_trades
        # MUST NOT raise -- fail-open.
        asyncio.run(_learn_from_closed_trades(["WDC"], mock_bq, mock_settings))

    # OutcomeTracker should still be constructed (just with model=None).
    assert constructed_with_model["value"] is None, (
        f"phase-31.1 fail-open BROKEN: expected model=None on make_client "
        f"failure, got {constructed_with_model['value']!r}"
    )


def test_phase_31_1_grep_symbol_present_in_autonomous_loop():
    """phase-31.1: the symbol `make_client` must appear inside
    `_learn_from_closed_trades` so a future refactor that removes the
    wiring breaks pytest."""
    from pathlib import Path
    src = (
        Path(__file__).resolve().parents[1]
        / "services" / "autonomous_loop.py"
    ).read_text(encoding="utf-8")

    # Find the _learn_from_closed_trades function block
    fn_idx = src.find("async def _learn_from_closed_trades")
    assert fn_idx >= 0, "_learn_from_closed_trades function not found"
    # Take the next ~80 lines (the function body)
    block = src[fn_idx : fn_idx + 4000]
    assert "make_client" in block, (
        "phase-31.1 wiring missing: _learn_from_closed_trades must call "
        "make_client to construct the OutcomeTracker reflection-model client"
    )
    assert "OutcomeTracker" in block, (
        "OutcomeTracker reference must remain in _learn_from_closed_trades"
    )


def test_phase_31_1_main_startup_logs_provider_routing():
    """phase-31.1: backend/main.py lifespan must log the standard-tier
    provider AND warn when gemini_model is misnamed (non-Gemini)."""
    from pathlib import Path
    src = (
        Path(__file__).resolve().parents[1]
        / "main.py"
    ).read_text(encoding="utf-8")

    assert "phase-31.1 model routing" in src, (
        "main.py lifespan must contain the phase-31.1 model-routing "
        "startup log line"
    )
    assert "standard-tier provider" in src, (
        "main.py must log the resolved standard-tier provider"
    )
    assert "non-Gemini model" in src, (
        "main.py must warn explicitly when gemini_model is misnamed"
    )
