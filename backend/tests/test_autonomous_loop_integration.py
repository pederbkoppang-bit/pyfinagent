"""phase-3.1 tests for AutonomousLoopOrchestrator integration.

Covers the two phase-3.1-close wiring changes:
 - `_load_real_context` reads real TSV + JSON when files exist.
 - `_load_real_context` falls back to mocks when files missing.

The orchestrator's __init__ calls `bigquery.Client(project=...)` which
requires auth; we monkeypatch it for all tests.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def orchestrator():
    with patch("google.cloud.bigquery.Client", MagicMock()):
        from backend.autonomous_loop import AutonomousLoopOrchestrator

        return AutonomousLoopOrchestrator(
            project_id="test-proj", dataset_id="test-ds"
        )


def test_load_real_context_returns_expected_shape(orchestrator):
    """Files exist in repo -> returns lists/dicts with plausible content."""
    recent, params = orchestrator._load_real_context(current_best_sharpe=1.17)
    assert isinstance(recent, list)
    assert isinstance(params, dict)
    assert len(recent) >= 1
    # Every row from the real path has these keys
    for r in recent:
        assert "sharpe" in r
        assert "features" in r
        assert isinstance(r["features"], list)
    # params from optimizer_best.json has many keys; require a minimum floor
    assert len(params) >= 4  # at least 4 params even in fallback


def test_load_real_context_fallback_when_files_missing(tmp_path, orchestrator, monkeypatch):
    """Patch the Path resolution to miss both files -> fallback mocks activate."""
    # Monkeypatch pathlib.Path.exists to return False for our target paths only.
    # Simplest approach: patch the module's Path object so parents[0] points
    # into an empty tmp dir.
    from backend import autonomous_loop as al

    # Replace the class attribute not feasible; instead we swap the
    # `__file__` anchor by monkeypatching `Path` behavior inside the helper
    # via a minimal mock: force a fresh method call on an orchestrator
    # whose helper sees empty paths. The helper already short-circuits
    # to mocks if the files don't exist, so we simulate by removing the
    # known files temporarily via `.exists()` monkeypatch.

    from pathlib import Path as _RealPath
    real_resolve = _RealPath.resolve

    with patch.object(_RealPath, "exists", lambda self: False):
        recent, params = orchestrator._load_real_context(
            current_best_sharpe=1.0
        )

    # fallback path -> exactly one mock row, 4 legacy params
    assert len(recent) == 1
    assert recent[0]["sharpe"] == 1.0  # echoed current_best_sharpe
    assert len(params) == 4
    assert set(params.keys()) == {"ma_short", "ma_long", "rsi_threshold", "vol_lookback"}
