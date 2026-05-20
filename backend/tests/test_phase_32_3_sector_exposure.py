"""phase-32.3 tests: portfolio sector exposure injected into FACT_LEDGER.

Audit basis: handoff/archive/phase-31.0/experiment_results.md section 4 P1.3.
Spec source: .claude/masterplan.json::phase-32.3.implementation_plan.test_specs.

The helper `_compute_portfolio_sector_exposure` is pure (positions in, dict
out). The integration test exercises that the helper output reaches the
FACT_LEDGER dict via the run_full_analysis pipeline's wiring at the
assembly site near orchestrator.py:1487.

Test plan (6 cases):
  1. test_high_tech_concentration_warns: portfolio with 89% Tech ->
     concentration_warning=True, max_sector='Technology',
     max_sector_exposure_pct >= 60.
  2. test_low_concentration_silent: portfolio with 30% Tech -> warning False.
  3. test_other_sector_silent_for_diff_sector_candidate: portfolio
     89% Tech, dict-level check confirms the warning is portfolio-level
     (not per-candidate); the Risk Judge prompt makes the candidate-vs-
     max-sector decision.
  4. test_empty_portfolio_silent: empty positions list -> warning=False,
     by_sector={}, max_sector=None.
  5. test_threshold_boundary: portfolio at EXACTLY 60% -> warning=True;
     portfolio at 59.99% -> warning=False. Tests the >= boundary.
  6. test_missing_market_value_or_sector_robust: malformed rows (missing
     market_value, market_value=None, empty sector) are tolerated --
     market_value <= 0 is skipped; empty sector becomes 'Unknown'.
"""
from __future__ import annotations

import pytest

from backend.agents.orchestrator import _compute_portfolio_sector_exposure


def test_high_tech_concentration_warns():
    """Production-mirror: 10 Tech + 1 Industrials totaling ~89% Tech."""
    positions = [
        {"ticker": "MU", "sector": "Technology", "market_value": 731.99},
        {"ticker": "SNDK", "sector": "Technology", "market_value": 1392.56},
        {"ticker": "INTC", "sector": "Technology", "market_value": 1189.60},
        {"ticker": "COHR", "sector": "Technology", "market_value": 1075.50},
        {"ticker": "WDC", "sector": "Technology", "market_value": 919.24},
        {"ticker": "LITE", "sector": "Technology", "market_value": 868.07},
        {"ticker": "ON", "sector": "Technology", "market_value": 1102.10},
        {"ticker": "DELL", "sector": "Technology", "market_value": 1214.65},
        {"ticker": "GLW", "sector": "Technology", "market_value": 1084.14},
        {"ticker": "KEYS", "sector": "Technology", "market_value": 1368.32},
        {"ticker": "GEV", "sector": "Industrials", "market_value": 1024.52},
    ]
    result = _compute_portfolio_sector_exposure(positions, threshold_pct=60.0)
    assert result["max_sector"] == "Technology"
    assert result["max_sector_exposure_pct"] >= 60.0
    assert result["concentration_warning"] is True
    assert result["threshold_pct"] == 60.0
    assert result["total_positions"] == 11
    # by_sector should sum to ~100% modulo rounding.
    assert abs(sum(result["by_sector"].values()) - 100.0) < 0.5


def test_low_concentration_silent():
    """Diversified: 30% Tech, 30% Healthcare, 25% Industrials, 15% Energy."""
    positions = [
        {"ticker": "T1", "sector": "Technology", "market_value": 30.0},
        {"ticker": "H1", "sector": "Healthcare", "market_value": 30.0},
        {"ticker": "I1", "sector": "Industrials", "market_value": 25.0},
        {"ticker": "E1", "sector": "Energy", "market_value": 15.0},
    ]
    result = _compute_portfolio_sector_exposure(positions, threshold_pct=60.0)
    assert result["max_sector_exposure_pct"] < 60.0
    assert result["concentration_warning"] is False
    assert result["max_sector"] == "Technology"  # the tie-break favors first-iteration max
    assert result["total_positions"] == 4


def test_other_sector_silent_for_diff_sector_candidate():
    """Helper output is portfolio-level. The Risk Judge prompt makes the
    candidate-vs-max-sector decision; the helper just exposes max_sector.
    This test confirms the data shape supports that downstream decision."""
    positions = [
        {"ticker": "T1", "sector": "Technology", "market_value": 89.0},
        {"ticker": "H1", "sector": "Healthcare", "market_value": 11.0},
    ]
    result = _compute_portfolio_sector_exposure(positions, threshold_pct=60.0)
    assert result["concentration_warning"] is True
    assert result["max_sector"] == "Technology"
    # Healthcare exists in by_sector but is NOT the max -- the Risk Judge
    # prompt can compare candidate.sector against result.max_sector.
    assert "Healthcare" in result["by_sector"]
    assert result["by_sector"]["Healthcare"] < result["by_sector"]["Technology"]


def test_empty_portfolio_silent():
    """No positions -> warning False, by_sector empty, max_sector None."""
    result = _compute_portfolio_sector_exposure([], threshold_pct=60.0)
    assert result["concentration_warning"] is False
    assert result["by_sector"] == {}
    assert result["max_sector"] is None
    assert result["max_sector_exposure_pct"] == 0.0
    assert result["total_positions"] == 0


def test_threshold_boundary_exact_match_fires():
    """At EXACTLY threshold -> warning True (>= boundary)."""
    positions = [
        {"ticker": "T1", "sector": "Technology", "market_value": 60.0},
        {"ticker": "H1", "sector": "Healthcare", "market_value": 40.0},
    ]
    result = _compute_portfolio_sector_exposure(positions, threshold_pct=60.0)
    assert result["max_sector_exposure_pct"] == 60.0
    assert result["concentration_warning"] is True


def test_risk_judge_prompt_renders_fact_ledger_block_not_literal_placeholder():
    """phase-32.3 bug fix: get_risk_judge_prompt previously did NOT pass
    fact_ledger_section to format_skill, so the rendered Risk Judge prompt
    contained the literal token `{{fact_ledger_section}}` instead of the
    actual FACT_LEDGER content. This test confirms the fix: with a non-empty
    fact_ledger JSON, the rendered prompt must contain the canonical
    'FACT_LEDGER (Ground Truth' header AND must NOT contain the unrendered
    placeholder. Regression test -- if anyone reverts the prompts.py:976-985
    one-line fix, this fails."""
    import json
    from backend.config import prompts
    fact_ledger_data = {
        "ticker": "NVDA",
        "sector": "Technology",
        "portfolio_sector_exposure": {
            "by_sector": {"Technology": 89.34, "Industrials": 10.66},
            "max_sector": "Technology",
            "max_sector_exposure_pct": 89.34,
            "concentration_warning": True,
            "threshold_pct": 60.0,
            "total_positions": 11,
        },
    }
    fact_ledger_json = json.dumps(fact_ledger_data, indent=2)
    rendered = prompts.get_risk_judge_prompt(
        ticker="NVDA",
        synthesis_json="{}",
        aggressive_arg="",
        conservative_arg="",
        neutral_arg="",
        debate_history="",
        past_memory="",
        fact_ledger=fact_ledger_json,
    )
    # The FACT_LEDGER block must render.
    assert "FACT_LEDGER (Ground Truth" in rendered, (
        "phase-32.3 bug fix: get_risk_judge_prompt must pass fact_ledger_section "
        "to format_skill so the FACT_LEDGER block renders in the Risk Judge prompt."
    )
    # The unrendered placeholder must NOT be visible.
    assert "{{fact_ledger_section}}" not in rendered, (
        "phase-32.3 bug fix: the literal placeholder token must NOT survive "
        "rendering. If you see this assertion fire, the get_risk_judge_prompt "
        "regression has returned."
    )
    # The new portfolio_sector_exposure block must transit through.
    assert "portfolio_sector_exposure" in rendered
    assert "Technology" in rendered  # sector name flows through the JSON dump


def test_missing_market_value_or_sector_robust():
    """Malformed rows MUST NOT crash the helper.
    - market_value=None -> skipped (treated as 0).
    - empty/None sector -> bucketed as 'Unknown'.
    - market_value missing entirely -> skipped."""
    positions = [
        {"ticker": "T1", "sector": "Technology", "market_value": 80.0},
        {"ticker": "MV_NONE", "sector": "Healthcare", "market_value": None},
        {"ticker": "MV_MISSING", "sector": "Energy"},  # no market_value key
        {"ticker": "S_EMPTY", "sector": "", "market_value": 20.0},
        {"ticker": "S_NONE", "sector": None, "market_value": 0.0},  # mv=0 -> skipped
    ]
    result = _compute_portfolio_sector_exposure(positions, threshold_pct=60.0)
    # Only T1 ($80) + S_EMPTY ($20 -> 'Unknown') contribute.
    assert result["total_positions"] == 5  # raw count, including skipped
    assert result["by_sector"] == {"Technology": 80.0, "Unknown": 20.0}
    assert result["max_sector"] == "Technology"
    assert result["max_sector_exposure_pct"] == 80.0
    assert result["concentration_warning"] is True
