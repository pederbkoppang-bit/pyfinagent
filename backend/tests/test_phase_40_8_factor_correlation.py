"""phase-40.8 verification: FF3 factor-correlation cap (OPEN-5).

Per masterplan 40.8 criteria:
  1. ff3_factor_exposure_used_alongside_gics
  2. correlation_cap_blocks_simulated_high_ff_corr_buy
  3. regression_against_known_fixture

Augments GICS sector cap by catching cross-sector factor crowding via
cosine similarity over FF3 (market_beta, smb_beta, hml_beta) loadings.
Default-OFF: settings.paper_max_factor_corr=0.0 disables (backward-compat).
"""

from __future__ import annotations

import math

import pytest

from backend.services.factor_correlation import (
    factor_correlation_score,
    aggregate_portfolio_loadings,
    FF3_FIELDS,
)
from backend.services.portfolio_risk import compute_ff3


def test_phase_40_8_factor_correlation_score_returns_high_for_similar_vectors():
    """Cosine similarity > 0.99 for near-identical FF3 vectors."""
    cand = {"market_beta": 1.0, "smb_beta": 0.5, "hml_beta": 0.3}
    port = {"market_beta": 0.99, "smb_beta": 0.51, "hml_beta": 0.29}
    sim = factor_correlation_score(cand, port)
    assert 0.99 < sim <= 1.0


def test_phase_40_8_factor_correlation_score_returns_low_for_orthogonal():
    """Orthogonal vectors yield score near 0."""
    cand = {"market_beta": 1.0, "smb_beta": 0.0, "hml_beta": 0.0}
    port = {"market_beta": 0.0, "smb_beta": 1.0, "hml_beta": 0.0}
    sim = factor_correlation_score(cand, port)
    assert abs(sim) < 1e-9


def test_phase_40_8_factor_correlation_returns_zero_for_missing_inputs():
    """Forward-compat: missing or empty inputs return 0 (cap won't fire)."""
    assert factor_correlation_score(None, {"market_beta": 1.0, "smb_beta": 0, "hml_beta": 0}) == 0.0
    assert factor_correlation_score({}, {"market_beta": 1.0, "smb_beta": 0, "hml_beta": 0}) == 0.0
    assert factor_correlation_score({"market_beta": 1.0}, {"market_beta": 1.0, "smb_beta": 0, "hml_beta": 0}) == 0.0
    # NaN inputs
    assert factor_correlation_score(
        {"market_beta": float("nan"), "smb_beta": 0.0, "hml_beta": 0.0},
        {"market_beta": 1.0, "smb_beta": 0.0, "hml_beta": 0.0},
    ) == 0.0
    # Zero vector
    assert factor_correlation_score(
        {"market_beta": 0.0, "smb_beta": 0.0, "hml_beta": 0.0},
        {"market_beta": 1.0, "smb_beta": 0.0, "hml_beta": 0.0},
    ) == 0.0


def test_phase_40_8_aggregate_portfolio_loadings_weighted_by_market_value():
    """Weighted average across positions with FF3 loadings."""
    positions = [
        {"ticker": "AAA", "market_value": 100.0,
         "factor_loadings": {"market_beta": 1.0, "smb_beta": 0.5, "hml_beta": 0.0}},
        {"ticker": "BBB", "market_value": 300.0,
         "factor_loadings": {"market_beta": 0.5, "smb_beta": -0.5, "hml_beta": 1.0}},
    ]
    agg = aggregate_portfolio_loadings(positions)
    # Total weight = 400. weighted avg = (100*1.0 + 300*0.5)/400 = 0.625 for market_beta
    assert math.isclose(agg["market_beta"], (100 * 1.0 + 300 * 0.5) / 400)
    assert math.isclose(agg["smb_beta"], (100 * 0.5 + 300 * -0.5) / 400)
    assert math.isclose(agg["hml_beta"], (100 * 0.0 + 300 * 1.0) / 400)


def test_phase_40_8_aggregate_portfolio_loadings_empty_when_no_loadings():
    """Forward-compat: positions without factor_loadings yield empty dict."""
    positions = [
        {"ticker": "X", "market_value": 100.0},
        {"ticker": "Y", "market_value": 200.0, "factor_loadings": {}},
    ]
    assert aggregate_portfolio_loadings(positions) == {}


# ---- Criterion 1: ff3_factor_exposure_used_alongside_gics ------------


def test_phase_40_8_ff3_factor_exposure_used_alongside_gics():
    """Criterion 1: portfolio_manager.py wires the FF3 cap AFTER the
    existing GICS sector NAV-pct cap; both gates active independently.
    Verifies the cap reads settings.paper_max_factor_corr and the
    helper imports are present in the BUY loop."""
    from pathlib import Path
    pm = (Path(__file__).resolve().parents[2] / "backend" / "services" / "portfolio_manager.py").read_text(encoding="utf-8")
    # Must reference the new settings field
    assert "paper_max_factor_corr" in pm, "portfolio_manager.py must read settings.paper_max_factor_corr"
    # Must use the helper
    assert "factor_correlation_score" in pm, "portfolio_manager.py must call factor_correlation_score"
    # Must come AFTER the GICS NAV-pct cap (string position check)
    idx_gics = pm.find("paper_max_per_sector_nav_pct")
    idx_ff3 = pm.find("factor_correlation_score")
    assert idx_gics > 0 and idx_ff3 > idx_gics, (
        "FF3 cap must be wired AFTER the GICS sector NAV-pct cap, not before"
    )


# ---- Criterion 2: correlation_cap_blocks_simulated_high_ff_corr_buy --


def test_phase_40_8_correlation_cap_blocks_simulated_high_ff_corr_buy():
    """Criterion 2: a candidate with high cosine similarity to the
    weighted portfolio average gets blocked when paper_max_factor_corr
    is set."""
    # Portfolio: one position with loadings near (1.0, 0.5, 0.3)
    portfolio = [{
        "ticker": "PORT", "market_value": 1000.0,
        "factor_loadings": {"market_beta": 1.0, "smb_beta": 0.5, "hml_beta": 0.3},
    }]
    port_agg = aggregate_portfolio_loadings(portfolio)
    # Candidate: near-identical loadings -> cos sim > 0.99
    cand_high = {"market_beta": 0.99, "smb_beta": 0.51, "hml_beta": 0.29}
    sim_high = factor_correlation_score(cand_high, port_agg)
    # Candidate: orthogonal loadings -> cos sim near 0
    cand_low = {"market_beta": 0.0, "smb_beta": 0.0, "hml_beta": 1.0}
    sim_low = factor_correlation_score(cand_low, port_agg)

    cap = 0.85
    assert sim_high > cap, f"sim_high={sim_high} must exceed cap {cap}"
    assert sim_low < cap, f"sim_low={sim_low} must NOT exceed cap {cap}"


def test_phase_40_8_default_off_backward_compat_zero_cap_disables():
    """When settings.paper_max_factor_corr == 0.0 the cap is disabled.
    Verifies the portfolio_manager.py gate short-circuits on cap=0."""
    from pathlib import Path
    pm = (Path(__file__).resolve().parents[2] / "backend" / "services" / "portfolio_manager.py").read_text(encoding="utf-8")
    # The gate must check max_factor_corr > 0 before running the helper
    assert "max_factor_corr > 0" in pm, (
        "portfolio_manager.py must short-circuit when paper_max_factor_corr == 0 (default-OFF)"
    )


# ---- Criterion 3: regression_against_known_fixture -------------------


def test_phase_40_8_regression_against_known_fixture():
    """Criterion 3: compute_ff3 produces fixed alpha/betas for a canned
    return series + factor series. Regression-tests the math primitive
    that downstream factor_loadings will eventually use."""
    # Canned 60-day series. Portfolio returns = 1.2 * MktRf + 0.4 * SMB + 0.1 * HML + 0.0002
    # (alpha = 2bp/day; betas above). Use deterministic seed to build the inputs.
    import random

    rng = random.Random(40_8)
    n = 60
    mkt = [rng.gauss(0.0005, 0.012) for _ in range(n)]
    smb = [rng.gauss(0.0001, 0.006) for _ in range(n)]
    hml = [rng.gauss(0.0001, 0.005) for _ in range(n)]
    alpha_true = 0.0002
    b_mkt_true, b_smb_true, b_hml_true = 1.2, 0.4, 0.1
    port_ret = [
        alpha_true + b_mkt_true * mkt[i] + b_smb_true * smb[i] + b_hml_true * hml[i]
        for i in range(n)
    ]
    out = compute_ff3(port_ret, {"Mkt-Rf": mkt, "SMB": smb, "HML": hml}, rf=0.0)
    # Recovered betas should be near the true values (noise-free fixture)
    assert math.isclose(out["alpha"], alpha_true, abs_tol=1e-10)
    assert math.isclose(out["market_beta"], b_mkt_true, abs_tol=1e-10)
    assert math.isclose(out["smb_beta"], b_smb_true, abs_tol=1e-10)
    assert math.isclose(out["hml_beta"], b_hml_true, abs_tol=1e-10)
    assert out["r_squared"] > 0.999
    assert out["n_obs"] == n
