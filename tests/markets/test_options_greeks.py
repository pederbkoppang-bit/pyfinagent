"""phase-5.6 unit tests for Black-Scholes greeks + OCC symbol parser.

12 tests covering greeks correctness (delta sanity, sign conventions,
put-call parity), edge cases (T=0, sigma=0), OCC symbol parsing, and
inline reproduction of the masterplan immutable verification.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.markets.options.greeks import (  # noqa: E402
    black_scholes_greeks,
    parse_occ_symbol,
)


# ----------------------
# Greeks: delta correctness
# ----------------------

def test_immutable_verification_atm_call_delta():
    """Reproduce the masterplan immutable verification call delta check."""
    g = black_scholes_greeks(
        S=450, K=450, T=30 / 365, r=0.05, sigma=0.20, option_type="call"
    )
    assert 0.4 < g["delta"] < 0.6
    # Researcher computed 0.5400 in-venv; assert tighter range.
    assert 0.50 < g["delta"] < 0.58


def test_atm_put_delta_negative_and_paired():
    """ATM put delta is negative and pairs with call: |call| + |put| ~= 1 (q=0)."""
    call = black_scholes_greeks(450, 450, 30 / 365, 0.05, 0.20, "call")
    put = black_scholes_greeks(450, 450, 30 / 365, 0.05, 0.20, "put")
    assert put["delta"] < 0
    # Put-call parity for delta (q=0): call_delta - put_delta = 1
    assert call["delta"] - put["delta"] == pytest.approx(1.0, abs=1e-10)


def test_deep_itm_call_delta_near_one():
    """Deep ITM call (S >> K) should have delta -> 1."""
    g = black_scholes_greeks(S=600, K=450, T=30 / 365, r=0.05, sigma=0.20, option_type="call")
    assert g["delta"] > 0.95


def test_deep_otm_call_delta_near_zero():
    """Deep OTM call (S << K) should have delta -> 0."""
    g = black_scholes_greeks(S=300, K=450, T=30 / 365, r=0.05, sigma=0.20, option_type="call")
    assert 0 < g["delta"] < 0.05


# ----------------------
# Greeks: sign conventions
# ----------------------

def test_gamma_positive_for_long_options():
    call = black_scholes_greeks(450, 450, 30 / 365, 0.05, 0.20, "call")
    put = black_scholes_greeks(450, 450, 30 / 365, 0.05, 0.20, "put")
    assert call["gamma"] > 0
    assert put["gamma"] > 0


def test_vega_positive_for_long_options():
    call = black_scholes_greeks(450, 450, 30 / 365, 0.05, 0.20, "call")
    put = black_scholes_greeks(450, 450, 30 / 365, 0.05, 0.20, "put")
    assert call["vega"] > 0
    assert put["vega"] > 0


def test_theta_negative_for_long_options():
    """Theta in per-day units should be negative for both calls and puts."""
    call = black_scholes_greeks(450, 450, 30 / 365, 0.05, 0.20, "call")
    put = black_scholes_greeks(450, 450, 30 / 365, 0.05, 0.20, "put")
    assert call["theta"] < 0
    assert put["theta"] < 0


# ----------------------
# Greeks: edge cases
# ----------------------

def test_expired_call_intrinsic_only():
    """T <= 0 -> intrinsic value, delta = 1 if ITM else 0, others = 0."""
    g = black_scholes_greeks(S=460, K=450, T=0, r=0.05, sigma=0.20, option_type="call")
    assert g["price"] == pytest.approx(10.0)
    assert g["delta"] == pytest.approx(1.0)
    assert g["gamma"] == 0.0
    assert g["theta"] == 0.0
    assert g["vega"] == 0.0


def test_zero_sigma_does_not_raise():
    """sigma=0 floors to MIN_SIGMA; numerical stability, no div-by-zero."""
    g = black_scholes_greeks(450, 450, 30 / 365, 0.05, 0.0, "call")
    assert "delta" in g
    assert g["price"] >= 0


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        black_scholes_greeks(S=0, K=450, T=30 / 365, r=0.05, sigma=0.20)
    with pytest.raises(ValueError):
        black_scholes_greeks(S=450, K=0, T=30 / 365, r=0.05, sigma=0.20)
    with pytest.raises(ValueError):
        black_scholes_greeks(S=450, K=450, T=30 / 365, r=0.05, sigma=0.20, option_type="other")


# ----------------------
# OCC symbol parser
# ----------------------

def test_parse_occ_unpadded():
    """Common compact form: 'AAPL240119C00150000'."""
    out = parse_occ_symbol("AAPL240119C00150000")
    assert out["ticker"] == "AAPL"
    assert out["expiration"] == "2024-01-19"
    assert out["option_type"] == "call"
    assert out["strike"] == 150.0


def test_parse_occ_put():
    out = parse_occ_symbol("SPY240315P00450000")
    assert out["ticker"] == "SPY"
    assert out["expiration"] == "2024-03-15"
    assert out["option_type"] == "put"
    assert out["strike"] == 450.0


def test_parse_occ_invalid_raises():
    with pytest.raises(ValueError):
        parse_occ_symbol("too_short")
    with pytest.raises(ValueError):
        parse_occ_symbol("AAPL240119X00150000")  # X is not C or P
    with pytest.raises(ValueError):
        parse_occ_symbol(12345)  # type: ignore[arg-type]
