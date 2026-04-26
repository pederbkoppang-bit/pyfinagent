"""phase-5.6 Black-Scholes greeks for options pricing.

Standard 1973 Black-Scholes formulas for European options on a
non-dividend-paying underlying (q=0). All greeks computed analytically
via scipy.stats.norm. Returns a dict with keys:
    delta, gamma, theta, vega, rho, price

Conventions (per Wikipedia Greeks + OCC standard):
- Theta is in PER-DAY units (annual theta / 365)
- Vega is in PER-1%-vol units (raw vega / 100), the practitioner default
- Delta of a long call is positive; long put is negative
- Gamma is positive for both long calls and long puts
- Vega is positive for both long calls and long puts
- Theta is negative for both long calls and long puts (time decay)

Edge cases:
- T <= 0: returns intrinsic value with delta = +/-1 (ITM) or 0 (OTM/ATM),
  other greeks = 0
- sigma <= 0: floors at 1e-6 to avoid div-by-zero (numerical guard)
- S <= 0 or K <= 0: raises ValueError (these are nonsensical inputs)

Pure module: no I/O, no env reads, no module-level side effects.
"""
from __future__ import annotations

import math
from typing import Any

from scipy.stats import norm

CALL = "call"
PUT = "put"
DAYS_PER_YEAR = 365.0
MIN_SIGMA = 1e-6  # numerical guard


def _intrinsic(S: float, K: float, option_type: str) -> float:
    """Intrinsic value at expiration."""
    if option_type == CALL:
        return max(S - K, 0.0)
    return max(K - S, 0.0)


def _expired_greeks(S: float, K: float, option_type: str) -> dict[str, float]:
    """Greeks at T <= 0: intrinsic value, delta = sign-of-moneyness, others = 0."""
    intrinsic = _intrinsic(S, K, option_type)
    if option_type == CALL:
        delta = 1.0 if S > K else (0.5 if S == K else 0.0)
    else:
        delta = -1.0 if S < K else (-0.5 if S == K else 0.0)
    return {
        "delta": delta,
        "gamma": 0.0,
        "theta": 0.0,
        "vega": 0.0,
        "rho": 0.0,
        "price": intrinsic,
    }


def black_scholes_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = CALL,
    q: float = 0.0,
) -> dict[str, float]:
    """Compute Black-Scholes price + greeks for a European option.

    Args:
        S: spot price
        K: strike price
        T: time to expiration in YEARS (e.g. 30/365 for 30 days)
        r: risk-free rate (annual, decimal, e.g. 0.05 for 5%)
        sigma: implied volatility (annual, decimal, e.g. 0.20 for 20%)
        option_type: 'call' or 'put'
        q: continuous dividend yield (default 0; non-dividend assumption)

    Returns:
        dict with keys: delta, gamma, theta, vega, rho, price
        theta in per-day units; vega in per-1%-vol units.

    Raises:
        ValueError if S<=0 or K<=0 or option_type not in {'call','put'}
    """
    if S <= 0 or K <= 0:
        raise ValueError(f"S and K must be positive; got S={S}, K={K}")
    ot = (option_type or "").lower().strip()
    if ot not in (CALL, PUT):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    if T <= 0:
        return _expired_greeks(S, K, ot)

    sig = max(float(sigma), MIN_SIGMA)
    sqrt_T = math.sqrt(T)

    d1 = (math.log(S / K) + (r - q + 0.5 * sig * sig) * T) / (sig * sqrt_T)
    d2 = d1 - sig * sqrt_T

    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    Nmd1 = norm.cdf(-d1)
    Nmd2 = norm.cdf(-d2)
    pdf_d1 = norm.pdf(d1)

    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)

    if ot == CALL:
        price = S * disc_q * Nd1 - K * disc_r * Nd2
        delta = disc_q * Nd1
        rho = K * T * disc_r * Nd2 / 100.0  # per-1%-rate
        theta_annual = (
            -(S * disc_q * pdf_d1 * sig) / (2.0 * sqrt_T)
            - r * K * disc_r * Nd2
            + q * S * disc_q * Nd1
        )
    else:  # put
        price = K * disc_r * Nmd2 - S * disc_q * Nmd1
        delta = -disc_q * Nmd1
        rho = -K * T * disc_r * Nmd2 / 100.0
        theta_annual = (
            -(S * disc_q * pdf_d1 * sig) / (2.0 * sqrt_T)
            + r * K * disc_r * Nmd2
            - q * S * disc_q * Nmd1
        )

    gamma = (disc_q * pdf_d1) / (S * sig * sqrt_T)
    vega = S * disc_q * pdf_d1 * sqrt_T / 100.0  # per-1%-vol
    theta_per_day = theta_annual / DAYS_PER_YEAR

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta_per_day),
        "vega": float(vega),
        "rho": float(rho),
        "price": float(price),
    }


def parse_occ_symbol(occ: str) -> dict[str, Any]:
    """Parse a 21-character OCC option symbol.

    Format: ticker(6, left-padded with spaces) + YYMMDD(6) + C|P(1) + strike8(price * 1000, zero-padded)

    Example: "AAPL  240119C00150000" -> {ticker: "AAPL", expiration: "2024-01-19",
                                           option_type: "call", strike: 150.0}

    Raises ValueError on malformed input.
    """
    if not isinstance(occ, str):
        raise ValueError(f"occ must be a string, got {type(occ).__name__}")
    s = occ.strip()
    # Tolerate the (very common) form with stripped padding e.g. "AAPL240119C00150000"
    # by detecting suffix length backwards.
    if len(s) < 15:
        raise ValueError(f"occ symbol too short: {occ!r}")
    # Last 8 chars: strike (zero-padded; integer = price * 1000)
    strike_str = s[-8:]
    # Char before that: option type C or P
    type_char = s[-9].upper()
    # 6 chars before that: YYMMDD
    yymmdd = s[-15:-9]
    # Whatever's left at the front is the ticker (trim)
    ticker = s[:-15].strip()
    if not ticker:
        raise ValueError(f"occ symbol missing ticker: {occ!r}")
    if type_char not in ("C", "P"):
        raise ValueError(f"occ symbol option_type must be C or P, got {type_char!r}")
    try:
        year = 2000 + int(yymmdd[0:2])
        month = int(yymmdd[2:4])
        day = int(yymmdd[4:6])
        strike = int(strike_str) / 1000.0
    except Exception as exc:
        raise ValueError(f"occ symbol parse error: {occ!r}: {exc!r}") from exc

    return {
        "ticker": ticker,
        "expiration": f"{year:04d}-{month:02d}-{day:02d}",
        "option_type": "call" if type_char == "C" else "put",
        "strike": strike,
    }


__all__ = ["black_scholes_greeks", "parse_occ_symbol", "CALL", "PUT"]
