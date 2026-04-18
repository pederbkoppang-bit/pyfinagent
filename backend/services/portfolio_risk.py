"""phase-4.8 step 4.8.2 Portfolio risk primitives.

Three functions:

1. `compute_cvar(returns, alpha=0.975)` -- Conditional Value-at-Risk
   at confidence `alpha`. Historical (empirical) method per
   Rockafellar-Uryasev 2000: CVaR_alpha = E[L | L >= VaR_alpha],
   where L = -return (loss). Returns a POSITIVE number representing
   expected loss magnitude (e.g., 0.025 = 2.5% daily CVaR).

2. `compute_ff3(portfolio_returns, factor_returns)` -- OLS
   regression of portfolio excess return on Fama-French 3 factors
   (Mkt-Rf, SMB, HML) via numpy.linalg.lstsq. Returns {alpha,
   market_beta, smb_beta, hml_beta, r_squared, n_obs}.

3. `daily_check()` -- gate decision on current NAV history.
   Returns {cvar_97_5, ff3, gate: {new_positions_allowed, reasons}}.
   If NAV history is absent, seeds deterministic 252-day returns
   and flags `data_source: "seeded"`.
"""
from __future__ import annotations

import hashlib
import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


CVAR_LIMIT_PCT = 0.02   # 2% daily CVaR_97.5 ceiling
BETA_CAP = 1.5          # absolute market-beta cap
CVAR_ALPHA = 0.975


def compute_cvar(returns: list[float] | np.ndarray, alpha: float = CVAR_ALPHA) -> float:
    """Historical-simulation CVaR at confidence `alpha`.

    Loss distribution is `-returns`. CVaR = mean of losses that are
    at or beyond the alpha-quantile of the loss distribution.
    Returns a non-negative float in return-space units (e.g., 0.025).
    """
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    losses = -r
    # Rank-based quantile: for alpha=0.975 on 252 obs, threshold index
    # is ceil(0.975 * 252) - 1 = 246 (6 tail-loss observations).
    threshold = float(np.quantile(losses, alpha))
    tail = losses[losses >= threshold]
    if tail.size == 0:
        return max(threshold, 0.0)
    return max(float(tail.mean()), 0.0)


def compute_ff3(
    portfolio_returns: list[float] | np.ndarray,
    factor_returns: dict[str, list[float] | np.ndarray],
    rf: list[float] | np.ndarray | float = 0.0,
) -> dict[str, float]:
    """OLS regression: port_return - rf ~ alpha + b_mkt*MktRf + b_smb*SMB + b_hml*HML.

    Parameters
    ----------
    portfolio_returns : 1D array of length T.
    factor_returns : dict with keys 'Mkt-Rf', 'SMB', 'HML'; each a
        1D array of length T.
    rf : scalar or 1D array of length T (risk-free rate).
    """
    y_raw = np.asarray(portfolio_returns, dtype=float)
    mkt = np.asarray(factor_returns.get("Mkt-Rf", []), dtype=float)
    smb = np.asarray(factor_returns.get("SMB", []), dtype=float)
    hml = np.asarray(factor_returns.get("HML", []), dtype=float)
    if not (y_raw.size == mkt.size == smb.size == hml.size) or y_raw.size < 4:
        return {"alpha": 0.0, "market_beta": 0.0, "smb_beta": 0.0,
                "hml_beta": 0.0, "r_squared": 0.0, "n_obs": int(y_raw.size)}
    rf_arr = np.asarray(rf, dtype=float) if not np.isscalar(rf) else np.full_like(y_raw, float(rf))
    y = y_raw - rf_arr
    X = np.column_stack([np.ones_like(mkt), mkt, smb, hml])
    coeffs, _resid, _rank, _sv = np.linalg.lstsq(X, y, rcond=None)
    alpha = float(coeffs[0])
    b_mkt, b_smb, b_hml = map(float, coeffs[1:])
    y_hat = X @ coeffs
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return {
        "alpha": alpha,
        "market_beta": b_mkt,
        "smb_beta": b_smb,
        "hml_beta": b_hml,
        "r_squared": max(0.0, min(1.0, r2)),
        "n_obs": int(y.size),
    }


def _seed_returns(seed: str, n: int = 252, sigma: float = 0.008) -> tuple[np.ndarray, dict]:
    """Deterministic returns: a normal-ish sample with fixed seed.

    Default sigma 0.8% daily puts CVaR_97.5 around 1.5% (comfortably
    below the 2% gate ceiling) for a reasonable "benign" fixture.
    Callers who need a CVaR-tripping fixture should supply a fat-tail
    series via `portfolio_returns=` (see scripts/audit/portfolio_
    risk_audit.py).
    """
    h = int(hashlib.sha1(seed.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(h)
    # ~6 bps daily drift; 0.8% daily vol ~ calm book (not realistic for
    # growth equities; sufficient for the signature contract test).
    returns = rng.normal(loc=0.0006, scale=sigma, size=n)
    return returns, {
        "seed": seed,
        "n": n,
        "sigma": sigma,
        "mean": float(returns.mean()),
        "std": float(returns.std()),
    }


def _seed_factors(n: int = 252) -> dict[str, np.ndarray]:
    """Deterministic 3-factor returns aligned with port seed length."""
    rng = np.random.default_rng(42)
    return {
        "Mkt-Rf": rng.normal(0.0004, 0.01, n),
        "SMB": rng.normal(0.0001, 0.006, n),
        "HML": rng.normal(0.0001, 0.005, n),
    }


def daily_check(
    portfolio_returns: list[float] | np.ndarray | None = None,
    factor_returns: dict[str, list[float]] | None = None,
    seed: str = "daily-check-default",
) -> dict[str, Any]:
    """Gate decision for new positions. Returns a self-describing dict:

        {
          "cvar_97_5": {"value": float, "limit": CVAR_LIMIT_PCT,
                        "exceeds_limit": bool, "alpha": 0.975},
          "ff3": {"alpha", "market_beta", ...},
          "gate": {"new_positions_allowed": bool,
                   "blocking_reasons": [..]},
          "data_source": "live" | "seeded",
          "meta": {...}
        }
    """
    meta: dict[str, Any] = {}
    data_source = "live"
    if portfolio_returns is None:
        returns, seed_meta = _seed_returns(seed)
        meta["seed"] = seed_meta
        data_source = "seeded"
    else:
        returns = np.asarray(portfolio_returns, dtype=float)
    if factor_returns is None:
        factor_returns = {k: v for k, v in _seed_factors(len(returns)).items()}
        data_source = "seeded" if data_source == "seeded" else "partial_seeded"
        meta["factors_seeded"] = True

    cvar = compute_cvar(returns, alpha=CVAR_ALPHA)
    ff3 = compute_ff3(returns, factor_returns)

    reasons: list[str] = []
    if cvar > CVAR_LIMIT_PCT:
        reasons.append(
            f"cvar_exceeded ({cvar:.4f} > {CVAR_LIMIT_PCT:.4f})"
        )
    if abs(ff3["market_beta"]) > BETA_CAP:
        reasons.append(
            f"beta_cap_exceeded (|{ff3['market_beta']:.3f}| > {BETA_CAP})"
        )

    return {
        "cvar_97_5": {
            "value": round(float(cvar), 6),
            "limit": CVAR_LIMIT_PCT,
            "alpha": CVAR_ALPHA,
            "exceeds_limit": cvar > CVAR_LIMIT_PCT,
        },
        "ff3": ff3,
        "gate": {
            "new_positions_allowed": not reasons,
            "blocking_reasons": reasons,
            "cvar_limit_pct": CVAR_LIMIT_PCT,
            "beta_cap": BETA_CAP,
        },
        "data_source": data_source,
        "meta": meta,
    }


__all__ = [
    "BETA_CAP",
    "CVAR_ALPHA",
    "CVAR_LIMIT_PCT",
    "compute_cvar",
    "compute_ff3",
    "daily_check",
]
