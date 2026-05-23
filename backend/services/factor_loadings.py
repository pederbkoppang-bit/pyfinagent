"""phase-40.8.1 (P3): wire compute_ff3 producer for the FF3 correlation cap.

Activates the dormant phase-40.8 cap by populating `factor_loadings` on
screener candidate dicts. Uses the existing math primitive at
`backend/services/portfolio_risk.py:58::compute_ff3` (OLS regression).

Scope (per CLAUDE.md "honest dual-interpretation"):
- IN-MEMORY only this cycle.
- BQ persistence + Kenneth French ingestion deferred to phase-40.8.2.
- This cycle uses a STUBBED factor-return generator (deterministic
  synthetic) so the pipeline can be tested end-to-end without an
  Internet fetch.

References:
- Kenneth French Data Library (canonical FF3 daily file)
- AQR "Measuring Factor Exposures" (Israel & Ross 2017)
- arXiv 2001.04185 (Volpati equity factor crowding)
"""

from __future__ import annotations

import math
import random
from typing import Mapping, Optional, Sequence

from backend.services.portfolio_risk import compute_ff3

FF3_FIELDS = ("market_beta", "smb_beta", "hml_beta")


def synthetic_ff3_returns(window_days: int = 60, seed: int = 4081) -> dict:
    rng = random.Random(seed)
    return {
        "Mkt-Rf": [rng.gauss(0.0005, 0.011) for _ in range(window_days)],
        "SMB": [rng.gauss(0.0001, 0.006) for _ in range(window_days)],
        "HML": [rng.gauss(0.0001, 0.005) for _ in range(window_days)],
    }


def _safe_returns(prices: Sequence[float]) -> list[float]:
    out: list[float] = []
    prev = None
    for p in prices:
        try:
            pf = float(p)
        except (TypeError, ValueError):
            continue
        if prev is not None and prev != 0:
            out.append(pf / prev - 1.0)
        prev = pf
    return out


def compute_candidate_loadings(
    candidates: list[dict],
    price_histories: Mapping[str, Sequence[float]],
    factor_returns: Optional[Mapping[str, Sequence[float]]] = None,
    window_days: int = 60,
) -> list[dict]:
    if factor_returns is None:
        factor_returns = synthetic_ff3_returns(window_days=window_days)
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        ticker = cand.get("ticker") or cand.get("symbol")
        if not ticker:
            cand["factor_loadings"] = None
            continue
        price_window = price_histories.get(ticker)
        if not price_window:
            cand["factor_loadings"] = None
            continue
        returns = _safe_returns(price_window)
        n = min(len(returns), window_days)
        if n < 4:
            cand["factor_loadings"] = None
            continue
        out = compute_ff3(
            returns[-n:],
            {
                "Mkt-Rf": list(factor_returns["Mkt-Rf"])[-n:],
                "SMB": list(factor_returns["SMB"])[-n:],
                "HML": list(factor_returns["HML"])[-n:],
            },
            rf=0.0,
        )
        if any(math.isnan(float(out.get(k, 0.0))) for k in FF3_FIELDS):
            cand["factor_loadings"] = None
            continue
        cand["factor_loadings"] = {
            "market_beta": float(out["market_beta"]),
            "smb_beta": float(out["smb_beta"]),
            "hml_beta": float(out["hml_beta"]),
        }
    return candidates
