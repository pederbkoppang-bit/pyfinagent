"""phase-40.8 (OPEN-5): Fama-French 3-factor correlation cap helper.

Augments the existing GICS sector cap (portfolio_manager.py) by catching
cross-sector factor crowding. Two stocks in different GICS sectors can
still be highly correlated by FF3 loadings (e.g., both high-momentum +
small-value); GICS alone misses this. Cosine similarity over the
(market_beta, smb_beta, hml_beta) vector identifies the crowding.

Default-OFF gate: settings.paper_max_factor_corr=0.0 disables.
Recommended live value: 0.85 (per cosine-similarity convention; no
canonical FF3-beta cap exists in AQR 2017 / Two Sigma 2025 reviewed).
Operator quiet-logs for 1-2 weeks before enabling.

Math primitive (compute_ff3) lives in backend/services/portfolio_risk.py:58.
This module is wiring + similarity scoring, not new regression math.

References:
- AQR "Measuring Factor Exposures: Uses and Abuses" (Israel & Ross 2017)
- arXiv 2001.04185 "Zooming In on Equity Factor Crowding" (Volpati 2020)
- Resonanz Capital "Crowding, Deleveraging" (2025)
"""

from __future__ import annotations

import math
from typing import Mapping, Optional

FF3_FIELDS = ("market_beta", "smb_beta", "hml_beta")


def factor_correlation_score(
    cand_loadings: Optional[Mapping[str, float]],
    port_loadings: Optional[Mapping[str, float]],
) -> float:
    if not cand_loadings or not port_loadings:
        return 0.0
    try:
        cand_vec = [float(cand_loadings[f]) for f in FF3_FIELDS]
        port_vec = [float(port_loadings[f]) for f in FF3_FIELDS]
    except (KeyError, TypeError, ValueError):
        return 0.0
    if any(math.isnan(v) for v in cand_vec) or any(math.isnan(v) for v in port_vec):
        return 0.0
    cand_norm = math.sqrt(sum(v * v for v in cand_vec))
    port_norm = math.sqrt(sum(v * v for v in port_vec))
    if cand_norm == 0.0 or port_norm == 0.0:
        return 0.0
    dot = sum(c * p for c, p in zip(cand_vec, port_vec))
    return max(-1.0, min(1.0, dot / (cand_norm * port_norm)))


def aggregate_portfolio_loadings(
    positions: list,
) -> dict:
    total_weight = 0.0
    accum = {f: 0.0 for f in FF3_FIELDS}
    for pos in positions:
        if not isinstance(pos, Mapping):
            continue
        ld = pos.get("factor_loadings")
        if not isinstance(ld, Mapping) or not ld:
            continue
        try:
            mv_f = float(pos.get("market_value", 0) or 0)
        except (TypeError, ValueError):
            continue
        if mv_f <= 0:
            continue
        try:
            vals = {f: float(ld[f]) for f in FF3_FIELDS}
        except (KeyError, TypeError, ValueError):
            continue
        if any(math.isnan(v) for v in vals.values()):
            continue
        for f in FF3_FIELDS:
            accum[f] += vals[f] * mv_f
        total_weight += mv_f
    if total_weight == 0.0:
        return {}
    return {f: accum[f] / total_weight for f in FF3_FIELDS}
